"""
L3/L4 API smoke test：验证最小鉴权与草稿链路。

一、测试目标
1. 验证写接口在无 API Key 时返回 401。
2. 验证带正确 API Key 时可正常创建工单。
3. 验证读接口仍保持开放，便于演示和排障。
4. 验证 `/agent` 能完成“创建草稿 -> 续办 -> 自动建单”的最小闭环。
5. 验证 L4-2 的幂等与草稿所有权约束已通过 API 暴露正确语义。

二、隔离策略
1. 使用独立的 SQLite 内存数据库。
2. 覆盖 `get_db` 依赖，不使用开发环境的会话工厂。
3. 关闭启动迁移，避免测试依赖外部数据库状态。
4. 对检索、LLM 和字段抽取做 monkeypatch，避免真实外部依赖。
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import src.api.app as api_app_module
from src.api.schemas import ToolPlan
from src.api.db import Base
from src.api.deps import get_db
from src.api import crud, services


_TEST_API_KEY = "test-api-key"


def _build_test_session() -> Session:
    """创建独立的内存数据库会话。"""
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return local_session()



def _patch_agent_dependencies(monkeypatch) -> None:
    """固定 `/agent` 相关外部依赖，避免调用真实检索和模型。"""
    monkeypatch.setattr(api_app_module, "ensure_schema_ready", lambda: None)
    monkeypatch.setattr(
        services,
        "retrieve",
        lambda text: [
            {
                "doc_id": "henu_network_manual",
                "page": 2,
                "score": 0.85,
                "snippet": "宿舍区网络报修需提供地点和联系方式。",
            }
        ],
    )
    monkeypatch.setattr(
        services,
        "answer_with_citations",
        lambda text, hits: {
            "answer": "请补充地点和联系方式后，我可以继续为你提交报修工单。",
            "citations": [
                {
                    "doc_id": "henu_network_manual",
                    "page": 2,
                    "snippet": "宿舍区网络报修需提供地点和联系方式。",
                }
            ],
            "meta": {
                "attempt_stage": "primary",
                "json_ok": True,
                "repair_used": False,
                "failure_reason": None,
            },
        },
    )
    monkeypatch.setattr(
        services,
        "extract_ticket_payload",
        lambda text, user, department: {
            "creator": user,
            "department": department,
            "category": "network",
            "priority": "P1",
            "title": "宿舍网络报修",
            "description": text or "补充信息",
            "contact": None,
            "location": None,
            "missing_fields": ["location", "contact"],
            "extractor": "rule_fallback",
        },
    )


def _set_test_api_key(monkeypatch) -> None:
    """为测试显式配置一个假的 API Key，避免依赖源码默认值。"""
    monkeypatch.setenv("POLICY_API_KEY", _TEST_API_KEY)



def test_ticket_api_requires_key_and_keeps_read_open(monkeypatch) -> None:
    """POST 应鉴权，GET 应开放，形成最小安全边界。"""
    db = _build_test_session()
    _set_test_api_key(monkeypatch)

    def _override_get_db():
        """把请求绑定到测试会话。"""
        try:
            yield db
        finally:
            pass

    monkeypatch.setattr(api_app_module, "ensure_schema_ready", lambda: None)
    api_app_module.app.dependency_overrides[get_db] = _override_get_db

    try:
        with TestClient(api_app_module.app) as client:
            no_key_response = client.post(
                "/tickets",
                json={
                    "title": "未带鉴权头",
                    "description": "应被拒绝",
                },
            )
            assert no_key_response.status_code == 401

            with_key_response = client.post(
                "/tickets",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "title": "带 Key 建单",
                    "description": "应成功创建",
                },
            )
            assert with_key_response.status_code == 200
            payload = with_key_response.json()
            assert str(payload["ticket_id"]).startswith("TCK-")

            list_response = client.get("/tickets")
            assert list_response.status_code == 200
            items = list_response.json()
            assert isinstance(items, list)
            assert len(items) == 1
            assert items[0]["title"] == "带 Key 建单"
    finally:
        api_app_module.app.dependency_overrides.clear()
        db.close()



def test_agent_api_supports_idempotent_draft_resume(monkeypatch) -> None:
    """同一草稿连续提交两次时，API 应返回同一 ticket_id。"""
    db = _build_test_session()
    _set_test_api_key(monkeypatch)

    def _override_get_db():
        """把请求绑定到测试会话。"""
        try:
            yield db
        finally:
            pass

    _patch_agent_dependencies(monkeypatch)
    api_app_module.app.dependency_overrides[get_db] = _override_get_db

    try:
        with TestClient(api_app_module.app) as client:
            first_response = client.post(
                "/agent",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "text": "我宿舍断网了，帮我报修工单。",
                    "user": "alice",
                    "department": "IT",
                },
            )
            assert first_response.status_code == 200
            first_payload = first_response.json()
            assert first_payload["route"] == "NEED_MORE_INFO"
            draft_id = str(first_payload["draft"]["draft_id"])

            second_response = client.post(
                "/agent",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "text": "",
                    "user": "alice",
                    "department": "IT",
                    "draft_id": draft_id,
                    "fields": {
                        "location": "金明校区",
                        "contact": "13812345678",
                    },
                },
            )
            third_response = client.post(
                "/agent",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "text": "",
                    "user": "alice",
                    "department": "IT",
                    "draft_id": draft_id,
                    "fields": {
                        "location": "金明校区",
                        "contact": "13812345678",
                    },
                },
            )

            assert second_response.status_code == 200
            assert third_response.status_code == 200
            second_payload = second_response.json()
            third_payload = third_response.json()
            assert second_payload["route"] == "CREATE_TICKET"
            assert third_payload["route"] == "CREATE_TICKET"
            assert second_payload["ticket"]["ticket_id"] == third_payload["ticket"]["ticket_id"]

            ticket = crud.get_ticket_by_public_id(db, second_payload["ticket"]["ticket_id"])
            assert ticket is not None
            assert ticket.source_draft_id == draft_id
    finally:
        api_app_module.app.dependency_overrides.clear()
        db.close()



def test_agent_api_hides_forbidden_draft_as_not_found(monkeypatch) -> None:
    """其他用户拿到 draft_id 时，API 应返回 404，避免暴露草稿存在性。"""
    db = _build_test_session()
    _set_test_api_key(monkeypatch)

    def _override_get_db():
        """把请求绑定到测试会话。"""
        try:
            yield db
        finally:
            pass

    _patch_agent_dependencies(monkeypatch)
    api_app_module.app.dependency_overrides[get_db] = _override_get_db

    try:
        with TestClient(api_app_module.app) as client:
            first_response = client.post(
                "/agent",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "text": "我宿舍断网了，帮我报修工单。",
                    "user": "alice",
                    "department": "IT",
                },
            )
            draft_id = str(first_response.json()["draft"]["draft_id"])

            forbidden_response = client.post(
                "/agent",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "text": "",
                    "user": "bob",
                    "department": "IT",
                    "draft_id": draft_id,
                    "fields": {
                        "location": "金明校区",
                        "contact": "13812345678",
                    },
                },
            )
            assert forbidden_response.status_code == 404
            assert forbidden_response.json()["detail"] == "draft_not_found"
    finally:
        api_app_module.app.dependency_overrides.clear()
        db.close()


def test_ticket_tool_endpoints_and_agent_lookup(monkeypatch) -> None:
    """L5 工具接口应可直接调用，Agent 也能路由到查单。"""
    db = _build_test_session()
    _set_test_api_key(monkeypatch)

    def _override_get_db():
        """把请求绑定到测试会话。"""
        try:
            yield db
        finally:
            pass

    monkeypatch.setattr(api_app_module, "ensure_schema_ready", lambda: None)
    api_app_module.app.dependency_overrides[get_db] = _override_get_db

    try:
        with TestClient(api_app_module.app) as client:
            create_response = client.post(
                "/tickets",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "title": "L5 测试工单",
                    "description": "用于验证工具调用接口",
                    "creator": "alice",
                },
            )
            assert create_response.status_code == 200
            ticket_id = str(create_response.json()["ticket_id"])

            comment_response = client.post(
                f"/tickets/{ticket_id}/comments",
                headers={"X-API-Key": _TEST_API_KEY},
                json={"actor": "alice", "comment": "补充：交换机在走廊"},
            )
            assert comment_response.status_code == 200
            assert [item["content"] for item in comment_response.json().get("comments") or []] == ["补充：交换机在走廊"]

            second_comment_response = client.post(
                f"/tickets/{ticket_id}/comments",
                headers={"X-API-Key": _TEST_API_KEY},
                json={"actor": "bob", "comment": "补充：配线间门已打开"},
            )
            assert second_comment_response.status_code == 200
            assert [item["content"] for item in second_comment_response.json().get("comments") or []] == [
                "补充：交换机在走廊",
                "补充：配线间门已打开",
            ]

            detail_response = client.get(f"/tickets/{ticket_id}")
            assert detail_response.status_code == 200
            assert [item["content"] for item in detail_response.json().get("comments") or []] == [
                "补充：交换机在走廊",
                "补充：配线间门已打开",
            ]

            escalate_response = client.post(
                f"/tickets/{ticket_id}/escalate",
                headers={"X-API-Key": _TEST_API_KEY},
                json={"actor": "alice", "reason": "请加急处理"},
            )
            assert escalate_response.status_code == 200
            assert escalate_response.json()["status"] == "in_progress"

            agent_lookup = client.post(
                "/agent",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "text": f"帮我查一下工单 {ticket_id} 的状态",
                    "user": "alice",
                    "department": "IT",
                },
            )
            assert agent_lookup.status_code == 200
            payload = agent_lookup.json()
            assert payload["route"] == "LOOKUP_TICKET"
            assert payload["ticket"]["ticket_id"] == ticket_id

            cancel_response = client.post(
                f"/tickets/{ticket_id}/cancel",
                headers={"X-API-Key": _TEST_API_KEY},
                json={"actor": "alice", "reason": "问题已自行恢复"},
            )
            assert cancel_response.status_code == 200
            assert cancel_response.json()["status"] == "cancelled"
    finally:
        api_app_module.app.dependency_overrides.clear()
        db.close()


def test_agent_api_llm_cancel_requires_confirmation_then_confirms(monkeypatch) -> None:
    """`/agent` 在 llm 模式下应返回 confirm_token，并可二次确认完成取消。"""
    db = _build_test_session()
    _set_test_api_key(monkeypatch)

    def _override_get_db():
        try:
            yield db
        finally:
            pass

    monkeypatch.setenv("AGENT_PLANNER_MODE", "llm")
    monkeypatch.setattr(api_app_module, "ensure_schema_ready", lambda: None)
    monkeypatch.setattr(
        services.planner,
        "run_global_planner",
        lambda user_text, tools_json, context: ToolPlan(
            tool="ticket_tool_planner",
            args={"ticket_id": str(context.get("provided_ticket_id") or ""), "raw_text": user_text},
            need_confirmation=False,
            missing_fields=[],
        ),
    )
    monkeypatch.setattr(
        services.planner,
        "run_ticket_tool_planner",
        lambda user_text, provided_ticket_id, tools_json: ToolPlan(
            tool="cancel_ticket",
            args={"ticket_id": provided_ticket_id, "reason": "问题已恢复"},
            need_confirmation=True,
            missing_fields=[],
        ),
    )
    api_app_module.app.dependency_overrides[get_db] = _override_get_db

    try:
        with TestClient(api_app_module.app) as client:
            create_response = client.post(
                "/tickets",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "title": "待取消工单",
                    "description": "验证 confirm_token API",
                    "creator": "alice",
                },
            )
            ticket_id = str(create_response.json()["ticket_id"])

            first_response = client.post(
                "/agent",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "text": f"请取消工单 {ticket_id}",
                    "user": "alice",
                    "department": "IT",
                },
            )
            assert first_response.status_code == 200
            first_payload = first_response.json()
            assert first_payload["route"] == "NEED_CONFIRMATION"
            confirm_token = str(first_payload["confirm_token"])

            second_response = client.post(
                "/agent",
                headers={"X-API-Key": _TEST_API_KEY},
                json={
                    "text": "确认取消",
                    "user": "alice",
                    "department": "IT",
                    "confirm_token": confirm_token,
                },
            )
            assert second_response.status_code == 200
            second_payload = second_response.json()
            assert second_payload["route"] == "CANCEL_TICKET"
            assert second_payload["ticket"]["status"] == "cancelled"
    finally:
        api_app_module.app.dependency_overrides.clear()
        db.close()
