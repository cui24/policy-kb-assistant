"""
L3/L4 服务层测试：覆盖 `services.py` 的核心工作流。

一、测试目标
1. 验证 `run_ask_workflow(...)` 会正确产出问答结果，并写入 `kb_queries` 与 `audit_logs`。
2. 验证 `create_ticket_workflow(...)` 会创建工单并留下 CREATE_TICKET 审计记录。
3. 验证 `run_agent_workflow(...)` 在直接建单路径上能串起 ASK、CREATE_TICKET、AGENT_ROUTE。
4. 验证 L4-2 的草稿续办具备幂等、所有权绑定与过期处理。

二、测试隔离策略
1. 使用独立的 SQLite 内存数据库。
2. 不连接开发环境的 Postgres。
3. 对检索、LLM 和字段抽取做 monkeypatch，避免真实外部依赖。
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import crud, models, planner as planner_module, planner_eval, services, skills
from src.api.db import Base
from src.api.schemas import ToolPlan



def _build_test_session() -> Session:
    """创建独立的内存数据库会话，避免污染开发库。"""
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
    """把检索、回答和字段抽取固定成稳定测试夹具。"""
    monkeypatch.setattr(
        services,
        "retrieve",
        lambda text: [
            {
                "doc_id": "henu_network_manual",
                "page": 2,
                "score": 0.86,
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



def test_run_ask_workflow_persists_query_and_audit(monkeypatch) -> None:
    """问答工作流应写入一条问答记录和一条 ASK 审计日志。"""
    db = _build_test_session()
    try:
        monkeypatch.setattr(
            services,
            "retrieve",
            lambda question: [
                {
                    "doc_id": "henu_network_manual",
                    "page": 5,
                    "score": 0.88,
                    "snippet": "统一身份认证登录地址 https://ids.henu.edu.cn",
                }
            ],
        )
        monkeypatch.setattr(
            services,
            "answer_with_citations",
            lambda question, hits: {
                "answer": "统一身份认证的登录地址是 https://ids.henu.edu.cn。",
                "citations": [
                    {
                        "doc_id": "henu_network_manual",
                        "page": 5,
                        "snippet": "统一身份认证登录地址 https://ids.henu.edu.cn",
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

        result = services.run_ask_workflow(
            db,
            question="统一身份认证的登录地址是什么？",
            user="alice",
            department="IT",
        )

        assert result["request_id"].startswith("req_")
        assert result["answer"].startswith("统一身份认证的登录地址")
        assert result["meta"]["valid_json"] is True

        kb_records = crud.list_kb_queries(db, request_id=result["request_id"], limit=10)
        assert len(kb_records) == 1
        assert kb_records[0].question == "统一身份认证的登录地址是什么？"

        audit_records = crud.list_audit_logs(db, request_id=result["request_id"], limit=10)
        assert len(audit_records) == 1
        assert audit_records[0].action_type == "ASK"
    finally:
        db.close()



def test_create_ticket_workflow_persists_ticket_and_audit() -> None:
    """建单工作流应创建工单，并留下 CREATE_TICKET 审计日志。"""
    db = _build_test_session()
    try:
        result = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="宿舍区无法上网",
            description="宿舍区断网，需要排查。",
            contact="13812345678",
            context={"location": "金明校区"},
            request_id="req_manual_ticket",
        )

        assert result["ticket_id"].startswith("TCK-")
        assert result["status"] == "open"

        ticket = crud.get_ticket_by_public_id(db, result["ticket_id"])
        assert ticket is not None
        assert ticket.title == "宿舍区无法上网"

        audit_records = crud.list_audit_logs(db, ticket_id=result["ticket_id"], limit=10)
        assert len(audit_records) == 1
        assert audit_records[0].action_type == "CREATE_TICKET"
        assert audit_records[0].request_id == "req_manual_ticket"
    finally:
        db.close()



def test_run_agent_workflow_create_ticket_path_writes_full_chain(monkeypatch) -> None:
    """Agent 直接建单路径应形成 ASK、CREATE_TICKET、AGENT_ROUTE 三段链路。"""
    db = _build_test_session()
    try:
        monkeypatch.setattr(
            services,
            "retrieve",
            lambda text: [
                {
                    "doc_id": "henu_network_manual",
                    "page": 3,
                    "score": 0.91,
                    "snippet": "宿舍区网络报修可联系服务大厅。",
                }
            ],
        )
        monkeypatch.setattr(
            services,
            "answer_with_citations",
            lambda text, hits: {
                "answer": "建议先检查认证状态，如仍异常可提交网络报修。",
                "citations": [
                    {
                        "doc_id": "henu_network_manual",
                        "page": 3,
                        "snippet": "宿舍区网络报修可联系服务大厅。",
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
                "description": text,
                "contact": "13812345678",
                "location": "金明校区",
                "missing_fields": [],
                "extractor": "rule_fallback",
            },
        )

        result = services.run_agent_workflow(
            db,
            text="我宿舍网络连不上，帮我提交报修工单。地点金明校区，手机号13812345678。",
            user="alice",
            department="IT",
        )

        assert result["route"] == "CREATE_TICKET"
        ticket_id = str(result["ticket"]["ticket_id"])
        assert ticket_id.startswith("TCK-")
        assert result["kb"]["request_id"].startswith("req_")

        audit_stmt = select(models.AuditLog).order_by(models.AuditLog.created_at.asc())
        audit_logs = db.execute(audit_stmt).scalars().all()
        action_types = [item.action_type for item in audit_logs]
        assert "ASK" in action_types
        assert "CREATE_TICKET" in action_types
        assert "AGENT_ROUTE" in action_types

        ticket = crud.get_ticket_by_public_id(db, ticket_id)
        assert ticket is not None
        assert ticket.context_json.get("kb_request_id") == result["kb"]["request_id"]
    finally:
        db.close()



def test_run_agent_workflow_draft_resume_is_idempotent(monkeypatch) -> None:
    """同一个 draft 连续续办两次时，应只创建一张工单并返回同一 ticket_id。"""
    db = _build_test_session()
    try:
        _patch_agent_dependencies(monkeypatch)

        first_result = services.run_agent_workflow(
            db,
            text="我宿舍断网了，帮我报修工单。",
            user="alice",
            department="IT",
        )
        draft_id = str(first_result["draft"]["draft_id"])

        second_result = services.run_agent_workflow(
            db,
            text="",
            user="alice",
            department="IT",
            draft_id=draft_id,
            fields={"location": "金明校区", "contact": "13812345678"},
        )
        third_result = services.run_agent_workflow(
            db,
            text="",
            user="alice",
            department="IT",
            draft_id=draft_id,
            fields={"location": "金明校区", "contact": "13812345678"},
        )

        first_ticket_id = str(second_result["ticket"]["ticket_id"])
        second_ticket_id = str(third_result["ticket"]["ticket_id"])
        assert first_ticket_id == second_ticket_id

        updated_draft = crud.get_ticket_draft_by_draft_id(db, draft_id)
        assert updated_draft is not None
        assert updated_draft.status == "consumed"

        tickets = db.execute(
            select(models.Ticket).where(models.Ticket.source_draft_id == draft_id)
        ).scalars().all()
        assert len(tickets) == 1

        chain_request_id = str(updated_draft.kb_request_id)
        audit_records = crud.list_audit_logs(db, request_id=chain_request_id, limit=30)
        action_types = [item.action_type for item in audit_records]
        assert "DRAFT_CREATED" in action_types
        assert "DRAFT_CONSUMED" in action_types
        assert "DRAFT_ALREADY_CONSUMED" in action_types
    finally:
        db.close()



def test_run_agent_workflow_draft_resume_rejects_foreign_user(monkeypatch) -> None:
    """草稿应绑定创建者，其他用户拿到 draft_id 也不能续办。"""
    db = _build_test_session()
    try:
        _patch_agent_dependencies(monkeypatch)

        first_result = services.run_agent_workflow(
            db,
            text="我宿舍断网了，帮我报修工单。",
            user="alice",
            department="IT",
        )
        draft_id = str(first_result["draft"]["draft_id"])

        with pytest.raises(PermissionError):
            services.run_agent_workflow(
                db,
                text="",
                user="bob",
                department="IT",
                draft_id=draft_id,
                fields={"location": "金明校区", "contact": "13812345678"},
            )

        draft = crud.get_ticket_draft_by_draft_id(db, draft_id)
        assert draft is not None
        chain_request_id = str(draft.kb_request_id)
        audit_records = crud.list_audit_logs(db, request_id=chain_request_id, limit=20)
        action_types = [item.action_type for item in audit_records]
        assert "DRAFT_FORBIDDEN" in action_types

        tickets = db.execute(select(models.Ticket)).scalars().all()
        assert len(tickets) == 0
    finally:
        db.close()



def test_run_agent_workflow_draft_resume_marks_expired(monkeypatch) -> None:
    """草稿过期后续办应返回 DRAFT_EXPIRED，并留下过期审计。"""
    db = _build_test_session()
    try:
        _patch_agent_dependencies(monkeypatch)

        first_result = services.run_agent_workflow(
            db,
            text="我宿舍断网了，帮我报修工单。",
            user="alice",
            department="IT",
        )
        draft_id = str(first_result["draft"]["draft_id"])

        draft = crud.get_ticket_draft_by_draft_id(db, draft_id)
        assert draft is not None
        draft.expires_at = services._utc_now() - timedelta(minutes=1)
        db.add(draft)
        db.commit()

        second_result = services.run_agent_workflow(
            db,
            text="",
            user="alice",
            department="IT",
            draft_id=draft_id,
            fields={"location": "金明校区", "contact": "13812345678"},
        )

        assert second_result["route"] == "DRAFT_EXPIRED"
        updated_draft = crud.get_ticket_draft_by_draft_id(db, draft_id)
        assert updated_draft is not None
        assert updated_draft.status == "expired"

        chain_request_id = str(updated_draft.kb_request_id)
        audit_records = crud.list_audit_logs(db, request_id=chain_request_id, limit=20)
        action_types = [item.action_type for item in audit_records]
        assert "DRAFT_EXPIRED" in action_types
    finally:
        db.close()


def test_run_agent_workflow_rules_infers_recent_draft_from_short_term_memory(monkeypatch) -> None:
    """短期记忆应能在未显式传入 draft_id 时恢复最近一次草稿。"""
    db = _build_test_session()
    try:
        _patch_agent_dependencies(monkeypatch)

        first_result = services.run_agent_workflow(
            db,
            text="我宿舍断网了，帮我报修工单。",
            user="alice",
            department="IT",
        )
        assert first_result["route"] == "NEED_MORE_INFO"
        remembered_draft_id = str(first_result["draft"]["draft_id"])

        second_result = services.run_agent_workflow(
            db,
            text="继续刚才那个问题",
            user="alice",
            department="IT",
            fields={"location": "金明校区", "contact": "13812345678"},
        )

        assert second_result["route"] == "CREATE_TICKET"
        assert second_result["draft"]["draft_id"] == remembered_draft_id

        memory_record = crud.get_agent_conversation_memory(db, "alice")
        assert memory_record is not None
        assert memory_record.last_ticket_id == second_result["ticket"]["ticket_id"]
        assert memory_record.last_draft_id is None
    finally:
        db.close()


def test_run_agent_workflow_prompts_for_ticket_id_when_reference_cannot_be_resolved() -> None:
    """引用“上一单/那张单”但没有可恢复对象时，应直接追问 ticket_id。"""
    db = _build_test_session()
    try:
        result = services.run_agent_workflow(
            db,
            text="上一单补一条信息",
            user="alice",
            department="IT",
        )

        assert result["route"] == "NEED_MORE_INFO"
        assert result["missing_fields"] == ["ticket_id"]
        assert "我没找到你提到的“上一单”" in str(result["message"])
        assert "TCK-2026-XXXXXX" in str(result["message"])
    finally:
        db.close()


def test_ticket_tool_workflows_append_comments_and_update_context() -> None:
    """评论应追加到独立表，催办/取消仍应更新工单上下文。"""
    db = _build_test_session()
    try:
        created = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="测试工单",
            description="用于验证 L5 工具调用。",
            request_id="req_tool_flow",
        )
        ticket_id = str(created["ticket_id"])

        first_commented = services.add_ticket_comment_workflow(db, ticket_id, "请补充交换机位置。", "alice")
        second_commented = services.add_ticket_comment_workflow(db, ticket_id, "交换机在走廊东侧。", "bob")
        assert [item["content"] for item in second_commented["comments"]] == [
            "请补充交换机位置。",
            "交换机在走廊东侧。",
        ]
        assert "comments" not in (second_commented.get("context") or {})
        assert first_commented["updated_at"] <= second_commented["updated_at"]

        ticket = crud.get_ticket_by_public_id(db, ticket_id)
        assert ticket is not None
        comment_rows = list(crud.list_ticket_comments(db, ticket.id, limit=10))
        assert len(comment_rows) == 2
        assert [row.content for row in reversed(comment_rows)] == [
            "请补充交换机位置。",
            "交换机在走廊东侧。",
        ]

        escalated = services.escalate_ticket_workflow(db, ticket_id, "alice", "需要加急处理")
        escalated_again = services.escalate_ticket_workflow(db, ticket_id, "alice", "再次催办")
        assert escalated["status"] == "in_progress"
        assert int((escalated_again.get("context") or {}).get("escalation_count") or 0) == 2

        cancelled = services.cancel_ticket_workflow(db, ticket_id, "alice", "用户已自行恢复")
        assert cancelled["status"] == "cancelled"
        assert (cancelled.get("context") or {}).get("cancel_reason") == "用户已自行恢复"
    finally:
        db.close()


def test_ticket_tool_skill_registry_exposes_contract_and_dispatches_by_name() -> None:
    """统一 registry 应同时暴露 global 分支工具与既有工单工具，ticket 工具仍可按名分发。"""
    db = _build_test_session()
    try:
        global_skill_items = services.list_global_planner_skill_contracts()
        assert [item["name"] for item in global_skill_items] == [
            "continue_ticket_draft",
            "ticket_tool_planner",
            "kb_answer",
            "create_ticket",
        ]
        assert all(item["planner_scope"] == "global" for item in global_skill_items)

        skill_items = services.list_ticket_tool_skill_contracts()
        assert [item["name"] for item in skill_items] == [
            "lookup_ticket",
            "add_ticket_comment",
            "escalate_ticket",
            "cancel_ticket",
        ]
        assert all(item["planner_scope"] == "ticket" for item in skill_items)

        cancel_skill = next(item for item in skill_items if item["name"] == "cancel_ticket")
        assert cancel_skill["risk_level"] == "HIGH"
        assert cancel_skill["auth_rule"] == "owner_or_admin"
        assert cancel_skill["input_schema"]["required"] == ["ticket_id", "reason"]

        created = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="注册表分发测试",
            description="用于验证 registry dispatch。",
        )
        ticket_id = str(created["ticket_id"])

        registry = skills.get_agent_skill_registry()
        assert registry.get("kb_answer") is not None
        lookup_skill = registry.get_by_route("LOOKUP_TICKET")
        assert lookup_skill is not None
        assert lookup_skill.name == "lookup_ticket"

        ticket_detail = registry.dispatch(
            "lookup_ticket",
            db=db,
            args={"ticket_id": ticket_id},
            actor="alice",
            raw_text=f"帮我查一下工单 {ticket_id}",
        )
        assert ticket_detail["ticket_id"] == ticket_id
        assert ticket_detail["status"] == "open"
    finally:
        db.close()


def test_run_agent_workflow_routes_existing_ticket_tools() -> None:
    """Agent 应能识别既有工单工具调用：查单、补充、催办、取消。"""
    db = _build_test_session()
    try:
        created = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="测试工单",
            description="用于验证 Agent 工具路由。",
        )
        ticket_id = str(created["ticket_id"])

        lookup = services.run_agent_workflow(db, text=f"帮我查一下工单 {ticket_id} 的进度", user="alice", department="IT")
        assert lookup["route"] == "LOOKUP_TICKET"
        assert lookup["ticket"]["ticket_id"] == ticket_id

        comment = services.run_agent_workflow(db, text=f"请给工单 {ticket_id} 补充说明 交换机在走廊", user="alice", department="IT")
        assert comment["route"] == "ADD_TICKET_COMMENT"

        escalate = services.run_agent_workflow(db, text=f"请催办工单 {ticket_id}", user="alice", department="IT")
        assert escalate["route"] == "ESCALATE_TICKET"
        assert escalate["ticket"]["status"] == "in_progress"

        cancel = services.run_agent_workflow(db, text=f"请取消工单 {ticket_id}，因为已经恢复", user="alice", department="IT")
        assert cancel["route"] == "CANCEL_TICKET"
        assert cancel["ticket"]["status"] == "cancelled"
    finally:
        db.close()


def test_ticket_tool_planner_repairs_invalid_json(monkeypatch) -> None:
    """Planner 首次输出损坏 JSON 时，应能通过修复 prompt 得到合法 ToolPlan。"""
    call_outputs = iter(
        [
            "tool=lookup_ticket, args=ticket_id",
            '{"tool":"lookup_ticket","args":{"ticket_id":"TCK-2026-ABC123"},"need_confirmation":false,"missing_fields":[]}',
        ]
    )

    monkeypatch.setattr(planner_module, "_call_planner_llm", lambda system_prompt, user_prompt: next(call_outputs))

    plan = planner_module.run_ticket_tool_planner(
        user_text="帮我查一下工单 TCK-2026-ABC123",
        provided_ticket_id="TCK-2026-ABC123",
        tools_json=services.list_ticket_tool_skill_contracts(),
    )

    assert plan.tool == "lookup_ticket"
    assert plan.args["ticket_id"] == "TCK-2026-ABC123"


def test_ticket_tool_planner_supports_langchain_structured_backend(monkeypatch) -> None:
    """设置 `langchain_structured` 后，应直接走结构化后端而不是 raw repair 分支。"""
    monkeypatch.setenv("AGENT_PLANNER_BACKEND", "langchain_structured")
    monkeypatch.setattr(
        planner_module,
        "_call_planner_langchain_structured",
        lambda system_prompt, user_prompt: ToolPlan(
            tool="lookup_ticket",
            args={"ticket_id": "TCK-2026-ABC123"},
            need_confirmation=False,
            missing_fields=[],
        ),
    )

    plan = planner_module.run_ticket_tool_planner(
        user_text="帮我查一下工单 TCK-2026-ABC123",
        provided_ticket_id="TCK-2026-ABC123",
        tools_json=services.list_ticket_tool_skill_contracts(),
    )

    assert plan.tool == "lookup_ticket"
    assert plan.args["ticket_id"] == "TCK-2026-ABC123"


def test_global_planner_supports_langchain_structured_backend(monkeypatch) -> None:
    """Global Planner 也应支持 `langchain_structured` 后端。"""
    monkeypatch.setenv("AGENT_PLANNER_BACKEND", "langchain_structured")
    monkeypatch.setattr(
        planner_module,
        "_call_planner_langchain_structured",
        lambda system_prompt, user_prompt: ToolPlan(
            tool="kb_answer",
            args={"query": "统一身份认证怎么登录？"},
            need_confirmation=False,
            missing_fields=[],
        ),
    )

    plan = planner_module.run_global_planner(
        user_text="统一身份认证怎么登录？",
        tools_json=services.list_global_planner_skill_contracts(),
        context={"has_ticket_id": False, "has_draft_id": False},
    )

    assert plan.tool == "kb_answer"
    assert plan.args["query"] == "统一身份认证怎么登录？"


def test_ticket_tool_planner_supports_langchain_tools_backend(monkeypatch) -> None:
    """设置 `langchain_tools` 后，ticket 子 Planner 应走 tool-calling 后端。"""
    monkeypatch.setenv("AGENT_PLANNER_BACKEND", "langchain_tools")
    monkeypatch.setattr(
        planner_module,
        "_call_planner_langchain_tools",
        lambda system_prompt, user_prompt, tools_json: ToolPlan(
            tool="cancel_ticket",
            args={"ticket_id": "TCK-2026-ABC123", "reason": "已经恢复"},
            need_confirmation=True,
            missing_fields=[],
        ),
    )

    plan = planner_module.run_ticket_tool_planner(
        user_text="帮我取消工单 TCK-2026-ABC123，因为已经恢复",
        provided_ticket_id="TCK-2026-ABC123",
        tools_json=services.list_ticket_tool_skill_contracts(),
    )

    assert plan.tool == "cancel_ticket"
    assert plan.need_confirmation is True
    assert plan.args["ticket_id"] == "TCK-2026-ABC123"


def test_global_planner_supports_langchain_tools_backend(monkeypatch) -> None:
    """Global Planner 也应支持 `langchain_tools` 后端。"""
    monkeypatch.setenv("AGENT_PLANNER_BACKEND", "langchain_tools")
    monkeypatch.setattr(
        planner_module,
        "_call_planner_langchain_tools",
        lambda system_prompt, user_prompt, tools_json: ToolPlan(
            tool="ticket_tool_planner",
            args={"ticket_id": "TCK-2026-ABC123", "raw_text": "TCK-2026-ABC123 太慢了"},
            need_confirmation=False,
            missing_fields=[],
        ),
    )

    plan = planner_module.run_global_planner(
        user_text="TCK-2026-ABC123 太慢了",
        tools_json=services.list_global_planner_skill_contracts(),
        context={"has_ticket_id": True, "provided_ticket_id": "TCK-2026-ABC123"},
    )

    assert plan.tool == "ticket_tool_planner"
    assert plan.args["ticket_id"] == "TCK-2026-ABC123"


def test_global_planner_langchain_tools_narrows_candidates_for_follow_up(monkeypatch) -> None:
    """无对象的跟进表达应收窄候选集，避免把 `create_ticket` 暴露给 tool-calling。"""
    monkeypatch.setenv("AGENT_PLANNER_BACKEND", "langchain_tools")
    captured: dict[str, list[str]] = {}

    def _fake_langchain_tools(system_prompt, user_prompt, tools_json):
        captured["tool_names"] = [str(item.get("name") or "") for item in tools_json]
        return ToolPlan(
            tool="ticket_tool_planner",
            args={"ticket_id": "", "raw_text": "给工单补个联系方式：138xxxx"},
            need_confirmation=False,
            missing_fields=[],
        )

    monkeypatch.setattr(planner_module, "_call_planner_langchain_tools", _fake_langchain_tools)

    planner_module.run_global_planner(
        user_text="给工单补个联系方式：138xxxx",
        tools_json=services.list_global_planner_skill_contracts(),
        context={"has_ticket_id": False, "has_draft_id": False, "ticket_tool_mode": False, "draft_mode": False},
    )

    assert captured["tool_names"] == ["ticket_tool_planner"]
    assert "create_ticket" not in captured["tool_names"]


def test_global_planner_langchain_tools_keeps_create_ticket_for_new_request(monkeypatch) -> None:
    """明确报修场景下，候选集仍应保留建单工具。"""
    monkeypatch.setenv("AGENT_PLANNER_BACKEND", "langchain_tools")
    captured: dict[str, list[str]] = {}

    def _fake_langchain_tools(system_prompt, user_prompt, tools_json):
        captured["tool_names"] = [str(item.get("name") or "") for item in tools_json]
        return ToolPlan(
            tool="create_ticket",
            args={"text": "我宿舍断网了，帮我报修", "fields": {}},
            need_confirmation=False,
            missing_fields=[],
        )

    monkeypatch.setattr(planner_module, "_call_planner_langchain_tools", _fake_langchain_tools)

    planner_module.run_global_planner(
        user_text="我宿舍断网了，帮我报修",
        tools_json=services.list_global_planner_skill_contracts(),
        context={"has_ticket_id": False, "has_draft_id": False, "ticket_tool_mode": False, "draft_mode": False},
    )

    assert captured["tool_names"] == ["create_ticket"]


def test_global_planner_tool_doc_rag_retrieves_relevant_docs() -> None:
    """Global Planner 的 tool-doc RAG 应能为不同输入返回更相关的工具说明。"""
    ticket_docs = planner_module.retrieve_global_planner_tool_docs(
        user_text="TCK-2026-AB12 太慢了，能不能快点",
        context={"has_ticket_id": True, "ticket_tool_mode": True},
        top_k=2,
    )
    assert [item["tool_name"] for item in ticket_docs[:1]] == ["ticket_tool_planner"]

    kb_docs = planner_module.retrieve_global_planner_tool_docs(
        user_text="宿舍网络报修流程是什么？",
        context={"has_ticket_id": False, "ticket_tool_mode": False},
        top_k=2,
    )
    assert [item["tool_name"] for item in kb_docs[:1]] == ["kb_answer"]


def test_global_planner_regression_cases_are_available() -> None:
    """模糊表达回归集应已落地，便于后续批量回归。"""
    cases = planner_module.load_global_planner_regression_cases()
    assert len(cases) == 30
    assert cases[0]["category"] == "省略对象/指代"
    assert cases[-1]["expected_tool"] == "ticket_tool_planner"


def test_evaluate_global_planner_cases_rules_strategy_reports_expected_matches() -> None:
    """离线评测在 rules 策略下应能对简单样例给出稳定统计。"""
    report = planner_eval.evaluate_global_planner_cases(
        [
            {
                "utterance": "统一身份认证登录流程是什么？",
                "category": "kb",
                "expected_tool": "kb_answer",
                "notes": "应走知识问答。",
            },
            {
                "utterance": "TCK-2026-AB12 太慢了",
                "category": "ticket",
                "expected_tool": "escalate_ticket",
                "notes": "全局层面应先进入 ticket_tool_planner。",
            },
        ],
        strategy="rules",
    )

    summary = report["summary"]
    assert summary["total_cases"] == 2
    assert summary["effective_valid_plan_count"] == 2
    assert summary["effective_branch_match_count"] == 2
    assert summary["fallback_count"] == 0


def test_evaluate_global_planner_cases_hybrid_reports_fallback(monkeypatch) -> None:
    """hybrid 评测在 planner 失败时应统计 fallback。"""
    monkeypatch.setattr(
        planner_eval.planner,
        "run_global_planner",
        lambda user_text, tools_json, context: (_ for _ in ()).throw(
            planner_module.PlannerError("planner_failed", code="global_repair_failed", fallback_eligible=True)
        ),
    )

    report = planner_eval.evaluate_global_planner_cases(
        [
            {
                "utterance": "统一身份认证登录流程是什么？",
                "category": "kb",
                "expected_tool": "kb_answer",
                "notes": "",
            }
        ],
        strategy="hybrid",
    )

    summary = report["summary"]
    assert summary["total_cases"] == 1
    assert summary["fallback_count"] == 1
    assert summary["effective_branch_match_count"] == 1
    assert summary["planner_error_count"] == 1


def test_evaluate_agent_workflow_cases_accepts_clarification_for_missing_ticket_reference() -> None:
    """workflow 级评测应把无对象时的 `NEED_MORE_INFO` 记为正确澄清。"""
    report = planner_eval.evaluate_agent_workflow_cases(
        [
            {
                "utterance": "上一单补一条信息",
                "category": "省略对象/指代",
                "expected_tool": "add_ticket_comment",
                "notes": "缺 ticket_id 时应追问。",
            }
        ],
        strategy="rules",
    )

    summary = report["summary"]
    row = report["results"][0]
    assert summary["evaluation_level"] == "workflow"
    assert summary["route_match_count"] == 1
    assert summary["clarification_match_count"] == 1
    assert row["response_route"] == "NEED_MORE_INFO"
    assert row["matched"] is True


def test_evaluate_agent_workflow_cases_seeds_explicit_ticket_for_lookup() -> None:
    """workflow 级评测应预置显式工单对象，让查单样例走到最终 route。"""
    report = planner_eval.evaluate_agent_workflow_cases(
        [
            {
                "utterance": "查一下 TCK-2026-AB12 的状态",
                "category": "ticket",
                "expected_tool": "lookup_ticket",
                "notes": "应命中查单。",
            }
        ],
        strategy="rules",
    )

    summary = report["summary"]
    row = report["results"][0]
    assert summary["route_match_count"] == 1
    assert row["response_route"] == "LOOKUP_TICKET"
    assert row["matched"] is True


def test_run_agent_workflow_llm_planner_executes_ticket_tool_and_logs_plan_events(monkeypatch) -> None:
    """`llm` 模式下，ticket 工具应走 Planner -> Validator -> 执行，并写计划审计。"""
    db = _build_test_session()
    try:
        monkeypatch.setenv("AGENT_PLANNER_MODE", "llm")
        created = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="LLM 查单测试",
            description="验证 Planner 路径。",
        )
        ticket_id = str(created["ticket_id"])
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
                tool="lookup_ticket",
                args={"ticket_id": provided_ticket_id},
                need_confirmation=False,
                missing_fields=[],
            ),
        )

        result = services.run_agent_workflow(
            db,
            text=f"帮我看下工单 {ticket_id} 的状态",
            user="alice",
            department="IT",
        )

        assert result["route"] == "LOOKUP_TICKET"
        audit_records = crud.list_audit_logs(db, ticket_id=ticket_id, limit=20)
        action_types = [item.action_type for item in audit_records]
        assert "PLAN_PROPOSED" in action_types
        assert "PLAN_EXECUTED" in action_types
    finally:
        db.close()


def test_run_agent_workflow_llm_infers_recent_ticket_from_short_term_memory(monkeypatch) -> None:
    """短期记忆应能在未显式给出 ticket_id 时恢复最近一次工单。"""
    db = _build_test_session()
    try:
        monkeypatch.setenv("AGENT_PLANNER_MODE", "llm")
        created = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="短期记忆工单",
            description="用于验证 recent ticket 引用恢复。",
        )
        ticket_id = str(created["ticket_id"])

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

        def _fake_ticket_tool_planner(user_text, provided_ticket_id, tools_json):
            if "查一下" in user_text:
                return ToolPlan(
                    tool="lookup_ticket",
                    args={"ticket_id": provided_ticket_id},
                    need_confirmation=False,
                    missing_fields=[],
                )
            return ToolPlan(
                tool="escalate_ticket",
                args={"ticket_id": provided_ticket_id, "reason": "用户请求催办。"},
                need_confirmation=False,
                missing_fields=[],
            )

        monkeypatch.setattr(services.planner, "run_ticket_tool_planner", _fake_ticket_tool_planner)

        first = services.run_agent_workflow(
            db,
            text=f"帮我查一下工单 {ticket_id} 的状态",
            user="alice",
            department="IT",
        )
        assert first["route"] == "LOOKUP_TICKET"

        second = services.run_agent_workflow(
            db,
            text="帮我催一下之前那个",
            user="alice",
            department="IT",
        )
        assert second["route"] == "ESCALATE_TICKET"
        assert second["ticket"]["ticket_id"] == ticket_id

        memory_record = crud.get_agent_conversation_memory(db, "alice")
        assert memory_record is not None
        assert memory_record.last_ticket_id == ticket_id
    finally:
        db.close()


def test_run_agent_workflow_create_ticket_uses_long_term_memory_defaults(monkeypatch) -> None:
    """第二次建单缺少 location/contact 时，应先用长期记忆补齐，避免再次追问。"""
    db = _build_test_session()
    try:
        monkeypatch.setenv("AGENT_PLANNER_MODE", "rules")
        monkeypatch.setattr(
            services,
            "retrieve",
            lambda text: [
                {
                    "doc_id": "henu_network_manual",
                    "page": 3,
                    "score": 0.9,
                    "snippet": "网络报修建议提供地点和联系方式。",
                }
            ],
        )
        monkeypatch.setattr(
            services,
            "answer_with_citations",
            lambda text, hits: {
                "answer": "可以提交网络报修。",
                "citations": [
                    {
                        "doc_id": "henu_network_manual",
                        "page": 3,
                        "snippet": "网络报修建议提供地点和联系方式。",
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

        def _extract_with_full_fields(text, user, department):
            return {
                "creator": user,
                "department": department,
                "category": "network",
                "priority": "P1",
                "title": "宿舍网络报修",
                "description": text,
                "contact": "13812345678",
                "location": "金明校区 3 号楼",
                "missing_fields": [],
                "extractor": "rule_fallback",
            }

        monkeypatch.setattr(services, "extract_ticket_payload", _extract_with_full_fields)
        first = services.run_agent_workflow(
            db,
            text="我宿舍网络断了，帮我提交工单。",
            user="alice",
            department="IT",
        )
        assert first["route"] == "CREATE_TICKET"

        memory_record = crud.get_user_memory(db, "alice")
        assert memory_record is not None
        assert memory_record.default_contact == "13812345678"
        assert memory_record.default_location == "金明校区 3 号楼"

        monkeypatch.setattr(
            services,
            "extract_ticket_payload",
            lambda text, user, department: {
                "creator": user,
                "department": department,
                "category": "network",
                "priority": "P1",
                "title": "宿舍网络报修",
                "description": text,
                "contact": None,
                "location": None,
                "missing_fields": ["location", "contact"],
                "extractor": "rule_fallback",
            },
        )

        second = services.run_agent_workflow(
            db,
            text="网络又断了，帮我再提交报修工单。",
            user="alice",
            department="IT",
        )

        assert second["route"] == "CREATE_TICKET"
        assert second["memory_applied"] == {
            "location": "金明校区 3 号楼",
            "contact": "13812345678",
            "source": "user_memory",
        }
        assert "已沿用上次地点和联系方式作为默认信息。" in str(second["message"])
        second_ticket = crud.get_ticket_by_public_id(db, str(second["ticket"]["ticket_id"]))
        assert second_ticket is not None
        assert second_ticket.contact == "13812345678"
        assert str((second_ticket.context_json or {}).get("location") or "") == "金明校区 3 号楼"
    finally:
        db.close()


def test_run_agent_workflow_llm_global_planner_routes_to_kb_answer(monkeypatch) -> None:
    """Global Planner 选中 `kb_answer` 时，应走 ASK 路径并保留问答链路。"""
    db = _build_test_session()
    try:
        monkeypatch.setenv("AGENT_PLANNER_MODE", "llm")
        monkeypatch.setattr(
            services.planner,
            "run_global_planner",
            lambda user_text, tools_json, context: ToolPlan(
                tool="kb_answer",
                args={"query": user_text},
                need_confirmation=False,
                missing_fields=[],
            ),
        )
        monkeypatch.setattr(
            services,
            "retrieve",
            lambda text: [
                {
                    "doc_id": "policy_manual",
                    "page": 1,
                    "score": 0.92,
                    "snippet": "统一身份认证登录地址请访问 ids.henu.edu.cn。",
                }
            ],
        )
        monkeypatch.setattr(
            services,
            "answer_with_citations",
            lambda text, hits: {
                "answer": "统一身份认证登录地址是 ids.henu.edu.cn。",
                "citations": [
                    {
                        "doc_id": "policy_manual",
                        "page": 1,
                        "snippet": "统一身份认证登录地址请访问 ids.henu.edu.cn。",
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

        result = services.run_agent_workflow(
            db,
            text="统一身份认证怎么登录？",
            user="alice",
            department="IT",
        )

        assert result["route"] == "ASK"
        assert "ids.henu.edu.cn" in str(result["kb"]["answer"])
    finally:
        db.close()


def test_run_agent_workflow_llm_global_planner_routes_to_create_ticket(monkeypatch) -> None:
    """Global Planner 选中 `create_ticket` 时，应复用既有建单链路，并让保守 fields 填补缺口。"""
    db = _build_test_session()
    try:
        monkeypatch.setenv("AGENT_PLANNER_MODE", "llm")
        monkeypatch.setattr(
            services.planner,
            "run_global_planner",
            lambda user_text, tools_json, context: ToolPlan(
                tool="create_ticket",
                args={
                    "text": user_text,
                    "fields": {"location": "金明校区", "contact": "13812345678"},
                },
                need_confirmation=False,
                missing_fields=[],
            ),
        )
        monkeypatch.setattr(
            services,
            "retrieve",
            lambda text: [
                {
                    "doc_id": "henu_network_manual",
                    "page": 4,
                    "score": 0.89,
                    "snippet": "网络异常可提交报修工单。",
                }
            ],
        )
        monkeypatch.setattr(
            services,
            "answer_with_citations",
            lambda text, hits: {
                "answer": "我可以继续为你提交网络报修。",
                "citations": [
                    {
                        "doc_id": "henu_network_manual",
                        "page": 4,
                        "snippet": "网络异常可提交报修工单。",
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
                "description": text,
                "contact": None,
                "location": None,
                "missing_fields": ["location", "contact"],
                "extractor": "rule_fallback",
            },
        )

        result = services.run_agent_workflow(
            db,
            text="宿舍网络断了，麻烦处理一下",
            user="alice",
            department="IT",
        )

        assert result["route"] == "CREATE_TICKET"
        ticket_id = str(result["ticket"]["ticket_id"])
        ticket = crud.get_ticket_by_public_id(db, ticket_id)
        assert ticket is not None
        assert ticket.contact == "13812345678"
        assert str((ticket.context_json or {}).get("location") or "") == "金明校区"
    finally:
        db.close()


def test_run_agent_workflow_hybrid_falls_back_to_rules_when_planner_fails(monkeypatch) -> None:
    """`hybrid` 模式下，Planner 失败应回退到旧规则路由。"""
    db = _build_test_session()
    try:
        monkeypatch.setenv("AGENT_PLANNER_MODE", "hybrid")
        created = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="Hybrid 回退测试",
            description="验证 fallback。",
        )
        ticket_id = str(created["ticket_id"])
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
            lambda user_text, provided_ticket_id, tools_json: (_ for _ in ()).throw(
                planner_module.PlannerError("planner_failed", code="repair_failed", fallback_eligible=True)
            ),
        )

        result = services.run_agent_workflow(
            db,
            text=f"请给工单 {ticket_id} 补充说明 交换机在走廊",
            user="alice",
            department="IT",
        )

        assert result["route"] == "ADD_TICKET_COMMENT"
        assert [item["content"] for item in result["ticket_detail"]["comments"]] == ["请给工单 补充说明 交换机在走廊"]
        audit_records = crud.list_audit_logs(db, ticket_id=ticket_id, limit=20)
        assert "PLAN_REJECTED" in [item.action_type for item in audit_records]
    finally:
        db.close()


def test_run_agent_workflow_cancel_requires_confirmation_then_executes(monkeypatch) -> None:
    """高风险取消应先发放 confirm_token，二次提交后才真正执行。"""
    db = _build_test_session()
    try:
        monkeypatch.setenv("AGENT_PLANNER_MODE", "llm")
        created = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="取消确认测试",
            description="验证 confirm_token。",
        )
        ticket_id = str(created["ticket_id"])
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
                args={"ticket_id": provided_ticket_id, "reason": "已经恢复"},
                need_confirmation=True,
                missing_fields=[],
            ),
        )

        first = services.run_agent_workflow(
            db,
            text=f"请取消工单 {ticket_id}，因为已经恢复",
            user="alice",
            department="IT",
        )
        assert first["route"] == "NEED_CONFIRMATION"
        confirm_token = str(first["confirm_token"])
        pending_action = crud.get_pending_action_by_confirm_id(db, confirm_token)
        assert pending_action is not None
        assert pending_action.status == "pending"

        second = services.run_agent_workflow(
            db,
            text="确认取消",
            user="alice",
            department="IT",
            confirm_token=confirm_token,
        )
        assert second["route"] == "CANCEL_TICKET"
        assert second["ticket"]["status"] == "cancelled"

        refreshed_pending = crud.get_pending_action_by_confirm_id(db, confirm_token)
        assert refreshed_pending is not None
        assert refreshed_pending.status == "consumed"
    finally:
        db.close()


def test_run_agent_workflow_llm_planner_rejects_unauthorized_cancel(monkeypatch) -> None:
    """`owner_or_admin` 未通过时，Validator 应拦截取消。"""
    db = _build_test_session()
    try:
        monkeypatch.setenv("AGENT_PLANNER_MODE", "llm")
        created = services.create_ticket_workflow(
            db,
            creator="alice",
            department="IT",
            category="network",
            priority="P1",
            title="权限拒绝测试",
            description="验证 cancel 鉴权。",
        )
        ticket_id = str(created["ticket_id"])
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
                args={"ticket_id": provided_ticket_id, "reason": "不是本人"},
                need_confirmation=True,
                missing_fields=[],
            ),
        )

        result = services.run_agent_workflow(
            db,
            text=f"请取消工单 {ticket_id}",
            user="bob",
            department="IT",
        )

        assert result["route"] == "PLAN_REJECTED"
        ticket = crud.get_ticket_by_public_id(db, ticket_id)
        assert ticket is not None
        assert ticket.status == "open"
    finally:
        db.close()
