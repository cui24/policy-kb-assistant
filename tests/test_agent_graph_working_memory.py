from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.agent_graph import run_agent_graph
from src.api import ask_pipeline, models
from src.api.db import Base


def _patch_in_memory_ask_cache(monkeypatch) -> dict:
    store: dict[str, dict] = {}

    def _fake_set_json(key: str, value: dict, **kwargs) -> bool:
        store[key] = value
        return True

    monkeypatch.setattr(ask_pipeline.ask_cache.redis_client, "get_json", lambda key: store.get(key))
    monkeypatch.setattr(ask_pipeline.ask_cache.redis_client, "set_json", _fake_set_json)
    return store


def _build_test_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return local_session()


def _working_memory_audits(db: Session) -> list[models.AuditLog]:
    stmt = (
        select(models.AuditLog)
        .where(models.AuditLog.action_type == "AGENT_WORKING_MEMORY")
        .order_by(models.AuditLog.created_at.asc())
    )
    return list(db.execute(stmt).scalars().all())


def test_agent_graph_l0_working_memory_audits_ask_path(monkeypatch) -> None:
    monkeypatch.setenv("ASK_CACHE_ENABLED", "false")
    db = _build_test_session()
    try:
        monkeypatch.setattr(
            ask_pipeline,
            "run_retrieve_step",
            lambda question: (
                [
                    {
                        "doc_id": "henu_network_manual",
                        "page": 5,
                        "score": 0.88,
                        "snippet": "统一身份认证登录地址 https://ids.henu.edu.cn",
                    }
                ],
                12,
            ),
        )
        monkeypatch.setattr(
            ask_pipeline,
            "run_answer_step",
            lambda question, hits: (
                {
                    "answer": "统一身份认证的登录地址是 https://ids.henu.edu.cn。",
                    "citations": [
                        {
                            "doc_id": "henu_network_manual",
                            "page": 5,
                            "snippet": "统一身份认证登录地址 https://ids.henu.edu.cn",
                        }
                    ],
                    "meta": {"attempt_stage": "primary", "json_ok": True},
                },
                30,
            ),
        )

        response = run_agent_graph(
            db,
            text="统一身份认证的登录地址是什么？",
            user="alice",
            department="IT",
            actor_user_id="alice",
            actor_role="user",
        )

        assert response["route"] == "ASK"
        audits = _working_memory_audits(db)
        assert len(audits) == 1
        payload = audits[0].payload_json
        assert payload["route_source"] == "rules"
        assert payload["intent"] == "ASK"
        assert payload["selected_tool"] == "kb_answer"
        assert payload["tool_result_summary"]["route"] == "ASK"
        assert "error_code" not in payload
    finally:
        db.close()


def test_agent_graph_ask_node_uses_shared_redis_cache(monkeypatch) -> None:
    monkeypatch.setenv("ASK_CACHE_ENABLED", "true")
    monkeypatch.setenv("ASK_CACHE_NAMESPACE_VERSION", "test-graph-cache")
    store = _patch_in_memory_ask_cache(monkeypatch)
    calls = {"retrieve": 0, "answer": 0}

    def _fake_retrieve(question: str):
        calls["retrieve"] += 1
        return [
            {
                "doc_id": "henu_network_manual",
                "page": 5,
                "score": 0.88,
                "snippet": "统一身份认证登录地址 https://ids.henu.edu.cn",
            }
        ], 12

    def _fake_answer(question: str, hits: list[dict]):
        calls["answer"] += 1
        return {
            "answer": "统一身份认证的登录地址是 https://ids.henu.edu.cn。",
            "citations": [
                {
                    "doc_id": "henu_network_manual",
                    "page": 5,
                    "snippet": "统一身份认证登录地址 https://ids.henu.edu.cn",
                }
            ],
            "meta": {"attempt_stage": "primary", "json_ok": True},
        }, 30

    monkeypatch.setattr(ask_pipeline, "run_retrieve_step", _fake_retrieve)
    monkeypatch.setattr(ask_pipeline, "run_answer_step", _fake_answer)

    db = _build_test_session()
    try:
        first = run_agent_graph(
            db,
            text="统一身份认证的登录地址是什么？",
            user="alice",
            department="IT",
            actor_user_id="alice",
            actor_role="user",
        )
        second = run_agent_graph(
            db,
            text="  统一身份认证的登录地址是什么？  ",
            user="alice",
            department="IT",
            actor_user_id="alice",
            actor_role="user",
        )

        assert len(store) == 1
        assert calls == {"retrieve": 1, "answer": 1}
        assert first["kb"]["meta"]["cache"]["status"] == "stored"
        assert second["kb"]["meta"]["cache"]["hit"] is True
        assert second["kb"]["meta"]["attempt_stage"] == "cache_hit"
    finally:
        db.close()


def test_agent_graph_l0_working_memory_records_reference_error() -> None:
    db = _build_test_session()
    try:
        response = run_agent_graph(
            db,
            text="帮我查一下上一单",
            user="alice",
            department="IT",
            actor_user_id="alice",
            actor_role="user",
        )

        assert response["route"] == "NEED_MORE_INFO"
        audits = _working_memory_audits(db)
        assert len(audits) == 1
        payload = audits[0].payload_json
        assert payload["error_code"] == "ticket_reference_missing"
        assert payload["error_stage"] == "resolve_references"
        assert payload["tool_result_summary"]["route"] == "NEED_MORE_INFO"
    finally:
        db.close()
