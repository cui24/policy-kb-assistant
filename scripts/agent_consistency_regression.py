#!/usr/bin/env python3
"""三路线一致性回归脚本：legacy / graph+local / graph+remote(strict path)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.agent_graph.executor import run_agent_graph
import src.agent_graph.nodes_execute_mcp as nodes_execute_mcp
from src.api import ask_pipeline, services
from src.api.db import Base
import src.mcp_wrapper.executor as mcp_executor
from src.mcp_wrapper import ToolCallRequest, invoke_tool


@contextmanager
def _patched_env(overrides: dict[str, str]):
    previous: dict[str, str | None] = {k: os.getenv(k) for k in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _patch_services() -> None:
    ask_pipeline.run_retrieve_step = lambda question: (  # type: ignore[assignment]
        [{"doc_id": "demo_doc", "page": 1, "score": 0.9, "snippet": "示例政策片段"}],
        1,
    )
    ask_pipeline.run_answer_step = lambda question, hits: (  # type: ignore[assignment]
        {
            "answer": "这是示例回答。",
            "citations": [{"doc_id": "demo_doc", "page": 1, "snippet": "示例政策片段"}],
            "meta": {"attempt_stage": "primary", "json_ok": True, "repair_used": False, "failure_reason": None},
        },
        1,
    )

    services.retrieve = lambda text: [  # type: ignore[assignment]
        {"doc_id": "demo_doc", "page": 1, "score": 0.9, "snippet": "示例政策片段"}
    ]
    services.answer_with_citations = lambda text, hits: {  # type: ignore[assignment]
        "answer": "这是示例回答。",
        "citations": [{"doc_id": "demo_doc", "page": 1, "snippet": "示例政策片段"}],
        "meta": {"attempt_stage": "primary", "json_ok": True, "repair_used": False, "failure_reason": None},
    }

    def _extractor(text: str, user: str, department: str) -> dict[str, Any]:
        if "缺字段" in text:
            return {
                "creator": user,
                "department": department,
                "category": "network",
                "priority": "P1",
                "title": "网络报修",
                "description": text,
                "contact": None,
                "location": None,
                "missing_fields": ["location", "contact"],
                "extractor": "rule_fallback",
            }
        return {
            "creator": user,
            "department": department,
            "category": "network",
            "priority": "P1",
            "title": "网络报修",
            "description": text,
            "contact": "13812345678",
            "location": "金明校区",
            "missing_fields": [],
            "extractor": "rule_fallback",
        }

    services.extract_ticket_payload = _extractor  # type: ignore[assignment]


def _normalize(result: dict[str, Any]) -> dict[str, Any]:
    ticket = result.get("ticket") if isinstance(result.get("ticket"), dict) else {}
    return {
        "route": str(result.get("route") or ""),
        "message": str(result.get("message") or ""),
        "missing_fields": list(result.get("missing_fields") or []),
        "ticket_id": str(ticket.get("ticket_id") or ""),
        "confirm_token_present": bool(str(result.get("confirm_token") or "").strip()),
    }


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def set(self, key: str, value: str, ex: int | None = None, nx: bool = False, xx: bool = False) -> bool:
        if nx and key in self._store:
            return False
        if xx and key not in self._store:
            return False
        self._store[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0


def _assert_consistent(case_name: str, outputs: dict[str, dict[str, Any]]) -> None:
    base = outputs["legacy"]
    for name, value in outputs.items():
        comparable_base = dict(base)
        comparable_value = dict(value)
        comparable_base["ticket_id_present"] = bool(str(comparable_base.get("ticket_id") or "").strip())
        comparable_value["ticket_id_present"] = bool(str(comparable_value.get("ticket_id") or "").strip())
        comparable_base.pop("ticket_id", None)
        comparable_value.pop("ticket_id", None)
        if comparable_value != comparable_base:
            raise AssertionError(
                f"[{case_name}] mismatch {name} != legacy\nlegacy={comparable_base}\n{name}={comparable_value}"
            )


def main() -> int:
    _patch_services()
    mcp_executor.get_redis = lambda: _FakeRedis()  # type: ignore[assignment]

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    cases = [
        ("ask", "统一身份认证登录地址是什么？"),
        ("create_ticket_success", "我宿舍网络断了，地点金明校区，电话13812345678，帮我报修。"),
        ("create_ticket_need_more_info", "我宿舍网络断了，缺字段，帮我报修。"),
    ]

    with Session() as db:
        original_call_remote = nodes_execute_mcp.call_remote_tool

        def _fake_remote_call_tool(*, server_url: str, tool: str, args: dict[str, Any], raw_text: str, timeout_seconds: int, **_: Any):
            req = ToolCallRequest(
                tool=tool,
                args=dict(args or {}),
                actor="alice",
                actor_user_id="alice-id",
                actor_role="user",
                department="IT",
                raw_text=raw_text,
            )
            return invoke_tool(db, req)

        nodes_execute_mcp.call_remote_tool = _fake_remote_call_tool  # type: ignore[assignment]
        try:
            for case_name, text in cases:
                legacy = services.run_agent_workflow(
                    db,
                    text=text,
                    user="alice",
                    department="IT",
                    actor_user_id="alice-id",
                    actor_role="user",
                )
                with _patched_env(
                    {
                        "AGENT_GRAPH_ENABLED": "true",
                        "AGENT_MCP_CLIENT_ENABLED": "false",
                    }
                ):
                    graph_local = run_agent_graph(
                        db,
                        text=text,
                        user="alice",
                        department="IT",
                        actor_user_id="alice-id",
                        actor_role="user",
                    )
                with _patched_env(
                    {
                        "AGENT_GRAPH_ENABLED": "true",
                        "AGENT_MCP_CLIENT_ENABLED": "true",
                    }
                ):
                    graph_remote = run_agent_graph(
                        db,
                        text=text,
                        user="alice",
                        department="IT",
                        actor_user_id="alice-id",
                        actor_role="user",
                    )

                outputs = {
                    "legacy": _normalize(legacy),
                    "graph_local": _normalize(graph_local),
                    "graph_remote_strict": _normalize(graph_remote),
                }
                _assert_consistent(case_name, outputs)
                print(f"[PASS] {case_name}: {outputs['legacy']}")
        finally:
            nodes_execute_mcp.call_remote_tool = original_call_remote  # type: ignore[assignment]

    print("[PASS] consistency regression completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
