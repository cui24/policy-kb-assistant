"""
MCP tools 测试：通过 in-memory client/server session 验证 stdio tool 逻辑。

一、测试目标
1. 验证工具列表可见且包含预期的 ticket tools。
2. 验证查单、评论、催办、两段式取消都能闭环。
3. 验证 MCP 调用会写入 `payload_json.source="mcp"` 的审计日志。
"""

from __future__ import annotations

import asyncio
import json

from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import crud, services
from src.api.db import Base
from src.mcp_stdio_server import build_mcp_server


def _build_test_session_factory():
    """创建独立内存数据库的会话工厂。"""
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def _call_tool(app, name: str, arguments: dict) -> dict:
    """通过 SDK in-memory transport 调用单个 MCP tool。"""

    async def _run():
        async with create_connected_server_and_client_session(app) as client:
            result = await client.call_tool(name, arguments)
            if result.structuredContent is not None:
                return result.structuredContent
            text_payload = "".join(
                block.text
                for block in result.content
                if getattr(block, "type", "") == "text"
            )
            return json.loads(text_payload or "{}")

    return asyncio.run(_run())


def _seed_ticket(db: Session, actor: str = "alice") -> dict:
    """创建一张测试工单。"""
    return services.create_ticket_workflow(
        db,
        creator=actor,
        department="IT",
        category="network",
        priority="P1",
        title="宿舍区无法上网",
        description="宿舍区断网，需要排查。",
        contact="13812345678",
        context={"location": "金明校区"},
        request_id="req_seed_ticket",
    )


def test_tools_list_exposes_expected_names() -> None:
    """MCP server 应暴露本阶段约定的五个工具。"""
    session_factory = _build_test_session_factory()
    app = build_mcp_server(actor_user_id="alice", session_factory=session_factory)

    async def _run():
        async with create_connected_server_and_client_session(app) as client:
            result = await client.list_tools()
            return sorted(tool.name for tool in result.tools)

    tool_names = asyncio.run(_run())

    assert tool_names == [
        "add_ticket_comment",
        "confirm_cancel_ticket",
        "escalate_ticket",
        "lookup_ticket",
        "request_cancel_ticket",
    ]


def test_lookup_ticket_ok() -> None:
    """lookup_ticket 应返回完整工单详情，并写 MCP 审计。"""
    session_factory = _build_test_session_factory()
    with session_factory() as db:
        seeded = _seed_ticket(db)

    app = build_mcp_server(actor_user_id="alice", session_factory=session_factory)
    payload = _call_tool(app, "lookup_ticket", {"ticket_id": seeded["ticket_id"]})

    assert payload["ticket_id"] == seeded["ticket_id"]
    assert payload["status"] == "open"
    assert isinstance(payload["comments"], list)

    with session_factory() as db:
        audit_records = crud.list_audit_logs(db, ticket_id=seeded["ticket_id"], limit=20)
        assert any(
            record.action_type == "MCP_TOOL_CALL" and (record.payload_json or {}).get("source") == "mcp"
            for record in audit_records
        )


def test_add_comment_appends() -> None:
    """add_ticket_comment 应真正追加一条 append-only 评论。"""
    session_factory = _build_test_session_factory()
    with session_factory() as db:
        seeded = _seed_ticket(db)

    app = build_mcp_server(actor_user_id="alice", session_factory=session_factory)
    payload = _call_tool(
        app,
        "add_ticket_comment",
        {
            "ticket_id": seeded["ticket_id"],
            "comment": "补充一下，今晚 8 点后有人在宿舍。",
        },
    )

    assert payload["ticket_id"] == seeded["ticket_id"]
    assert len(payload["comments"]) == 1
    assert payload["comments"][0]["content"].startswith("补充一下")

    with session_factory() as db:
        ticket = crud.get_ticket_by_public_id(db, seeded["ticket_id"])
        assert ticket is not None
        comment_rows = list(crud.list_ticket_comments(db, ticket.id, limit=10))
        assert len(comment_rows) == 1
        assert comment_rows[0].content.startswith("补充一下")

        audit_records = crud.list_audit_logs(db, ticket_id=seeded["ticket_id"], limit=20)
        assert any(
            record.action_type == "ADD_TICKET_COMMENT" and (record.payload_json or {}).get("source") == "mcp"
            for record in audit_records
        )


def test_escalate_increments() -> None:
    """escalate_ticket 应推进状态并累加 escalation_count。"""
    session_factory = _build_test_session_factory()
    with session_factory() as db:
        seeded = _seed_ticket(db)

    app = build_mcp_server(actor_user_id="alice", session_factory=session_factory)
    payload = _call_tool(
        app,
        "escalate_ticket",
        {
            "ticket_id": seeded["ticket_id"],
            "reason": "已影响今晚作业提交。",
        },
    )

    assert payload["ticket_id"] == seeded["ticket_id"]
    assert payload["status"] == "in_progress"
    assert int((payload["context"] or {}).get("escalation_count") or 0) == 1

    with session_factory() as db:
        audit_records = crud.list_audit_logs(db, ticket_id=seeded["ticket_id"], limit=20)
        assert any(
            record.action_type == "ESCALATE_TICKET" and (record.payload_json or {}).get("source") == "mcp"
            for record in audit_records
        )


def test_cancel_requires_confirmation() -> None:
    """request_cancel_ticket 第一步只发放 token，不改变工单状态。"""
    session_factory = _build_test_session_factory()
    with session_factory() as db:
        seeded = _seed_ticket(db)

    app = build_mcp_server(actor_user_id="alice", session_factory=session_factory)
    payload = _call_tool(
        app,
        "request_cancel_ticket",
        {
            "ticket_id": seeded["ticket_id"],
            "reason": "已经自己恢复了。",
        },
    )

    assert payload["ticket_id"] == seeded["ticket_id"]
    assert str(payload["confirm_token"])
    assert payload["message"].startswith("这是高风险操作")

    with session_factory() as db:
        ticket = crud.get_ticket_by_public_id(db, seeded["ticket_id"])
        pending_action = crud.get_pending_action_by_confirm_id(db, str(payload["confirm_token"]))
        assert ticket is not None
        assert ticket.status == "open"
        assert pending_action is not None
        assert pending_action.status == "pending"

        audit_records = crud.list_audit_logs(db, ticket_id=seeded["ticket_id"], limit=20)
        assert any(
            record.action_type == "NEED_CONFIRMATION" and (record.payload_json or {}).get("source") == "mcp"
            for record in audit_records
        )


def test_cancel_confirm_executes() -> None:
    """confirm_cancel_ticket 第二步才真正取消工单并消费 token。"""
    session_factory = _build_test_session_factory()
    with session_factory() as db:
        seeded = _seed_ticket(db)

    app = build_mcp_server(actor_user_id="alice", session_factory=session_factory)
    first_payload = _call_tool(
        app,
        "request_cancel_ticket",
        {
            "ticket_id": seeded["ticket_id"],
            "reason": "已经自己恢复了。",
        },
    )
    second_payload = _call_tool(
        app,
        "confirm_cancel_ticket",
        {
            "confirm_token": str(first_payload["confirm_token"]),
        },
    )

    ticket_detail = second_payload["ticket_detail"]
    assert second_payload["message"] == "已取消工单。"
    assert ticket_detail["ticket_id"] == seeded["ticket_id"]
    assert ticket_detail["status"] == "cancelled"

    with session_factory() as db:
        ticket = crud.get_ticket_by_public_id(db, seeded["ticket_id"])
        pending_action = crud.get_pending_action_by_confirm_id(db, str(first_payload["confirm_token"]))
        assert ticket is not None
        assert ticket.status == "cancelled"
        assert pending_action is not None
        assert pending_action.status == "consumed"

        audit_records = crud.list_audit_logs(db, ticket_id=seeded["ticket_id"], limit=30)
        assert any(
            record.action_type == "CANCEL_TICKET" and (record.payload_json or {}).get("source") == "mcp"
            for record in audit_records
        )
