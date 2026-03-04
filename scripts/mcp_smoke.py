#!/usr/bin/env python3
"""MCP in-process smoke test：不依赖 Host，直接验证 stdio server 的核心工具链。"""

from __future__ import annotations

import argparse
import asyncio
import json

from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import services
from src.api.db import Base
from src.mcp_stdio_server import build_mcp_server


def _parse_args() -> argparse.Namespace:
    """读取命令行参数。"""
    parser = argparse.ArgumentParser(description="Run an in-memory MCP smoke test for the local stdio server.")
    parser.add_argument("--actor", default="mcp-demo-user", help="Fixed actor used by the smoke MCP server.")
    parser.add_argument("--department", default="IT", help="Department label injected into the demo server.")
    return parser.parse_args()


def _build_test_session_factory():
    """创建隔离的内存数据库会话工厂。"""
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def _seed_ticket(session_factory, actor: str) -> str:
    """创建一张演示工单，供后续工具调用。"""
    with session_factory() as db:
        created = services.create_ticket_workflow(
            db,
            creator=actor,
            department="IT",
            category="network",
            priority="P1",
            title="MCP smoke demo ticket",
            description="Smoke test for local MCP tools.",
            contact="13812345678",
            context={"location": "金明校区"},
            request_id="req_mcp_smoke_seed",
        )
        return str(created["ticket_id"])


def _payload_from_result(result) -> dict:
    """把 MCP CallToolResult 统一转成 dict。"""
    if result.structuredContent is not None:
        return result.structuredContent

    text_payload = "".join(
        block.text
        for block in result.content
        if getattr(block, "type", "") == "text"
    )
    return json.loads(text_payload or "{}")


async def _run_smoke(*, actor: str, department: str) -> int:
    """构建 in-memory MCP server 并顺序验证全部工具。"""
    session_factory = _build_test_session_factory()
    ticket_id = _seed_ticket(session_factory, actor)
    app = build_mcp_server(
        actor_user_id=actor,
        department=department,
        session_factory=session_factory,
    )

    async with create_connected_server_and_client_session(app) as client:
        tools_result = await client.list_tools()
        tool_names = sorted(tool.name for tool in tools_result.tools)
        print("[TOOLS]", ", ".join(tool_names))

        lookup_payload = _payload_from_result(
            await client.call_tool("lookup_ticket", {"ticket_id": ticket_id})
        )
        print("[LOOKUP]", json.dumps(lookup_payload, ensure_ascii=False))

        comment_payload = _payload_from_result(
            await client.call_tool(
                "add_ticket_comment",
                {
                    "ticket_id": ticket_id,
                    "comment": "补充说明：今晚 8 点后可到宿舍排查。",
                },
            )
        )
        print("[COMMENT]", json.dumps(comment_payload, ensure_ascii=False))

        escalate_payload = _payload_from_result(
            await client.call_tool(
                "escalate_ticket",
                {
                    "ticket_id": ticket_id,
                    "reason": "已影响今晚作业提交。",
                },
            )
        )
        print("[ESCALATE]", json.dumps(escalate_payload, ensure_ascii=False))

        request_cancel_payload = _payload_from_result(
            await client.call_tool(
                "request_cancel_ticket",
                {
                    "ticket_id": ticket_id,
                    "reason": "已自行恢复。",
                },
            )
        )
        print("[REQUEST_CANCEL]", json.dumps(request_cancel_payload, ensure_ascii=False))

        confirm_cancel_payload = _payload_from_result(
            await client.call_tool(
                "confirm_cancel_ticket",
                {
                    "confirm_token": str(request_cancel_payload["confirm_token"]),
                },
            )
        )
        print("[CONFIRM_CANCEL]", json.dumps(confirm_cancel_payload, ensure_ascii=False))

    final_ticket = confirm_cancel_payload["ticket_detail"]
    assert final_ticket["ticket_id"] == ticket_id
    assert final_ticket["status"] == "cancelled"
    print("[PASS] MCP smoke test completed successfully.")
    return 0


def main() -> int:
    """命令行入口。"""
    args = _parse_args()
    return asyncio.run(
        _run_smoke(
            actor=str(args.actor or "").strip() or "mcp-demo-user",
            department=str(args.department or "").strip() or "IT",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
