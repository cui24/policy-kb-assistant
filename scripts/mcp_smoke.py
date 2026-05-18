#!/usr/bin/env python3
"""MCP in-process smoke test：不依赖 Host，直接验证主工具集合。"""

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


def _extract_data(payload: dict) -> dict:
    if isinstance(payload.get("data"), dict):
        return dict(payload["data"])
    return dict(payload or {})


async def _run_smoke(*, actor: str, department: str) -> int:
    """构建 in-memory MCP server 并验证主工具可见性与基础查单能力。"""
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
        expected_tools = {
            "ask_policy",
            "create_ticket",
            "continue_ticket_draft",
            "confirm_action",
            "ticket_tool_planner",
            "get_ticket_detail",
        }
        missing = sorted(expected_tools - set(tool_names))
        assert not missing, f"missing tools: {missing}"

        detail_payload = _payload_from_result(
            await client.call_tool("get_ticket_detail", {"ticket_id": ticket_id})
        )
        print("[GET_TICKET_DETAIL]", json.dumps(detail_payload, ensure_ascii=False))
        assert str(detail_payload.get("contract_version") or "") == "v1"
        assert bool(detail_payload.get("success")) is True
        detail_data = _extract_data(detail_payload)
        assert str(detail_data.get("ticket_id") or "") == ticket_id

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
