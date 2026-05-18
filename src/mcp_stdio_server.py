"""
stdio MCP server 入口：以固定单用户 actor 暴露一组 ticket tools。

一、程序目标
1. 提供一个可直接运行的 FastMCP stdio server。
2. 固定 actor，不接受外部参数伪造用户身份。
3. 通过薄封装复用现有 ticket validator / registry / workflow。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from sqlalchemy.orm import Session

from src.api.db import SessionLocal
from src.api.deps import load_runtime_settings
from src.api.migrations import ensure_schema_ready
from src.mcp_wrapper import ToolCallRequest, invoke_tool


logger = logging.getLogger(__name__)


@contextmanager
def _db_session(session_factory: Callable[[], Session]) -> Session:
    """为每次 MCP tool 调用提供独立数据库会话。"""
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def build_mcp_server(
    *,
    actor_user_id: str,
    department: str = "IT",
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
    session_factory: Callable[[], Session] = SessionLocal,
) -> FastMCP:
    """构造一个绑定固定 actor 的 FastMCP server。"""
    normalized_actor = str(actor_user_id or "").strip()
    if not normalized_actor:
        raise ValueError("MCP_ACTOR_USER_ID 不能为空。")

    normalized_department = str(department or "IT").strip() or "IT"
    app = FastMCP(
        "policy-kb-itsm",
        instructions=(
            "Demo mode: single-user fixed actor. "
            f"actor_user_id={normalized_actor}; department={normalized_department}."
        ),
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
    )

    def _invoke_tool(db: Session, tool: str, args: dict[str, Any], raw_text: str = "") -> dict[str, Any]:
        result = invoke_tool(
            db,
            ToolCallRequest(
                tool=tool,
                args=dict(args or {}),
                actor=normalized_actor,
                actor_user_id=normalized_actor,
                actor_role="user",
                department=normalized_department,
                raw_text=raw_text,
            ),
        )
        return dict(result.payload or {})

    @app.tool()
    def ask_policy(question: str) -> dict[str, Any]:
        """政策问答工具：返回答案与引用。"""
        with _db_session(session_factory) as db:
            return _invoke_tool(
                db,
                "ask_policy",
                {
                    "question": question,
                },
                raw_text=question,
            )

    @app.tool()
    def create_ticket(
        text: str,
        fields: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """创建工单工具：支持按文本抽取并建单。"""
        with _db_session(session_factory) as db:
            return _invoke_tool(
                db,
                "create_ticket",
                {
                    "text": text,
                    "fields": dict(fields or {}),
                    "idempotency_key": str(idempotency_key or ""),
                },
                raw_text=text,
            )

    @app.tool()
    def continue_ticket_draft(
        draft_id: str,
        text: str,
        fields: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """续办草稿工具：补充字段并继续执行建单流程。"""
        with _db_session(session_factory) as db:
            return _invoke_tool(
                db,
                "continue_ticket_draft",
                {
                    "draft_id": draft_id,
                    "text": text,
                    "fields": dict(fields or {}),
                    "idempotency_key": str(idempotency_key or ""),
                },
                raw_text=text,
            )

    @app.tool()
    def confirm_action(
        confirm_token: str,
        text: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """确认态工具：消费确认令牌并执行待确认动作。"""
        with _db_session(session_factory) as db:
            return _invoke_tool(
                db,
                "confirm_action",
                {
                    "confirm_token": confirm_token,
                    "text": str(text or ""),
                    "idempotency_key": str(idempotency_key or ""),
                },
                raw_text=str(text or ""),
            )

    @app.tool()
    def ticket_tool_planner(
        ticket_id: str,
        raw_text: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """工单工具规划器：按文本意图对目标工单执行 comment/escalate/cancel 等操作。"""
        with _db_session(session_factory) as db:
            return _invoke_tool(
                db,
                "ticket_tool_planner",
                {
                    "ticket_id": ticket_id,
                    "raw_text": raw_text,
                    "idempotency_key": str(idempotency_key or ""),
                },
                raw_text=raw_text,
            )

    @app.tool()
    def get_ticket_detail(ticket_id: str) -> dict[str, Any]:
        """查询工单详情工具。"""
        with _db_session(session_factory) as db:
            return _invoke_tool(
                db,
                "get_ticket_detail",
                {"ticket_id": ticket_id},
                raw_text=f"MCP get detail {ticket_id}",
            )

    return app


def main() -> None:
    """以 stdio transport 启动 MCP server。"""
    logging.basicConfig(level=logging.INFO)
    load_runtime_settings()
    ensure_schema_ready()

    actor_user_id = str(os.getenv("MCP_ACTOR_USER_ID") or "").strip()
    if not actor_user_id:
        raise RuntimeError("MCP_ACTOR_USER_ID is required for stdio MCP demo mode.")
    department = str(os.getenv("MCP_DEPARTMENT") or "IT").strip() or "IT"

    logger.info("Starting MCP stdio server for actor=%s department=%s", actor_user_id, department)
    build_mcp_server(
        actor_user_id=actor_user_id,
        department=department,
    ).run(transport="stdio")


if __name__ == "__main__":
    main()
