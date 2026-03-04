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
from src.api import services_mcp


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
    )

    @app.tool()
    def lookup_ticket(ticket_id: str) -> dict[str, Any]:
        """按工单号查询工单详情与评论。"""
        with _db_session(session_factory) as db:
            return services_mcp.invoke_ticket_tool(
                db,
                tool_name="lookup_ticket",
                args={"ticket_id": ticket_id},
                actor=normalized_actor,
                raw_text=f"MCP lookup {ticket_id}",
            )

    @app.tool()
    def add_ticket_comment(ticket_id: str, comment: str) -> dict[str, Any]:
        """向现有工单追加说明，评论按 append-only 保存。"""
        with _db_session(session_factory) as db:
            return services_mcp.invoke_ticket_tool(
                db,
                tool_name="add_ticket_comment",
                args={"ticket_id": ticket_id, "comment": comment},
                actor=normalized_actor,
                raw_text=comment,
            )

    @app.tool()
    def escalate_ticket(ticket_id: str, reason: str | None = None) -> dict[str, Any]:
        """记录催办请求，并在允许时推进工单状态。"""
        with _db_session(session_factory) as db:
            args = {"ticket_id": ticket_id}
            if reason is not None:
                args["reason"] = reason
            return services_mcp.invoke_ticket_tool(
                db,
                tool_name="escalate_ticket",
                args=args,
                actor=normalized_actor,
                raw_text=reason or "",
            )

    @app.tool()
    def request_cancel_ticket(ticket_id: str, reason: str | None = None) -> dict[str, Any]:
        """第一步：申请取消工单并获取 confirm_token。"""
        with _db_session(session_factory) as db:
            return services_mcp.request_cancel_ticket_workflow(
                db,
                ticket_id=ticket_id,
                actor=normalized_actor,
                reason=reason,
            )

    @app.tool()
    def confirm_cancel_ticket(confirm_token: str) -> dict[str, Any]:
        """第二步：消费 confirm_token，真正执行取消。"""
        with _db_session(session_factory) as db:
            ticket_detail = services_mcp.confirm_cancel_ticket_workflow(
                db,
                confirm_token=confirm_token,
                actor=normalized_actor,
            )
            return {
                "message": "已取消工单。",
                "ticket_detail": ticket_detail,
            }

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
