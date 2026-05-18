"""MCP wrapper 工具注册表：仅包装现有实现，不改底层业务逻辑。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from src.api import services, services_mcp
from src.mcp_wrapper.contracts import ToolCallRequest
from src.mcp_wrapper.errors import ToolNotFoundError

ToolHandler = Callable[[Session, ToolCallRequest], dict[str, Any]]


TOOL_ALIASES: dict[str, str] = {
    "kb_answer": "ask_policy",
    "lookup_ticket": "get_ticket_detail",
}


def _text_arg(request: ToolCallRequest) -> str:
    # 获取工具调用请求中的 text 参数，如果参数不存在，则返回用户原始输入文本
    return str(request.args.get("text") or request.raw_text or "")


def _query_arg(request: ToolCallRequest) -> str:
    # 获取工具调用请求中的 query 参数，如果参数不存在，则返回用户原始输入文本
    return str(request.args.get("query") or request.args.get("question") or request.raw_text or "")


def _handle_ask_policy(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    # ask_policy 只信任系统上下文中的 department，不接受参数覆盖。
    department = str(request.department or "general")
    return services._handle_kb_intent(
        db,
        text=_query_arg(request),
        actor=request.actor,
        actor_user_id=request.actor_user_id,
        actor_department=department,
    )


def _handle_create_ticket(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    return services._handle_create_ticket_intent(
        db,
        text=_text_arg(request),
        actor=request.actor,
        actor_user_id=request.actor_user_id,
        actor_department=request.department,
        planner_fields=request.args.get("fields") if isinstance(request.args.get("fields"), dict) else None,
    )


def _handle_continue_ticket_draft(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    return services._resume_ticket_draft_workflow(
        db,
        draft_id=str(request.args.get("draft_id") or ""),
        text=_text_arg(request),
        fields=request.args.get("fields") if isinstance(request.args.get("fields"), dict) else None,
        actor=request.actor,
        actor_user_id=request.actor_user_id,
        actor_role=request.actor_role,
        actor_department=request.department,
    )


def _handle_ticket_tool_planner(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    return services._handle_ticket_tool_route_with_planner(
        db,
        ticket_id=str(request.args.get("ticket_id") or ""),
        text=str(request.args.get("raw_text") or request.raw_text or ""),
        actor=request.actor,
        actor_user_id=request.actor_user_id,
        actor_role=request.actor_role,
        mode=str(request.mode or services._agent_planner_mode()),
    )


def _handle_confirm_action(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    return services._handle_confirmed_pending_action(
        db,
        confirm_token=str(request.args.get("confirm_token") or ""),
        actor=request.actor,
        actor_user_id=request.actor_user_id,
        actor_role=request.actor_role,
        text=_text_arg(request),
    )


def _handle_lookup_ticket(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    return services_mcp.invoke_ticket_tool(
        db,
        tool_name="lookup_ticket",
        args={"ticket_id": str(request.args.get("ticket_id") or "")},
        actor=request.actor,
        raw_text=str(request.raw_text or ""),
    )


def _handle_add_ticket_comment(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    return services_mcp.invoke_ticket_tool(
        db,
        tool_name="add_ticket_comment",
        args={
            "ticket_id": str(request.args.get("ticket_id") or ""),
            "comment": str(request.args.get("comment") or ""),
        },
        actor=request.actor,
        raw_text=str(request.raw_text or ""),
    )


def _handle_escalate_ticket(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    args: dict[str, Any] = {"ticket_id": str(request.args.get("ticket_id") or "")}
    reason = request.args.get("reason")
    if reason is not None:
        args["reason"] = str(reason)
    return services_mcp.invoke_ticket_tool(
        db,
        tool_name="escalate_ticket",
        args=args,
        actor=request.actor,
        raw_text=str(request.raw_text or ""),
    )


def _handle_request_cancel_ticket(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    reason = request.args.get("reason")
    return services_mcp.request_cancel_ticket_workflow(
        db,
        ticket_id=str(request.args.get("ticket_id") or ""),
        actor=request.actor,
        reason=str(reason) if reason is not None else None,
    )


def _handle_confirm_cancel_ticket(db: Session, request: ToolCallRequest) -> dict[str, Any]:
    detail = services_mcp.confirm_cancel_ticket_workflow(
        db,
        confirm_token=str(request.args.get("confirm_token") or ""),
        actor=request.actor,
    )
    return {
        "message": "已取消工单。",
        "ticket_detail": detail,
    }


TOOL_REGISTRY: dict[str, ToolHandler] = {
    "ask_policy": _handle_ask_policy,
    "create_ticket": _handle_create_ticket,
    "continue_ticket_draft": _handle_continue_ticket_draft,
    "ticket_tool_planner": _handle_ticket_tool_planner,
    "confirm_action": _handle_confirm_action,
    "get_ticket_detail": _handle_lookup_ticket,
    "add_ticket_comment": _handle_add_ticket_comment,
    "escalate_ticket": _handle_escalate_ticket,
    "request_cancel_ticket": _handle_request_cancel_ticket,
    "confirm_cancel_ticket": _handle_confirm_cancel_ticket,
}


def normalize_tool_name(raw_tool: str) -> str:
    """将工具名归一化到主工具名。"""
    normalized = str(raw_tool or "").strip()
    if not normalized:
        return ""
    return TOOL_ALIASES.get(normalized, normalized)


def dispatch_tool(db: Session, request: ToolCallRequest) -> tuple[dict[str, Any], str]:
    """按工具名调用对应处理器，并返回归一化工具名。"""
    raw_tool = str(request.tool or "").strip()
    normalized_tool = normalize_tool_name(raw_tool)
    handler = TOOL_REGISTRY.get(normalized_tool)
    if handler is None:
        raise ToolNotFoundError(raw_tool)
    return handler(db, request), normalized_tool
