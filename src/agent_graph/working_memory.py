"""L0 Working Memory helpers.

L0 只描述一次 Agent 请求执行中的临时状态：输入规范化、路由判断、
引用恢复、工具选择、约束校验和工具结果摘要。它不落独立表，也不保存
跨轮用户事实；需要长期复用的内容应进入 L1/L3/L4 或审计。
"""

from __future__ import annotations

from typing import Any

from src.agent_graph.state import AgentState

_TEXT_PREVIEW_LIMIT = 180
_MESSAGE_PREVIEW_LIMIT = 180


def _preview(value: Any, limit: int = _TEXT_PREVIEW_LIMIT) -> str:
    return str(value or "").strip()[:limit]


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def update_working_memory(state: AgentState, **updates: Any) -> AgentState:
    """Merge partial L0 fields into state."""
    working = dict(state.get("working") or {})
    for key, value in updates.items():
        if value is not None:
            working[key] = value
    state["working"] = working
    return state


def record_error(
    state: AgentState,
    *,
    code: str,
    stage: str,
    reason: str | None = None,
) -> AgentState:
    """Record a structured error reason for this request only."""
    return update_working_memory(
        state,
        error_code=str(code or "unknown_error"),
        error_stage=str(stage or "unknown"),
        error_reason=_preview(reason or code, _MESSAGE_PREVIEW_LIMIT),
    )


def tool_args_preview(args: dict[str, Any] | None) -> dict[str, Any]:
    """Build a safe, compact preview of tool args for L0/audit."""
    preview: dict[str, Any] = {}
    for key, value in _safe_dict(args).items():
        if key == "confirm_token":
            token = str(value or "")
            preview[key] = f"{token[:8]}..." if token else ""
        elif key in {"text", "raw_text", "query", "comment", "reason"}:
            preview[key] = _preview(value)
        elif key == "fields":
            preview[key] = _safe_dict(value)
        elif key == "idempotency_key":
            preview[key] = bool(value)
        else:
            preview[key] = value
    return preview


def tool_result_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    """Build a compact tool result summary without large business payloads."""
    normalized = _safe_dict(result)
    ticket = _safe_dict(normalized.get("ticket"))
    draft = _safe_dict(normalized.get("draft"))
    tool_result = _safe_dict(normalized.get("tool_result"))
    summary: dict[str, Any] = {
        "route": normalized.get("route"),
        "message": _preview(normalized.get("message"), _MESSAGE_PREVIEW_LIMIT),
    }
    if normalized.get("missing_fields"):
        summary["missing_fields"] = list(normalized.get("missing_fields") or [])
    if ticket.get("ticket_id"):
        summary["ticket_id"] = ticket.get("ticket_id")
    if draft.get("draft_id"):
        summary["draft_id"] = draft.get("draft_id")
    if tool_result:
        summary["tool_ok"] = bool(tool_result.get("ok"))
        summary["error_code"] = tool_result.get("error_code")
        summary["retryable"] = bool(tool_result.get("retryable"))
    return {key: value for key, value in summary.items() if value not in (None, "", [])}


def audit_summary(state: AgentState) -> dict[str, Any]:
    """Return the L0 fields safe enough to persist into audit payloads."""
    working = dict(state.get("working") or {})
    keys = (
        "request_id",
        "route_source",
        "intent",
        "selected_tool",
        "resolved_refs",
        "missing_fields",
        "auth_context",
        "permission_context",
        "risk_context",
        "memory_context",
        "tool_result_summary",
        "error_code",
        "error_stage",
        "error_reason",
    )
    return {key: working.get(key) for key in keys if working.get(key) not in (None, "", [], {})}
