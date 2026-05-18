"""Agent Graph 通用执行节点：通过 MCP wrapper 发起工具调用。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from sqlalchemy.orm import Session

from src.agent_graph.audit import append_simple_event
from src.agent_graph.config import load_agent_graph_config
from src.agent_graph.mcp_client import call_remote_tool
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import record_error, tool_args_preview, tool_result_summary, update_working_memory
from src.mcp_wrapper import ToolCallRequest, invoke_tool

AUTO_IDEMPOTENT_TOOLS = {
    "create_ticket",
    "continue_ticket_draft",
    "confirm_action",
    "ticket_tool_planner",
}


def _auto_idempotency_window_seconds() -> int:
    raw = str(os.getenv("AGENT_AUTO_IDEMPOTENCY_WINDOW_SECONDS") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 60
    return max(15, min(value, 300))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compute_auto_idempotency_key(
    *,
    tool: str,
    actor_user_id: str,
    raw_text: str,
    args: dict[str, Any],
) -> str:
    # 不把 idempotency_key 自身计入指纹，避免嵌套污染。
    fingerprint_args = {k: v for k, v in args.items() if str(k) != "idempotency_key"}
    payload = {
        "tool": str(tool or ""),
        "actor_user_id": str(actor_user_id or "anonymous"),
        "raw_text": str(raw_text or ""),
        "args": fingerprint_args,
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:24]
    bucket = int(time.time() // _auto_idempotency_window_seconds())
    return f"auto-{tool}-{bucket}-{digest}"


def _inject_auto_idempotency_if_needed(
    *,
    tool: str,
    args: dict[str, Any],
    actor_user_id: str,
    raw_text: str,
) -> tuple[dict[str, Any], str | None]:
    normalized_tool = str(tool or "").strip()
    normalized_args = dict(args or {})
    provided_key = str(normalized_args.get("idempotency_key") or "").strip()
    if provided_key:
        return normalized_args, provided_key
    if normalized_tool not in AUTO_IDEMPOTENT_TOOLS:
        return normalized_args, None

    auto_key = _compute_auto_idempotency_key(
        tool=normalized_tool,
        actor_user_id=actor_user_id,
        raw_text=raw_text,
        args=normalized_args,
    )
    normalized_args["idempotency_key"] = auto_key
    return normalized_args, auto_key


def run_execute_mcp_tool_node(db: Session, state: AgentState) -> AgentState:
    """执行 planner 产出的工具调用申请。"""
    request = dict(state.get("request") or {})
    planner_state = dict(state.get("planner") or {})
    tool_request = dict(planner_state.get("tool_request") or {})

    tool = str(tool_request.get("tool") or planner_state.get("tool") or "").strip()
    original_args = dict(tool_request.get("args") or planner_state.get("validated_args") or {})
    if not tool:
        state["execution"] = {
            "route": "PLAN_REJECTED",
            "message": "tool_request_missing_tool",
        }
        record_error(
            state,
            code="tool_request_missing_tool",
            stage="execute_mcp_tool",
            reason="planner_tool_request_missing_tool",
        )
        append_simple_event(state, "NODE_EXECUTED", {"node": "execute_mcp_tool", "ok": False})
        return state
    request_actor_user_id = str(request.get("actor_user_id") or "anonymous")
    request_raw_text = str(request.get("text") or "")
    args, auto_or_provided_idempotency_key = _inject_auto_idempotency_if_needed(
        tool=tool,
        args=original_args,
        actor_user_id=request_actor_user_id,
        raw_text=request_raw_text,
    )
    update_working_memory(
        state,
        selected_tool=tool,
        tool_args_preview=tool_args_preview(args),
    )

    invoke_request = ToolCallRequest(
        tool=tool,
        args=args,
        request_id=str(tool_request.get("request_id") or planner_state.get("request_id") or ""),
        actor=str(request.get("user") or "anonymous"),
        actor_user_id=request_actor_user_id,
        actor_role=str(request.get("actor_role") or ""),
        department=str(request.get("department") or "general"),
        idempotency_key=auto_or_provided_idempotency_key,
        mode=str(tool_request.get("mode") or planner_state.get("mode") or ""),
        raw_text=request_raw_text,
    )
    config = load_agent_graph_config()
    if config.mcp_client_enabled:
        result = call_remote_tool(
            server_url=config.mcp_server_url,
            tool=invoke_request.tool,
            args=invoke_request.args,
            raw_text=str(invoke_request.raw_text or ""),
            timeout_seconds=config.mcp_client_timeout_seconds,
            circuit_breaker_enabled=bool(config.mcp_circuit_breaker_enabled),
            circuit_fail_threshold=int(config.mcp_circuit_fail_threshold),
            circuit_open_seconds=int(config.mcp_circuit_open_seconds),
        )
    else:
        result = invoke_tool(db, invoke_request)

    payload = dict(result.payload or {})
    data = payload.get("data")
    if isinstance(data, dict):
        execution = dict(data)
    else:
        execution = dict(payload)

    execution["tool_result"] = {
        "ok": bool(result.ok),
        "route": str(result.route or execution.get("route") or payload.get("route") or ""),
        "error_code": result.error_code,
        "message": result.message,
        "retryable": bool(result.retryable),
        "contract_version": str(payload.get("contract_version") or ""),
        "success": bool(payload.get("success")) if "success" in payload else bool(result.ok),
        "error": payload.get("error"),
        "raw_tool": (payload.get("_tool_meta") or {}).get("raw_tool"),
        "normalized_tool": (payload.get("_tool_meta") or {}).get("normalized_tool"),
        "idempotency_key": auto_or_provided_idempotency_key,
    }
    if "route" not in execution or not str(execution.get("route") or ""):
        execution["route"] = str(result.route or payload.get("route") or "PLAN_REJECTED")

    state["execution"] = execution
    update_working_memory(
        state,
        tool_result_summary=tool_result_summary(execution),
    )
    if not result.ok:
        record_error(
            state,
            code=str(result.error_code or execution.get("route") or "tool_execution_failed"),
            stage="execute_mcp_tool",
            reason=result.message,
        )
    append_simple_event(
        state,
        "NODE_EXECUTED",
        {
            "node": "execute_mcp_tool",
            "tool": tool,
            "ok": bool(result.ok),
        },
    )
    return state
