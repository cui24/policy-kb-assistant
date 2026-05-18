"""Agent Graph 远程 MCP client 适配层。"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.mcp_wrapper.contracts import ToolCallResult

_CIRCUIT_LOCK = threading.Lock()
_CIRCUIT_FAILURES = 0
_CIRCUIT_OPEN_UNTIL = 0.0


def reset_remote_mcp_circuit_breaker() -> None:
    """重置远程 MCP 熔断状态（用于测试或人工恢复）。"""
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    with _CIRCUIT_LOCK:
        _CIRCUIT_FAILURES = 0
        _CIRCUIT_OPEN_UNTIL = 0.0


def _is_circuit_open() -> tuple[bool, int]:
    with _CIRCUIT_LOCK:
        now = time.time()
        if _CIRCUIT_OPEN_UNTIL > now:
            return True, max(1, int(_CIRCUIT_OPEN_UNTIL - now))
        return False, 0


def _record_success() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    with _CIRCUIT_LOCK:
        _CIRCUIT_FAILURES = 0
        _CIRCUIT_OPEN_UNTIL = 0.0


def _record_failure(*, fail_threshold: int, open_seconds: int) -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    with _CIRCUIT_LOCK:
        _CIRCUIT_FAILURES += 1
        if _CIRCUIT_FAILURES >= max(1, int(fail_threshold)):
            _CIRCUIT_OPEN_UNTIL = time.time() + max(1, int(open_seconds))


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        normalized = dict(payload)
    else:
        normalized = {"value": payload}

    route = str(normalized.get("route") or "")
    if not route:
        if "ticket_detail" in normalized or "ticket" in normalized:
            normalized["route"] = "TICKET_TOOL"
        else:
            normalized["route"] = "MCP_TOOL"
    return normalized


def _payload_from_call_result(result) -> dict[str, Any]:
    if getattr(result, "structuredContent", None) is not None:
        return _normalize_payload(getattr(result, "structuredContent"))

    text_payload = "".join(
        block.text
        for block in getattr(result, "content", [])
        if getattr(block, "type", "") == "text"
    )
    if not text_payload:
        return _normalize_payload({})
    try:
        parsed = json.loads(text_payload)
    except json.JSONDecodeError:
        return _normalize_payload({"raw_text": text_payload})
    return _normalize_payload(parsed)


def _map_tool_to_remote(
    tool: str,
    args: dict[str, Any],
    *,
    raw_text: str,
) -> tuple[str, dict[str, Any]] | None:
    normalized_tool = str(tool or "").strip()
    if normalized_tool in {"kb_answer", "ask_policy"}:
        return (
            "ask_policy",
            {
                "question": str(args.get("query") or raw_text or ""),
            },
        )
    if normalized_tool == "continue_ticket_draft":
        return (
            "continue_ticket_draft",
            {
                "draft_id": str(args.get("draft_id") or ""),
                "text": str(args.get("text") or raw_text or ""),
                "fields": dict(args.get("fields") or {}),
                "idempotency_key": str(args.get("idempotency_key") or ""),
            },
        )
    if normalized_tool == "confirm_action":
        return (
            "confirm_action",
            {
                "confirm_token": str(args.get("confirm_token") or ""),
                "text": str(args.get("text") or raw_text or ""),
                "idempotency_key": str(args.get("idempotency_key") or ""),
            },
        )
    if normalized_tool == "ticket_tool_planner":
        return (
            "ticket_tool_planner",
            {
                "ticket_id": str(args.get("ticket_id") or ""),
                "raw_text": str(args.get("raw_text") or raw_text or ""),
                "idempotency_key": str(args.get("idempotency_key") or ""),
            },
        )
    if normalized_tool == "create_ticket":
        return (
            "create_ticket",
            {
                "text": str(args.get("text") or raw_text or ""),
                "fields": dict(args.get("fields") or {}),
                "idempotency_key": str(args.get("idempotency_key") or ""),
            },
        )
    if normalized_tool in {"lookup_ticket", "get_ticket_detail"}:
        return (
            "get_ticket_detail",
            {
                "ticket_id": str(args.get("ticket_id") or ""),
            },
        )
    return None


async def _call_remote_tool_async(
    *,
    server_url: str,
    tool: str,
    args: dict[str, Any],
    raw_text: str,
    timeout_seconds: int,
) -> ToolCallResult:
    mapped = _map_tool_to_remote(
        tool,
        args,
        raw_text=raw_text,
    )
    if mapped is None:
        return ToolCallResult(
            ok=False,
            route="PLAN_REJECTED",
            payload={"route": "PLAN_REJECTED", "message": f"remote_tool_unsupported:{tool}"},
            error_code="remote_tool_unsupported",
            message=f"remote_tool_unsupported:{tool}",
            retryable=False,
        )

    remote_tool, remote_args = mapped
    try:
        async with streamablehttp_client(
            str(server_url),
            timeout=float(timeout_seconds),
            sse_read_timeout=float(max(timeout_seconds, 60)),
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                call_result = await session.call_tool(remote_tool, arguments=remote_args)
                payload = _payload_from_call_result(call_result)
                payload_error = payload.get("error")
                payload_success = bool(payload.get("success")) if "success" in payload else True
                return ToolCallResult(
                    ok=payload_success,
                    route=str(payload.get("route") or "MCP_TOOL"),
                    payload=payload,
                    error_code=(
                        str(payload_error.get("error_code") or "")
                        if isinstance(payload_error, dict)
                        else None
                    ),
                    message=str(payload.get("message") or "") or None,
                    retryable=bool(payload_error.get("retryable")) if isinstance(payload_error, dict) else False,
                )
    except Exception as exc:
        return ToolCallResult(
            ok=False,
            route="PLAN_REJECTED",
            payload={"route": "PLAN_REJECTED", "message": "远程 MCP 调用失败。"},
            error_code="remote_mcp_error",
            message=f"remote_mcp_error:{exc.__class__.__name__}",
            retryable=True,
        )


async def _check_remote_mcp_health_async(*, server_url: str, timeout_seconds: int) -> tuple[bool, str]:
    try:
        async with streamablehttp_client(
            str(server_url),
            timeout=float(timeout_seconds),
            sse_read_timeout=float(max(timeout_seconds, 30)),
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                tool_count = len(getattr(listed, "tools", []) or [])
                return True, f"ok:{tool_count}_tools"
    except Exception as exc:
        return False, f"remote_mcp_unhealthy:{exc.__class__.__name__}"


def check_remote_mcp_health(*, server_url: str, timeout_seconds: int) -> tuple[bool, str]:
    """主动探测远程 MCP 可用性（initialize + list_tools）。"""
    async def _runner() -> tuple[bool, str]:
        return await _check_remote_mcp_health_async(
            server_url=server_url,
            timeout_seconds=max(1, int(timeout_seconds)),
        )

    return anyio.run(_runner)


def call_remote_tool(
    *,
    server_url: str,
    tool: str,
    args: dict[str, Any],
    raw_text: str,
    timeout_seconds: int,
    circuit_breaker_enabled: bool = True,
    circuit_fail_threshold: int = 3,
    circuit_open_seconds: int = 30,
) -> ToolCallResult:
    """同步上下文下调用远程 MCP 工具。"""
    if circuit_breaker_enabled:
        opened, retry_after = _is_circuit_open()
        if opened:
            return ToolCallResult(
                ok=False,
                route="PLAN_REJECTED",
                payload={"route": "PLAN_REJECTED", "message": "远程 MCP 熔断中，请稍后重试。"},
                error_code="remote_mcp_circuit_open",
                message=f"remote_mcp_circuit_open:retry_after={retry_after}",
                retryable=True,
            )

    async def _runner() -> ToolCallResult:
        return await _call_remote_tool_async(
            server_url=server_url,
            tool=tool,
            args=dict(args or {}),
            raw_text=str(raw_text or ""),
            timeout_seconds=max(1, int(timeout_seconds)),
        )

    result = anyio.run(_runner)
    if not circuit_breaker_enabled:
        return result

    if result.ok:
        _record_success()
        return result

    if str(result.error_code or "") == "remote_mcp_error":
        _record_failure(
            fail_threshold=max(1, int(circuit_fail_threshold)),
            open_seconds=max(1, int(circuit_open_seconds)),
        )
    else:
        _record_success()
    return result
