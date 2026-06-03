"""MCP wrapper 执行入口：统一调度工具调用。"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from sqlalchemy.orm import Session

from src.api.idempotency import (
    IdempotencyDecision,
    IdempotencyStoreError,
    abort_idempotent_request,
    begin_idempotent_request,
    finish_idempotent_success,
)
from src.api.redis_client import get_redis
from src.mcp_wrapper.contracts import ToolCallRequest, ToolCallResult
from src.mcp_wrapper.errors import (
    BusinessRejectedError,
    IdempotencyConflictError,
    IdempotencyStoreUnavailableError,
    InvalidArgsError,
    MCPToolError,
    PermissionDeniedError,
    RateLimitedError,
    ResourceNotFoundError,
)
from src.mcp_wrapper.registry import dispatch_tool, normalize_tool_name

logger = logging.getLogger(__name__)

_CONCURRENCY_LOCK = threading.Lock()
_CONCURRENCY_LIMIT: int | None = None
_CONCURRENCY_SEMAPHORE: threading.BoundedSemaphore | None = None


FORBIDDEN_ARGUMENT_KEYS = {
    "is_admin",
    "role",
    "db_session",
    "session",
    "actor",
    "actor_user_id",
    "user_id",
    "request_id",
}

FORBIDDEN_TOP_LEVEL_KEYS = {
    "department",
}

IDEMPOTENT_TOOLS = {
    "create_ticket",
    "continue_ticket_draft",
    "confirm_action",
    "ticket_tool_planner",
}


def _mcp_max_concurrent_calls() -> int:
    """读取单进程 MCP 工具并发上限；0 表示关闭本地闸门。"""
    raw = str(os.getenv("MCP_WRAPPER_MAX_CONCURRENT_CALLS") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 16
    return max(0, min(value, 1000))


def _mcp_concurrency_wait_seconds() -> float:
    """读取获取并发槽位的最长等待时间。"""
    raw = str(os.getenv("MCP_WRAPPER_CONCURRENCY_WAIT_SECONDS") or "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.2
    return max(0.0, min(value, 30.0))


def _get_concurrency_semaphore() -> threading.BoundedSemaphore | None:
    """按当前配置返回进程内并发闸门。"""
    global _CONCURRENCY_LIMIT, _CONCURRENCY_SEMAPHORE
    limit = _mcp_max_concurrent_calls()
    if limit <= 0:
        return None
    with _CONCURRENCY_LOCK:
        if _CONCURRENCY_SEMAPHORE is None or _CONCURRENCY_LIMIT != limit:
            _CONCURRENCY_LIMIT = limit
            _CONCURRENCY_SEMAPHORE = threading.BoundedSemaphore(limit)
        return _CONCURRENCY_SEMAPHORE


def _acquire_mcp_call_slot() -> threading.BoundedSemaphore | None:
    """尝试获取一个工具执行槽位，超限时返回可重试限流错误。"""
    semaphore = _get_concurrency_semaphore()
    if semaphore is None:
        return None

    wait_seconds = _mcp_concurrency_wait_seconds()
    if wait_seconds <= 0:
        acquired = semaphore.acquire(blocking=False)
    else:
        acquired = semaphore.acquire(timeout=wait_seconds)
    if not acquired:
        raise RateLimitedError("mcp_concurrency_limit_exceeded")
    return semaphore


def _iter_forbidden_key_paths(data: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            key_text = str(key)
            key_lower = key_text.lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_lower in FORBIDDEN_ARGUMENT_KEYS:
                found.append(key_path)
            found.extend(_iter_forbidden_key_paths(value, path=key_path))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            item_path = f"{path}[{idx}]"
            found.extend(_iter_forbidden_key_paths(item, path=item_path))
    return found


def _validate_request(request: ToolCallRequest) -> None:
    #检查tool不能是空字符串，args必须是字典
    if not str(request.tool or "").strip():
        raise InvalidArgsError("tool is required")
    if not isinstance(request.args, dict):
        raise InvalidArgsError("args must be an object")
    #检查args不能是空字典
    if not request.args:
        raise InvalidArgsError("args must be an object")
    forbidden_top_level = [
        str(key)
        for key in request.args.keys()
        if str(key).lower() in FORBIDDEN_TOP_LEVEL_KEYS
    ]
    if forbidden_top_level:
        raise InvalidArgsError(f"forbidden top-level keys: {', '.join(forbidden_top_level)}")
    forbidden_paths = _iter_forbidden_key_paths(request.args)
    if forbidden_paths:
        raise InvalidArgsError(f"forbidden argument keys: {', '.join(forbidden_paths)}")


def _normalize_payload(payload: Any) -> dict[str, Any]:
    #如果payload是字典，则直接返回，否则返回{"value": payload}
    if isinstance(payload, dict):
        normalized = dict(payload)
    else:
        normalized = {"value": payload}

    #自动补充 route 字段（如果缺失）
    route = str(normalized.get("route") or "")
    if not route:
        if "ticket_detail" in normalized or "ticket" in normalized:
            normalized["route"] = "TICKET_TOOL"
        else:
            normalized["route"] = "MCP_TOOL"
    return normalized


def _build_success_contract(
    *,
    raw_tool: str,
    normalized_tool: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    route = str(data.get("route") or "")
    is_rejected = route.upper() == "PLAN_REJECTED"
    message = str(data.get("message") or "") or None
    error_payload = None
    if is_rejected:
        error_payload = {
            "error_code": "plan_rejected",
            "message": message or "plan rejected",
            "retryable": False,
            "details": {},
        }
    contract = {
        "contract_version": "v1",
        "success": not is_rejected,
        "message": message,
        "data": data,
        "error": error_payload,
        "_tool_meta": {
            "raw_tool": raw_tool,
            "normalized_tool": normalized_tool,
            "deprecated_alias": bool(raw_tool and raw_tool != normalized_tool),
        },
    }
    if route:
        contract["route"] = route
    return contract


def _map_known_exception(exc: Exception) -> MCPToolError | None:
    """将常见业务/权限异常映射为统一错误码。"""
    if isinstance(exc, MCPToolError):
        return exc
    if isinstance(exc, PermissionError):
        return PermissionDeniedError(str(exc) or "permission denied")
    if isinstance(exc, LookupError):
        return ResourceNotFoundError(str(exc) or "resource not found")
    if isinstance(exc, ValueError):
        detail = str(exc) or "invalid value"
        lowered = detail.lower()
        if "rate_limit" in lowered or "too_many" in lowered:
            return RateLimitedError(detail)
        if "idempotency" in lowered:
            return IdempotencyConflictError(detail)
        if lowered.startswith("unsupported_") or lowered.startswith("invalid_"):
            return InvalidArgsError(detail)
        return BusinessRejectedError(detail)
    return None


def _build_error_contract(
    *,
    raw_tool: str,
    normalized_tool: str,
    error_code: str,
    message: str,
    retryable: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": "v1",
        "success": False,
        "message": message,
        "data": None,
        "error": {
            "error_code": error_code,
            "message": message,
            "retryable": bool(retryable),
            "details": dict(details or {}),
        },
        "route": "PLAN_REJECTED",
        "_tool_meta": {
            "raw_tool": raw_tool,
            "normalized_tool": normalized_tool,
            "deprecated_alias": bool(raw_tool and raw_tool != normalized_tool),
        },
    }


def _result_from_contract_payload(payload: dict[str, Any]) -> ToolCallResult:
    route = str(payload.get("route") or "PLAN_REJECTED")
    payload_error = payload.get("error")
    success = bool(payload.get("success")) if "success" in payload else route.upper() != "PLAN_REJECTED"
    error_code = str(payload_error.get("error_code") or "") if isinstance(payload_error, dict) else None
    retryable = bool(payload_error.get("retryable")) if isinstance(payload_error, dict) else False
    return ToolCallResult(
        ok=success,
        route=route,
        payload=payload,
        error_code=error_code,
        message=str(payload.get("message") or "") or None,
        retryable=retryable,
    )


def _extract_effective_idempotency_key(request: ToolCallRequest, args: dict[str, Any]) -> str | None:
    key_from_request = str(request.idempotency_key or "").strip()
    key_from_args = str(args.pop("idempotency_key", "") or "").strip()
    key = key_from_request or key_from_args
    return key or None


def _idempotency_scope(normalized_tool: str) -> str:
    return f"mcp:{normalized_tool or 'unknown'}"


def _idempotency_payload(*, tool: str, args: dict[str, Any], raw_text: str | None) -> dict[str, Any]:
    return {
        "tool": str(tool or ""),
        "args": dict(args or {}),
        "raw_text": str(raw_text or ""),
    }


def _start_idempotent_execution(
    *,
    request: ToolCallRequest,
    normalized_tool: str,
) -> tuple[IdempotencyDecision | None, ToolCallResult | None]:
    if normalized_tool not in IDEMPOTENT_TOOLS:
        return None, None

    idempotency_key = str(request.idempotency_key or "").strip()
    if not idempotency_key:
        return None, None

    try:
        redis = get_redis()
        decision = begin_idempotent_request(
            redis,
            scope=_idempotency_scope(normalized_tool),
            user_id=str(request.actor_user_id or request.actor or "anonymous"),
            idempotency_key=idempotency_key,
            payload=_idempotency_payload(
                tool=normalized_tool,
                args=request.args,
                raw_text=request.raw_text,
            ),
        )
    except IdempotencyStoreError as exc:
        raise IdempotencyStoreUnavailableError(str(exc) or "idempotency_store_unavailable") from exc
    except ValueError as exc:
        raise InvalidArgsError(str(exc) or "invalid_idempotency_key") from exc

    if decision.state == "replay":
        cached = decision.cached_response
        if not isinstance(cached, dict):
            raise IdempotencyConflictError("idempotency_cache_invalid")
        return decision, _result_from_contract_payload(cached)
    if decision.state == "conflict":
        raise IdempotencyConflictError("idempotency_key_conflict")
    if decision.state == "in_progress":
        raise IdempotencyConflictError("idempotency_request_in_progress")
    return decision, None


def _finish_idempotent_execution(decision: IdempotencyDecision | None, payload: dict[str, Any]) -> None:
    if decision is None:
        return
    try:
        finish_idempotent_success(get_redis(), decision, payload)
    except IdempotencyStoreError:
        # 主流程成功时，不因缓存写失败而失败。
        pass


def _abort_idempotent_execution(decision: IdempotencyDecision | None) -> None:
    if decision is None:
        return
    try:
        abort_idempotent_request(get_redis(), decision)
    except IdempotencyStoreError:
        # 撤销失败不覆盖原始业务异常。
        pass


def _log_tool_call(
    *,
    request: ToolCallRequest,
    raw_tool: str,
    normalized_tool: str,
    success: bool,
    error_code: str | None,
    retryable: bool,
    started_at: float,
) -> None:
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        "mcp_tool_call request_id=%s tool=%s normalized_tool=%s actor=%s source=%s latency_ms=%s success=%s error_code=%s retryable=%s",
        str(request.request_id or ""),
        raw_tool,
        normalized_tool,
        str(request.actor_user_id or request.actor or "anonymous"),
        str(request.mode or "unknown"),
        latency_ms,
        bool(success),
        str(error_code or ""),
        bool(retryable),
    )


def invoke_tool(db: Session, request: ToolCallRequest) -> ToolCallResult:
    """执行工具调用并统一封装输出。"""
    started_at = time.perf_counter()
    raw_tool = str(request.tool or "").strip()
    normalized_tool = normalize_tool_name(raw_tool)
    effective_args = dict(request.args or {})
    effective_idempotency_key = _extract_effective_idempotency_key(request, effective_args)
    effective_request = ToolCallRequest(
        tool=request.tool,
        args=effective_args,
        request_id=request.request_id,
        actor=request.actor,
        actor_user_id=request.actor_user_id,
        actor_role=request.actor_role,
        department=request.department,
        idempotency_key=effective_idempotency_key,
        mode=request.mode,
        raw_text=request.raw_text,
    )
    idempotency_decision: IdempotencyDecision | None = None
    acquired_semaphore: threading.BoundedSemaphore | None = None
    try:
        # 验证请求参数
        _validate_request(effective_request)
        idempotency_decision, replay_result = _start_idempotent_execution(
            request=effective_request,
            normalized_tool=normalized_tool,
        )
        if replay_result is not None:
            _log_tool_call(
                request=effective_request,
                raw_tool=raw_tool,
                normalized_tool=normalized_tool,
                success=bool(replay_result.ok),
                error_code=replay_result.error_code,
                retryable=bool(replay_result.retryable),
                started_at=started_at,
            )
            return replay_result

        acquired_semaphore = _acquire_mcp_call_slot()
        try:
            payload_raw, normalized_tool = dispatch_tool(db, effective_request)
        finally:
            if acquired_semaphore is not None:
                acquired_semaphore.release()
                acquired_semaphore = None
        data = _normalize_payload(payload_raw)
        payload = _build_success_contract(
            raw_tool=raw_tool,
            normalized_tool=normalized_tool,
            data=data,
        )
        result = _result_from_contract_payload(payload)
        _finish_idempotent_execution(idempotency_decision, payload)
        _log_tool_call(
            request=effective_request,
            raw_tool=raw_tool,
            normalized_tool=normalized_tool,
            success=bool(result.ok),
            error_code=result.error_code,
            retryable=bool(result.retryable),
            started_at=started_at,
        )
        return result
    except MCPToolError as exc:
        if acquired_semaphore is not None:
            acquired_semaphore.release()
        _abort_idempotent_execution(idempotency_decision)
        # 工具调用失败，返回PLAN_REJECTED结果
        payload = _build_error_contract(
            raw_tool=raw_tool,
            normalized_tool=normalized_tool,
            error_code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
        )
        result = ToolCallResult(
            ok=False,
            route="PLAN_REJECTED",
            payload=payload,
            error_code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
        )
        _log_tool_call(
            request=effective_request,
            raw_tool=raw_tool,
            normalized_tool=normalized_tool,
            success=False,
            error_code=exc.code,
            retryable=exc.retryable,
            started_at=started_at,
        )
        return result
    except (PermissionError, LookupError, ValueError) as exc:
        if acquired_semaphore is not None:
            acquired_semaphore.release()
        _abort_idempotent_execution(idempotency_decision)
        mapped = _map_known_exception(exc)
        assert mapped is not None
        payload = _build_error_contract(
            raw_tool=raw_tool,
            normalized_tool=normalized_tool,
            error_code=mapped.code,
            message=mapped.message,
            retryable=mapped.retryable,
            details={"exception_type": exc.__class__.__name__},
        )
        result = ToolCallResult(
            ok=False,
            route="PLAN_REJECTED",
            payload=payload,
            error_code=mapped.code,
            message=mapped.message,
            retryable=mapped.retryable,
        )
        _log_tool_call(
            request=effective_request,
            raw_tool=raw_tool,
            normalized_tool=normalized_tool,
            success=False,
            error_code=mapped.code,
            retryable=mapped.retryable,
            started_at=started_at,
        )
        return result
    except Exception as exc:
        if acquired_semaphore is not None:
            acquired_semaphore.release()
        _abort_idempotent_execution(idempotency_decision)
        # 工具调用失败，返回PLAN_REJECTED结果
        message = f"internal_error:{exc.__class__.__name__}"
        payload = _build_error_contract(
            raw_tool=raw_tool,
            normalized_tool=normalized_tool,
            error_code="internal_error",
            message="工具执行失败，当前未执行操作。",
            retryable=False,
            details={"exception_type": exc.__class__.__name__},
        )
        result = ToolCallResult(
            ok=False,
            route="PLAN_REJECTED",
            payload=payload,
            error_code="internal_error",
            message=message,
            retryable=False,
        )
        _log_tool_call(
            request=effective_request,
            raw_tool=raw_tool,
            normalized_tool=normalized_tool,
            success=False,
            error_code="internal_error",
            retryable=False,
            started_at=started_at,
        )
        return result
