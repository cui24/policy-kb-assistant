"""
幂等处理模块：为写接口提供“同键只执行一次”的基础能力。

当前先服务于 `POST /tickets`，后续可复用到其他写接口。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from redis import Redis
from redis.exceptions import RedisError


IdempotencyState = Literal["new", "replay", "in_progress", "conflict"]

_DEFAULT_PROCESSING_TTL_SECONDS = 90
_DEFAULT_SUCCESS_TTL_SECONDS = 24 * 60 * 60


class IdempotencyStoreError(RuntimeError):
    """幂等存储层异常（Redis 异常）。"""


@dataclass(frozen=True)
class IdempotencyDecision:
    """一次幂等判定结果。"""

    state: IdempotencyState
    redis_key: str
    payload_fingerprint: str
    cached_response: dict[str, Any] | None = None


def _processing_ttl_seconds() -> int:
    """读取 processing 状态 TTL（秒）并做边界保护。"""
    raw = str(os.getenv("IDEMPOTENCY_PROCESSING_TTL_SECONDS") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_PROCESSING_TTL_SECONDS
    return max(15, min(value, 600))


def _success_ttl_seconds() -> int:
    """读取 success 状态 TTL（秒）并做边界保护。"""
    raw = str(os.getenv("IDEMPOTENCY_SUCCESS_TTL_SECONDS") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_SUCCESS_TTL_SECONDS
    return max(60, min(value, 7 * 24 * 60 * 60))


def _normalize_user_id(user_id: str) -> str:
    """规范化 user_id；空值统一落到 anonymous，避免 key 结构异常。"""
    normalized = str(user_id or "").strip()
    return normalized or "anonymous"


def _normalize_idempotency_key(idempotency_key: str) -> str:
    """
    校验并规范化幂等键。

    约束：
    1. 不能为空。
    2. 长度不超过 128（避免恶意超长 key）。
    """
    normalized = str(idempotency_key or "").strip()
    if not normalized:
        raise ValueError("idempotency_key_required")
    if len(normalized) > 128:
        raise ValueError("idempotency_key_too_long")
    return normalized


def _build_redis_key(*, scope: str, user_id: str, idempotency_key: str) -> str:
    """构造 Redis 幂等键，按“前缀+作用域+用户+业务键”分段。"""
    prefix = str(os.getenv("IDEMPOTENCY_REDIS_PREFIX") or "idem").strip() or "idem"
    normalized_scope = str(scope or "").strip() or "unknown_scope"
    return f"{prefix}:{normalized_scope}:{_normalize_user_id(user_id)}:{idempotency_key}"


def _json_dumps(payload: dict[str, Any]) -> str:
    """稳定序列化 JSON：固定键顺序，便于跨请求做一致性比较。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    """计算请求载荷指纹，用于判定“同 key 是否同请求”。"""
    raw = _json_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _decode_record(raw_value: str | bytes | None) -> dict[str, Any] | None:
    """把 Redis 原始字符串解码为 dict；非法 JSON 返回 None。"""
    if raw_value is None:
        return None
    if isinstance(raw_value, bytes):
        text = raw_value.decode("utf-8", errors="replace")
    else:
        text = str(raw_value)
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _safe_cached_response(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """从幂等记录中安全提取缓存响应，仅接受 dict 结构。"""
    if not isinstance(record, dict):
        return None
    payload = record.get("response")
    return payload if isinstance(payload, dict) else None


def begin_idempotent_request(
    redis: Redis,
    *,
    scope: str,
    user_id: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> IdempotencyDecision:
    """
    开始一次幂等请求。

    处理步骤：
    1. 先用 `SET key value NX EX` 抢执行权，抢到则返回 `new`。
    2. 未抢到则读取已有记录，对比 payload 指纹：
       - 指纹不同：`conflict`
       - 已 success：`replay`
       - 其余状态：`in_progress`
    3. 若读取时恰好 key 消失（并发间隙），再做一次兜底抢锁。

    返回状态含义：
    - `new`：本次请求可继续执行业务。
    - `replay`：直接返回历史成功响应。
    - `in_progress`：已有同键请求处理中。
    - `conflict`：同 key 但 payload 不一致。
    """

    normalized_key = _normalize_idempotency_key(idempotency_key)
    normalized_payload = payload if isinstance(payload, dict) else {}
    payload_fingerprint = _fingerprint_payload(normalized_payload)
    redis_key = _build_redis_key(scope=scope, user_id=user_id, idempotency_key=normalized_key)
    processing_record = {
        "status": "processing",
        "fingerprint": payload_fingerprint,
        "updated_at": int(time.time()),
    }

    try:
        # NX 原子抢占执行权，避免同 key 并发双写。
        locked = bool(redis.set(redis_key, _json_dumps(processing_record), ex=_processing_ttl_seconds(), nx=True))
        if locked:
            return IdempotencyDecision(
                state="new",
                redis_key=redis_key,
                payload_fingerprint=payload_fingerprint,
            )
        current = _decode_record(redis.get(redis_key))
    except RedisError as exc:
        raise IdempotencyStoreError(f"idempotency_store_unavailable:{exc.__class__.__name__}") from exc

    if current is None:
        # key 可能在并发间隙中过期/被清理，这里给一次兜底重试。
        try:
            locked = bool(redis.set(redis_key, _json_dumps(processing_record), ex=_processing_ttl_seconds(), nx=True))
        except RedisError as exc:
            raise IdempotencyStoreError(f"idempotency_store_unavailable:{exc.__class__.__name__}") from exc
        if locked:
            return IdempotencyDecision(
                state="new",
                redis_key=redis_key,
                payload_fingerprint=payload_fingerprint,
            )
        return IdempotencyDecision(
            state="in_progress",
            redis_key=redis_key,
            payload_fingerprint=payload_fingerprint,
        )

    existing_fingerprint = str(current.get("fingerprint") or "")
    if existing_fingerprint and existing_fingerprint != payload_fingerprint:
        return IdempotencyDecision(
            state="conflict",
            redis_key=redis_key,
            payload_fingerprint=payload_fingerprint,
        )

    status = str(current.get("status") or "")
    if status == "success":
        return IdempotencyDecision(
            state="replay",
            redis_key=redis_key,
            payload_fingerprint=payload_fingerprint,
            cached_response=_safe_cached_response(current),
        )

    return IdempotencyDecision(
        state="in_progress",
        redis_key=redis_key,
        payload_fingerprint=payload_fingerprint,
    )


def finish_idempotent_success(redis: Redis, decision: IdempotencyDecision, response_payload: dict[str, Any]) -> None:
    """
    在业务成功后写入 success 记录。

    记录中包含：
    1. payload 指纹（用于后续冲突校验）。
    2. 成功响应 JSON（用于 replay）。
    3. 更新时间戳。
    """
    normalized_response = response_payload if isinstance(response_payload, dict) else {}
    success_record = {
        "status": "success",
        "fingerprint": decision.payload_fingerprint,
        "response": normalized_response,
        "updated_at": int(time.time()),
    }
    try:
        redis.set(decision.redis_key, _json_dumps(success_record), ex=_success_ttl_seconds())
    except RedisError as exc:
        raise IdempotencyStoreError(f"idempotency_store_unavailable:{exc.__class__.__name__}") from exc


def abort_idempotent_request(redis: Redis, decision: IdempotencyDecision) -> None:
    """
    业务失败时尝试清理 processing 状态。

    安全条件：
    1. 当前记录状态仍是 processing。
    2. 当前记录 fingerprint 与本次请求一致。

    这样可避免误删“后来者”写入的 success 结果。
    """
    try:
        current = _decode_record(redis.get(decision.redis_key))
        if not isinstance(current, dict):
            return
        if str(current.get("status") or "") != "processing":
            return
        if str(current.get("fingerprint") or "") != decision.payload_fingerprint:
            return
        redis.delete(decision.redis_key)
    except RedisError as exc:
        raise IdempotencyStoreError(f"idempotency_store_unavailable:{exc.__class__.__name__}") from exc
