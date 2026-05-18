"""
限流模块：基于 Redis 固定时间窗口实现请求频率控制。

当前用于：
1. `/auth/login` 防暴力尝试。
2. `/agent` 防高频滥用。
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

from fastapi import Request
from redis import Redis
from redis.exceptions import RedisError


_DEFAULT_AUTH_LOGIN_LIMIT = 8
_DEFAULT_AUTH_LOGIN_WINDOW_SECONDS = 60
_DEFAULT_AUTH_LOGIN_IP_LIMIT = 30
_DEFAULT_AUTH_LOGIN_IP_WINDOW_SECONDS = 60
_DEFAULT_AGENT_LIMIT = 20
_DEFAULT_AGENT_WINDOW_SECONDS = 60


class RateLimitStoreError(RuntimeError):
    """Redis 限流存储异常。"""


@dataclass(frozen=True)
class RateLimitDecision:
    """一次限流判定结果。"""

    allowed: bool
    limit: int
    current: int
    remaining: int
    retry_after_seconds: int
    window_seconds: int
    redis_key: str


def _int_from_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """读取整型环境变量并做边界保护，解析失败时回退默认值。"""
    raw = str(os.getenv(name) or "").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = default
    return max(minimum, min(parsed, maximum))


def auth_login_limit() -> int:
    """登录接口（按 identifier 维度）的单位窗口最大请求数。"""
    return _int_from_env("AUTH_LOGIN_RATE_LIMIT_MAX", _DEFAULT_AUTH_LOGIN_LIMIT, minimum=1, maximum=500)


def auth_login_window_seconds() -> int:
    """登录接口（按 identifier 维度）的限流窗口秒数。"""
    return _int_from_env(
        "AUTH_LOGIN_RATE_LIMIT_WINDOW_SECONDS",
        _DEFAULT_AUTH_LOGIN_WINDOW_SECONDS,
        minimum=1,
        maximum=3600,
    )


def auth_login_ip_limit() -> int:
    """登录接口（按 IP 维度）的单位窗口最大请求数。"""
    return _int_from_env("AUTH_LOGIN_IP_RATE_LIMIT_MAX", _DEFAULT_AUTH_LOGIN_IP_LIMIT, minimum=1, maximum=2000)


def auth_login_ip_window_seconds() -> int:
    """登录接口（按 IP 维度）的限流窗口秒数。"""
    return _int_from_env(
        "AUTH_LOGIN_IP_RATE_LIMIT_WINDOW_SECONDS",
        _DEFAULT_AUTH_LOGIN_IP_WINDOW_SECONDS,
        minimum=1,
        maximum=3600,
    )


def agent_limit() -> int:
    """`/agent` 接口的单位窗口最大请求数。"""
    return _int_from_env("AGENT_RATE_LIMIT_MAX", _DEFAULT_AGENT_LIMIT, minimum=1, maximum=2000)


def agent_window_seconds() -> int:
    """`/agent` 接口的限流窗口秒数。"""
    return _int_from_env("AGENT_RATE_LIMIT_WINDOW_SECONDS", _DEFAULT_AGENT_WINDOW_SECONDS, minimum=1, maximum=3600)


def extract_client_ip(request: Request) -> str:
    """
    从请求中提取客户端 IP。
    兼容反向代理场景：优先 `X-Forwarded-For`，否则回退 `request.client.host`。
    """
    forwarded_for = str(request.headers.get("x-forwarded-for") or "").strip()
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return str(request.client.host).strip()
    return "unknown"


def _normalize_subject(subject: str) -> str:
    """规范化限流主体（user/ip/identifier 组合）；空值回退 anonymous。"""
    normalized = str(subject or "").strip()
    return normalized or "anonymous"


def _build_rate_limit_key(scope: str, subject: str, window_seconds: int, now_ts: int) -> str:
    """
    构造固定窗口限流 key。

    结构：
    `prefix:scope:window_seconds:slot:subject_hash`
    - `slot = now_ts // window_seconds` 用于时间分桶。
    - `subject_hash` 避免把敏感明文直接写入 Redis key。
    """
    prefix = str(os.getenv("RATE_LIMIT_REDIS_PREFIX") or "rate").strip() or "rate"
    slot = now_ts // window_seconds
    subject_hash = hashlib.sha256(_normalize_subject(subject).encode("utf-8")).hexdigest()[:24]
    normalized_scope = str(scope or "").strip() or "unknown_scope"
    return f"{prefix}:{normalized_scope}:{window_seconds}:{slot}:{subject_hash}"


def _retry_after_seconds(window_seconds: int, now_ts: int) -> int:
    """计算超限后建议重试秒数（到当前窗口结束）。"""
    elapsed = now_ts % window_seconds
    retry_after = window_seconds - elapsed
    return max(1, retry_after)


def consume_rate_limit(
    redis: Redis,
    *,
    scope: str,
    subject: str,
    limit: int,
    window_seconds: int,
) -> RateLimitDecision:
    """
    消耗一次请求额度并返回判定结果（固定窗口算法）。

    逻辑：
    1. 通过 `INCR` 递增窗口计数。
    2. 首次计数时设置 key 过期（窗口秒数 + 2 秒冗余）。
    3. 当 `current > limit` 时判定为限流。
    """
    safe_limit = max(1, int(limit))
    safe_window = max(1, int(window_seconds))
    now_ts = int(time.time())
    redis_key = _build_rate_limit_key(scope=scope, subject=subject, window_seconds=safe_window, now_ts=now_ts)

    try:
        current = int(redis.incr(redis_key))
        if current == 1:
            redis.expire(redis_key, safe_window + 2)
    except RedisError as exc:
        raise RateLimitStoreError(f"rate_limit_store_unavailable:{exc.__class__.__name__}") from exc

    remaining = max(safe_limit - current, 0)
    allowed = current <= safe_limit
    retry_after = 0 if allowed else _retry_after_seconds(safe_window, now_ts)
    return RateLimitDecision(
        allowed=allowed,
        limit=safe_limit,
        current=current,
        remaining=remaining,
        retry_after_seconds=retry_after,
        window_seconds=safe_window,
        redis_key=redis_key,
    )


def rate_limit_headers(decision: RateLimitDecision) -> dict[str, str]:
    """把限流判定转成响应头。"""
    headers = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Window": str(decision.window_seconds),
    }
    if not decision.allowed and decision.retry_after_seconds > 0:
        headers["Retry-After"] = str(decision.retry_after_seconds)
    return headers
