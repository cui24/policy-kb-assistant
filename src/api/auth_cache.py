"""Redis 中的当前用户鉴权快照。

使用 Hash 而不是 JSON 字符串，便于后续单独 HSET 某个字段，例如禁用用户时
只改 `is_active`，角色变更时只改 `role`。
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from redis import Redis
from redis.exceptions import RedisError

from src.api import models


AUTH_USER_KEY_PREFIX = "auth:user:"
DEFAULT_AUTH_USER_TTL_SECONDS = 24 * 3600
DEFAULT_LOCAL_AUTH_FALLBACK_TTL_SECONDS = 30
DEFAULT_LOCAL_AUTH_FALLBACK_MAXSIZE = 500

_LOCAL_CACHE_LOCK = threading.Lock()
_LOCAL_CACHE: OrderedDict[str, tuple[AuthUserSnapshot, float]] = OrderedDict()


class AuthCacheError(RuntimeError):
    """鉴权缓存不可用或内容异常。"""


@dataclass(frozen=True)
class AuthUserSnapshot:
    """Redis 中保存的轻量当前用户快照。"""

    id: str
    username: str
    role: str
    is_active: bool
    email: str | None = None
    phone: str | None = None


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def local_auth_fallback_enabled() -> bool:
    """Redis 鉴权快照读取异常时，是否允许使用进程内短 TTL 快照兜底。"""
    return _bool_env("AUTH_LOCAL_FALLBACK_ENABLED", True)


def local_auth_fallback_ttl_seconds() -> int:
    """进程内鉴权兜底缓存 TTL。保持很短，降低用户禁用/改权后的不一致窗口。"""
    return _int_env(
        "AUTH_LOCAL_FALLBACK_TTL_SECONDS",
        DEFAULT_LOCAL_AUTH_FALLBACK_TTL_SECONDS,
        1,
        300,
    )


def local_auth_fallback_maxsize() -> int:
    """进程内鉴权兜底缓存最大用户数，避免 Worker 常驻内存无界增长。"""
    return _int_env(
        "AUTH_LOCAL_FALLBACK_MAXSIZE",
        DEFAULT_LOCAL_AUTH_FALLBACK_MAXSIZE,
        1,
        100_000,
    )


def jwt_only_fallback_enabled() -> bool:
    """Redis 鉴权快照不可用且本地无快照时，是否允许基于已签名 JWT 短暂放行。"""
    return _bool_env("AUTH_JWT_ONLY_FALLBACK_ENABLED", True)


def reset_local_auth_cache() -> None:
    """清空进程内鉴权兜底缓存，主要供测试使用。"""
    with _LOCAL_CACHE_LOCK:
        _LOCAL_CACHE.clear()


def remember_local_auth_user(snapshot: AuthUserSnapshot) -> None:
    """记录最近一次 Redis 成功读取的用户快照，供短暂 Redis 故障兜底。"""
    if not local_auth_fallback_enabled():
        return
    if not snapshot.is_active:
        return
    expires_at = time.time() + local_auth_fallback_ttl_seconds()
    with _LOCAL_CACHE_LOCK:
        now = time.time()
        for key, (_, item_expires_at) in list(_LOCAL_CACHE.items()):
            if item_expires_at <= now:
                _LOCAL_CACHE.pop(key, None)
        _LOCAL_CACHE[str(snapshot.id)] = (snapshot, expires_at)
        _LOCAL_CACHE.move_to_end(str(snapshot.id))
        while len(_LOCAL_CACHE) > local_auth_fallback_maxsize():
            _LOCAL_CACHE.popitem(last=False)


def read_local_auth_user(user_id: str) -> AuthUserSnapshot | None:
    """读取进程内短 TTL 用户快照；过期或禁用态一律不返回。"""
    if not local_auth_fallback_enabled():
        return None
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        return None
    with _LOCAL_CACHE_LOCK:
        entry = _LOCAL_CACHE.get(normalized_user_id)
        if entry is None:
            return None
        snapshot, expires_at = entry
        if expires_at <= time.time() or not snapshot.is_active:
            _LOCAL_CACHE.pop(normalized_user_id, None)
            return None
        _LOCAL_CACHE.move_to_end(normalized_user_id)
        return snapshot


def auth_user_from_jwt_payload(payload: dict) -> AuthUserSnapshot | None:
    """Redis 与本地快照都不可用时，从已验签 JWT 构造最小用户快照。"""
    if not jwt_only_fallback_enabled():
        return None
    user_id = str(payload.get("sub") or "").strip()
    username = str(payload.get("username") or "").strip()
    role = str(payload.get("role") or "user").strip() or "user"
    if not user_id or not username:
        return None
    return AuthUserSnapshot(
        id=user_id,
        username=username,
        role=role,
        is_active=True,
    )


def auth_user_key(user_id: str) -> str:
    """生成用户鉴权快照 Redis key。"""
    return f"{AUTH_USER_KEY_PREFIX}{str(user_id or '').strip()}"


def _role_value(role: object) -> str:
    if hasattr(role, "value"):
        return str(getattr(role, "value"))
    return str(role or "")


def _bool_string(value: object) -> str:
    return "1" if bool(value) else "0"


def _parse_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def user_hash_from_model(user: models.User) -> dict[str, str]:
    """把 ORM 用户转换成 Redis Hash 字段。"""
    return {
        "id": str(user.id),
        "username": str(user.username),
        "role": _role_value(user.role),
        "is_active": _bool_string(user.is_active),
        "email": str(user.email or ""),
        "phone": str(user.phone or ""),
    }


def write_auth_user(
    redis: Redis,
    user: models.User,
    *,
    ttl_seconds: int = DEFAULT_AUTH_USER_TTL_SECONDS,
) -> None:
    """写入或刷新当前用户鉴权快照。"""
    key = auth_user_key(str(user.id))
    try:
        redis.hset(key, mapping=user_hash_from_model(user))
        redis.expire(key, max(60, int(ttl_seconds)))
    except Exception as exc:
        raise AuthCacheError(f"auth_cache_write_failed:{exc.__class__.__name__}") from exc


def refresh_auth_user_ttl(
    redis: Redis,
    user_id: str,
    *,
    ttl_seconds: int = DEFAULT_AUTH_USER_TTL_SECONDS,
) -> None:
    """成功鉴权后刷新 Redis 用户快照 TTL，让活跃用户不会自然过期。"""
    key = auth_user_key(user_id)
    try:
        redis.expire(key, max(60, int(ttl_seconds)))
    except Exception as exc:
        raise AuthCacheError(f"auth_cache_refresh_failed:{exc.__class__.__name__}") from exc


def read_auth_user(redis: Redis, user_id: str) -> AuthUserSnapshot:
    """读取当前用户鉴权快照；缺失视为未登录态不可用。"""
    key = auth_user_key(user_id)
    try:
        raw = redis.hgetall(key)
    except Exception as exc:
        raise AuthCacheError(f"auth_cache_read_failed:{exc.__class__.__name__}") from exc
    if not raw:
        raise AuthCacheError("auth_cache_miss")

    data = {str(key): str(value) for key, value in dict(raw).items()}
    snapshot_user_id = str(data.get("id") or "").strip()
    username = str(data.get("username") or "").strip()
    role = str(data.get("role") or "").strip()
    if not snapshot_user_id or snapshot_user_id != str(user_id or "").strip():
        raise AuthCacheError("auth_cache_invalid_user_id")
    if not username:
        raise AuthCacheError("auth_cache_missing_username")
    if not role:
        role = "user"

    return AuthUserSnapshot(
        id=snapshot_user_id,
        username=username,
        role=role,
        is_active=_parse_bool(data.get("is_active")),
        email=str(data.get("email") or "") or None,
        phone=str(data.get("phone") or "") or None,
    )
