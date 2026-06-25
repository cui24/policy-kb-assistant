"""
鉴权依赖模块：提供当前用户解析与角色校验。

一、程序目标
1. 从 `Authorization: Bearer <token>` 解析当前用户。
2. 校验 token 有效性与用户启用状态。
3. 提供简单的角色依赖工厂，便于路由层声明权限。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis import Redis

from src.api.auth_cache import (
    AuthCacheError,
    auth_user_from_jwt_payload,
    read_auth_user,
    read_local_auth_user,
    refresh_auth_user_ttl,
    remember_local_auth_user,
)
from src.api.deps import get_redis_dep
from src.api.request_timing import get_current_timing, timing_span_sync
from src.api.security import decode_access_token


bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    """由已签名 JWT 还原出的轻量当前用户。"""

    id: str
    username: str
    role: str
    is_active: bool = True
    email: str | None = None
    phone: str | None = None


def _role_value(role: object) -> str:
    """把枚举或字符串角色统一转成小写字符串。"""
    if hasattr(role, "value"):
        return str(getattr(role, "value"))
    return str(role or "")


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    redis: Redis = Depends(get_redis_dep),
) -> AuthenticatedUser:
    """解析并返回当前登录用户。"""
    with timing_span_sync("auth"):
        if credentials is None or str(credentials.scheme).lower() != "bearer":
            raise HTTPException(status_code=401, detail="not_authenticated")
        token = str(credentials.credentials or "").strip()
        if not token:
            raise HTTPException(status_code=401, detail="empty_token")
        try:
            payload = decode_access_token(token)
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail="invalid_token") from exc

        user_id = str(payload.get("sub") or "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="invalid_token_subject")
        try:
            snapshot = read_auth_user(redis, user_id)
        except AuthCacheError as exc:
            detail = str(exc)
            if detail == "auth_cache_miss":
                raise HTTPException(status_code=401, detail="auth_cache_miss") from exc
            snapshot = read_local_auth_user(user_id)
            if snapshot is None:
                snapshot = auth_user_from_jwt_payload(payload)
                if snapshot is None:
                    raise HTTPException(status_code=503, detail=detail) from exc
                request.state.auth_fallback = "jwt_only"
                request.state.auth_fallback_reason = detail
            else:
                request.state.auth_fallback = "local_snapshot"
                request.state.auth_fallback_reason = detail
        else:
            if snapshot.is_active:
                try:
                    refresh_auth_user_ttl(redis, user_id)
                except AuthCacheError:
                    pass
            remember_local_auth_user(snapshot)
        if not snapshot.is_active:
            raise HTTPException(status_code=403, detail="user_inactive")

        request.state.user_id = user_id
        request.state.username = snapshot.username
        request.state.user_role = snapshot.role
        request.state.user = snapshot
        current_user = AuthenticatedUser(
            id=snapshot.id,
            username=snapshot.username,
            role=snapshot.role,
            is_active=snapshot.is_active,
            email=snapshot.email,
            phone=snapshot.phone,
        )
    timing = get_current_timing()
    request.state.auth_ms = int((timing.snapshot().get("auth", 0) if timing else 0) or 0)
    return current_user


def get_current_active_user(current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    """返回已登录且启用中的用户。"""
    if not bool(current_user.is_active):
        raise HTTPException(status_code=403, detail="user_inactive")
    return current_user


def require_roles(*roles: str) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    """角色依赖工厂：仅允许指定角色访问。"""
    allowed_roles = {str(role).strip().lower() for role in roles if str(role).strip()}

    def _dependency(current_user: AuthenticatedUser = Depends(get_current_active_user)) -> AuthenticatedUser:
        normalized_role = _role_value(current_user.role).strip().lower()
        if allowed_roles and normalized_role not in allowed_roles:
            raise HTTPException(status_code=403, detail="forbidden_role")
        return current_user

    return _dependency
