"""
鉴权依赖模块：提供当前用户解析与角色校验。

一、程序目标
1. 从 `Authorization: Bearer <token>` 解析当前用户。
2. 校验 token 有效性与用户启用状态。
3. 提供简单的角色依赖工厂，便于路由层声明权限。
"""

from __future__ import annotations

from collections.abc import Callable

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.api import crud, models
from src.api.deps import get_db
from src.api.security import decode_access_token


bearer_scheme = HTTPBearer(auto_error=False)


def _role_value(role: object) -> str:
    """把枚举或字符串角色统一转成小写字符串。"""
    if hasattr(role, "value"):
        return str(getattr(role, "value"))
    return str(role or "")


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    """解析并返回当前登录用户。"""
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

    user = crud.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="user_not_found")
    return user


def get_current_active_user(current_user: models.User = Depends(get_current_user)) -> models.User:
    """返回已登录且启用中的用户。"""
    if not bool(current_user.is_active):
        raise HTTPException(status_code=403, detail="user_inactive")
    return current_user


def require_roles(*roles: str) -> Callable[[models.User], models.User]:
    """角色依赖工厂：仅允许指定角色访问。"""
    allowed_roles = {str(role).strip().lower() for role in roles if str(role).strip()}

    def _dependency(current_user: models.User = Depends(get_current_active_user)) -> models.User:
        normalized_role = _role_value(current_user.role).strip().lower()
        if allowed_roles and normalized_role not in allowed_roles:
            raise HTTPException(status_code=403, detail="forbidden_role")
        return current_user

    return _dependency
