"""
`/auth` 路由：提供注册、登录与当前用户信息接口。

一、程序目标
1. 提供基础账号注册能力。
2. 提供账号登录并签发 JWT 访问令牌。
3. 提供 `me` 接口验证 token 与获取当前用户信息。
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from redis import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.api import crud, models
from src.api.deps import get_db, get_redis_dep
from src.api.deps_auth import get_current_active_user
from src.api.rate_limit import (
    RateLimitStoreError,
    auth_login_ip_limit,
    auth_login_ip_window_seconds,
    auth_login_limit,
    auth_login_window_seconds,
    consume_rate_limit,
    extract_client_ip,
    rate_limit_headers,
)
from src.api.schemas import AuthLoginRequest, AuthRegisterRequest, AuthTokenResponse, UserProfileResponse
from src.api.security import access_token_expire_seconds, create_access_token, hash_password, verify_password


router = APIRouter(prefix="/auth", tags=["auth"])


def _iso_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _role_value(role: object) -> str:
    if hasattr(role, "value"):
        return str(getattr(role, "value"))
    return str(role or "")


def _serialize_user(user: models.User) -> UserProfileResponse:
    """把用户 ORM 对象序列化为对外响应结构。"""
    return UserProfileResponse(
        id=str(user.id),
        username=str(user.username),
        role=_role_value(user.role),
        email=user.email,
        phone=user.phone,
        is_active=bool(user.is_active),
        created_at=user.created_at.isoformat(),
        updated_at=user.updated_at.isoformat(),
        last_login_at=_iso_or_none(user.last_login_at),
    )


def _build_auth_response(user: models.User) -> AuthTokenResponse:
    """统一构建鉴权响应。"""
    role_value = _role_value(user.role)
    token = create_access_token(
        user_id=str(user.id),
        username=str(user.username),
        role=role_value,
    )
    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=access_token_expire_seconds(),
        user=_serialize_user(user),
    )


@router.post("/register", response_model=AuthTokenResponse)
def register(payload: AuthRegisterRequest, db: Session = Depends(get_db)) -> AuthTokenResponse:
    """注册新用户，并直接返回登录态 token。"""
    username = str(payload.username or "").strip()
    email = str(payload.email or "").strip() or None
    phone = str(payload.phone or "").strip() or None
    if not username:
        raise HTTPException(status_code=422, detail="username_required")

    if crud.get_user_by_username(db, username) is not None:
        raise HTTPException(status_code=409, detail="username_already_exists")
    if email and crud.get_user_by_email(db, email) is not None:
        raise HTTPException(status_code=409, detail="email_already_exists")
    if phone and crud.get_user_by_phone(db, phone) is not None:
        raise HTTPException(status_code=409, detail="phone_already_exists")

    password_hash = hash_password(payload.password)
    try:
        user = crud.create_user(
            db,
            {
                "username": username,
                "password_hash": password_hash,
                "email": email,
                "phone": phone,
                "role": models.RoleEnum.USER,
                "is_active": True,
            },
        )
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="user_already_exists") from exc

    user = crud.update_user_last_login(db, user)
    return _build_auth_response(user)


@router.post("/login", response_model=AuthTokenResponse)
def login(
    payload: AuthLoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_redis_dep),
) -> AuthTokenResponse:
    """用户登录，校验凭证后签发 token。"""
    identifier = str(payload.identifier or "").strip()
    client_ip = extract_client_ip(request)

    try:
        ip_decision = consume_rate_limit(
            redis,
            scope="auth:login:ip",
            subject=f"ip:{client_ip}",
            limit=auth_login_ip_limit(),
            window_seconds=auth_login_ip_window_seconds(),
        )
        if not ip_decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="too_many_login_attempts",
                headers=rate_limit_headers(ip_decision),
            )

        login_decision = consume_rate_limit(
            redis,
            scope="auth:login:principal",
            subject=f"ip:{client_ip}|identifier:{identifier.lower()}",
            limit=auth_login_limit(),
            window_seconds=auth_login_window_seconds(),
        )
    except RateLimitStoreError as exc:
        raise HTTPException(status_code=503, detail="rate_limit_store_unavailable") from exc

    if not login_decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="too_many_login_attempts",
            headers=rate_limit_headers(login_decision),
        )
    for key, value in rate_limit_headers(login_decision).items():
        response.headers[key] = value

    user = crud.get_user_by_login_identifier(db, identifier)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if not verify_password(payload.password, str(user.password_hash or "")):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if not bool(user.is_active):
        raise HTTPException(status_code=403, detail="user_inactive")

    user = crud.update_user_last_login(db, user)
    return _build_auth_response(user)


@router.get("/me", response_model=UserProfileResponse)
def me(current_user: models.User = Depends(get_current_active_user)) -> UserProfileResponse:
    """返回当前登录用户信息。"""
    return _serialize_user(current_user)
