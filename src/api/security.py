"""
鉴权安全工具：密码哈希与 JWT 令牌编解码。

一、程序目标
1. 统一登录密码哈希算法，避免明文存储。
2. 统一 Access Token 的签发与校验逻辑。
3. 为后续 `get_current_user` 依赖提供基础能力。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt


PBKDF2_ALGORITHM = "sha256"
PBKDF2_ITERATIONS = 310_000
JWT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 60


def _jwt_secret_key() -> str:
    """读取 JWT 密钥；开发环境提供回退值，便于本地联调。"""
    return str(os.getenv("JWT_SECRET_KEY") or "dev-insecure-jwt-key-change-me")


def _jwt_issuer() -> str:
    """读取 JWT 签发者。"""
    return str(os.getenv("JWT_ISSUER") or "policy-kb-assistant")


def _jwt_audience() -> str:
    """读取 JWT 接收方。"""
    return str(os.getenv("JWT_AUDIENCE") or "policy-kb-clients")


def _access_token_expire_minutes() -> int:
    """读取 Access Token 过期分钟数。"""
    raw = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
    if raw is None:
        return DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES


def access_token_expire_seconds() -> int:
    """返回 Access Token 过期秒数。"""
    return _access_token_expire_minutes() * 60


def hash_password(password: str) -> str:
    """
    生成密码哈希。

    存储格式：
    `pbkdf2_sha256$<iterations>$<salt_b64>$<digest_b64>`
    """
    normalized = str(password or "")
    if not normalized:
        raise ValueError("password_empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        normalized.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, password_hash: str) -> bool:
    """校验用户输入密码是否匹配数据库中的哈希。"""
    normalized = str(password or "")
    encoded_hash = str(password_hash or "")
    parts = encoded_hash.split("$")
    if len(parts) != 4:
        return False
    scheme, iterations_raw, salt_b64, digest_b64 = parts
    if scheme != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected_digest = base64.b64decode(digest_b64.encode("ascii"))
    except Exception:
        return False
    provided_digest = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        normalized.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(provided_digest, expected_digest)


def create_access_token(
    *,
    user_id: str,
    username: str,
    role: str,
    expires_delta: timedelta | None = None,
) -> str:
    """签发 Access Token。"""
    now = datetime.now(timezone.utc)
    expire_at = now + (expires_delta or timedelta(minutes=_access_token_expire_minutes()))
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": str(username),
        "role": str(role),
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expire_at.timestamp()),
        "iss": _jwt_issuer(),
        "aud": _jwt_audience(),
        "typ": "access",
    }
    return str(jwt.encode(payload, _jwt_secret_key(), algorithm=JWT_ALGORITHM))


def decode_access_token(token: str) -> dict[str, Any]:
    """解码并校验 Access Token，失败时抛出 PyJWT 异常。"""
    normalized = str(token or "").strip()
    if not normalized:
        raise jwt.InvalidTokenError("token_empty")
    return jwt.decode(
        normalized,
        _jwt_secret_key(),
        algorithms=[JWT_ALGORITHM],
        audience=_jwt_audience(),
        issuer=_jwt_issuer(),
    )
