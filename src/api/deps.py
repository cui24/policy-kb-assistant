"""
L2/L3 依赖模块：负责运行时环境、数据库会话与最小鉴权依赖。

一、程序目标
1. 统一加载 `.env`。
2. 统一设置 API 默认运行在 `l2` 配置。
3. 统一提供 FastAPI 的数据库依赖。
4. 为写接口提供最小 API Key 鉴权。
"""

from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from fastapi import Header, HTTPException
from sqlalchemy.orm import Session

from src.api.db import SessionLocal



def load_runtime_settings() -> None:
    """加载环境变量，并确保 API 默认使用 L2 级别配置。"""
    load_dotenv()
    os.environ.setdefault("APP_LEVEL", "l2")


load_runtime_settings()



def get_db() -> Generator[Session, None, None]:
    """为每个请求提供独立数据库会话，请求结束后关闭。"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def expected_api_key() -> str:
    """读取当前服务要求的 API Key；未配置时返回空字符串。"""
    return str(os.getenv("POLICY_API_KEY") or "").strip()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """校验写接口的 `X-API-Key`；服务未配置 key 时拒绝写操作。"""
    configured_key = expected_api_key()
    if not configured_key:
        raise HTTPException(status_code=500, detail="api_key_not_configured")
    normalized_key = (x_api_key or "").strip()
    if normalized_key != configured_key:
        raise HTTPException(status_code=401, detail="invalid_api_key")
