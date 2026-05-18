"""MCP wrapper 契约定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallRequest:
    """统一工具调用入参。"""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    actor: str = "anonymous"
    actor_user_id: str = "anonymous"
    actor_role: str = ""
    department: str = "general"
    idempotency_key: str | None = None
    mode: str | None = None  
    raw_text: str | None = None  # 用户原始输入文本


@dataclass(frozen=True)
class ToolCallResult:
    """统一工具调用出参。"""

    ok: bool
    route: str  # 工具调用结果路由（调用了什么结果路由）
    payload: dict[str, Any] = field(default_factory=dict)  # 主要业务数据负载
    error_code: str | None = None  # 错误码
    message: str | None = None  # 错误消息
    retryable: bool = False  # 是否可重试
