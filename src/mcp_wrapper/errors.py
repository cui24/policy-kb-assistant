"""MCP wrapper 异常与错误码。"""

from __future__ import annotations


class MCPToolError(Exception):
    """统一工具调用异常。"""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = str(code or "internal_error")
        self.message = str(message or "internal_error")
        self.retryable = bool(retryable)


class ToolNotFoundError(MCPToolError):
    def __init__(self, tool: str) -> None:
        super().__init__("tool_not_found", f"unknown tool: {tool}", retryable=False)


class InvalidArgsError(MCPToolError):
    def __init__(self, detail: str) -> None:
        super().__init__("invalid_args", detail, retryable=False)


class PermissionDeniedError(MCPToolError):
    def __init__(self, detail: str = "permission denied") -> None:
        super().__init__("permission_denied", detail, retryable=False)


class ResourceNotFoundError(MCPToolError):
    def __init__(self, detail: str = "resource not found") -> None:
        super().__init__("resource_not_found", detail, retryable=False)


class RateLimitedError(MCPToolError):
    def __init__(self, detail: str = "rate limited") -> None:
        super().__init__("rate_limited", detail, retryable=True)


class IdempotencyConflictError(MCPToolError):
    def __init__(self, detail: str = "idempotency conflict") -> None:
        super().__init__("idempotency_conflict", detail, retryable=True)


class BusinessRejectedError(MCPToolError):
    def __init__(self, detail: str = "business rejected") -> None:
        super().__init__("business_rejected", detail, retryable=False)


class IdempotencyStoreUnavailableError(MCPToolError):
    def __init__(self, detail: str = "idempotency store unavailable") -> None:
        super().__init__("idempotency_store_unavailable", detail, retryable=True)
