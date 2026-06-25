"""轻量请求打点工具。

目标是给压测前的定位提供低侵入观测：
- middleware 为每个请求初始化 timing collector；
- 业务代码使用 `async with timing_span("stage")` 包住阶段；
- 响应可复用 `request.state.timing.snapshot()` 放入 diagnostics；
- 日志只输出结构化摘要，不引入 Jaeger/Zipkin 这类重设施。
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger("src.api.request_timing")
_current_timing: ContextVar["RequestTiming | None"] = ContextVar("request_timing", default=None)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass
class RequestTiming:
    """单个请求内的分段耗时收集器。"""

    request_id: str = ""
    timings_ms: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed_ms: int) -> None:
        """记录一个阶段耗时；重复阶段会累加，适合循环/重试场景。"""
        key = str(name or "").strip()
        if not key:
            return
        self.timings_ms[key] = int(self.timings_ms.get(key, 0)) + max(0, int(elapsed_ms))

    def snapshot(self) -> dict[str, int]:
        """返回稳定副本，供响应 diagnostics 或日志使用。"""
        return dict(self.timings_ms)


def get_current_timing() -> RequestTiming | None:
    """读取当前请求的 timing collector。"""
    return _current_timing.get()


def record_timing(name: str, elapsed_ms: int) -> None:
    """记录已由下游组件产出的耗时值。"""
    timing = get_current_timing()
    if timing is not None:
        timing.add(name, elapsed_ms)


@asynccontextmanager
async def timing_span(name: str) -> AsyncIterator[None]:
    """异步阶段计时器，用于 async 路由/工作流。"""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        record_timing(name, elapsed_ms)


@contextmanager
def timing_span_sync(name: str) -> Iterator[None]:
    """同步阶段计时器，用于仍是 sync 的小段逻辑。"""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        record_timing(name, elapsed_ms)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """为每个请求初始化 timing collector，并可选输出结构化日志。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        timing = RequestTiming()
        request.state.timing = timing
        token = _current_timing.set(timing)
        status_code = 500
        try:
            async with timing_span("http_total"):
                response = await call_next(request)
            status_code = int(response.status_code)
            response.headers["X-Request-Timing-Total-Ms"] = str(timing.snapshot().get("http_total", 0))
            auth_fallback = str(getattr(request.state, "auth_fallback", "") or "").strip()
            if auth_fallback:
                response.headers["X-Auth-Fallback"] = "true"
                response.headers["X-Auth-Fallback-Mode"] = auth_fallback
                logger.warning(
                    "auth_fallback mode=%s reason=%s path=%s",
                    auth_fallback,
                    str(getattr(request.state, "auth_fallback_reason", "") or ""),
                    request.url.path,
                )
            return response
        finally:
            if _bool_env("REQUEST_TIMING_LOG_ENABLED", False):
                logger.info(
                    "request_timing method=%s path=%s status=%s timing_ms=%s",
                    request.method,
                    request.url.path,
                    status_code,
                    timing.snapshot(),
                )
            _current_timing.reset(token)
