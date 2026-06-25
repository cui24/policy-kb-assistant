"""
L2 API 入口：把 L0/L1 问答系统包装成可被业务系统调用的 FastAPI 服务。

一、程序目标
1. 提供 `/ask`、`/tickets`、`/agent`、`/kb_queries`、`/audit_logs` 等接口。
2. 在启动时确保数据库 schema 已迁移到最新版本。
3. 让当前项目从“可演示网页”扩展到“可被系统集成的 API”。

二、运行顺序
1. 导入模块时先加载运行时环境。
2. 创建 FastAPI 应用。
3. 注册启动事件：调用 `ensure_schema_ready()`。
4. 挂载三组路由。
5. 提供 `/health` 便于联调和部署探活。

三、输入输出
1. 输入：HTTP 请求。
2. 输出：JSON 响应。
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from src.agent_graph.config import load_agent_graph_config
from src.agent_graph.mcp_client import check_remote_mcp_health
from src.api import ask_persist_queue
from src.api.deps import load_runtime_settings
from src.api.migrations import ensure_schema_ready
from src.api.request_timing import RequestTimingMiddleware
from src.api.routes.agent import router as agent_router
from src.api.routes.auth import router as auth_router
from src.api.routes.ask import router as ask_router
from src.api.routes.history import router as history_router
from src.api.routes.ops import router as ops_router
from src.api.routes.tickets import router as tickets_router
from src.kb.retrieve import warmup_retrieval_stack


load_runtime_settings()
logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}

app = FastAPI(
    title="Policy KB Assistant API",
    version="0.2.0",
)
app.add_middleware(RequestTimingMiddleware)


@app.on_event("startup")
def on_startup() -> None:
    """启动时优先执行 Alembic 迁移，确保表结构处于最新版本。"""
    ensure_schema_ready()
    if _env_flag("RETRIEVAL_WARMUP_ON_STARTUP", True):
        try:
            detail = warmup_retrieval_stack()
            logger.info("Retrieval stack warmup completed: %s", detail)
        except Exception:
            logger.exception("Retrieval stack warmup failed; continuing startup")
    ask_persist_queue.start_worker()
    graph_config = load_agent_graph_config()
    if graph_config.mcp_client_enabled and graph_config.mcp_check_on_startup:
        ok, detail = check_remote_mcp_health(
            server_url=graph_config.mcp_server_url,
            timeout_seconds=graph_config.mcp_healthcheck_timeout_seconds,
        )
        if ok:
            logger.info("Remote MCP health check passed: %s", detail)
        else:
            message = f"Remote MCP health check failed: {detail}"
            if graph_config.mcp_startup_strict:
                raise RuntimeError(message)
            logger.warning("%s (non-strict mode, continue startup)", message)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """服务停止时尽量刷完 ASK 异步持久化队列。"""
    await ask_persist_queue.stop_worker()


@app.get("/health")
def health() -> dict:
    """探活接口。"""
    return {"status": "ok", "stage": "l2"}


app.include_router(ask_router)
app.include_router(tickets_router)
app.include_router(agent_router)
app.include_router(history_router)
app.include_router(auth_router)
app.include_router(ops_router)
