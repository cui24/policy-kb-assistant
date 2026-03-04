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

from fastapi import FastAPI

from src.api.deps import load_runtime_settings
from src.api.migrations import ensure_schema_ready
from src.api.routes.agent import router as agent_router
from src.api.routes.ask import router as ask_router
from src.api.routes.history import router as history_router
from src.api.routes.tickets import router as tickets_router


load_runtime_settings()

app = FastAPI(
    title="Policy KB Assistant API",
    version="0.2.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """启动时优先执行 Alembic 迁移，确保表结构处于最新版本。"""
    ensure_schema_ready()


@app.get("/health")
def health() -> dict:
    """探活接口。"""
    return {"status": "ok", "stage": "l2"}


app.include_router(ask_router)
app.include_router(tickets_router)
app.include_router(agent_router)
app.include_router(history_router)
