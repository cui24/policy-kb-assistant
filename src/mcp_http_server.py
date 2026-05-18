"""
MCP HTTP server 入口：以 Streamable HTTP 常驻运行，供远端客户端调用。

一、程序目标
1. 把 MCP 从本地 stdio 升级为服务器常驻服务。
2. 复用已有 tool 定义与业务逻辑，不重复实现。
3. 通过环境变量配置监听地址与路径，便于容器化部署。
"""

from __future__ import annotations

import logging
import os

from src.api.deps import load_runtime_settings
from src.api.migrations import ensure_schema_ready
from src.mcp_stdio_server import build_mcp_server


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，解析失败时回退默认值。"""
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def main() -> None:
    """以 streamable-http transport 启动 MCP server。"""
    logging.basicConfig(level=logging.INFO)
    load_runtime_settings()
    ensure_schema_ready()

    actor_user_id = str(os.getenv("MCP_ACTOR_USER_ID") or "").strip()
    if not actor_user_id:
        raise RuntimeError("MCP_ACTOR_USER_ID is required for MCP HTTP server.")

    department = str(os.getenv("MCP_DEPARTMENT") or "IT").strip() or "IT"
    host = str(os.getenv("MCP_HOST") or "0.0.0.0").strip() or "0.0.0.0"
    port = _env_int("MCP_PORT", 9000)
    streamable_http_path = str(os.getenv("MCP_STREAMABLE_HTTP_PATH") or "/mcp").strip() or "/mcp"

    logger.info(
        "Starting MCP HTTP server actor=%s department=%s host=%s port=%s path=%s",
        actor_user_id,
        department,
        host,
        port,
        streamable_http_path,
    )
    build_mcp_server(
        actor_user_id=actor_user_id,
        department=department,
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
    ).run(transport="streamable-http")


if __name__ == "__main__":
    main()
