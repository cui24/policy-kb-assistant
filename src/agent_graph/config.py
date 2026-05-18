"""
Agent Graph 配置模块。

作用：
1. 读取图运行参数（开关、超时、重试、阈值）。
2. 统一管理环境变量默认值与边界。
3. 为 executor/graph/nodes 提供稳定配置来源。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentGraphConfig:
    """Agent Graph 运行时配置。"""

    enabled: bool
    prefer_langgraph: bool
    mcp_client_enabled: bool
    mcp_server_url: str
    mcp_client_timeout_seconds: int
    mcp_check_on_startup: bool
    mcp_startup_strict: bool
    mcp_healthcheck_timeout_seconds: int
    mcp_circuit_breaker_enabled: bool
    mcp_circuit_fail_threshold: int
    mcp_circuit_open_seconds: int


def _read_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_agent_graph_config() -> AgentGraphConfig:
    """读取 Agent Graph 运行配置。"""
    return AgentGraphConfig(
        enabled=_read_bool("AGENT_GRAPH_ENABLED", True),
        prefer_langgraph=_read_bool("AGENT_GRAPH_PREFER_LANGGRAPH", True),
        # 默认启用远程 MCP，更贴近业务部署场景。
        # 本地开发若需回退内置 wrapper，可显式设为 false。
        mcp_client_enabled=_read_bool("AGENT_MCP_CLIENT_ENABLED", True),
        mcp_server_url=str(os.getenv("AGENT_MCP_SERVER_URL", "http://127.0.0.1:9000/mcp")).strip(),
        mcp_client_timeout_seconds=max(1, int(str(os.getenv("AGENT_MCP_CLIENT_TIMEOUT_SECONDS", "30")).strip() or "30")),
        mcp_check_on_startup=_read_bool("AGENT_MCP_CHECK_ON_STARTUP", False),
        mcp_startup_strict=_read_bool("AGENT_MCP_STARTUP_STRICT", False),
        mcp_healthcheck_timeout_seconds=max(
            1,
            int(str(os.getenv("AGENT_MCP_HEALTHCHECK_TIMEOUT_SECONDS", "5")).strip() or "5"),
        ),
        mcp_circuit_breaker_enabled=_read_bool("AGENT_MCP_CIRCUIT_BREAKER_ENABLED", True),
        mcp_circuit_fail_threshold=max(
            1,
            int(str(os.getenv("AGENT_MCP_CIRCUIT_FAIL_THRESHOLD", "3")).strip() or "3"),
        ),
        mcp_circuit_open_seconds=max(
            3,
            int(str(os.getenv("AGENT_MCP_CIRCUIT_OPEN_SECONDS", "30")).strip() or "30"),
        ),
    )
