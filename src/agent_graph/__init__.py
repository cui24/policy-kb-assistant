"""
Agent Graph 包入口。

作用：
1. 聚合基于 LangGraph 的 Agent 编排模块。
2. 为 API/services 层提供稳定的导入路径。
"""

from src.agent_graph.executor import run_agent_graph

__all__ = ["run_agent_graph"]
