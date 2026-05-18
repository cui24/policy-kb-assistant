"""
Agent Graph 异常定义模块。

作用：
1. 定义 Agent 图领域异常类型。
2. 让上层按异常类型映射 HTTP 状态。
3. 避免依赖字符串匹配判断错误分支。
"""

from __future__ import annotations


class AgentGraphError(Exception):
    """Agent Graph 领域基础异常。"""


class AgentGraphRoutingError(AgentGraphError):
    """路由阶段异常。"""


class AgentGraphExecutionError(AgentGraphError):
    """节点执行阶段异常。"""


class AgentGraphValidationError(AgentGraphError):
    """输入状态校验异常。"""
