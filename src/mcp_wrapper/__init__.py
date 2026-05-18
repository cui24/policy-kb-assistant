"""MCP wrapper 包入口。"""

from src.mcp_wrapper.contracts import ToolCallRequest, ToolCallResult
from src.mcp_wrapper.executor import invoke_tool

__all__ = [
    "ToolCallRequest",
    "ToolCallResult",
    "invoke_tool",
]
'''
公开接口声明：
    从 contracts 模块导入 ToolCallRequest 和 ToolCallResult 
    —— 这是前面分析过的统一工具调用的输入/输出数据结构。

    从 executor 模块导入 invoke_tool 
    —— 这是一个执行工具调用的核心函数，其签名大致为 (ToolCallRequest) -> ToolCallResult 
    或直接接受工具名和参数。

    通过 __all__ 列表明确指定：当外部使用 from src.mcp_wrapper import * 时，
    只会导入这三个符号。

简化导入路径:
    如果不这样做，外部调用者可能需要写：
        from src.mcp_wrapper.contracts import ToolCallRequest 
        from src.mcp_wrapper.executor import invoke_tool。

    现在可以直接写：
        from src.mcp_wrapper import ToolCallRequest, ToolCallResult, invoke_tool
        更加简洁且封装了内部模块结构。
'''