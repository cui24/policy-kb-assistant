"""
Agent Graph 提示词模块。

作用：
1. 集中管理各节点提示词与模板。
2. 降低提示词散落在节点代码中的维护成本。
3. 便于后续版本化、灰度与 A/B 调整。
"""

from __future__ import annotations

PROMPTS = {
    "global_planner": "你是全局路由规划器，负责在 ask/create_ticket/ticket_tool/draft_continue 中选择。",
    "ticket_tool_planner": "你是工单工具规划器，负责在 lookup/comment/escalate/cancel 中选择。",
}
