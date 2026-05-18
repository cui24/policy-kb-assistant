"""
Agent Graph 路由决策模块。

作用：
1. 根据当前状态判断下一跳节点。
2. 统一管理 ask/create_ticket/draft/confirm/tool 的路由规则。
3. 提供可独立测试的路由纯函数。
"""

from __future__ import annotations

from src.agent_graph import adapters
from src.agent_graph.state import AgentState


def route_next_node(state: AgentState) -> str:
    """根据请求上下文和解析结果选择下一节点。"""
    request = dict(state.get("request") or {})
    text = str(request.get("text") or "")
    confirm_token = str(request.get("confirm_token") or "").strip()
    effective_draft_id = str(request.get("effective_draft_id") or "").strip()
    resolved_ticket_id = str(request.get("resolved_ticket_id") or "").strip()

    if confirm_token:
        return "confirm"
    if effective_draft_id:
        return "draft"
    if resolved_ticket_id:
        return "ticket_tool"

    if adapters.should_route_to_ticket(text):
        return "ticket_create"
    return "ask"
