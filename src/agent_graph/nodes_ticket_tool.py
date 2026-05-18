"""
Agent Graph 工单工具节点模块。

作用：
1. 执行既有工单动作（lookup/comment/escalate/cancel）。
2. 对接工具注册中心与参数校验。
3. 回填工具执行结果到状态。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.audit import append_simple_event
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import tool_result_summary, update_working_memory


def run_ticket_tool_node(db: Session, state: AgentState) -> AgentState:
    """执行工单工具分支。"""
    request = dict(state.get("request") or {})
    response = adapters.handle_ticket_tool_route(
        db,
        ticket_id=str(request.get("resolved_ticket_id") or ""),
        text=str(request.get("text") or ""),
        actor=str(request.get("user") or "anonymous"),
        actor_user_id=str(request.get("actor_user_id") or ""),
        actor_role=str(request.get("actor_role") or ""),
    )
    state["execution"] = adapters.normalize_graph_response(response)
    update_working_memory(
        state,
        intent="ticket_tool",
        selected_tool="ticket_tool_planner",
        tool_result_summary=tool_result_summary(state["execution"]),
    )
    append_simple_event(state, "NODE_EXECUTED", {"node": "ticket_tool"})
    return state
