"""
Agent Graph 建单节点模块。

作用：
1. 抽取并标准化建单字段。
2. 判定缺失字段并决定是否进入草稿补全。
3. 字段齐全时调用建单业务并回填状态。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.audit import append_simple_event
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import tool_result_summary, update_working_memory


def run_ticket_create_node(db: Session, state: AgentState) -> AgentState:
    """执行建单分支（含缺失字段落草稿）。"""
    request = dict(state.get("request") or {})
    response = adapters.handle_create_ticket_intent(
        db,
        text=str(request.get("text") or ""),
        actor=str(request.get("user") or "anonymous"),
        actor_user_id=str(request.get("actor_user_id") or ""),
        actor_department=str(request.get("department") or "general"),
        planner_fields=request.get("fields"),
    )
    state["execution"] = adapters.normalize_graph_response(response)
    update_working_memory(
        state,
        intent="create_ticket",
        selected_tool="create_ticket",
        missing_fields=list(state["execution"].get("missing_fields") or []),
        tool_result_summary=tool_result_summary(state["execution"]),
    )
    append_simple_event(state, "NODE_EXECUTED", {"node": "ticket_create"})
    return state
