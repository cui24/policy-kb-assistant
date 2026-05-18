"""
Agent Graph 草稿节点模块。

作用：
1. 管理草稿创建、续办与过期流程。
2. 合并补充字段并重新判定可执行性。
3. 控制草稿相关状态流转。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.audit import append_simple_event
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import tool_result_summary, update_working_memory


def run_draft_node(db: Session, state: AgentState) -> AgentState:
    """执行草稿续办分支。"""
    request = dict(state.get("request") or {})
    response = adapters.resume_ticket_draft_workflow(
        db,
        draft_id=str(request.get("effective_draft_id") or request.get("draft_id") or ""),
        text=str(request.get("text") or ""),
        fields=request.get("fields"),
        actor=str(request.get("user") or "anonymous"),
        actor_user_id=str(request.get("actor_user_id") or ""),
        actor_role=str(request.get("actor_role") or ""),
        actor_department=str(request.get("department") or "general"),
    )
    state["execution"] = adapters.normalize_graph_response(response)
    update_working_memory(
        state,
        intent="continue_ticket_draft",
        selected_tool="continue_ticket_draft",
        tool_result_summary=tool_result_summary(state["execution"]),
    )
    append_simple_event(state, "NODE_EXECUTED", {"node": "draft"})
    return state
