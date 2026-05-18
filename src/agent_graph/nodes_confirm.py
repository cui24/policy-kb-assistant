"""
Agent Graph 确认态节点模块。

作用：
1. 管理高风险动作二次确认。
2. 发放与校验 confirm token。
3. 确认通过后触发实际执行。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.audit import append_simple_event
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import record_error, tool_result_summary, update_working_memory


def run_confirm_node(db: Session, state: AgentState) -> AgentState:
    """执行确认态消费逻辑。"""
    request = dict(state.get("request") or {})
    response = adapters.handle_confirmed_pending_action(
        db,
        confirm_token=str(request.get("confirm_token") or ""),
        actor=str(request.get("user") or "anonymous"),
        actor_user_id=str(request.get("actor_user_id") or ""),
        actor_role=str(request.get("actor_role") or ""),
        text=str(request.get("text") or ""),
    )
    state["execution"] = adapters.normalize_graph_response(response)
    update_working_memory(
        state,
        route_source="confirm_token",
        intent="confirm_action",
        selected_tool="confirm_action",
        risk_context={
            "confirm_token_present": bool(request.get("confirm_token")),
            "confirm_token_prefix": str(request.get("confirm_token") or "")[:8],
        },
        tool_result_summary=tool_result_summary(state["execution"]),
    )
    if state["execution"].get("route") == "PLAN_REJECTED":
        record_error(
            state,
            code="confirm_action_failed",
            stage="confirm",
            reason=str(state["execution"].get("message") or ""),
        )
    append_simple_event(state, "NODE_EXECUTED", {"node": "confirm"})
    return state
