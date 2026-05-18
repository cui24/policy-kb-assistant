"""
Agent Graph 收尾节点模块。

作用：
1. 兜底补齐响应结构。
2. 回写短期记忆。
3. 更新图执行元信息。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.audit import append_simple_event
from src.agent_graph.memory import update_memory_after_response
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import audit_summary, record_error, tool_result_summary, update_working_memory
from src.api import crud


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_finalize_node(db: Session, state: AgentState) -> AgentState:
    """执行统一收尾。"""
    request = dict(state.get("request") or {})
    execution = dict(state.get("execution") or {})
    if not execution:
        execution = {
            "route": "PLAN_REJECTED",
            "message": "empty_execution_result",
        }
        record_error(
            state,
            code="empty_execution_result",
            stage="finalize",
            reason="empty_execution_result",
        )
    state["execution"] = adapters.normalize_graph_response(execution)
    update_working_memory(
        state,
        tool_result_summary=tool_result_summary(state["execution"]),
    )
    state = update_memory_after_response(db, state)

    meta = dict(state.get("meta") or {})
    meta["finished"] = True
    meta["ended_at"] = _now_iso()
    state["meta"] = meta

    working_summary = audit_summary(state)
    if working_summary:
        request_id = str(
            working_summary.get("request_id")
            or dict(state.get("planner") or {}).get("request_id")
            or state["execution"].get("request_id")
            or "agent_working_memory"
        )
        crud.create_audit_log(
            db,
            {
                "actor": str(request.get("user") or "anonymous"),
                "actor_user_id": str(request.get("actor_user_id") or ""),
                "action_type": "AGENT_WORKING_MEMORY",
                "target_type": "AGENT",
                "target_id": request_id,
                "request_id": request_id,
                "payload_json": working_summary,
            },
        )

    append_simple_event(state, "NODE_EXECUTED", {"node": "finalize"})
    return state
