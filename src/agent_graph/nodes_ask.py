"""
ASK 业务节点。

粒度定位：
- 对 Agent 图来说，ASK 是单个业务节点。
- 节点内部顺序执行：缓存命中检查 -> 检索/生成（未命中时）-> 落库/审计 -> 响应组装。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.agent_graph.audit import append_simple_event
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import tool_result_summary, update_working_memory
from src.api import ask_pipeline


def run_ask_node(db: Session, state: AgentState) -> AgentState:
    """执行 ASK 单节点业务编排。"""
    request = dict(state.get("request") or {})
    ask_state = dict(state.get("ask") or {})

    question = str(request.get("text") or "")
    actor = str(request.get("user") or "anonymous")
    actor_user_id = str(request.get("actor_user_id") or "anonymous")
    department = str(request.get("department") or "general")

    request_id = str(ask_state.get("request_id") or ask_pipeline.new_request_id())

    normalized = ask_pipeline.run_cached_ask_steps(question, department)

    kb_query = ask_pipeline.persist_kb_query(
        db,
        request_id=request_id,
        actor=actor,
        actor_user_id=actor_user_id,
        department=department,
        question=question,
        normalized=normalized,
    )

    query_id = str(kb_query.id)
    ask_pipeline.write_ask_audit(
        db,
        actor=actor,
        actor_user_id=actor_user_id,
        request_id=request_id,
        target_query_id=query_id,
        question=question,
        department=department,
        normalized=normalized,
    )
    ask_pipeline.write_agent_route_audit(
        db,
        actor=actor,
        actor_user_id=actor_user_id,
        request_id=request_id,
        target_query_id=query_id,
        text=question,
        engine="agent_graph",
    )

    kb_result = ask_pipeline.build_kb_result(
        request_id=request_id,
        query_id=query_id,
        normalized=normalized,
    )

    ask_state["request_id"] = request_id
    ask_state["query_id"] = query_id
    ask_state["normalized"] = normalized
    state["ask"] = ask_state

    state["execution"] = {
        "route": "ASK",
        "kb": ask_pipeline.public_kb_response(kb_result),
        "message": None,
    }
    update_working_memory(
        state,
        request_id=request_id,
        intent="ASK",
        selected_tool="kb_answer",
        tool_result_summary=tool_result_summary(state["execution"]),
    )
    append_simple_event(state, "NODE_EXECUTED", {"node": "ask"})
    return state
