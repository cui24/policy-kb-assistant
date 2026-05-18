"""
Agent Graph 预处理节点模块。

作用：
1. 标准化请求上下文字段。
2. 补齐节点运行所需的默认结构。
3. 统一设置追踪元数据。
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.agent_graph.audit import append_simple_event
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import update_working_memory


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_prepare_node(state: AgentState) -> AgentState:
    """执行图入口预处理。"""
    request = dict(state.get("request") or {})
    request["text"] = str(request.get("text") or "").strip()
    request["user"] = str(request.get("user") or "anonymous")
    request["department"] = str(request.get("department") or "general")
    request["fields"] = dict(request.get("fields") or {})
    request["confirm_token"] = str(request.get("confirm_token") or "") or None
    request["draft_id"] = str(request.get("draft_id") or "") or None
    state["request"] = request

    state["planner"] = dict(state.get("planner") or {})
    state["memory"] = dict(state.get("memory") or {})
    state["working"] = dict(state.get("working") or {})
    state["execution"] = dict(state.get("execution") or {})
    state["audit"] = dict(state.get("audit") or {"events": []})

    meta = dict(state.get("meta") or {})
    meta["started_at"] = str(meta.get("started_at") or _now_iso())
    meta["graph_version"] = str(meta.get("graph_version") or "v1")
    meta["finished"] = False
    state["meta"] = meta

    update_working_memory(
        state,
        normalized_text=request["text"],
        request_id=request.get("request_id"),
        auth_context={
            "actor_user_id": request.get("actor_user_id"),
            "actor_role": request.get("actor_role"),
            "department": request.get("department"),
            "authenticated": bool(request.get("actor_user_id")),
        },
        risk_context={
            "confirm_token_present": bool(request.get("confirm_token")),
        },
    )

    append_simple_event(state, "NODE_EXECUTED", {"node": "prepare"})
    return state
