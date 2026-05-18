"""
Agent Graph 记忆模块。

作用：
1. 提供短期记忆/长期记忆统一读写接口。
2. 屏蔽底层存储实现（当前 DB，后续可加 Redis 缓存层）。
3. 服务于图节点中的记忆注入与更新。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import update_working_memory


def hydrate_memory_state(db: Session, state: AgentState) -> AgentState:
    """读取并注入短期/长期记忆快照。"""
    request = dict(state.get("request") or {})
    actor_user_id = str(request.get("actor_user_id") or "").strip()
    memory_state = dict(state.get("memory") or {})
    memory_state["short_term"] = adapters.load_short_term_memory(db, actor_user_id)
    memory_state["long_term"] = adapters.load_user_memory(db, actor_user_id)
    state["memory"] = memory_state
    update_working_memory(
        state,
        memory_context={
            "short_term_loaded": bool(memory_state.get("short_term")),
            "long_term_loaded": bool(memory_state.get("long_term")),
        },
    )
    return state


def update_memory_after_response(db: Session, state: AgentState) -> AgentState:
    """根据本轮输出更新短期记忆。"""
    request = dict(state.get("request") or {})
    actor_user_id = str(request.get("actor_user_id") or "").strip()
    text = str(request.get("text") or "")
    execution = dict(state.get("execution") or {})
    adapters.update_short_term_memory_from_response(db, actor_user_id=actor_user_id, text=text, response=execution)
    memory_state = dict(state.get("memory") or {})
    memory_state["updated"] = True
    state["memory"] = memory_state
    return state
