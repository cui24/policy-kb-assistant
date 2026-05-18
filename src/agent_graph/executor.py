"""
Agent Graph 执行入口模块。

作用：
1. 对外暴露统一执行入口（如 run_agent_graph）。
2. 负责初始状态构建与图调用。
3. 返回可被 API/services 消费的标准结果。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.config import load_agent_graph_config
from src.agent_graph.graph import run_graph
from src.agent_graph.state import AgentState


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_initial_state(
    text: str,
    user: str | None,
    department: str | None,
    draft_id: str | None,
    fields: dict | None,
    confirm_token: str | None,
    actor_user_id: str | None,
    actor_role: str | None,
) -> AgentState:
    actor = str(user or "anonymous")
    resolved_actor_user_id = adapters.resolve_actor_user_id(actor_user_id, actor)
    return {
        "request": {
            "text": str(text or "").strip(),
            "user": actor,
            "department": str(department or "general"),
            "actor_user_id": resolved_actor_user_id,
            "actor_role": adapters.role_value(actor_role),
            "draft_id": str(draft_id or "") or None,
            "confirm_token": str(confirm_token or "") or None,
            "fields": dict(fields or {}),
        },
        "planner": {},
        "memory": {},
        "working": {},
        "draft": {},
        "confirm": {},
        "execution": {},
        "error": {},
        "audit": {"events": []},
        "meta": {
            "started_at": _now_iso(),
            "finished": False,
            "graph_version": "v1",
            "engine": "graph+mcp_wrapper",
        },
    }


def run_agent_graph(
    db: Session,
    text: str,
    user: str | None = None,
    department: str | None = None,
    draft_id: str | None = None,
    fields: dict | None = None,
    confirm_token: str | None = None,
    actor_user_id: str | None = None,
    actor_role: str | None = None,
) -> dict[str, Any]:
    """执行 Agent Graph，并返回兼容 `/agent` 的响应结构。"""
    config = load_agent_graph_config()

    if not config.enabled:
        return adapters.run_legacy_agent_workflow(
            db,
            text=text,
            user=user,
            department=department,
            draft_id=draft_id,
            fields=fields,
            confirm_token=confirm_token,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
        )

    state = _build_initial_state(
        text=text,
        user=user,
        department=department,
        draft_id=draft_id,
        fields=fields,
        confirm_token=confirm_token,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
    )

    final_state = run_graph(db, state)
    response = dict((final_state or {}).get("execution") or {})
    return adapters.normalize_graph_response(response)
