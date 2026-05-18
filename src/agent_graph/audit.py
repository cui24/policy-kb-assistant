"""
Agent Graph 审计模块。

作用：
1. 统一 Agent 相关审计事件结构。
2. 提供集中写入接口，避免节点中重复拼装日志。
3. 保持不同分支审计字段一致性。
"""

from __future__ import annotations

from typing import Any


def append_audit_event(state: dict, event: dict[str, Any]) -> None:
    """向状态对象追加一条审计事件草稿。"""
    audit_state = dict(state.get("audit") or {})
    events = list(audit_state.get("events") or [])
    events.append(dict(event))
    audit_state["events"] = events
    state["audit"] = audit_state


def append_simple_event(state: dict, action: str, payload: dict[str, Any] | None = None) -> None:
    """追加简化审计事件。"""
    append_audit_event(
        state,
        {
            "action": str(action),
            "payload": dict(payload or {}),
        },
    )
