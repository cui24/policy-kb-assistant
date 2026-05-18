"""
Agent Graph 路由节点模块。

作用：
1. 恢复工单/草稿引用。
2. 生成规则模式下的分支决策。
3. 在非 rules 模式下执行全局规划。
"""

from __future__ import annotations

from src.agent_graph import adapters
from src.agent_graph.audit import append_simple_event
from src.agent_graph.router import route_next_node
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import record_error, tool_args_preview, update_working_memory


def run_resolve_references_node(state: AgentState) -> AgentState:
    """解析显式与隐式 ticket/draft 引用。"""
    request = dict(state.get("request") or {})
    memory_state = dict(state.get("memory") or {})
    short_term = dict(memory_state.get("short_term") or {})
    text = str(request.get("text") or "")

    explicit_ticket_id = adapters.extract_ticket_id(text)
    effective_draft_id = str(request.get("draft_id") or "") or adapters.infer_draft_id_from_memory(
        text,
        short_term,
        explicit_ticket_id=explicit_ticket_id,
    )
    resolved_ticket_id = str(explicit_ticket_id or "") or str(
        adapters.infer_ticket_id_from_memory(
            text,
            short_term,
            explicit_draft_id=effective_draft_id,
        )
        or ""
    )

    request["explicit_ticket_id"] = explicit_ticket_id
    request["effective_draft_id"] = effective_draft_id or None
    request["resolved_ticket_id"] = resolved_ticket_id or None
    state["request"] = request
    update_working_memory(
        state,
        extracted_entities={
            "explicit_ticket_id": explicit_ticket_id,
            "draft_id": request.get("draft_id"),
        },
        resolved_refs={
            "ticket_id": request.get("resolved_ticket_id"),
            "draft_id": request.get("effective_draft_id"),
            "ticket_id_from_memory": bool(resolved_ticket_id and not explicit_ticket_id),
            "draft_id_from_memory": bool(effective_draft_id and not request.get("draft_id")),
        },
    )
    append_simple_event(
        state,
        "NODE_EXECUTED",
        {
            "node": "resolve_references",
            "resolved_ticket_id": bool(request.get("resolved_ticket_id")),
            "effective_draft_id": bool(request.get("effective_draft_id")),
        },
    )
    return state


def run_reference_clarify_node(state: AgentState) -> AgentState:
    """缺失引用时返回追问响应。"""
    state["execution"] = adapters.build_missing_ticket_reference_response()
    record_error(
        state,
        code="ticket_reference_missing",
        stage="resolve_references",
        reason="short_term_ticket_reference_unresolved",
    )
    append_simple_event(state, "NODE_EXECUTED", {"node": "reference_clarify"})
    return state


def run_rules_router_node(state: AgentState) -> AgentState:
    """在 rules 模式下决定进入哪个业务节点。"""
    planner_state = dict(state.get("planner") or {})
    next_node = route_next_node(state)
    planner_state["route"] = str(next_node)
    state["planner"] = planner_state
    update_working_memory(
        state,
        route_source="rules",
        intent=str(next_node),
    )

    meta = dict(state.get("meta") or {})
    meta["next_node"] = next_node
    state["meta"] = meta

    append_simple_event(state, "NODE_EXECUTED", {"node": "rules_router", "next_node": next_node})
    return state


def run_rules_tool_request_node(state: AgentState) -> AgentState:
    """把 rules 路由结果转换成统一工具调用申请。"""
    request = dict(state.get("request") or {})
    planner_state = dict(state.get("planner") or {})

    route = str(planner_state.get("route") or "")
    text = str(request.get("text") or "")
    fields = request.get("fields") if isinstance(request.get("fields"), dict) else {}
    request_id = str(planner_state.get("request_id") or adapters.new_request_id())
    mode = str(planner_state.get("mode") or adapters.planner_mode())

    tool = ""
    args: dict[str, object] = {}
    if route == "ask":
        tool = "kb_answer"
        args = {"query": text}
    elif route == "ticket_create":
        tool = "create_ticket"
        args = {"text": text, "fields": fields}
    elif route == "draft":
        tool = "continue_ticket_draft"
        args = {
            "draft_id": str(request.get("effective_draft_id") or request.get("draft_id") or ""),
            "text": text,
            "fields": fields,
        }
    elif route == "confirm":
        tool = "confirm_action"
        args = {
            "confirm_token": str(request.get("confirm_token") or ""),
            "text": text,
        }
    elif route == "ticket_tool":
        tool = "ticket_tool_planner"
        args = {
            "ticket_id": str(request.get("resolved_ticket_id") or ""),
            "raw_text": text,
        }
    else:
        state["execution"] = {
            "route": "PLAN_REJECTED",
            "message": f"unsupported_rules_route:{route}",
        }
        record_error(
            state,
            code="unsupported_rules_route",
            stage="rules_tool_request",
            reason=route,
        )
        append_simple_event(state, "NODE_EXECUTED", {"node": "rules_tool_request", "ok": False, "route": route})
        return state

    planner_state["tool"] = tool
    planner_state["args"] = dict(args)
    planner_state["tool_request"] = {
        "tool": tool,
        "args": dict(args),
        "request_id": request_id,
        "mode": mode,
    }
    planner_state["global_next"] = "execute"
    planner_state["request_id"] = request_id
    planner_state["mode"] = mode
    state["planner"] = planner_state
    update_working_memory(
        state,
        request_id=request_id,
        selected_tool=tool,
        tool_args_preview=tool_args_preview(args),
    )
    append_simple_event(
        state,
        "NODE_EXECUTED",
        {"node": "rules_tool_request", "ok": True, "tool": tool},
    )
    return state
