"""
Global Planner 节点模块。

目标：
1. `global_plan` 只产出 ToolPlan 与审计 proposed。
2. `global_validate` 只做校验与审计 executed/rejected。
3. 产出统一 `planner.tool_request`，交由通用执行节点处理。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.audit import append_simple_event
from src.agent_graph.state import AgentState
from src.agent_graph.working_memory import record_error, tool_args_preview, update_working_memory
from src.api.schemas import ToolPlan


def run_global_plan_node(db: Session, state: AgentState) -> AgentState:
    """执行全局规划（不执行工具）。"""
    request = dict(state.get("request") or {})
    memory_state = dict(state.get("memory") or {})
    planner_state = dict(state.get("planner") or {})

    text = str(request.get("text") or "")
    actor = str(request.get("user") or "anonymous")
    actor_user_id = str(request.get("actor_user_id") or "")
    mode = adapters.planner_mode()

    explicit_ticket_in_text = bool(adapters.extract_ticket_id(text))
    explicit_draft_provided = bool(str(request.get("draft_id") or "").strip())
    provided_ticket_id = str(request.get("resolved_ticket_id") or "")
    provided_draft_id = str(request.get("effective_draft_id") or "")
    request_id = adapters.new_request_id()

    planner_context = {
        "actor_user_id": actor_user_id,
        "provided_ticket_id": provided_ticket_id,
        "provided_draft_id": provided_draft_id,
        "has_ticket_id": bool(provided_ticket_id),
        "has_draft_id": bool(provided_draft_id),
        "confirm_token_present": False,
        "ticket_tool_mode": bool(provided_ticket_id),
        "draft_mode": bool(provided_draft_id),
        "short_term_memory": dict(memory_state.get("short_term") or {}),
        "memory_ticket_applied": bool(
            not explicit_ticket_in_text
            and (memory_state.get("short_term") or {}).get("last_ticket_id")
            and provided_ticket_id
        ),
        "memory_draft_applied": bool(
            not explicit_draft_provided
            and (memory_state.get("short_term") or {}).get("last_draft_id")
            and provided_draft_id
        ),
    }

    try:
        plan = adapters.run_global_plan_only(
            user_text=text,
            tools_json=adapters.list_global_skills(),
            context=planner_context,
        )
    except Exception as exc:
        error_code = adapters.planner_error_code(exc)
        adapters.audit_global_plan_event(
            db,
            actor=actor,
            actor_user_id=actor_user_id,
            action_type="PLAN_REJECTED",
            target_type="AGENT",
            target_id=request_id,
            request_id=request_id,
            payload_json={
                "reason": error_code,
                "planner_mode": mode,
                "planner_scope": "global",
            },
        )
        if mode == "hybrid" and adapters.planner_fallback_eligible(exc):
            planner_state["global_next"] = "fallback_rules"
        else:
            planner_state["global_next"] = "finalize"
            state["execution"] = adapters.build_plan_rejected_response("智能规划失败，当前未执行任何操作。")
        planner_state["request_id"] = request_id
        planner_state["mode"] = mode
        state["planner"] = planner_state
        update_working_memory(
            state,
            request_id=request_id,
            route_source="global_planner",
        )
        record_error(
            state,
            code=error_code,
            stage="global_plan",
            reason=str(exc),
        )
        append_simple_event(state, "NODE_EXECUTED", {"node": "global_plan", "mode": mode, "ok": False})
        return state

    target_type, target_id = adapters.global_plan_target(
        plan,
        request_id=request_id,
        provided_ticket_id=provided_ticket_id,
        provided_draft_id=provided_draft_id,
    )
    adapters.audit_global_plan_event(
        db,
        actor=actor,
        actor_user_id=actor_user_id,
        action_type="PLAN_PROPOSED",
        target_type=target_type,
        target_id=target_id,
        request_id=request_id,
        payload_json={
            "tool": plan.tool,
            "args": adapters.plan_args_summary(plan.args),
            "need_confirmation": bool(plan.need_confirmation),
            "missing_fields": list(plan.missing_fields or []),
            "planner_mode": mode,
            "planner_scope": "global",
        },
    )

    planner_state.update(
        {
            "mode": mode,
            "request_id": request_id,
            "tool": str(plan.tool),
            "args": dict(plan.args or {}),
            "need_confirmation": bool(plan.need_confirmation),
            "missing_fields": list(plan.missing_fields or []),
            "target_type": target_type,
            "target_id": target_id,
            "provided_ticket_id": provided_ticket_id,
            "provided_draft_id": provided_draft_id,
            "global_next": "validate",
        }
    )
    state["planner"] = planner_state
    update_working_memory(
        state,
        request_id=request_id,
        route_source="global_planner",
        intent=str(plan.tool),
        selected_tool=str(plan.tool),
        missing_fields=list(plan.missing_fields or []),
        risk_context={
            "need_confirmation": bool(plan.need_confirmation),
            "confirm_token_present": False,
        },
        tool_args_preview=tool_args_preview(dict(plan.args or {})),
    )
    append_simple_event(state, "NODE_EXECUTED", {"node": "global_plan", "mode": mode, "ok": True})
    return state


def run_global_validate_node(db: Session, state: AgentState) -> AgentState:
    """执行全局 plan 校验（不执行工具）。"""
    request = dict(state.get("request") or {})
    planner_state = dict(state.get("planner") or {})

    if str(planner_state.get("global_next") or "") != "validate":
        return state

    actor = str(request.get("user") or "anonymous")
    actor_user_id = str(request.get("actor_user_id") or "")
    request_id = str(planner_state.get("request_id") or adapters.new_request_id())
    mode = str(planner_state.get("mode") or adapters.planner_mode())
    tool = str(planner_state.get("tool") or "")
    args = dict(planner_state.get("args") or {})
    missing_fields = list(planner_state.get("missing_fields") or [])

    plan = ToolPlan(
        tool=tool,
        args=args,
        need_confirmation=bool(planner_state.get("need_confirmation")),
        missing_fields=missing_fields,
    )

    validation_result = adapters.validate_global_plan(
        db,
        plan=plan,
        actor=actor,
        actor_user_id=actor_user_id,
        request_id=request_id,
        normalized_text=str(request.get("text") or ""),
        provided_ticket_id=str(planner_state.get("provided_ticket_id") or ""),
        provided_draft_id=str(planner_state.get("provided_draft_id") or ""),
    )

    status = str(validation_result.get("status") or "")
    if status == "response":
        planner_state["global_next"] = "finalize"
        state["execution"] = adapters.normalize_graph_response(validation_result.get("response") or {})
        state["planner"] = planner_state
        update_working_memory(
            state,
            missing_fields=list(state["execution"].get("missing_fields") or missing_fields),
            permission_context={"validation_status": status},
        )
        record_error(
            state,
            code="missing_required_field",
            stage="global_validate",
            reason="missing_fields_from_plan",
        )
        append_simple_event(state, "NODE_EXECUTED", {"node": "global_validate", "status": status})
        return state

    if status == "fallback":
        adapters.audit_global_plan_event(
            db,
            actor=actor,
            actor_user_id=actor_user_id,
            action_type="PLAN_REJECTED",
            target_type=str(validation_result.get("target_type") or planner_state.get("target_type") or "AGENT"),
            target_id=str(validation_result.get("target_id") or planner_state.get("target_id") or request_id),
            request_id=request_id,
            payload_json={
                "tool": tool,
                "reason": str(validation_result.get("reason") or "validation_failed"),
                "planner_scope": "global",
            },
        )
        if mode == "hybrid":
            planner_state["global_next"] = "fallback_rules"
        else:
            planner_state["global_next"] = "finalize"
            state["execution"] = adapters.build_plan_rejected_response("规划结果未通过校验，当前未执行任何操作。")
        state["planner"] = planner_state
        update_working_memory(
            state,
            permission_context={"validation_status": status},
        )
        record_error(
            state,
            code=str(validation_result.get("reason") or "planner_validation_failed"),
            stage="global_validate",
            reason=str(validation_result.get("reason") or "validation_failed"),
        )
        append_simple_event(state, "NODE_EXECUTED", {"node": "global_validate", "status": status})
        return state

    validated_args = dict(validation_result.get("args") or {})
    adapters.audit_global_plan_event(
        db,
        actor=actor,
        actor_user_id=actor_user_id,
        action_type="PLAN_EXECUTED",
        target_type=str(validation_result.get("target_type") or planner_state.get("target_type") or "AGENT"),
        target_id=str(validation_result.get("target_id") or planner_state.get("target_id") or request_id),
        request_id=request_id,
        payload_json={
            "tool": tool,
            "args": adapters.plan_args_summary(validated_args),
            "planner_scope": "global",
        },
    )
    planner_state["validated_args"] = validated_args
    planner_state["tool_request"] = {
        "tool": tool,
        "args": validated_args,
        "request_id": request_id,
        "mode": mode,
    }
    planner_state["global_next"] = "execute"
    state["planner"] = planner_state
    update_working_memory(
        state,
        selected_tool=tool,
        tool_args_preview=tool_args_preview(validated_args),
        permission_context={"validation_status": "validated"},
    )
    append_simple_event(state, "NODE_EXECUTED", {"node": "global_validate", "status": "validated"})
    return state
