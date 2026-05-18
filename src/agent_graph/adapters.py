"""
Agent Graph 适配层模块。

作用：
1. 适配现有 services/crud/planner 接口，降低迁移耦合。
2. 转换 legacy 输入输出到 graph 统一结构。
3. 作为并存迁移阶段的桥接层。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.api import planner, services
from src.api.planner import PlannerError
from src.api.schemas import ToolPlan


def resolve_actor_user_id(actor_user_id: str | None, actor: str) -> str:
    """统一解析用户标识。"""
    return services._resolve_actor_user_id(actor_user_id, actor)


def role_value(role: object | None) -> str:
    """统一角色值。"""
    return services._role_value(role)


def planner_mode() -> str:
    """读取当前 planner 模式。"""
    return services._agent_planner_mode()


def should_route_to_ticket(text: str) -> bool:
    """复用既有规则判断是否走建单路径。"""
    return services._should_route_to_ticket(text)


def extract_ticket_id(text: str) -> str | None:
    """从文本中提取工单号。"""
    return services._extract_ticket_public_id(text)


def infer_ticket_id_from_memory(text: str, memory_snapshot: dict | None, explicit_draft_id: str | None = None) -> str | None:
    """通过短期记忆推断工单号。"""
    return services._infer_ticket_id_from_memory(text, memory_snapshot, explicit_draft_id=explicit_draft_id)


def infer_draft_id_from_memory(text: str, memory_snapshot: dict | None, explicit_ticket_id: str | None = None) -> str | None:
    """通过短期记忆推断草稿号。"""
    return services._infer_draft_id_from_memory(text, memory_snapshot, explicit_ticket_id=explicit_ticket_id)


def needs_ticket_reference_clarification(text: str, resolved_ticket_id: str | None, effective_draft_id: str | None) -> bool:
    """判断是否需要先追问工单引用。"""
    return services._needs_ticket_reference_clarification(
        text,
        resolved_ticket_id=resolved_ticket_id,
        effective_draft_id=effective_draft_id,
    )


def build_missing_ticket_reference_response() -> dict[str, Any]:
    """构造缺失工单号的追问响应。"""
    return services._build_missing_ticket_reference_response()


def load_short_term_memory(db: Session, actor_user_id: str) -> dict[str, Any]:
    """读取短期记忆。"""
    return services._load_short_term_memory(db, actor_user_id)


def load_user_memory(db: Session, actor_user_id: str) -> dict[str, Any]:
    """读取长期记忆。"""
    return services._load_user_memory(db, actor_user_id)


def update_short_term_memory_from_response(db: Session, actor_user_id: str, text: str, response: dict | None) -> None:
    """根据响应更新短期记忆。"""
    services._update_short_term_memory_from_response(db, actor_user_id=actor_user_id, text=text, response=response)


def handle_kb_intent(db: Session, text: str, actor: str, actor_user_id: str | None, actor_department: str) -> dict[str, Any]:
    """执行问答分支。"""
    return services._handle_kb_intent(
        db,
        text=text,
        actor=actor,
        actor_user_id=actor_user_id,
        actor_department=actor_department,
    )


def handle_create_ticket_intent(
    db: Session,
    text: str,
    actor: str,
    actor_user_id: str | None,
    actor_department: str,
    planner_fields: dict | None = None,
) -> dict[str, Any]:
    """执行建单分支。"""
    return services._handle_create_ticket_intent(
        db,
        text=text,
        actor=actor,
        actor_user_id=actor_user_id,
        actor_department=actor_department,
        planner_fields=planner_fields,
    )


def resume_ticket_draft_workflow(
    db: Session,
    draft_id: str,
    text: str,
    fields: dict | None,
    actor: str,
    actor_user_id: str | None,
    actor_role: str | None,
    actor_department: str,
) -> dict[str, Any]:
    """执行草稿续办分支。"""
    return services._resume_ticket_draft_workflow(
        db,
        draft_id=draft_id,
        text=text,
        fields=fields,
        actor=actor,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_department=actor_department,
    )


def handle_ticket_tool_route(
    db: Session,
    ticket_id: str,
    text: str,
    actor: str,
    actor_user_id: str | None,
    actor_role: str | None,
) -> dict[str, Any]:
    """执行工单工具分支。"""
    return services._handle_ticket_tool_route(
        db,
        ticket_id=ticket_id,
        text=text,
        actor=actor,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
    )


def handle_confirmed_pending_action(
    db: Session,
    confirm_token: str,
    actor: str,
    actor_user_id: str | None,
    actor_role: str | None,
    text: str,
) -> dict[str, Any]:
    """执行确认态动作。"""
    return services._handle_confirmed_pending_action(
        db,
        confirm_token=confirm_token,
        actor=actor,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        text=text,
    )


def run_rules_workflow(
    db: Session,
    text: str,
    actor: str,
    actor_user_id: str | None,
    actor_role: str | None,
    actor_department: str,
    draft_id: str | None,
    resolved_ticket_id: str | None,
    fields: dict | None,
    confirm_token: str | None,
) -> dict[str, Any]:
    """执行规则路由全流程。"""
    return services._run_agent_workflow_rules(
        db,
        text=text,
        actor=actor,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_department=actor_department,
        draft_id=draft_id,
        resolved_ticket_id=resolved_ticket_id,
        fields=fields,
        confirm_token=confirm_token,
    )


def run_global_planner_route(
    db: Session,
    text: str,
    actor: str,
    actor_user_id: str | None,
    actor_role: str | None,
    actor_department: str,
    mode: str,
    draft_id: str | None,
    resolved_ticket_id: str | None,
    memory_snapshot: dict | None,
    fields: dict | None,
    confirm_token: str | None,
) -> dict[str, Any]:
    """执行全局规划流程。"""
    return services._handle_global_planner_route(
        db,
        text=text,
        actor=actor,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_department=actor_department,
        mode=mode,
        draft_id=draft_id,
        resolved_ticket_id=resolved_ticket_id,
        memory_snapshot=memory_snapshot,
        fields=fields,
        confirm_token=confirm_token,
    )


def new_request_id() -> str:
    """生成请求 ID。"""
    return services._new_request_id()


def list_global_skills() -> list[dict[str, Any]]:
    """读取 Global Planner 工具契约。"""
    return services.list_global_planner_skill_contracts()


def run_global_plan_only(
    *,
    user_text: str,
    tools_json: list[dict[str, Any]],
    context: dict[str, Any],
) -> ToolPlan:
    """仅执行全局规划，返回 ToolPlan。"""
    return planner.run_global_planner(
        user_text=user_text,
        tools_json=tools_json,
        context=context,
    )


def planner_error_code(exc: Exception) -> str:
    """提取规划异常代码。"""
    if isinstance(exc, PlannerError):
        return str(getattr(exc, "code", "") or "planner_failed")
    return "planner_failed"


def planner_fallback_eligible(exc: Exception) -> bool:
    """规划失败时是否允许回退。"""
    if isinstance(exc, PlannerError):
        return bool(getattr(exc, "fallback_eligible", False))
    return False


def global_plan_target(
    plan: ToolPlan,
    request_id: str,
    provided_ticket_id: str,
    provided_draft_id: str,
) -> tuple[str, str]:
    """根据 plan 计算审计目标。"""
    return services._global_plan_target(plan, request_id, provided_ticket_id, provided_draft_id)


def audit_global_plan_event(
    db: Session,
    *,
    actor: str,
    actor_user_id: str | None,
    action_type: str,
    target_type: str,
    target_id: str,
    request_id: str,
    payload_json: dict[str, Any],
) -> None:
    """写入全局规划审计事件。"""
    services._audit_global_plan_event(
        db,
        actor=actor,
        actor_user_id=actor_user_id,
        action_type=action_type,
        target_type=target_type,
        target_id=target_id,
        request_id=request_id,
        payload_json=payload_json,
    )


def validate_global_plan(
    db: Session,
    *,
    plan: ToolPlan,
    actor: str,
    actor_user_id: str | None,
    request_id: str,
    normalized_text: str,
    provided_ticket_id: str,
    provided_draft_id: str,
) -> dict[str, Any]:
    """执行全局 plan 校验。"""
    return services._validate_global_plan(
        db,
        plan=plan,
        actor=actor,
        actor_user_id=actor_user_id,
        request_id=request_id,
        normalized_text=normalized_text,
        provided_ticket_id=provided_ticket_id,
        provided_draft_id=provided_draft_id,
    )


def plan_args_summary(args: dict | None) -> dict[str, Any]:
    """压缩计划参数用于审计。"""
    return services._plan_args_summary(args)


def build_plan_rejected_response(message: str) -> dict[str, Any]:
    """统一 PLAN_REJECTED 响应。"""
    return services._build_plan_rejected_response(message)


def execute_global_tool(
    db: Session,
    *,
    tool: str,
    args: dict[str, Any],
    text: str,
    actor: str,
    actor_user_id: str | None,
    actor_role: str | None,
    actor_department: str,
    mode: str,
) -> dict[str, Any]:
    """按全局工具名执行对应分支。"""
    if tool == "continue_ticket_draft":
        return services._resume_ticket_draft_workflow(
            db,
            draft_id=str(args.get("draft_id") or ""),
            text=text,
            fields=args.get("fields"),
            actor=actor,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            actor_department=actor_department,
        )
    if tool == "ticket_tool_planner":
        return services._handle_ticket_tool_route_with_planner(
            db,
            ticket_id=str(args.get("ticket_id") or ""),
            text=str(args.get("raw_text") or text),
            actor=actor,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            mode=mode,
        )
    if tool == "kb_answer":
        return services._handle_kb_intent(
            db,
            text=str(args.get("query") or text),
            actor=actor,
            actor_user_id=actor_user_id,
            actor_department=actor_department,
        )
    return services._handle_create_ticket_intent(
        db,
        text=str(args.get("text") or text),
        actor=actor,
        actor_user_id=actor_user_id,
        actor_department=actor_department,
        planner_fields=args.get("fields"),
    )


def normalize_graph_response(response: dict[str, Any] | None) -> dict[str, Any]:
    """把业务响应收敛到 AgentResponse 兼容结构。"""
    normalized = dict(response or {})
    if "route" not in normalized:
        normalized["route"] = "PLAN_REJECTED"
        normalized["message"] = normalized.get("message") or "graph_result_missing_route"
    return normalized


def run_legacy_agent_workflow(
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
    """回退到旧版 `services.run_agent_workflow`。"""
    return services.run_agent_workflow(
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
