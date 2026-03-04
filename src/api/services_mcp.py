"""
MCP 服务适配层：把固定 actor 的 MCP 工具调用映射到现有 ticket 工作流。

一、程序目标
1. 为 stdio MCP server 提供稳定的、可测试的服务封装。
2. 复用现有 registry + validator + workflow，不绕过业务边界。
3. 为 MCP 调用补充 `payload_json.source="mcp"` 的审计标记。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.api import crud, models, services
from src.api.schemas import CancelTicketPlanArgs, ToolPlan
from src.api.skills import get_ticket_tool_registry


MCP_AUDIT_SOURCE = "mcp"


def _dump_model(model_instance) -> dict[str, Any]:
    """兼容 Pydantic v1/v2 的导出接口。"""
    if hasattr(model_instance, "model_dump"):
        return model_instance.model_dump(exclude_none=True)
    return model_instance.dict(exclude_none=True)


def _target_ref(ticket_id: str, request_id: str) -> tuple[str, str]:
    """为无对象失败场景选择一个可追踪的审计目标。"""
    normalized_ticket_id = str(ticket_id or "").strip()
    if normalized_ticket_id:
        return "TICKET", normalized_ticket_id
    return "AGENT", request_id


def _audit_log(
    db: Session,
    *,
    actor: str,
    action_type: str,
    target_type: str,
    target_id: str,
    request_id: str,
    payload_json: dict[str, Any],
) -> None:
    """写入一条带 MCP source 的审计日志。"""
    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": action_type,
            "target_type": target_type,
            "target_id": target_id,
            "request_id": request_id,
            "payload_json": services._with_audit_source(payload_json, MCP_AUDIT_SOURCE),
        },
    )


def _append_audit_log_uncommitted(
    db: Session,
    *,
    actor: str,
    action_type: str,
    target_type: str,
    target_id: str,
    request_id: str,
    payload_json: dict[str, Any],
) -> None:
    """在当前事务内追加一条带 source 的审计日志。"""
    db.add(
        models.AuditLog(
            actor=actor,
            action_type=action_type,
            target_type=target_type,
            target_id=target_id,
            request_id=request_id,
            payload_json=services._with_audit_source(payload_json, MCP_AUDIT_SOURCE),
        )
    )


def _reject_tool_call(
    db: Session,
    *,
    actor: str,
    ticket_id: str,
    request_id: str,
    tool_name: str,
    message: str,
    reason: str,
) -> None:
    """记录一条 MCP 拒绝审计，并抛出异常。"""
    target_type, target_id = _target_ref(ticket_id, request_id)
    _audit_log(
        db,
        actor=actor,
        action_type="MCP_TOOL_REJECTED",
        target_type=target_type,
        target_id=target_id,
        request_id=request_id,
        payload_json={
            "tool": tool_name,
            "reason": reason,
            "message": message,
        },
    )
    raise ValueError(message)


def invoke_ticket_tool(
    db: Session,
    *,
    tool_name: str,
    args: dict[str, Any],
    actor: str,
    raw_text: str = "",
) -> dict[str, Any]:
    """执行一个低风险 ticket 工具，复用现有校验后再 dispatch。"""
    request_id = services._new_request_id()
    plan = ToolPlan(
        tool=tool_name,
        args=dict(args or {}),
        need_confirmation=False,
        missing_fields=[],
    )
    validation_result = services._validate_ticket_tool_plan(
        db,
        plan=plan,
        actor=actor,
        request_id=request_id,
        confirmation_verified=False,
    )
    status = str(validation_result.get("status") or "")
    ticket_id = str((args or {}).get("ticket_id") or "")

    if status != "validated":
        response = validation_result.get("response") or {}
        message = str(response.get("message") or "MCP 工具调用未通过校验。")
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id=ticket_id,
            request_id=request_id,
            tool_name=tool_name,
            message=message,
            reason=f"validator_{status or 'unknown'}",
        )

    skill = validation_result["skill"]
    if str(skill.risk_level or "") == "HIGH":
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id=ticket_id,
            request_id=request_id,
            tool_name=tool_name,
            message="高风险取消操作请先调用 request_cancel_ticket。",
            reason="high_risk_requires_two_step_confirmation",
        )

    dispatch_args = dict(validation_result["args"] or {})
    dispatch_args["_audit_source"] = MCP_AUDIT_SOURCE
    ticket_detail = get_ticket_tool_registry().dispatch(
        skill.name,
        db=db,
        args=dispatch_args,
        actor=actor,
        raw_text=raw_text,
    )
    _audit_log(
        db,
        actor=actor,
        action_type="MCP_TOOL_CALL",
        target_type="TICKET",
        target_id=str(dispatch_args.get("ticket_id") or ""),
        request_id=request_id,
        payload_json={
            "tool": skill.name,
            "route": skill.route_name,
        },
    )
    return ticket_detail


def request_cancel_ticket_workflow(
    db: Session,
    *,
    ticket_id: str,
    actor: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """第一步：创建待确认动作，只返回 confirm_token。"""
    request_id = services._new_request_id()
    normalized_reason = (reason or "").strip() or "用户请求取消工单。"
    try:
        normalized_args = _dump_model(
            services._validate_pydantic_model(
                CancelTicketPlanArgs,
                {
                    "ticket_id": str(ticket_id or ""),
                    "reason": normalized_reason,
                },
            )
        )
    except Exception as exc:
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id=str(ticket_id or ""),
            request_id=request_id,
            tool_name="request_cancel_ticket",
            message="request_cancel_ticket 参数不合法。",
            reason=f"schema_invalid:{exc.__class__.__name__}",
        )

    normalized_ticket_id = str(normalized_args.get("ticket_id") or "")
    ticket = crud.get_ticket_by_public_id(db, normalized_ticket_id)
    if ticket is None:
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id=normalized_ticket_id,
            request_id=request_id,
            tool_name="request_cancel_ticket",
            message="目标工单不存在，当前未执行操作。",
            reason="ticket_not_found",
        )

    if not services._actor_satisfies_auth_rule(actor, "owner_or_admin", ticket):
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id=normalized_ticket_id,
            request_id=request_id,
            tool_name="request_cancel_ticket",
            message="当前用户无权执行该操作。",
            reason="auth_rejected",
        )

    pending_action = models.PendingAction(
        user_id=actor,
        tool_name="cancel_ticket",
        args_json={
            "ticket_id": normalized_ticket_id,
            "reason": str(normalized_args.get("reason") or normalized_reason),
        },
        status="pending",
        expires_at=services._pending_action_expiry(),
    )
    db.add(pending_action)
    db.flush()
    _append_audit_log_uncommitted(
        db,
        actor=actor,
        action_type="NEED_CONFIRMATION",
        target_type="TICKET",
        target_id=normalized_ticket_id,
        request_id=request_id,
        payload_json={
            "tool": "cancel_ticket",
            "confirm_token_prefix": str(pending_action.confirm_id)[:8],
            "expires_at": pending_action.expires_at.isoformat(),
        },
    )
    db.commit()
    db.refresh(pending_action)
    return {
        "ticket_id": normalized_ticket_id,
        "confirm_token": str(pending_action.confirm_id),
        "expires_at": pending_action.expires_at.isoformat(),
        "message": "这是高风险操作。请调用 confirm_cancel_ticket 并携带 confirm_token 完成确认。",
    }


def confirm_cancel_ticket_workflow(
    db: Session,
    *,
    confirm_token: str,
    actor: str,
) -> dict[str, Any]:
    """第二步：消费 confirm_token 并执行真正取消。"""
    request_id = services._new_request_id()
    normalized_token = str(confirm_token or "").strip()
    if not normalized_token:
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id="",
            request_id=request_id,
            tool_name="confirm_cancel_ticket",
            message="confirm_token 不能为空。",
            reason="confirm_token_missing",
        )

    pending_action = crud.get_pending_action_by_confirm_id(db, normalized_token)
    if pending_action is None:
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id="",
            request_id=request_id,
            tool_name="confirm_cancel_ticket",
            message="确认令牌无效或已失效。",
            reason="confirm_token_not_found",
        )

    ticket_id = str((pending_action.args_json or {}).get("ticket_id") or "")
    if str(pending_action.user_id or "") != actor:
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id=ticket_id,
            request_id=request_id,
            tool_name="confirm_cancel_ticket",
            message="该确认令牌不属于当前用户。",
            reason="confirm_token_actor_mismatch",
        )

    if str(pending_action.tool_name or "") != "cancel_ticket":
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id=ticket_id,
            request_id=request_id,
            tool_name="confirm_cancel_ticket",
            message="当前确认令牌不对应取消工单。",
            reason="unexpected_pending_tool",
        )

    expires_at = services._normalize_datetime(pending_action.expires_at)
    if str(pending_action.status or "") != "pending" or expires_at is None or expires_at <= services._utc_now():
        if str(pending_action.status or "") == "pending":
            crud.update_pending_action(db, pending_action, status="expired")
        _reject_tool_call(
            db,
            actor=actor,
            ticket_id=ticket_id,
            request_id=request_id,
            tool_name="confirm_cancel_ticket",
            message="确认令牌已过期或已使用。",
            reason="confirm_token_expired_or_consumed",
        )

    ticket_detail = services.cancel_ticket_workflow(
        db,
        ticket_id=ticket_id,
        actor=actor,
        reason=str((pending_action.args_json or {}).get("reason") or ""),
        audit_source=MCP_AUDIT_SOURCE,
    )
    crud.update_pending_action(db, pending_action, status="consumed")
    _audit_log(
        db,
        actor=actor,
        action_type="MCP_TOOL_CALL",
        target_type="TICKET",
        target_id=ticket_id,
        request_id=request_id,
        payload_json={
            "tool": "confirm_cancel_ticket",
            "confirm_token_prefix": normalized_token[:8],
        },
    )
    return ticket_detail
