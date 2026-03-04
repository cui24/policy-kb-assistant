"""
L2/L4 CRUD 层：把业务动作落到数据库，并提供查询能力。

一、程序目标
1. 封装数据库读写，避免路由层直接写 SQLAlchemy 细节。
2. 让 `/ask`、`/tickets`、`/agent` 共用同一套持久化逻辑。
3. 为 L3-3 提供 `kb_queries` 与 `audit_logs` 的过滤查询接口。
4. 为 L4-1 提供 `ticket_drafts` 的创建、读取与更新能力。

二、输入输出
1. 输入：业务层整理好的字段字典或过滤条件。
2. 输出：ORM 对象，供上层继续序列化或追加审计。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api import models



def create_kb_query(db: Session, payload: dict[str, Any]) -> models.KBQuery:
    """创建问答记录。"""
    record = models.KBQuery(**payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record



def create_ticket(db: Session, payload: dict[str, Any]) -> models.Ticket:
    """创建工单记录。"""
    record = models.Ticket(**payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record



def create_ticket_draft(db: Session, payload: dict[str, Any]) -> models.TicketDraft:
    """创建工单草稿。"""
    record = models.TicketDraft(**payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record



def get_ticket_by_public_id(db: Session, public_id: str) -> models.Ticket | None:
    """按工单号查询工单。"""
    stmt = select(models.Ticket).where(models.Ticket.public_id == public_id)
    return db.execute(stmt).scalar_one_or_none()


def get_ticket_by_public_id_for_update(db: Session, public_id: str) -> models.Ticket | None:
    """按工单号加锁查询工单，供并发写入场景使用。"""
    stmt = select(models.Ticket).where(models.Ticket.public_id == public_id).with_for_update()
    return db.execute(stmt).scalar_one_or_none()


def get_ticket_by_source_draft_id(db: Session, source_draft_id: str) -> models.Ticket | None:
    """按来源草稿号查询工单，用于幂等命中时复用既有结果。"""
    stmt = select(models.Ticket).where(models.Ticket.source_draft_id == source_draft_id)
    return db.execute(stmt).scalar_one_or_none()



def get_ticket_draft_by_draft_id(db: Session, draft_id: str) -> models.TicketDraft | None:
    """按 draft_id 查询单条工单草稿。"""
    stmt = select(models.TicketDraft).where(models.TicketDraft.draft_id == draft_id)
    return db.execute(stmt).scalar_one_or_none()


def get_pending_action_by_confirm_id(db: Session, confirm_id: str) -> models.PendingAction | None:
    """按确认 token 查询单条待确认动作。"""
    stmt = select(models.PendingAction).where(models.PendingAction.confirm_id == confirm_id)
    return db.execute(stmt).scalar_one_or_none()


def get_agent_conversation_memory(db: Session, user_id: str) -> models.AgentConversationMemory | None:
    """按 user_id 查询单条短期对话记忆。"""
    stmt = select(models.AgentConversationMemory).where(models.AgentConversationMemory.user_id == user_id)
    return db.execute(stmt).scalar_one_or_none()


def get_user_memory(db: Session, user_id: str) -> models.UserMemory | None:
    """按 user_id 查询单条用户长期记忆。"""
    stmt = select(models.UserMemory).where(models.UserMemory.user_id == user_id)
    return db.execute(stmt).scalar_one_or_none()



def get_kb_query_by_request_id(db: Session, request_id: str) -> models.KBQuery | None:
    """按 request_id 查询单条问答记录。"""
    stmt = select(models.KBQuery).where(models.KBQuery.request_id == request_id)
    return db.execute(stmt).scalar_one_or_none()



def list_tickets(db: Session, status: str | None = None) -> Sequence[models.Ticket]:
    """列出工单，可按状态筛选。"""
    stmt = select(models.Ticket).order_by(models.Ticket.created_at.desc())
    if status:
        stmt = stmt.where(models.Ticket.status == status)
    return db.execute(stmt).scalars().all()


def list_ticket_comments(
    db: Session,
    ticket_row_id: str,
    limit: int = 20,
) -> Sequence[models.TicketComment]:
    """按工单内部主键列出最近评论，默认只取最近 20 条。"""
    safe_limit = max(1, min(int(limit), 200))
    stmt = (
        select(models.TicketComment)
        .where(models.TicketComment.ticket_id == ticket_row_id)
        .order_by(models.TicketComment.created_at.desc())
        .limit(safe_limit)
    )
    return db.execute(stmt).scalars().all()



def list_kb_queries(
    db: Session,
    user_name: str | None = None,
    department: str | None = None,
    request_id: str | None = None,
    limit: int = 20,
) -> Sequence[models.KBQuery]:
    """列出问答记录，支持按核心字段过滤。"""
    safe_limit = max(1, min(int(limit), 200))
    stmt = select(models.KBQuery).order_by(models.KBQuery.created_at.desc())
    if user_name:
        stmt = stmt.where(models.KBQuery.user_name == user_name)
    if department:
        stmt = stmt.where(models.KBQuery.department == department)
    if request_id:
        stmt = stmt.where(models.KBQuery.request_id == request_id)
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()



def update_ticket_status(db: Session, ticket: models.Ticket, status: str) -> models.Ticket:
    """更新工单状态。"""
    ticket.status = status
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket



def update_ticket_draft(db: Session, draft: models.TicketDraft, **updates: Any) -> models.TicketDraft:
    """更新工单草稿字段。"""
    for key, value in updates.items():
        setattr(draft, key, value)
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return draft


def update_pending_action(db: Session, pending_action: models.PendingAction, **updates: Any) -> models.PendingAction:
    """更新待确认动作字段。"""
    for key, value in updates.items():
        setattr(pending_action, key, value)
    db.add(pending_action)
    db.commit()
    db.refresh(pending_action)
    return pending_action


def upsert_agent_conversation_memory(
    db: Session,
    user_id: str,
    **updates: Any,
) -> models.AgentConversationMemory:
    """创建或更新一条用户级短期对话记忆。"""
    record = get_agent_conversation_memory(db, user_id)
    if record is None:
        record = models.AgentConversationMemory(user_id=user_id)
    for key, value in updates.items():
        setattr(record, key, value)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def upsert_user_memory(
    db: Session,
    user_id: str,
    **updates: Any,
) -> models.UserMemory:
    """创建或更新一条用户长期记忆。"""
    record = get_user_memory(db, user_id)
    if record is None:
        record = models.UserMemory(user_id=user_id)
    for key, value in updates.items():
        setattr(record, key, value)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record



def create_audit_log(db: Session, payload: dict[str, Any]) -> models.AuditLog:
    """创建审计日志。"""
    record = models.AuditLog(**payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record



def list_audit_logs(
    db: Session,
    request_id: str | None = None,
    ticket_id: str | None = None,
    action_type: str | None = None,
    actor: str | None = None,
    limit: int = 50,
) -> Sequence[models.AuditLog]:
    """列出审计日志，支持按 request_id、ticket_id、动作和操作者过滤。"""
    safe_limit = max(1, min(int(limit), 300))
    stmt = select(models.AuditLog).order_by(models.AuditLog.created_at.desc())
    if request_id:
        stmt = stmt.where(models.AuditLog.request_id == request_id)
    if ticket_id:
        stmt = stmt.where(
            models.AuditLog.target_type == "TICKET",
            models.AuditLog.target_id == ticket_id,
        )
    if action_type:
        stmt = stmt.where(models.AuditLog.action_type == action_type)
    if actor:
        stmt = stmt.where(models.AuditLog.actor == actor)
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()
