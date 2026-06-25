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

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from src.api import models


def _strip_nul_chars(value: Any) -> Any:
    """
    递归移除字符串中的 NUL 字符（\\x00），避免 PostgreSQL 文本/JSONB 落库报错。

    Postgres 不接受包含 NUL 的 text/jsonb 值：
    `unsupported Unicode escape sequence: \\u0000 cannot be converted to text`
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_strip_nul_chars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_nul_chars(item) for item in value)
    if isinstance(value, dict):
        return {key: _strip_nul_chars(item) for key, item in value.items()}
    return value



def create_user(db: Session, payload: dict[str, Any]) -> models.User:
    """创建用户记录。"""
    record = models.User(**payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_user_by_id(db: Session, user_id: str) -> models.User | None:
    """按用户主键查询。"""
    stmt = select(models.User).where(models.User.id == user_id)
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_username(db: Session, username: str) -> models.User | None:
    """按用户名查询。"""
    stmt = select(models.User).where(models.User.username == username)
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_email(db: Session, email: str) -> models.User | None:
    """按邮箱查询。"""
    stmt = select(models.User).where(models.User.email == email)
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_phone(db: Session, phone: str) -> models.User | None:
    """按手机号查询。"""
    stmt = select(models.User).where(models.User.phone == phone)
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_login_identifier(db: Session, identifier: str) -> models.User | None:
    """按登录标识查询：支持 username / email / phone。"""
    normalized = str(identifier or "").strip()
    if not normalized:
        return None
    stmt = select(models.User).where(
        or_(
            models.User.username == normalized,
            models.User.email == normalized,
            models.User.phone == normalized,
        )
    )
    return db.execute(stmt).scalar_one_or_none()


def update_user_last_login(db: Session, user: models.User) -> models.User:
    """刷新用户最近登录时间。"""
    user.last_login_at = datetime.now(timezone.utc)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def list_users(
    db: Session,
    role: str | None = None,
    is_active: bool | None = None,
    limit: int = 50,
) -> Sequence[models.User]:
    """列出用户，可按角色与启用状态过滤。"""
    safe_limit = max(1, min(int(limit), 300))
    stmt = select(models.User).order_by(models.User.created_at.desc())
    if role:
        stmt = stmt.where(models.User.role == role)
    if is_active is not None:
        stmt = stmt.where(models.User.is_active.is_(bool(is_active)))
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()


def create_kb_query(db: Session, payload: dict[str, Any]) -> models.KBQuery:
    """创建问答记录。"""
    normalized_payload = _strip_nul_chars(dict(payload or {}))
    if "actor_user_id" not in normalized_payload:
        normalized_payload["actor_user_id"] = str(normalized_payload.get("user_name") or "anonymous")
    record = models.KBQuery(**normalized_payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def create_kb_query_with_audit(
    db: Session,
    kb_payload: dict[str, Any],
    audit_payload: dict[str, Any],
) -> tuple[models.KBQuery, models.AuditLog]:
    """在同一事务中创建问答记录和对应审计日志。"""
    normalized_kb_payload = _strip_nul_chars(dict(kb_payload or {}))
    if "actor_user_id" not in normalized_kb_payload:
        normalized_kb_payload["actor_user_id"] = str(normalized_kb_payload.get("user_name") or "anonymous")

    kb_record = models.KBQuery(**normalized_kb_payload)
    db.add(kb_record)
    db.flush()

    normalized_audit_payload = _strip_nul_chars(dict(audit_payload or {}))
    if "actor_user_id" not in normalized_audit_payload:
        normalized_audit_payload["actor_user_id"] = str(normalized_audit_payload.get("actor") or "anonymous")
    if not normalized_audit_payload.get("target_id"):
        normalized_audit_payload["target_id"] = kb_record.id

    audit_record = models.AuditLog(**normalized_audit_payload)
    db.add(audit_record)
    db.commit()
    return kb_record, audit_record


async def create_kb_query_async(db: Session, payload: dict[str, Any]) -> models.KBQuery:
    """
    异步问答落库入口。

    这里保持与同步版本相同的事务/提交路径，避免改变既有一致性语义。
    同步提交是阻塞调用，丢进线程池执行，避免冻结 event loop（同一请求内串行 await，
    不会出现多线程并发访问同一 Session）。
    """
    return await asyncio.to_thread(create_kb_query, db, payload)


async def create_kb_query_with_audit_async(
    db: Session,
    kb_payload: dict[str, Any],
    audit_payload: dict[str, Any],
) -> tuple[models.KBQuery, models.AuditLog]:
    """异步组合落库入口：一次事务写问答记录和审计日志。"""
    return await asyncio.to_thread(create_kb_query_with_audit, db, kb_payload, audit_payload)



def create_ticket(db: Session, payload: dict[str, Any]) -> models.Ticket:
    """创建工单记录。"""
    normalized_payload = dict(payload or {})
    if "creator_user_id" not in normalized_payload:
        normalized_payload["creator_user_id"] = str(normalized_payload.get("creator") or "anonymous")
    record = models.Ticket(**normalized_payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record



def create_ticket_draft(db: Session, payload: dict[str, Any]) -> models.TicketDraft:
    """创建工单草稿。"""
    normalized_payload = dict(payload or {})
    if "owner_user_id" not in normalized_payload:
        normalized_payload["owner_user_id"] = str(normalized_payload.get("creator") or "anonymous")
    record = models.TicketDraft(**normalized_payload)
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


def list_tickets_by_creator_user_id(
    db: Session,
    creator_user_id: str,
    status: str | None = None,
    limit: int = 100,
) -> Sequence[models.Ticket]:
    """按创建人列出工单，可按状态筛选。"""
    safe_limit = max(1, min(int(limit), 500))
    stmt = (
        select(models.Ticket)
        .where(models.Ticket.creator_user_id == creator_user_id)
        .order_by(models.Ticket.created_at.desc())
    )
    if status:
        stmt = stmt.where(models.Ticket.status == status)
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()


def list_tickets_by_assignee_user_id(
    db: Session,
    assignee_user_id: str,
    status: str | None = None,
    limit: int = 100,
) -> Sequence[models.Ticket]:
    """按处理人列出工单，可按状态筛选。"""
    safe_limit = max(1, min(int(limit), 500))
    stmt = (
        select(models.Ticket)
        .where(models.Ticket.assignee_user_id == assignee_user_id)
        .order_by(models.Ticket.created_at.desc())
    )
    if status:
        stmt = stmt.where(models.Ticket.status == status)
    stmt = stmt.limit(safe_limit)
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
    actor_user_id: str | None = None,
    department: str | None = None,
    request_id: str | None = None,
    limit: int = 20,
) -> Sequence[models.KBQuery]:
    """列出问答记录，支持按核心字段过滤。"""
    safe_limit = max(1, min(int(limit), 200))
    stmt = select(models.KBQuery).order_by(models.KBQuery.created_at.desc())
    if user_name:
        stmt = stmt.where(models.KBQuery.user_name == user_name)
    if actor_user_id:
        stmt = stmt.where(models.KBQuery.actor_user_id == actor_user_id)
    if department:
        stmt = stmt.where(models.KBQuery.department == department)
    if request_id:
        stmt = stmt.where(models.KBQuery.request_id == request_id)
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()


def list_kb_queries_by_actor_user_id(
    db: Session,
    actor_user_id: str,
    limit: int = 50,
) -> Sequence[models.KBQuery]:
    """按用户主键列出问答记录。"""
    safe_limit = max(1, min(int(limit), 300))
    stmt = (
        select(models.KBQuery)
        .where(models.KBQuery.actor_user_id == actor_user_id)
        .order_by(models.KBQuery.created_at.desc())
        .limit(safe_limit)
    )
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
    normalized_payload = _strip_nul_chars(dict(payload or {}))
    if "actor_user_id" not in normalized_payload:
        normalized_payload["actor_user_id"] = str(normalized_payload.get("actor") or "anonymous")
    record = models.AuditLog(**normalized_payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


async def create_audit_log_async(db: Session, payload: dict[str, Any]) -> models.AuditLog:
    """
    异步审计落库入口。

    保留现有同步提交语义，确保和原链路一致。
    同步提交丢进线程池执行，避免阻塞 event loop。
    """
    return await asyncio.to_thread(create_audit_log, db, payload)



def list_audit_logs(
    db: Session,
    request_id: str | None = None,
    ticket_id: str | None = None,
    action_type: str | None = None,
    actor: str | None = None,
    actor_user_id: str | None = None,
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
    if actor_user_id:
        stmt = stmt.where(models.AuditLog.actor_user_id == actor_user_id)
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()


def list_audit_logs_by_actor_user_id(
    db: Session,
    actor_user_id: str,
    action_type: str | None = None,
    limit: int = 100,
) -> Sequence[models.AuditLog]:
    """按用户主键列出审计日志。"""
    safe_limit = max(1, min(int(limit), 500))
    stmt = (
        select(models.AuditLog)
        .where(models.AuditLog.actor_user_id == actor_user_id)
        .order_by(models.AuditLog.created_at.desc())
    )
    if action_type:
        stmt = stmt.where(models.AuditLog.action_type == action_type)
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()


def list_pending_actions_by_user_id(
    db: Session,
    user_id: str,
    status: str | None = None,
    only_unexpired: bool = False,
    limit: int = 50,
) -> Sequence[models.PendingAction]:
    """按用户列出待确认动作，可过滤状态与是否过期。"""
    safe_limit = max(1, min(int(limit), 300))
    stmt = (
        select(models.PendingAction)
        .where(models.PendingAction.user_id == user_id)
        .order_by(models.PendingAction.created_at.desc())
    )
    if status:
        stmt = stmt.where(models.PendingAction.status == status)
    if only_unexpired:
        stmt = stmt.where(models.PendingAction.expires_at > datetime.now(timezone.utc))
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()


def list_ticket_drafts_by_owner_user_id(
    db: Session,
    owner_user_id: str,
    status: str | None = None,
    limit: int = 50,
) -> Sequence[models.TicketDraft]:
    """按草稿拥有者列出草稿，可按状态筛选。"""
    safe_limit = max(1, min(int(limit), 300))
    stmt = (
        select(models.TicketDraft)
        .where(models.TicketDraft.owner_user_id == owner_user_id)
        .order_by(models.TicketDraft.updated_at.desc())
    )
    if status:
        stmt = stmt.where(models.TicketDraft.status == status)
    stmt = stmt.limit(safe_limit)
    return db.execute(stmt).scalars().all()
