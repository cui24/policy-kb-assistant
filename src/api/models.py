"""
L2/L5.1/L5 Planner ORM 模型定义：`tickets`、`ticket_comments`、`pending_actions`、`kb_queries`、`audit_logs`、`ticket_drafts`、`agent_conversation_memory`、`user_memory`。

一、程序目标
1. 把 L2 需要持久化的业务对象映射成数据库表。
2. 为 `/ask`、`/tickets`、`/agent` 提供统一的数据落库结构。
3. 为 L5.1 的评论追加能力提供独立持久化表。
4. 为高风险动作的确认态提供待确认动作存储。
5. 为 L4 的多轮补全能力提供工单草稿持久化。

二、八张表分别负责什么
1. `Ticket`
   - 保存工单主数据。
2. `TicketComment`
   - 保存工单评论，避免 JSON 读改写导致覆盖。
3. `PendingAction`
   - 保存待确认的高风险动作，供二次确认执行。
4. `KBQuery`
   - 保存一次问答的检索、回答、引用与耗时。
5. `AuditLog`
   - 保存所有动作的可追溯审计记录。
6. `TicketDraft`
   - 保存需要多轮补全的工单草稿。
7. `AgentConversationMemory`
   - 保存用户级短期对话记忆，用于恢复“那个工单/刚才那个问题”。
8. `UserMemory`
   - 保存用户级长期默认资料，用于自动补全默认地点与联系方式。

三、输入输出
1. 输入：由 CRUD 层实例化字段值。
2. 输出：由 SQLAlchemy 管理的 ORM 实体对象。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.api.db import Base


JsonType = JSON().with_variant(JSONB, "postgresql")



def utc_now() -> datetime:
    """统一生成 UTC 时间，避免不同表的时间源不一致。"""
    return datetime.now(timezone.utc)



def new_uuid() -> str:
    """生成字符串版 UUID，便于跨数据库兼容。"""
    return str(uuid4())


class Ticket(Base):
    """工单主表，承载 L2 的“可执行动作”结果。"""

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    creator: Mapped[str] = mapped_column(String(64), default="anonymous")
    assignee: Mapped[str | None] = mapped_column(String(64), nullable=True)
    department: Mapped[str] = mapped_column(String(64), default="IT")
    category: Mapped[str] = mapped_column(String(32), default="other")
    priority: Mapped[str] = mapped_column(String(8), default="P2")
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="open", index=True)
    contact: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_draft_id: Mapped[str | None] = mapped_column(String(32), nullable=True, unique=True, index=True)
    context_json: Mapped[dict] = mapped_column(JsonType, default=dict)


class TicketComment(Base):
    """工单评论表，使用 append-only 方式记录补充说明。"""

    __tablename__ = "ticket_comments"
    __table_args__ = (
        Index("ix_ticket_comments_ticket_id_created_at", "ticket_id", "created_at"),
    )

    comment_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    ticket_id: Mapped[str] = mapped_column(String(36), ForeignKey("tickets.id", ondelete="CASCADE"))
    actor_user_id: Mapped[str] = mapped_column(String(64), default="anonymous")
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PendingAction(Base):
    """待确认动作表，用于承载高风险工具的二次确认。"""

    __tablename__ = "pending_actions"

    confirm_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    args_json: Mapped[dict] = mapped_column(JsonType, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class KBQuery(Base):
    """问答记录表，承载一次完整的 `/ask` 执行轨迹。"""

    __tablename__ = "kb_queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    user_name: Mapped[str] = mapped_column(String(64), default="anonymous")
    department: Mapped[str] = mapped_column(String(64), default="general")
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[list] = mapped_column(JsonType, default=list)
    retrieve_topk_json: Mapped[list] = mapped_column(JsonType, default=list)
    attempt_stage: Mapped[str] = mapped_column(String(64), default="unknown")
    latency_retrieve_ms: Mapped[int] = mapped_column(default=0)
    latency_answer_ms: Mapped[int] = mapped_column(default=0)
    model: Mapped[str] = mapped_column(String(128), default="unknown")
    valid_json: Mapped[bool] = mapped_column(Boolean, default=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditLog(Base):
    """审计表，记录所有关键动作，满足 L2 的可追溯要求。"""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    actor: Mapped[str] = mapped_column(String(64), default="anonymous")
    action_type: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str] = mapped_column(String(64))
    target_id: Mapped[str] = mapped_column(String(64))
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    payload_json: Mapped[dict] = mapped_column(JsonType, default=dict)


class TicketDraft(Base):
    """工单草稿表，承载 L4 的“缺字段后续办”能力。"""

    __tablename__ = "ticket_drafts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    draft_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    creator: Mapped[str] = mapped_column(String(64), default="anonymous")
    owner_user_id: Mapped[str] = mapped_column(String(64), default="anonymous", index=True)
    department: Mapped[str] = mapped_column(String(64), default="IT")
    payload_json: Mapped[dict] = mapped_column(JsonType, default=dict)
    missing_fields_json: Mapped[list] = mapped_column(JsonType, default=list)
    status: Mapped[str] = mapped_column(String(32), default="open", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    kb_request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class AgentConversationMemory(Base):
    """用户级短期对话记忆，只保存结构化引用与简短主题摘要。"""

    __tablename__ = "agent_conversation_memory"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    last_ticket_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    last_draft_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    last_tool: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_topic_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class UserMemory(Base):
    """用户级长期记忆，只保存可安全复用的默认资料。"""

    __tablename__ = "user_memory"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    default_location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    default_contact: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_ticket_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
