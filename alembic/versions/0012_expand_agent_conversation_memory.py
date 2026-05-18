"""expand_agent_conversation_memory

Revision ID: 0012_expand_agent_conversation_memory
Revises: 0011_add_user_sessions
Create Date: 2026-05-15 10:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0012_expand_agent_conversation_memory"
down_revision = "0011_add_user_sessions"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(column.get("name") == column_name for column in inspector.get_columns(table_name))


def _has_index(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    """扩展 L1 Session Memory 字段。"""
    if not _has_table("agent_conversation_memory"):
        return

    if not _has_column("agent_conversation_memory", "current_goal"):
        op.add_column("agent_conversation_memory", sa.Column("current_goal", sa.Text(), nullable=True))
    if not _has_column("agent_conversation_memory", "pending_task_json"):
        op.add_column("agent_conversation_memory", sa.Column("pending_task_json", sa.JSON(), nullable=True))
    if not _has_column("agent_conversation_memory", "recent_turns_json"):
        op.add_column("agent_conversation_memory", sa.Column("recent_turns_json", sa.JSON(), nullable=True))
    if not _has_column("agent_conversation_memory", "expires_at"):
        op.add_column("agent_conversation_memory", sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))
    if not _has_index("agent_conversation_memory", "ix_agent_conversation_memory_expires_at"):
        op.create_index(
            "ix_agent_conversation_memory_expires_at",
            "agent_conversation_memory",
            ["expires_at"],
            unique=False,
        )


def downgrade() -> None:
    """回滚 L1 Session Memory 扩展字段。"""
    if not _has_table("agent_conversation_memory"):
        return

    if _has_index("agent_conversation_memory", "ix_agent_conversation_memory_expires_at"):
        op.drop_index("ix_agent_conversation_memory_expires_at", table_name="agent_conversation_memory")
    if _has_column("agent_conversation_memory", "expires_at"):
        op.drop_column("agent_conversation_memory", "expires_at")
    if _has_column("agent_conversation_memory", "recent_turns_json"):
        op.drop_column("agent_conversation_memory", "recent_turns_json")
    if _has_column("agent_conversation_memory", "pending_task_json"):
        op.drop_column("agent_conversation_memory", "pending_task_json")
    if _has_column("agent_conversation_memory", "current_goal"):
        op.drop_column("agent_conversation_memory", "current_goal")
