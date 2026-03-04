"""add_agent_conversation_memory

Revision ID: 0007_add_agent_conversation_memory
Revises: 0006_add_pending_actions
Create Date: 2026-03-02 23:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0007_add_agent_conversation_memory"
down_revision = "0006_add_pending_actions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """新增短期对话记忆表，支撑对象指代恢复。"""
    op.create_table(
        "agent_conversation_memory",
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_ticket_id", sa.String(length=32), nullable=True),
        sa.Column("last_draft_id", sa.String(length=32), nullable=True),
        sa.Column("last_tool", sa.String(length=64), nullable=True),
        sa.Column("last_topic_summary", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index(
        "ix_agent_conversation_memory_last_ticket_id",
        "agent_conversation_memory",
        ["last_ticket_id"],
        unique=False,
    )
    op.create_index(
        "ix_agent_conversation_memory_last_draft_id",
        "agent_conversation_memory",
        ["last_draft_id"],
        unique=False,
    )


def downgrade() -> None:
    """回滚短期对话记忆表。"""
    op.drop_index("ix_agent_conversation_memory_last_draft_id", table_name="agent_conversation_memory")
    op.drop_index("ix_agent_conversation_memory_last_ticket_id", table_name="agent_conversation_memory")
    op.drop_table("agent_conversation_memory")
