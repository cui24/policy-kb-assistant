"""add_user_memory

Revision ID: 0008_add_user_memory
Revises: 0007_add_agent_conversation_memory
Create Date: 2026-03-02 23:45:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0008_add_user_memory"
down_revision = "0007_add_agent_conversation_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """新增用户长期记忆表，保存默认地点与联系方式。"""
    op.create_table(
        "user_memory",
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("default_location", sa.String(length=255), nullable=True),
        sa.Column("default_contact", sa.String(length=128), nullable=True),
        sa.Column("source_ticket_id", sa.String(length=32), nullable=True),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index("ix_user_memory_source_ticket_id", "user_memory", ["source_ticket_id"], unique=False)


def downgrade() -> None:
    """回滚用户长期记忆表。"""
    op.drop_index("ix_user_memory_source_ticket_id", table_name="user_memory")
    op.drop_table("user_memory")
