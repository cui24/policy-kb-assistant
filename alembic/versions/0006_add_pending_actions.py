"""add_pending_actions

Revision ID: 0006_add_pending_actions
Revises: 0005_add_ticket_comments
Create Date: 2026-03-02 14:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0006_add_pending_actions"
down_revision = "0005_add_ticket_comments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """新增待确认动作表，支撑高风险工具的二次确认。"""
    op.create_table(
        "pending_actions",
        sa.Column("confirm_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("args_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("confirm_id"),
    )
    op.create_index("ix_pending_actions_user_id", "pending_actions", ["user_id"], unique=False)
    op.create_index("ix_pending_actions_tool_name", "pending_actions", ["tool_name"], unique=False)
    op.create_index("ix_pending_actions_status", "pending_actions", ["status"], unique=False)
    op.create_index("ix_pending_actions_expires_at", "pending_actions", ["expires_at"], unique=False)


def downgrade() -> None:
    """回滚待确认动作表。"""
    op.drop_index("ix_pending_actions_expires_at", table_name="pending_actions")
    op.drop_index("ix_pending_actions_status", table_name="pending_actions")
    op.drop_index("ix_pending_actions_tool_name", table_name="pending_actions")
    op.drop_index("ix_pending_actions_user_id", table_name="pending_actions")
    op.drop_table("pending_actions")
