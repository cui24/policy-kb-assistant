"""add_ticket_comments

Revision ID: 0005_add_ticket_comments
Revises: 0004_ticket_assignee
Create Date: 2026-03-02 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_add_ticket_comments"
down_revision = "0004_ticket_assignee"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """新增工单评论表，改为 append-only 记录评论。"""
    op.create_table(
        "ticket_comments",
        sa.Column("comment_id", sa.String(length=36), nullable=False),
        sa.Column("ticket_id", sa.String(length=36), nullable=False),
        sa.Column("actor_user_id", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["ticket_id"], ["tickets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("comment_id"),
    )
    op.create_index(
        "ix_ticket_comments_ticket_id_created_at",
        "ticket_comments",
        ["ticket_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """回滚工单评论表。"""
    op.drop_index("ix_ticket_comments_ticket_id_created_at", table_name="ticket_comments")
    op.drop_table("ticket_comments")
