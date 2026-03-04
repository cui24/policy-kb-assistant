"""add_ticket_assignee

Revision ID: 0004_ticket_assignee
Revises: 0003_draft_owner_ticket_src
Create Date: 2026-03-01 02:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_ticket_assignee"
down_revision = "0003_draft_owner_ticket_src"
branch_labels = None
depends_on = None



def upgrade() -> None:
    """为 tickets 增加 assignee 字段，支持查进度时展示处理人。"""
    with op.batch_alter_table("tickets") as batch_op:
        batch_op.add_column(sa.Column("assignee", sa.String(length=64), nullable=True))



def downgrade() -> None:
    """回滚 assignee 字段。"""
    with op.batch_alter_table("tickets") as batch_op:
        batch_op.drop_column("assignee")
