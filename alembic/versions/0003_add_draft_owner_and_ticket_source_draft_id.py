"""add_draft_owner_and_ticket_source_draft_id

Revision ID: 0003_draft_owner_ticket_src
Revises: 0002_add_ticket_drafts
Create Date: 2026-03-01 01:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_draft_owner_ticket_src"
down_revision = "0002_add_ticket_drafts"
branch_labels = None
depends_on = None



def upgrade() -> None:
    """为草稿绑定 owner，并为工单增加来源草稿唯一约束。"""
    with op.batch_alter_table("ticket_drafts") as batch_op:
        batch_op.add_column(sa.Column("owner_user_id", sa.String(length=64), nullable=True))

    op.execute("UPDATE ticket_drafts SET owner_user_id = creator WHERE owner_user_id IS NULL")

    with op.batch_alter_table("ticket_drafts") as batch_op:
        batch_op.alter_column("owner_user_id", existing_type=sa.String(length=64), nullable=False)
        batch_op.create_index(
            "ix_ticket_drafts_owner_user_id",
            ["owner_user_id"],
            unique=False,
        )

    with op.batch_alter_table("tickets") as batch_op:
        batch_op.add_column(sa.Column("source_draft_id", sa.String(length=32), nullable=True))
        batch_op.create_index(
            "ix_tickets_source_draft_id",
            ["source_draft_id"],
            unique=True,
        )



def downgrade() -> None:
    """回滚 owner 绑定与工单来源草稿字段。"""
    with op.batch_alter_table("tickets") as batch_op:
        batch_op.drop_index("ix_tickets_source_draft_id")
        batch_op.drop_column("source_draft_id")

    with op.batch_alter_table("ticket_drafts") as batch_op:
        batch_op.drop_index("ix_ticket_drafts_owner_user_id")
        batch_op.drop_column("owner_user_id")
