"""add_ticket_drafts

Revision ID: 0002_add_ticket_drafts
Revises: 0001_initial_l2_schema
Create Date: 2026-03-01 00:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_add_ticket_drafts"
down_revision = "0001_initial_l2_schema"
branch_labels = None
depends_on = None



def _json_type() -> sa.types.TypeEngine:
    """根据当前数据库方言选择 JSON 类型；Postgres 用 JSONB，其它方言回退到 JSON。"""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        return postgresql.JSONB(astext_type=sa.Text())
    return sa.JSON()



def upgrade() -> None:
    """创建工单草稿表及索引，支撑 L4 的多轮补全能力。"""
    op.create_table(
        "ticket_drafts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("draft_id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("creator", sa.String(length=64), nullable=False),
        sa.Column("department", sa.String(length=64), nullable=False),
        sa.Column("payload_json", _json_type(), nullable=False),
        sa.Column("missing_fields_json", _json_type(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kb_request_id", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_ticket_drafts_draft_id",
        "ticket_drafts",
        ["draft_id"],
        unique=True,
        if_not_exists=True,
    )
    op.create_index(
        "ix_ticket_drafts_status",
        "ticket_drafts",
        ["status"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_ticket_drafts_kb_request_id",
        "ticket_drafts",
        ["kb_request_id"],
        unique=False,
        if_not_exists=True,
    )



def downgrade() -> None:
    """删除工单草稿表及索引。"""
    op.drop_index("ix_ticket_drafts_kb_request_id", table_name="ticket_drafts", if_exists=True)
    op.drop_index("ix_ticket_drafts_status", table_name="ticket_drafts", if_exists=True)
    op.drop_index("ix_ticket_drafts_draft_id", table_name="ticket_drafts", if_exists=True)
    op.drop_table("ticket_drafts", if_exists=True)
