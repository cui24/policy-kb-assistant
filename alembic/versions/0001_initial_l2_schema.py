"""initial_l2_schema

Revision ID: 0001_initial_l2_schema
Revises: None
Create Date: 2026-03-01 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_l2_schema"
down_revision = None
branch_labels = None
depends_on = None



def _json_type() -> sa.types.TypeEngine:
    """根据当前数据库方言选择 JSON 类型；Postgres 用 JSONB，其它方言回退到 JSON。"""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        return postgresql.JSONB(astext_type=sa.Text())
    return sa.JSON()



def upgrade() -> None:
    """创建 L2 三张核心表及索引；已存在时跳过，便于旧库平滑接入 Alembic。"""
    op.create_table(
        "tickets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("public_id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("creator", sa.String(length=64), nullable=False),
        sa.Column("department", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.String(length=8), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("contact", sa.String(length=128), nullable=True),
        sa.Column("context_json", _json_type(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_tickets_public_id",
        "tickets",
        ["public_id"],
        unique=True,
        if_not_exists=True,
    )
    op.create_index(
        "ix_tickets_status",
        "tickets",
        ["status"],
        unique=False,
        if_not_exists=True,
    )

    op.create_table(
        "kb_queries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_name", sa.String(length=64), nullable=False),
        sa.Column("department", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("citations_json", _json_type(), nullable=False),
        sa.Column("retrieve_topk_json", _json_type(), nullable=False),
        sa.Column("attempt_stage", sa.String(length=64), nullable=False),
        sa.Column("latency_retrieve_ms", sa.Integer(), nullable=False),
        sa.Column("latency_answer_ms", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("valid_json", sa.Boolean(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_kb_queries_request_id",
        "kb_queries",
        ["request_id"],
        unique=True,
        if_not_exists=True,
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(length=64), nullable=False),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("payload_json", _json_type(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_audit_logs_action_type",
        "audit_logs",
        ["action_type"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_audit_logs_request_id",
        "audit_logs",
        ["request_id"],
        unique=False,
        if_not_exists=True,
    )



def downgrade() -> None:
    """删除索引与表；用于回滚当前初始版本。"""
    op.drop_index("ix_audit_logs_request_id", table_name="audit_logs", if_exists=True)
    op.drop_index("ix_audit_logs_action_type", table_name="audit_logs", if_exists=True)
    op.drop_table("audit_logs", if_exists=True)

    op.drop_index("ix_kb_queries_request_id", table_name="kb_queries", if_exists=True)
    op.drop_table("kb_queries", if_exists=True)

    op.drop_index("ix_tickets_status", table_name="tickets", if_exists=True)
    op.drop_index("ix_tickets_public_id", table_name="tickets", if_exists=True)
    op.drop_table("tickets", if_exists=True)
