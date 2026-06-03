"""add_kb_query_usage_metrics

Revision ID: 0013_add_kb_query_usage_metrics
Revises: 0012_expand_agent_conversation_memory
Create Date: 2026-05-31 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0013_add_kb_query_usage_metrics"
down_revision = "0012_expand_agent_conversation_memory"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(column.get("name") == column_name for column in inspector.get_columns(table_name))


def upgrade() -> None:
    """为 ASK 记录补充 JSON 修复、token usage 与成本估算字段。"""
    if not _has_table("kb_queries"):
        return

    if not _has_column("kb_queries", "repair_used"):
        op.add_column("kb_queries", sa.Column("repair_used", sa.Boolean(), nullable=False, server_default=sa.false()))
    if not _has_column("kb_queries", "prompt_tokens"):
        op.add_column("kb_queries", sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"))
    if not _has_column("kb_queries", "completion_tokens"):
        op.add_column("kb_queries", sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"))
    if not _has_column("kb_queries", "total_tokens"):
        op.add_column("kb_queries", sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"))
    if not _has_column("kb_queries", "token_usage_estimated"):
        op.add_column(
            "kb_queries",
            sa.Column("token_usage_estimated", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    if not _has_column("kb_queries", "estimated_cost_usd"):
        op.add_column("kb_queries", sa.Column("estimated_cost_usd", sa.Float(), nullable=False, server_default="0"))


def downgrade() -> None:
    """回滚 ASK usage/cost 字段。"""
    if not _has_table("kb_queries"):
        return

    for column_name in (
        "estimated_cost_usd",
        "token_usage_estimated",
        "total_tokens",
        "completion_tokens",
        "prompt_tokens",
        "repair_used",
    ):
        if _has_column("kb_queries", column_name):
            op.drop_column("kb_queries", column_name)
