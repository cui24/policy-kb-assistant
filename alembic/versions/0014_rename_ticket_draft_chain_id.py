"""rename_ticket_draft_chain_id

Revision ID: 0014_rename_ticket_draft_chain_id
Revises: 0013_add_kb_query_usage_metrics
Create Date: 2026-06-06 10:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0014_rename_ticket_draft_chain_id"
down_revision = "0013_add_kb_query_usage_metrics"
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
    return any(index.get("name") == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    """把工单草稿链路 ID 从问答语义改为 Agent 语义。"""
    if not _has_table("ticket_drafts"):
        return

    if not _has_column("ticket_drafts", "agent_request_id"):
        op.add_column("ticket_drafts", sa.Column("agent_request_id", sa.String(length=64), nullable=True))

    if _has_column("ticket_drafts", "kb_request_id"):
        op.execute(
            sa.text(
                "UPDATE ticket_drafts "
                "SET agent_request_id = COALESCE(agent_request_id, kb_request_id) "
                "WHERE kb_request_id IS NOT NULL"
            )
        )

    if not _has_index("ticket_drafts", "ix_ticket_drafts_agent_request_id"):
        op.create_index(
            "ix_ticket_drafts_agent_request_id",
            "ticket_drafts",
            ["agent_request_id"],
            unique=False,
        )

    if _has_index("ticket_drafts", "ix_ticket_drafts_kb_request_id"):
        op.drop_index("ix_ticket_drafts_kb_request_id", table_name="ticket_drafts")
    if _has_column("ticket_drafts", "kb_request_id"):
        with op.batch_alter_table("ticket_drafts") as batch_op:
            batch_op.drop_column("kb_request_id")


def downgrade() -> None:
    """回滚为旧的问答链路 ID 字段。"""
    if not _has_table("ticket_drafts"):
        return

    if not _has_column("ticket_drafts", "kb_request_id"):
        op.add_column("ticket_drafts", sa.Column("kb_request_id", sa.String(length=64), nullable=True))

    if _has_column("ticket_drafts", "agent_request_id"):
        op.execute(
            sa.text(
                "UPDATE ticket_drafts "
                "SET kb_request_id = COALESCE(kb_request_id, agent_request_id) "
                "WHERE agent_request_id IS NOT NULL"
            )
        )

    if not _has_index("ticket_drafts", "ix_ticket_drafts_kb_request_id"):
        op.create_index(
            "ix_ticket_drafts_kb_request_id",
            "ticket_drafts",
            ["kb_request_id"],
            unique=False,
        )

    if _has_index("ticket_drafts", "ix_ticket_drafts_agent_request_id"):
        op.drop_index("ix_ticket_drafts_agent_request_id", table_name="ticket_drafts")
    if _has_column("ticket_drafts", "agent_request_id"):
        with op.batch_alter_table("ticket_drafts") as batch_op:
            batch_op.drop_column("agent_request_id")
