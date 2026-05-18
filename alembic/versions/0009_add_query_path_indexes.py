"""add_query_path_indexes

Revision ID: 0009_add_query_path_indexes
Revises: 0008_add_user_memory
Create Date: 2026-04-08 00:00:00
"""

from __future__ import annotations

from alembic import op


revision = "0009_add_query_path_indexes"
down_revision = "0008_add_user_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """按当前查询路径补充复合索引与时间索引。"""
    op.create_index("ix_tickets_created_at", "tickets", ["created_at"], unique=False)
    op.create_index("ix_tickets_status_created_at", "tickets", ["status", "created_at"], unique=False)

    op.create_index(
        "ix_pending_actions_status_expires_at",
        "pending_actions",
        ["status", "expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_pending_actions_user_id_status_expires_at",
        "pending_actions",
        ["user_id", "status", "expires_at"],
        unique=False,
    )

    op.create_index("ix_kb_queries_created_at", "kb_queries", ["created_at"], unique=False)
    op.create_index(
        "ix_kb_queries_user_name_created_at",
        "kb_queries",
        ["user_name", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_kb_queries_department_created_at",
        "kb_queries",
        ["department", "created_at"],
        unique=False,
    )

    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)
    op.create_index(
        "ix_audit_logs_target_type_target_id_created_at",
        "audit_logs",
        ["target_type", "target_id", "created_at"],
        unique=False,
    )

    op.create_index("ix_ticket_drafts_expires_at", "ticket_drafts", ["expires_at"], unique=False)


def downgrade() -> None:
    """回滚本次新增索引。"""
    op.drop_index("ix_ticket_drafts_expires_at", table_name="ticket_drafts")

    op.drop_index("ix_audit_logs_target_type_target_id_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")

    op.drop_index("ix_kb_queries_department_created_at", table_name="kb_queries")
    op.drop_index("ix_kb_queries_user_name_created_at", table_name="kb_queries")
    op.drop_index("ix_kb_queries_created_at", table_name="kb_queries")

    op.drop_index("ix_pending_actions_user_id_status_expires_at", table_name="pending_actions")
    op.drop_index("ix_pending_actions_status_expires_at", table_name="pending_actions")

    op.drop_index("ix_tickets_status_created_at", table_name="tickets")
    op.drop_index("ix_tickets_created_at", table_name="tickets")
