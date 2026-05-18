"""add_user_sessions

Revision ID: 0011_add_user_sessions
Revises: 0010_auth_expand_schema
Create Date: 2026-04-08 01:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0011_add_user_sessions"
down_revision = "0010_auth_expand_schema"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    """新增用户会话表，承载 refresh token 多设备会话。"""
    if not _has_table("user_sessions"):
        op.create_table(
            "user_sessions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("refresh_token_hash", sa.String(length=255), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("user_agent", sa.String(length=255), nullable=True),
            sa.Column("ip_address", sa.String(length=45), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if _has_table("user_sessions"):
        if not _has_index("user_sessions", "ix_user_sessions_user_id"):
            op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"], unique=False)
        if not _has_index("user_sessions", "ix_user_sessions_refresh_token_hash"):
            op.create_index(
                "ix_user_sessions_refresh_token_hash",
                "user_sessions",
                ["refresh_token_hash"],
                unique=True,
            )
        if not _has_index("user_sessions", "ix_user_sessions_expires_at"):
            op.create_index("ix_user_sessions_expires_at", "user_sessions", ["expires_at"], unique=False)
        if not _has_index("user_sessions", "ix_user_sessions_revoked_at"):
            op.create_index("ix_user_sessions_revoked_at", "user_sessions", ["revoked_at"], unique=False)
        if not _has_index("user_sessions", "ix_user_sessions_last_used_at"):
            op.create_index("ix_user_sessions_last_used_at", "user_sessions", ["last_used_at"], unique=False)
        if not _has_index("user_sessions", "ix_user_sessions_user_id_revoked_at_expires_at"):
            op.create_index(
                "ix_user_sessions_user_id_revoked_at_expires_at",
                "user_sessions",
                ["user_id", "revoked_at", "expires_at"],
                unique=False,
            )


def downgrade() -> None:
    """回滚用户会话表。"""
    if _has_table("user_sessions"):
        if _has_index("user_sessions", "ix_user_sessions_user_id_revoked_at_expires_at"):
            op.drop_index("ix_user_sessions_user_id_revoked_at_expires_at", table_name="user_sessions")
        if _has_index("user_sessions", "ix_user_sessions_last_used_at"):
            op.drop_index("ix_user_sessions_last_used_at", table_name="user_sessions")
        if _has_index("user_sessions", "ix_user_sessions_revoked_at"):
            op.drop_index("ix_user_sessions_revoked_at", table_name="user_sessions")
        if _has_index("user_sessions", "ix_user_sessions_expires_at"):
            op.drop_index("ix_user_sessions_expires_at", table_name="user_sessions")
        if _has_index("user_sessions", "ix_user_sessions_refresh_token_hash"):
            op.drop_index("ix_user_sessions_refresh_token_hash", table_name="user_sessions")
        if _has_index("user_sessions", "ix_user_sessions_user_id"):
            op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
        op.drop_table("user_sessions")
