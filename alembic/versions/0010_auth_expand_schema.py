"""auth_expand_schema

Revision ID: 0010_auth_expand_schema
Revises: 0009_add_query_path_indexes
Create Date: 2026-04-08 00:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0010_auth_expand_schema"
down_revision = "0009_add_query_path_indexes"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col.get("name") == column_name for col in inspector.get_columns(table_name))


def _has_index(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(idx.get("name") == index_name for idx in inspector.get_indexes(table_name))


def _create_role_enum_if_needed() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    role_enum = sa.Enum("admin", "user", "support", name="role_enum")
    role_enum.create(bind, checkfirst=True)


def _drop_role_enum_if_needed() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    role_enum = sa.Enum("admin", "user", "support", name="role_enum")
    role_enum.drop(bind, checkfirst=True)


def _role_type() -> sa.types.TypeEngine:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # 该枚举类型由 _create_role_enum_if_needed() 负责创建，这里只复用。
        return postgresql.ENUM("admin", "user", "support", name="role_enum", create_type=False)
    return sa.String(length=32)


def upgrade() -> None:
    """第一阶段扩展鉴权相关 schema：先加表和列，不做强约束收紧。"""
    _create_role_enum_if_needed()

    if not _has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("username", sa.String(length=32), nullable=False),
            sa.Column("password_hash", sa.String(length=255), nullable=False),
            sa.Column("email", sa.String(length=128), nullable=True),
            sa.Column("phone", sa.String(length=32), nullable=True),
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("role", _role_type(), nullable=False, server_default=sa.text("'user'")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.PrimaryKeyConstraint("id"),
        )

    if _has_table("users"):
        if not _has_index("users", "ix_users_username"):
            op.create_index("ix_users_username", "users", ["username"], unique=True)
        if not _has_index("users", "ix_users_email"):
            op.create_index("ix_users_email", "users", ["email"], unique=True)
        if not _has_index("users", "ix_users_phone"):
            op.create_index("ix_users_phone", "users", ["phone"], unique=True)
        if not _has_index("users", "ix_users_role"):
            op.create_index("ix_users_role", "users", ["role"], unique=False)
        if not _has_index("users", "ix_users_is_active"):
            op.create_index("ix_users_is_active", "users", ["is_active"], unique=False)

    if _has_table("tickets"):
        with op.batch_alter_table("tickets") as batch_op:
            if not _has_column("tickets", "creator_user_id"):
                batch_op.add_column(sa.Column("creator_user_id", sa.String(length=36), nullable=True))
            if not _has_column("tickets", "assignee_user_id"):
                batch_op.add_column(sa.Column("assignee_user_id", sa.String(length=36), nullable=True))
        if not _has_index("tickets", "ix_tickets_creator_user_id"):
            op.create_index("ix_tickets_creator_user_id", "tickets", ["creator_user_id"], unique=False)
        if not _has_index("tickets", "ix_tickets_assignee_user_id"):
            op.create_index("ix_tickets_assignee_user_id", "tickets", ["assignee_user_id"], unique=False)

    if _has_table("kb_queries"):
        with op.batch_alter_table("kb_queries") as batch_op:
            if not _has_column("kb_queries", "actor_user_id"):
                batch_op.add_column(sa.Column("actor_user_id", sa.String(length=36), nullable=True))
        if not _has_index("kb_queries", "ix_kb_queries_actor_user_id"):
            op.create_index("ix_kb_queries_actor_user_id", "kb_queries", ["actor_user_id"], unique=False)

    if _has_table("audit_logs"):
        with op.batch_alter_table("audit_logs") as batch_op:
            if not _has_column("audit_logs", "actor_user_id"):
                batch_op.add_column(sa.Column("actor_user_id", sa.String(length=36), nullable=True))
        if not _has_index("audit_logs", "ix_audit_logs_actor_user_id"):
            op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"], unique=False)


def downgrade() -> None:
    """回滚鉴权 schema 扩展。"""
    if _has_table("audit_logs"):
        if _has_index("audit_logs", "ix_audit_logs_actor_user_id"):
            op.drop_index("ix_audit_logs_actor_user_id", table_name="audit_logs")
        if _has_column("audit_logs", "actor_user_id"):
            with op.batch_alter_table("audit_logs") as batch_op:
                batch_op.drop_column("actor_user_id")

    if _has_table("kb_queries"):
        if _has_index("kb_queries", "ix_kb_queries_actor_user_id"):
            op.drop_index("ix_kb_queries_actor_user_id", table_name="kb_queries")
        if _has_column("kb_queries", "actor_user_id"):
            with op.batch_alter_table("kb_queries") as batch_op:
                batch_op.drop_column("actor_user_id")

    if _has_table("tickets"):
        if _has_index("tickets", "ix_tickets_assignee_user_id"):
            op.drop_index("ix_tickets_assignee_user_id", table_name="tickets")
        if _has_index("tickets", "ix_tickets_creator_user_id"):
            op.drop_index("ix_tickets_creator_user_id", table_name="tickets")
        with op.batch_alter_table("tickets") as batch_op:
            if _has_column("tickets", "assignee_user_id"):
                batch_op.drop_column("assignee_user_id")
            if _has_column("tickets", "creator_user_id"):
                batch_op.drop_column("creator_user_id")

    if _has_table("users"):
        if _has_index("users", "ix_users_is_active"):
            op.drop_index("ix_users_is_active", table_name="users")
        if _has_index("users", "ix_users_role"):
            op.drop_index("ix_users_role", table_name="users")
        if _has_index("users", "ix_users_phone"):
            op.drop_index("ix_users_phone", table_name="users")
        if _has_index("users", "ix_users_email"):
            op.drop_index("ix_users_email", table_name="users")
        if _has_index("users", "ix_users_username"):
            op.drop_index("ix_users_username", table_name="users")
        op.drop_table("users")

    _drop_role_enum_if_needed()
