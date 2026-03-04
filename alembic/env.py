"""
Alembic 环境入口：负责把 SQLAlchemy 模型元数据接到迁移系统。

一、程序目标
1. 读取 `.env` 中的 `DATABASE_URL`。
2. 告诉 Alembic 当前项目的 `target_metadata` 是什么。
3. 支持离线迁移与在线迁移两种模式。

二、运行顺序
1. 读取 Alembic `config`
2. 加载 `.env`
3. 导入 `src.api.models` 以注册所有表
4. 设置 `target_metadata = Base.metadata`
5. 根据 Alembic 当前模式执行：
   - `run_migrations_offline()`
   - 或 `run_migrations_online()`
"""

from __future__ import annotations

import os

from alembic import context
from alembic.ddl import impl as alembic_ddl
from dotenv import load_dotenv
from sqlalchemy import Column, MetaData, PrimaryKeyConstraint, String, Table, engine_from_config, inspect, pool

from src.api.db import Base
from src.api import models


_ = models
config = context.config
load_dotenv()

_ALEMBIC_VERSION_NUM_LENGTH = 64

database_url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
config.set_main_option("sqlalchemy.url", database_url)

target_metadata = Base.metadata


def _version_table_impl_with_longer_version_num(
    self,
    *,
    version_table: str,
    version_table_schema: str | None,
    version_table_pk: bool,
    **_: object,
) -> Table:
    """让 Alembic 新建版本表时使用更长的 version_num。"""
    version_table_object = Table(
        version_table,
        MetaData(),
        Column("version_num", String(_ALEMBIC_VERSION_NUM_LENGTH), nullable=False),
        schema=version_table_schema,
    )
    if version_table_pk:
        version_table_object.append_constraint(
            PrimaryKeyConstraint("version_num", name=f"{version_table}_pkc")
        )
    return version_table_object


def _ensure_alembic_version_column_capacity(connection) -> None:
    """兼容旧数据库：若版本列过短，先扩容再执行迁移。"""
    inspector = inspect(connection)
    if "alembic_version" not in inspector.get_table_names():
        return

    version_column = next(
        (
            column
            for column in inspector.get_columns("alembic_version")
            if str(column.get("name") or "") == "version_num"
        ),
        None,
    )
    if version_column is None:
        return

    current_length = getattr(version_column.get("type"), "length", None)
    if current_length is not None and int(current_length) >= _ALEMBIC_VERSION_NUM_LENGTH:
        return

    if connection.dialect.name == "postgresql":
        connection.exec_driver_sql(
            f"ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR({_ALEMBIC_VERSION_NUM_LENGTH})"
        )
    elif connection.dialect.name in {"mysql", "mariadb"}:
        connection.exec_driver_sql(
            f"ALTER TABLE alembic_version MODIFY version_num VARCHAR({_ALEMBIC_VERSION_NUM_LENGTH}) NOT NULL"
        )


alembic_ddl.DefaultImpl.version_table_impl = _version_table_impl_with_longer_version_num



def run_migrations_offline() -> None:
    """不创建数据库连接，直接按 URL 文本生成迁移 SQL。"""
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()



def run_migrations_online() -> None:
    """创建真实数据库连接并执行迁移。"""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = database_url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        _ensure_alembic_version_column_capacity(connection)
        if connection.in_transaction():
            connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
