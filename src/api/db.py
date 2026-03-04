"""
L2/L3 数据库模块：负责数据库连接、会话工厂与开发兜底建表入口。

一、程序目标
1. 从 `.env` 读取数据库连接地址。
2. 创建 SQLAlchemy `engine` 与 `SessionLocal`。
3. 提供 `create_all_tables()`，作为开发环境的兜底建表入口。

二、运行顺序
1. 导入模块时先解析数据库地址。
2. 创建 `engine` 与会话工厂。
3. Alembic 或开发兜底逻辑会按需调用 `create_all_tables()`。

三、输入输出
1. 输入：`DATABASE_URL` 环境变量。
2. 输出：
   - `engine`
   - `SessionLocal`
   - `Base`
   - `create_all_tables()`

四、实现取舍
1. 当前默认主路径已经切到 Alembic 迁移。
2. `create_all()` 不再作为默认启动方式，只保留给开发模式兜底。
3. 这样既保留本地快速恢复能力，又让正式环境具备可演进的 schema 管理。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    """所有 ORM 模型共用的声明式基类。"""



def _resolve_database_url() -> str:
    """读取数据库地址；若未配置则回退到本地 SQLite，保证最小可运行。"""
    load_dotenv()
    return os.getenv("DATABASE_URL", "sqlite:///./policy_kb_l2.db")


DATABASE_URL = _resolve_database_url()
_SQLITE_CONNECT_ARGS = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args=_SQLITE_CONNECT_ARGS,
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)



def create_all_tables() -> None:
    """导入模型并直接建表；仅用于开发兜底，不作为默认迁移主路径。"""
    """
    这里把模型导入放在函数内部，是为了避免模块间循环导入。
    只有在真正建表时，才需要让 SQLAlchemy 知道有哪些 ORM 模型。
    """
    from src.api import models

    _ = models
    Base.metadata.create_all(bind=engine)


def init_db() -> None:
    """兼容旧调用；转发到 `create_all_tables()`。"""
    create_all_tables()
