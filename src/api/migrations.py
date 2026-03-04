"""
L3 数据库迁移模块：负责把 Alembic 接到当前 API 启动流程。

一、程序目标
1. 让 API 启动时优先执行 Alembic 迁移，而不是直接 `create_all()`。
2. 提供统一的 `upgrade_to_head()` 与 `ensure_schema_ready()` 入口。
3. 在开发模式下允许保留 `create_all()` 兜底，但不再把它当默认主路径。

二、运行顺序
1. 读取 `.env`
2. 构造 Alembic 配置对象
3. 执行 `alembic upgrade head`
4. 若迁移失败且开发兜底开启，则退回 `create_all()`

三、输入输出
1. 输入：
   - `DATABASE_URL`
   - `AUTO_MIGRATE_ON_STARTUP`
   - `DEV_DB_FALLBACK_CREATE_ALL`
2. 输出：
   - 迁移执行完成或抛出异常
"""

from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from dotenv import load_dotenv

from src.api.db import DATABASE_URL, create_all_tables


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
ALEMBIC_DIR = PROJECT_ROOT / "alembic"



def _flag_is_true(raw_value: str | None, default: bool) -> bool:
    """将环境变量字符串解析成布尔值。"""
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}



def get_alembic_config() -> Config:
    """构造当前项目可复用的 Alembic 配置对象。"""
    load_dotenv()
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL", DATABASE_URL))
    return config



def upgrade_to_head() -> None:
    """执行 `alembic upgrade head`。"""
    command.upgrade(get_alembic_config(), "head")



def ensure_schema_ready() -> None:
    """API 启动时调用：优先迁移，开发模式可按需回退到 `create_all()`。"""
    load_dotenv()
    auto_migrate = _flag_is_true(os.getenv("AUTO_MIGRATE_ON_STARTUP"), True)
    fallback_create_all = _flag_is_true(os.getenv("DEV_DB_FALLBACK_CREATE_ALL"), False)

    if not auto_migrate:
        if fallback_create_all:
            create_all_tables()
        return

    try:
        upgrade_to_head()
    except Exception:
        if fallback_create_all:
            create_all_tables()
            return
        raise
