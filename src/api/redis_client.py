"""
Redis 客户端模块：统一 Redis 连接、健康检查与基础读写能力。

一、程序目标
1. 从环境变量读取 Redis 地址并创建客户端。
2. 提供全局复用的连接入口，避免每次请求重复建连。
3. 提供幂等/限流可复用的基础方法（字符串、JSON、NX 写入）。

二、设计约束
1. 当前项目是同步 FastAPI + 同步 SQLAlchemy，因此这里使用同步 Redis 客户端。
2. 尽量保持“薄封装”：只做连接管理和通用读写，不直接耦合业务逻辑。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from redis import Redis
from redis.exceptions import RedisError


DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_REDIS_MAX_CONNECTIONS = 20


class RedisClientError(RuntimeError):
    """Redis 客户端异常，统一上抛给上层业务处理。"""


def _resolve_redis_url() -> str:
    """读取 Redis 地址；未配置时回退本地默认值。"""
    load_dotenv()
    normalized = str(os.getenv("REDIS_URL") or "").strip()
    return normalized or DEFAULT_REDIS_URL


def _redis_max_connections() -> int:
    """读取单 worker Redis 连接池上限。"""
    raw = str(os.getenv("REDIS_MAX_CONNECTIONS") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_REDIS_MAX_CONNECTIONS
    return max(2, min(value, 200))


class RedisClient:
    """项目级 Redis 客户端封装。"""

    def __init__(self, url: str | None = None) -> None:
        self.url = (url or _resolve_redis_url()).strip() or DEFAULT_REDIS_URL
        # decode_responses=True：统一返回 str，便于直接做 JSON 编解码。
        self.client: Redis = Redis.from_url(
            self.url,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            health_check_interval=30,
            retry_on_timeout=True,
            max_connections=_redis_max_connections(),
        )

    def get_redis(self) -> Redis:
        """返回底层 Redis 对象，供高级用法直接调用。"""
        return self.client

    def ping(self) -> bool:
        """测试 Redis 连通性。"""
        try:
            return bool(self.client.ping())
        except RedisError:
            return False

    def close(self) -> None:
        """主动关闭连接（常用于测试或进程结束前清理）。"""
        self.client.close()


@lru_cache(maxsize=1)
def get_redis_client(url: str | None = None) -> RedisClient:
    """
    返回进程级单例 RedisClient。

    说明：
    1. 绝大多数业务直接用默认连接，不传 url。
    2. 单测中可传入测试地址或调用 `reset_redis_client_cache()` 重新初始化。
    3. 第一次调用时创建一个 RedisClient，后面再调用时，复用同一个对象
    4. 不会每次请求都重新 new 一个客户端
    """
    return RedisClient(url=url)


def reset_redis_client_cache() -> None:
    """清理单例缓存并关闭旧连接，供测试场景复用。"""
    if get_redis_client.cache_info().currsize > 0:
        try:
            get_redis_client().close()
        except Exception:
            pass
    get_redis_client.cache_clear()


def get_redis() -> Redis:
    """返回全局 Redis 连接。"""
    return get_redis_client().get_redis()


def ping_redis() -> tuple[bool, str]:
    """返回 Redis 健康状态与描述信息。"""
    try:
        ok = get_redis_client().ping()
    except Exception as exc:  # pragma: no cover - 极端底层异常
        return False, f"redis_error:{exc.__class__.__name__}"
    return (True, "ok") if ok else (False, "ping_failed")


def set_string(
    key: str,
    value: str,
    *,
    ttl_seconds: int | None = None,
    nx: bool = False,
    xx: bool = False,
) -> bool:
    """写字符串值，支持 TTL 与 NX/XX 选项。
        NX：只有 key 不存在时才写入
        XX：只有 key 已存在时才更新
    """
    try:
        return bool(get_redis().set(key, value, ex=ttl_seconds, nx=nx, xx=xx))
    except RedisError as exc:
        raise RedisClientError(f"redis_set_failed:{exc.__class__.__name__}") from exc


def get_string(key: str) -> str | None:
    """读字符串值；不存在返回 None。"""
    try:
        raw = get_redis().get(key)
    except RedisError as exc:
        raise RedisClientError(f"redis_get_failed:{exc.__class__.__name__}") from exc
    return None if raw is None else str(raw)


def set_json(
    key: str,
    value: Any,
    *,
    ttl_seconds: int | None = None,
    nx: bool = False,
    xx: bool = False,
) -> bool:
    """Redis 本身存的是字符串/字节，不是 Python dict
    所以如果想存对象，需要自己做序列化
    先把 Python 对象转成 JSON 字符串
    再调用 set_string() 存进去
    写 JSON 值，内部序列化为 UTF-8 字符串。"""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return set_string(key, encoded, ttl_seconds=ttl_seconds, nx=nx, xx=xx)


def get_json(key: str) -> Any | None:
    """先读字符串，把 JSON 字符串解析回 Python 对象"""
    raw = get_string(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RedisClientError("redis_value_not_json") from exc


def delete_keys(*keys: str) -> int:
    """删除一个或多个 key，返回删除数量。"""
    if not keys:
        return 0
    try:
        return int(get_redis().delete(*keys))
    except RedisError as exc:
        raise RedisClientError(f"redis_delete_failed:{exc.__class__.__name__}") from exc
