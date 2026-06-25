"""ASK 问答记录与审计日志的本地异步批量落库队列。"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.api import crud, models
from src.api.db import SessionLocal


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskPersistItem:
    kb_payload: dict[str, Any]
    audit_payload: dict[str, Any]


_queue: asyncio.Queue[AskPersistItem] | None = None
_worker_task: asyncio.Task | None = None
_stop_event: asyncio.Event | None = None


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def is_enabled() -> bool:
    load_dotenv()
    return _bool_env("ASK_ASYNC_PERSIST_ENABLED", False)


def queue_maxsize() -> int:
    return _int_env("ASK_ASYNC_PERSIST_QUEUE_MAXSIZE", 1000, 1, 100_000)


def batch_size() -> int:
    return _int_env("ASK_ASYNC_PERSIST_BATCH_SIZE", 50, 1, 1000)


def flush_interval_seconds() -> float:
    return _float_env("ASK_ASYNC_PERSIST_FLUSH_INTERVAL_SECONDS", 1.0, 0.05, 60.0)


def shutdown_flush_timeout_seconds() -> float:
    return _float_env("ASK_ASYNC_PERSIST_SHUTDOWN_FLUSH_TIMEOUT_SECONDS", 2.0, 0.1, 30.0)


def queued_item_count() -> int:
    return 0 if _queue is None else int(_queue.qsize())


def start_worker() -> None:
    """启动当前 worker 进程内的后台刷库任务。"""
    global _queue, _worker_task, _stop_event
    if not is_enabled():
        return
    if _worker_task is not None and not _worker_task.done():
        return
    _queue = asyncio.Queue(maxsize=queue_maxsize())
    _stop_event = asyncio.Event()
    _worker_task = asyncio.create_task(_worker_loop(), name="ask-persist-writer")
    logger.info(
        "ASK async persist worker started: queue_maxsize=%s batch_size=%s flush_interval=%s",
        queue_maxsize(),
        batch_size(),
        flush_interval_seconds(),
    )


async def stop_worker() -> None:
    """停止后台刷库任务，并在超时内尽量刷完内存队列。"""
    global _worker_task
    if _worker_task is None:
        return
    remaining = queued_item_count()
    if remaining:
        logger.warning("Shutting down, flushing %s remaining ASK persist items...", remaining)
    if _stop_event is not None:
        _stop_event.set()
    try:
        await asyncio.wait_for(_worker_task, timeout=shutdown_flush_timeout_seconds())
    except asyncio.TimeoutError:
        logger.error(
            "ASK async persist shutdown flush timed out: remaining_items=%s timeout_seconds=%s",
            queued_item_count(),
            shutdown_flush_timeout_seconds(),
        )
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    _worker_task = None


async def enqueue_ask_persist(kb_payload: dict[str, Any], audit_payload: dict[str, Any]) -> str:
    """
    把 ASK 落库任务放入本地队列。

    返回：
    - disabled：未启用异步持久化。
    - queued：成功入队，请求路径不再等待 DB 写入。
    - sync_fallback：队列不可用或已满，退回同步写库，避免丢数据。
    """
    if not is_enabled():
        return "disabled"
    if _queue is None:
        await asyncio.to_thread(_persist_one_sync, kb_payload, audit_payload)
        return "sync_fallback"
    item = AskPersistItem(dict(kb_payload or {}), dict(audit_payload or {}))
    try:
        _queue.put_nowait(item)
    except asyncio.QueueFull:
        await asyncio.to_thread(_persist_one_sync, kb_payload, audit_payload)
        return "sync_fallback"
    return "queued"


async def _worker_loop() -> None:
    assert _queue is not None
    assert _stop_event is not None
    pending: list[AskPersistItem] = []
    while not _stop_event.is_set() or not _queue.empty() or pending:
        try:
            item = await asyncio.wait_for(_queue.get(), timeout=flush_interval_seconds())
            pending.append(item)
            _queue.task_done()
        except asyncio.TimeoutError:
            pass

        while len(pending) < batch_size() and not _queue.empty():
            pending.append(_queue.get_nowait())
            _queue.task_done()

        if pending and (len(pending) >= batch_size() or _stop_event.is_set() or _queue.empty()):
            batch = pending
            pending = []
            try:
                await asyncio.to_thread(_persist_batch_sync, batch)
            except Exception:
                logger.exception("ASK async persist batch failed; dropped_items=%s", len(batch))


def _persist_one_sync(kb_payload: dict[str, Any], audit_payload: dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        crud.create_kb_query_with_audit(db, kb_payload, audit_payload)
    finally:
        db.close()


def _persist_batch_sync(items: list[AskPersistItem]) -> None:
    if not items:
        return
    db = SessionLocal()
    try:
        _persist_batch_in_session(db, items)
    finally:
        db.close()


def _persist_batch_in_session(db: Session, items: list[AskPersistItem]) -> None:
    for item in items:
        kb_payload = crud._strip_nul_chars(dict(item.kb_payload or {}))
        if "actor_user_id" not in kb_payload:
            kb_payload["actor_user_id"] = str(kb_payload.get("user_name") or "anonymous")
        kb_record = models.KBQuery(**kb_payload)
        db.add(kb_record)
        db.flush()

        audit_payload = crud._strip_nul_chars(dict(item.audit_payload or {}))
        if "actor_user_id" not in audit_payload:
            audit_payload["actor_user_id"] = str(audit_payload.get("actor") or "anonymous")
        if not audit_payload.get("target_id"):
            audit_payload["target_id"] = kb_record.id
        audit_record = models.AuditLog(**audit_payload)
        db.add(audit_record)
    db.commit()
