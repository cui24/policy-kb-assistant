"""非流式 ASK 问答结果的 Redis 缓存辅助模块。

本模块只负责“可复用问答结果”的缓存读写，不负责 request_id 生成、
数据库落库或审计写入。这样做的原因是：

1. 缓存命中时也必须为本次请求生成新的 request_id。
2. 缓存命中时也必须写入新的 kb_queries 和 audit_logs，保证历史追溯完整。
3. 传统规则 ASK、LangGraph ASK、MCP ask_policy 等入口都可以复用同一套缓存语义。

缓存中保存的是标准化后的 answer/citations/trace/meta，而不是完整 API 响应。
完整响应里的 request_id/query_id 属于单次请求，不能被跨请求复用。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from src.api import redis_client


CACHE_SCHEMA_VERSION = "v1"
DEFAULT_TTL_SECONDS = 3600
MAX_TTL_SECONDS = 7 * 24 * 3600
MAX_QUESTION_CHARS = 500
DEFAULT_SEMANTIC_THRESHOLD = 0.97
DEFAULT_SEMANTIC_MAX_CANDIDATES = 200

_CACHE_META_KEYS = {
    "cache_hit",
    "cache_status",
    "cache_key",
    "cache_lookup_ms",
    "cache_created_at",
    "cache_origin_attempt_stage",
    "cache_origin_latency_ms",
    "cache_match_type",
    "cache_similarity",
    "cache_matched_question",
}


@dataclass(frozen=True)
class AskCacheLookup:
    """一次缓存读取尝试的结构化结果。

    字段说明：
    - enabled：当前进程是否启用了 ASK 缓存。
    - key：本次问题对应的 Redis key；未启用缓存时为空。
    - normalized：命中缓存时返回的标准化 ASK 中间结果；未命中时为空。
    - latency_ms：读取 Redis 的耗时，用于响应 meta 和审计观测。
    - status：读取状态，例如 disabled/miss/hit/error/invalid。
    - error：Redis 异常的类名；缓存失败时只记录，不阻断主链路。
    """

    enabled: bool
    key: str | None
    normalized: dict[str, Any] | None
    latency_ms: int
    status: str
    error: str | None = None
    query_vector: list[float] | None = None


def _bool_env(name: str, default: bool = False) -> bool:
    """把环境变量解析成布尔值。

    输入：
    - name：环境变量名。
    - default：环境变量不存在或为空时使用的默认值。

    输出：
    - True：环境变量值为 1/true/yes/y/on。
    - False：其他值或默认 false。

    这个函数只在模块内部使用，目的是避免多个开关各自重复写解析逻辑。
    """
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def is_enabled() -> bool:
    """判断当前进程是否启用 ASK Redis 缓存。

    读取 `ASK_CACHE_ENABLED`：
    - true/1/yes/on：启用缓存。
    - 空值或其他值：关闭缓存。

    默认关闭是有意设计的：本地测试或没有 Redis 的环境不会因为缓存功能产生
    额外 2 秒级 Redis 连接超时。需要压测或演示缓存时再显式打开。
    """
    load_dotenv()
    return _bool_env("ASK_CACHE_ENABLED", False)


def semantic_cache_enabled() -> bool:
    """判断是否启用 ASK 语义缓存；默认关闭，避免误命中。"""
    load_dotenv()
    return _bool_env("ASK_SEMANTIC_CACHE_ENABLED", False)


def semantic_threshold() -> float:
    """读取语义缓存相似度阈值，默认使用偏保守的 0.97。"""
    raw = str(os.getenv("ASK_SEMANTIC_CACHE_THRESHOLD") or "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = DEFAULT_SEMANTIC_THRESHOLD
    return min(0.9999, max(0.90, value))


def semantic_max_candidates() -> int:
    """读取语义缓存最多比较候选数，避免 Redis 小索引被无限扫描。"""
    raw = str(os.getenv("ASK_SEMANTIC_CACHE_MAX_CANDIDATES") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_SEMANTIC_MAX_CANDIDATES
    return max(1, min(value, 2000))


def ttl_seconds() -> int:
    """读取并规范化缓存 TTL。

    输入来源：
    - `ASK_CACHE_TTL_SECONDS` 环境变量。

    输出：
    - Redis key 的过期秒数。

    取舍：
    - 最小 60 秒，避免误配置成 0 导致缓存刚写入就失效。
    - 最大 7 天，避免知识库或模型更新后旧答案长期存在。
    - 默认 3600 秒，适合 demo 和普通企业制度问答场景。
    """
    raw = str(os.getenv("ASK_CACHE_TTL_SECONDS") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_TTL_SECONDS
    return max(60, min(value, MAX_TTL_SECONDS))


def normalize_question(question: str) -> str:
    """规范化问题文本，提升缓存命中率。

    输入：
    - 用户原始问题。

    处理：
    - 把连续空白折叠成单个空格。
    - 去掉首尾空白。
    - 截断到 MAX_QUESTION_CHARS，避免超长输入制造异常大的 key 原料。

    输出：
    - 用于生成缓存 key 的稳定问题文本。

    注意：
    - 这里不做同义改写，不做标点归一，不做 embedding 相似缓存。
    - 第一版只做精确问题缓存，避免误命中带来错误答案。
    """
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    return text[:MAX_QUESTION_CHARS]


def _env_value(name: str, default: str = "") -> str:
    """读取环境变量并提供字符串默认值。

    输入：
    - name：环境变量名。
    - default：环境变量为空时的默认字符串。

    输出：
    - 去掉首尾空白后的环境变量值，或 default。

    这个小函数用于构造缓存配置指纹，确保空值处理一致。
    """
    value = str(os.getenv(name) or "").strip()
    return value or default


def _context_fingerprint(department: str | None) -> dict[str, str]:
    """构造决定“缓存是否兼容”的上下文指纹。

    输入：
    - department：本次问答所属部门。不同部门可能对应不同制度范围，
      因此必须进入缓存 key。

    输出：
    - 一个稳定的字典，包含 schema、命名空间、部门、模型、collection、
      embedding 模型、检索模式和 rerank 配置等。

    为什么要这样做：
    - 同一个问题在不同知识库 collection 下答案可能不同。
    - 同一个问题在不同模型下格式、引用、回答风格可能不同。
    - 知识库重建或评测切换后，可以通过 `ASK_CACHE_NAMESPACE_VERSION`
      手动整体失效旧缓存。
    """
    load_dotenv()
    namespace = _env_value("ASK_CACHE_NAMESPACE_VERSION", "default")
    app_level = _env_value("APP_LEVEL", "l0")
    retrieval_mode = _env_value("RETRIEVAL_MODE", "")
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "department": str(department or "general").strip() or "general",
        "app_level": app_level,
        "openai_model": _env_value("OPENAI_MODEL", "deepseek-chat"),
        "qdrant_collection": _env_value("QDRANT_COLLECTION", "policy_kb_l0"),
        "embed_model": _env_value("EMBED_MODEL", "BAAI/bge-large-zh-v1.5"),
        "retrieval_mode": retrieval_mode,
        "rerank_enabled": _env_value("RERANK_ENABLED", ""),
        "rerank_model": _env_value("RERANK_MODEL", ""),
        "top_k": _env_value("RETRIEVAL_TOP_K", ""),
    }


def _context_digest(department: str | None) -> str:
    raw = repr(_context_fingerprint(department))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_key(question: str, department: str | None) -> str:
    """生成当前问题对应的 Redis 缓存 key。

    输入：
    - question：用户问题。
    - department：用户所在部门或业务域。

    处理：
    - 先规范化 question。
    - 再拼入上下文指纹。
    - 最后对完整 key 原料做 sha256，避免 Redis key 过长或包含敏感原文。

    输出：
    - 形如 `ask:cache:v1:<sha256>` 的 Redis key。
    """
    normalized_question = normalize_question(question)
    context = _context_fingerprint(department)
    raw = repr(
        {
            "question": normalized_question,
            "context": context,
        }
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"ask:cache:{CACHE_SCHEMA_VERSION}:{digest}"


def semantic_index_key(department: str | None) -> str:
    """同一上下文指纹下的语义缓存候选索引 key。"""
    return f"ask:cache:{CACHE_SCHEMA_VERSION}:semantic:{_context_digest(department)}"


def _question_embedding(question: str) -> list[float]:
    """复用检索 embedding 模型生成问题向量；只在语义缓存开启时调用。"""
    from src.kb.retrieve import _get_embedding_model

    load_dotenv()
    model_name = _env_value("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
    model = _get_embedding_model(model_name)
    vector = model.encode([normalize_question(question)], normalize_embeddings=True).tolist()[0]
    return [float(item) for item in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=False):
        dot += float(a) * float(b)
        left_norm += float(a) * float(a)
        right_norm += float(b) * float(b)
    denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
    if denominator <= 0:
        return 0.0
    return dot / denominator


def _strip_cache_meta(output_meta: dict[str, Any] | None) -> dict[str, Any]:
    """从 answer meta 中剥离缓存运行态字段。

    输入：
    - output_meta：一次 ASK 回答的 meta。

    输出：
    - 移除 cache_hit/cache_status/cache_lookup_ms 等字段后的 meta。

    为什么要剥离：
    - 缓存里应该保存“原始回答结果”，而不是某一次请求的缓存命中状态。
    - 否则缓存命中后再次写入缓存，会把旧请求的 cache meta 污染到新请求。
    """
    meta = dict(output_meta or {})
    for key in _CACHE_META_KEYS:
        meta.pop(key, None)
    return meta


def _cacheable_normalized(normalized: dict[str, Any]) -> dict[str, Any]:
    """抽取适合放入缓存的标准化 ASK 数据。

    输入：
    - normalized：`normalize_answer_payload()` 产出的中间结果。

    输出：
    - 只包含 answer、citations、output_meta、trace_hits、原始 retrieve/answer 延迟。

    这里不保存：
    - request_id：属于单次请求。
    - query_id：属于单次数据库记录。
    - actor/user：ASK 答案本身不应绑定到某个请求用户。
    """
    payload = {
        "answer": str(normalized.get("answer") or ""),
        "citations": list(normalized.get("citations") or []),
        "output_meta": _strip_cache_meta(dict(normalized.get("output_meta") or {})),
        "trace_hits": list(normalized.get("trace_hits") or []),
        "retrieve_ms": int(normalized.get("retrieve_ms") or 0),
        "answer_ms": int(normalized.get("answer_ms") or 0),
    }
    return payload


def _cached_lookup_result(
    *,
    key: str | None,
    cached: dict[str, Any],
    payload: dict[str, Any],
    latency_ms: int,
    status: str,
    match_type: str,
    similarity: float | None = None,
) -> AskCacheLookup:
    output_meta = dict(cached.get("output_meta") or {})
    origin_latency = {
        "retrieve": int(cached.get("retrieve_ms") or 0),
        "answer": int(cached.get("answer_ms") or 0),
    }
    output_meta["cache_hit"] = True
    output_meta["cache_status"] = status
    if key:
        output_meta["cache_key"] = key
    output_meta["cache_lookup_ms"] = latency_ms
    output_meta["cache_created_at"] = payload.get("created_at")
    output_meta["cache_origin_attempt_stage"] = str(output_meta.get("attempt_stage") or "unknown")
    output_meta["cache_origin_latency_ms"] = origin_latency
    output_meta["cache_match_type"] = match_type
    if similarity is not None:
        output_meta["cache_similarity"] = round(float(similarity), 6)
    if payload.get("question"):
        output_meta["cache_matched_question"] = str(payload.get("question"))
    output_meta["attempt_stage"] = "cache_hit" if match_type == "exact" else "semantic_cache_hit"

    cached["output_meta"] = output_meta
    cached["retrieve_ms"] = 0
    cached["answer_ms"] = 0
    return AskCacheLookup(True, key, cached, latency_ms, status)


def is_cacheable(normalized: dict[str, Any]) -> bool:
    """判断一次 ASK 结果是否允许写入缓存。

    缓存条件：
    - answer 非空。
    - `failure_reason` 为空。
    - `json_ok` 为 true，说明模型输出通过结构化解析。

    不缓存失败结果的原因：
    - 失败可能是临时模型抖动、解析错误或外部服务异常。
    - 把失败答案缓存起来会放大一次偶发错误，影响后续正常请求。
    """
    answer = str(normalized.get("answer") or "").strip()
    output_meta = dict(normalized.get("output_meta") or {})
    if not answer:
        return False
    if output_meta.get("failure_reason"):
        return False
    return bool(output_meta.get("json_ok", False))


def read_cached_answer(question: str, department: str | None) -> AskCacheLookup:
    """读取当前问题的缓存答案。

    输入：
    - question：用户问题。
    - department：用户部门，用于参与 key 指纹。

    输出：
    - AskCacheLookup，其中 normalized 非空表示命中缓存。

    关键行为：
    - 缓存关闭时直接返回 disabled。
    - Redis miss 时返回 miss，主链路继续检索和生成。
    - Redis 异常时返回 error，但不抛出异常，主链路继续执行。
    - 命中时把 `attempt_stage` 改成 cache_hit，并把本次 retrieve/answer
      延迟置 0，同时保留原始回答的 origin_latency_ms 供观测。
    """
    if not is_enabled():
        return AskCacheLookup(False, None, None, 0, "disabled")

    key = cache_key(question, department)
    started = time.perf_counter()
    try:
        payload = redis_client.get_json(key)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AskCacheLookup(True, key, None, latency_ms, "error", exc.__class__.__name__)

    latency_ms = int((time.perf_counter() - started) * 1000)
    if not isinstance(payload, dict):
        return _read_semantic_cached_answer(question, department, key, started)

    normalized = payload.get("normalized")
    if not isinstance(normalized, dict):
        return AskCacheLookup(True, key, None, latency_ms, "invalid")

    cached = _cacheable_normalized(normalized)
    return _cached_lookup_result(
        key=key,
        cached=cached,
        payload=payload,
        latency_ms=latency_ms,
        status="hit",
        match_type="exact",
    )


def _read_semantic_cached_answer(
    question: str,
    department: str | None,
    exact_key: str,
    started: float,
) -> AskCacheLookup:
    if not semantic_cache_enabled():
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AskCacheLookup(True, exact_key, None, latency_ms, "miss")
    try:
        query_vector = _question_embedding(question)
        index_payload = redis_client.get_json(semantic_index_key(department))
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AskCacheLookup(True, exact_key, None, latency_ms, "semantic_error", exc.__class__.__name__)
    if not isinstance(index_payload, dict):
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AskCacheLookup(True, exact_key, None, latency_ms, "semantic_miss", query_vector=query_vector)

    candidates = index_payload.get("items")
    if not isinstance(candidates, list):
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AskCacheLookup(True, exact_key, None, latency_ms, "semantic_miss", query_vector=query_vector)

    best_item: dict[str, Any] | None = None
    best_score = 0.0
    for item in candidates[: semantic_max_candidates()]:
        if not isinstance(item, dict):
            continue
        vector = item.get("embedding")
        if not isinstance(vector, list):
            continue
        score = _cosine_similarity(query_vector, [float(value) for value in vector])
        if score > best_score:
            best_score = score
            best_item = item

    if best_item is None or best_score < semantic_threshold():
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AskCacheLookup(True, exact_key, None, latency_ms, "semantic_miss", query_vector=query_vector)

    matched_key = str(best_item.get("key") or "")
    if not matched_key:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AskCacheLookup(True, exact_key, None, latency_ms, "semantic_miss", query_vector=query_vector)
    try:
        payload = redis_client.get_json(matched_key)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AskCacheLookup(True, exact_key, None, latency_ms, "semantic_error", exc.__class__.__name__)
    latency_ms = int((time.perf_counter() - started) * 1000)
    if not isinstance(payload, dict) or not isinstance(payload.get("normalized"), dict):
        return AskCacheLookup(True, exact_key, None, latency_ms, "semantic_invalid")

    cached = _cacheable_normalized(payload["normalized"])
    return _cached_lookup_result(
        key=matched_key,
        cached=cached,
        payload=payload,
        latency_ms=latency_ms,
        status="semantic_hit",
        match_type="semantic",
        similarity=best_score,
    )


def write_cached_answer(
    question: str,
    department: str | None,
    normalized: dict[str, Any],
    *,
    key: str | None = None,
) -> str:
    """把一次成功 ASK 结果写入 Redis。

    输入：
    - question：用户问题。
    - department：部门，用于生成 key。
    - normalized：标准化 ASK 中间结果。
    - key：可选。通常来自读取阶段生成的 key，避免重复计算。

    输出：
    - stored：成功写入。
    - disabled：缓存未启用。
    - not_cacheable：结果不满足缓存条件。
    - store_failed：Redis 写入失败。

    设计取舍：
    - 写失败不影响本次回答返回。
    - 缓存只作为性能优化层，不参与业务正确性的强依赖。
    """
    if not is_enabled():
        return "disabled"
    if not is_cacheable(normalized):
        return "not_cacheable"

    resolved_key = key or cache_key(question, department)
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context": _context_fingerprint(department),
        "question": normalize_question(question),
        "normalized": _cacheable_normalized(normalized),
    }
    try:
        redis_client.set_json(resolved_key, payload, ttl_seconds=ttl_seconds())
        if semantic_cache_enabled():
            _upsert_semantic_index_item(
                key=resolved_key,
                question=question,
                department=department,
                created_at=str(payload["created_at"]),
            )
    except Exception:
        return "store_failed"
    return "stored"


def _upsert_semantic_index_item(*, key: str, question: str, department: str | None, created_at: str) -> None:
    index_key = semantic_index_key(department)
    payload = redis_client.get_json(index_key)
    if not isinstance(payload, dict):
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "context": _context_fingerprint(department),
            "items": [],
        }
    items = payload.get("items")
    if not isinstance(items, list):
        items = []

    normalized_question = normalize_question(question)
    embedding = _question_embedding(normalized_question)
    fresh_item = {
        "key": key,
        "question": normalized_question,
        "embedding": embedding,
        "created_at": created_at,
    }
    deduped = [item for item in items if not isinstance(item, dict) or item.get("key") != key]
    payload["items"] = [fresh_item, *deduped][: semantic_max_candidates()]
    redis_client.set_json(index_key, payload, ttl_seconds=ttl_seconds())


def mark_cache_status(
    normalized: dict[str, Any],
    lookup: AskCacheLookup,
    *,
    status: str,
) -> dict[str, Any]:
    """给“新计算出来”的 ASK 结果补充缓存状态。

    输入：
    - normalized：刚刚经过检索/生成得到的结果。
    - lookup：本次缓存读取结果。
    - status：写缓存后的状态，例如 stored/not_cacheable/store_failed。

    输出：
    - 带 cache meta 的 normalized 副本。

    为什么需要这个函数：
    - miss 后返回给用户的答案也应该说明缓存状态，便于压测和调试。
    - 这里返回副本，不直接修改传入对象，降低调用方意外共享状态的风险。
    """
    output_meta = dict(normalized.get("output_meta") or {})
    output_meta["cache_hit"] = False
    output_meta["cache_status"] = status
    if lookup.key:
        output_meta["cache_key"] = lookup.key
    output_meta["cache_lookup_ms"] = int(lookup.latency_ms or 0)
    if lookup.error:
        output_meta["cache_error"] = lookup.error
    if lookup.status:
        output_meta["cache_lookup_status"] = lookup.status
    updated = dict(normalized)
    updated["output_meta"] = output_meta
    return updated


def cache_meta_from_output_meta(output_meta: dict[str, Any] | None) -> dict[str, Any]:
    """从内部 output_meta 中提取可对外展示/审计的缓存摘要。

    输入：
    - output_meta：ASK 回答内部 meta。

    输出：
    - 一个适合放入 API `meta.cache` 和审计 payload 的小字典。

    输出字段：
    - hit：是否命中缓存。
    - status：缓存状态。
    - lookup_ms：缓存读取耗时。
    - created_at/origin_attempt_stage/origin_latency_ms：仅命中时出现。

    注意：
    - 不输出完整 Redis key，避免把内部 key 设计暴露给外部调用方。
    """
    meta = dict(output_meta or {})
    result: dict[str, Any] = {
        "hit": bool(meta.get("cache_hit", False)),
        "status": str(meta.get("cache_status") or "disabled"),
        "lookup_ms": int(meta.get("cache_lookup_ms") or 0),
    }
    if meta.get("cache_created_at"):
        result["created_at"] = meta.get("cache_created_at")
    if meta.get("cache_origin_attempt_stage"):
        result["origin_attempt_stage"] = meta.get("cache_origin_attempt_stage")
    if isinstance(meta.get("cache_origin_latency_ms"), dict):
        result["origin_latency_ms"] = meta.get("cache_origin_latency_ms")
    if meta.get("cache_match_type"):
        result["match_type"] = meta.get("cache_match_type")
    if meta.get("cache_similarity") is not None:
        result["similarity"] = meta.get("cache_similarity")
    if meta.get("cache_matched_question"):
        result["matched_question"] = meta.get("cache_matched_question")
    return result
