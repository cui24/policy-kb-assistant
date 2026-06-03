"""
ASK 工作流可复用步骤。

目标：
1. 把 ASK 拆成可复用的小步骤（检索/回答/落库/审计/响应组装）。
2. 供 LangGraph 节点与传统 `run_ask_workflow` 共同复用。
3. 避免在节点内直接塞入大工作流函数。
"""

from __future__ import annotations

import os
import time
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from src.api import ask_cache, crud
from src.kb.answer import answer_with_citations, answer_with_citations_async
from src.kb.retrieve import retrieve, retrieve_async


def new_request_id() -> str:
    """生成 ASK 请求标识。"""
    return f"req_{uuid4().hex[:12]}"


def model_name() -> str:
    """读取当前回答模型名，便于落库。"""
    return os.getenv("OPENAI_MODEL", "deepseek-chat")


def trim_hits_for_trace(hits: list[dict], limit: int = 6) -> list[dict]:
    """压缩检索证据，只保留 API trace 必需字段。"""
    trimmed: list[dict] = []
    for hit in hits[:limit]:
        trimmed.append(
            {
                "doc_id": hit.get("doc_id"),
                "page": hit.get("page"),
                "score": float(hit.get("score", 0.0) or 0.0),
                "snippet": str(hit.get("snippet") or "")[:220],
            }
        )
    return trimmed


def run_retrieve_step(question: str) -> tuple[list[dict], int]:
    """执行检索步骤，返回命中和耗时（ms）。"""
    started = time.perf_counter()
    hits = retrieve(question)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return hits, latency_ms


def run_answer_step(question: str, hits: list[dict]) -> tuple[dict[str, Any], int]:
    """执行回答步骤，返回模型输出和耗时（ms）。"""
    started = time.perf_counter()
    output = answer_with_citations(question, hits)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return output, latency_ms


async def run_retrieve_step_async(question: str) -> tuple[list[dict], int]:
    """执行异步检索步骤，返回命中和耗时（ms）。"""
    started = time.perf_counter()
    hits = await retrieve_async(question)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return hits, latency_ms


async def run_answer_step_async(question: str, hits: list[dict]) -> tuple[dict[str, Any], int]:
    """执行异步回答步骤，返回模型输出和耗时（ms）。"""
    started = time.perf_counter()
    output = await answer_with_citations_async(question, hits)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return output, latency_ms


def run_cached_ask_steps(question: str, department: str | None = None) -> dict[str, Any]:
    """执行可缓存的同步 ASK 中间步骤，供传统服务层和 LangGraph 共用。"""
    cache_lookup = ask_cache.read_cached_answer(question, department)
    if cache_lookup.normalized is not None:
        return cache_lookup.normalized

    hits, retrieve_ms = run_retrieve_step(question)
    output, answer_ms = run_answer_step(question, hits)
    normalized = normalize_answer_payload(
        output=output,
        hits=hits,
        retrieve_ms=retrieve_ms,
        answer_ms=answer_ms,
    )
    cache_status = ask_cache.write_cached_answer(
        question,
        department,
        normalized,
        key=cache_lookup.key,
    )
    return ask_cache.mark_cache_status(normalized, cache_lookup, status=cache_status)


async def run_cached_ask_steps_async(question: str, department: str | None = None) -> dict[str, Any]:
    """执行可缓存的异步 ASK 中间步骤，供 `/ask` 异步接口共用。"""
    cache_lookup = ask_cache.read_cached_answer(question, department)
    if cache_lookup.normalized is not None:
        return cache_lookup.normalized

    hits, retrieve_ms = await run_retrieve_step_async(question)
    output, answer_ms = await run_answer_step_async(question, hits)
    normalized = normalize_answer_payload(
        output=output,
        hits=hits,
        retrieve_ms=retrieve_ms,
        answer_ms=answer_ms,
    )
    cache_status = ask_cache.write_cached_answer(
        question,
        department,
        normalized,
        key=cache_lookup.key,
    )
    return ask_cache.mark_cache_status(normalized, cache_lookup, status=cache_status)


def normalize_answer_payload(
    output: dict[str, Any],
    hits: list[dict],
    retrieve_ms: int,
    answer_ms: int,
) -> dict[str, Any]:
    """归一化 ASK 中间结果，供落库与响应复用。"""
    citations = output.get("citations", []) or []
    output_meta = output.get("meta", {}) or {}
    trace_hits = trim_hits_for_trace(hits)
    answer = str(output.get("answer") or "")
    return {
        "answer": answer,
        "citations": citations,
        "output_meta": output_meta,
        "trace_hits": trace_hits,
        "retrieve_ms": int(retrieve_ms),
        "answer_ms": int(answer_ms),
    }


def usage_metrics_from_meta(output_meta: dict[str, Any] | None) -> dict[str, Any]:
    """从回答 meta 中提取可落库的 token 与成本指标。"""
    meta = dict(output_meta or {})
    usage = meta.get("usage")
    usage_payload = dict(usage) if isinstance(usage, dict) else {}

    def int_metric(key: str) -> int:
        try:
            return max(0, int(usage_payload.get(key, 0) or 0))
        except Exception:
            return 0

    def float_metric(key: str) -> float:
        try:
            return max(0.0, float(meta.get(key, 0.0) or 0.0))
        except Exception:
            return 0.0

    return {
        "repair_used": bool(meta.get("repair_used", False)),
        "prompt_tokens": int_metric("prompt_tokens"),
        "completion_tokens": int_metric("completion_tokens"),
        "total_tokens": int_metric("total_tokens"),
        "token_usage_estimated": bool(meta.get("token_usage_estimated", False)),
        "estimated_cost_usd": float_metric("estimated_cost_usd"),
    }


def persist_kb_query(
    db: Session,
    *,
    request_id: str,
    actor: str,
    actor_user_id: str,
    department: str,
    question: str,
    normalized: dict[str, Any],
):
    """把 ASK 结果落库到 kb_queries。"""
    output_meta = dict(normalized.get("output_meta") or {})
    usage_metrics = usage_metrics_from_meta(output_meta)
    return crud.create_kb_query(
        db,
        {
            "request_id": request_id,
            "user_name": actor,
            "actor_user_id": actor_user_id,
            "department": department,
            "question": question,
            "answer": str(normalized.get("answer") or ""),
            "citations_json": list(normalized.get("citations") or []),
            "retrieve_topk_json": list(normalized.get("trace_hits") or []),
            "attempt_stage": str(output_meta.get("attempt_stage") or "unknown"),
            "latency_retrieve_ms": int(normalized.get("retrieve_ms") or 0),
            "latency_answer_ms": int(normalized.get("answer_ms") or 0),
            "model": model_name(),
            "valid_json": bool(output_meta.get("json_ok", False)),
            "failure_reason": output_meta.get("failure_reason"),
            **usage_metrics,
        },
    )


def write_ask_audit(
    db: Session,
    *,
    actor: str,
    actor_user_id: str,
    request_id: str,
    target_query_id: str,
    question: str,
    department: str,
    normalized: dict[str, Any],
    streamed: bool = False,
) -> None:
    """写 ASK 审计日志。"""
    output_meta = dict(normalized.get("output_meta") or {})
    trace_hits = list(normalized.get("trace_hits") or [])
    payload = {
        "question": question,
        "department": department,
        "attempt_stage": output_meta.get("attempt_stage"),
        "top_hit": trace_hits[0] if trace_hits else None,
        "latency_ms": {
            "retrieve": int(normalized.get("retrieve_ms") or 0),
            "answer": int(normalized.get("answer_ms") or 0),
        },
        "failure_reason": output_meta.get("failure_reason"),
        "usage": usage_metrics_from_meta(output_meta),
        "cache": ask_cache.cache_meta_from_output_meta(output_meta),
    }
    if streamed:
        payload["streamed"] = True

    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "actor_user_id": actor_user_id,
            "action_type": "ASK",
            "target_type": "KB_QUERY",
            "target_id": target_query_id,
            "request_id": request_id,
            "payload_json": payload,
        },
    )


def write_agent_route_audit(
    db: Session,
    *,
    actor: str,
    actor_user_id: str,
    request_id: str,
    target_query_id: str,
    text: str,
    engine: str,
) -> None:
    """写 Agent 路由审计日志。"""
    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "actor_user_id": actor_user_id,
            "action_type": "AGENT_ROUTE",
            "target_type": "KB_QUERY",
            "target_id": target_query_id,
            "request_id": request_id,
            "payload_json": {
                "route": "ASK",
                "text": text,
                "engine": engine,
            },
        },
    )


def build_kb_result(*, request_id: str, query_id: str, normalized: dict[str, Any]) -> dict[str, Any]:
    """组装 ASK 统一结果结构。"""
    output_meta = dict(normalized.get("output_meta") or {})
    return {
        "request_id": request_id,
        "query_id": query_id,
        "answer": str(normalized.get("answer") or ""),
        "citations": list(normalized.get("citations") or []),
        "meta": {
            "attempt_stage": str(output_meta.get("attempt_stage") or "unknown"),
            "valid_json": bool(output_meta.get("json_ok", False)),
            "repair_used": bool(output_meta.get("repair_used", False)),
            "failure_reason": output_meta.get("failure_reason"),
            "retrieve_topk": list(normalized.get("trace_hits") or []),
            "latency_ms": {
                "retrieve": int(normalized.get("retrieve_ms") or 0),
                "answer": int(normalized.get("answer_ms") or 0),
            },
            "usage": usage_metrics_from_meta(output_meta),
            "cache": ask_cache.cache_meta_from_output_meta(output_meta),
        },
    }


def public_kb_response(kb_result: dict[str, Any]) -> dict[str, Any]:
    """去掉内部 query_id，输出给外部调用方。"""
    visible = dict(kb_result)
    visible.pop("query_id", None)
    return visible
