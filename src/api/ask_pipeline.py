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

from src.api import crud
from src.kb.answer import answer_with_citations
from src.kb.retrieve import retrieve


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
        },
    }


def public_kb_response(kb_result: dict[str, Any]) -> dict[str, Any]:
    """去掉内部 query_id，输出给外部调用方。"""
    visible = dict(kb_result)
    visible.pop("query_id", None)
    return visible
