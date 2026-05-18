"""
L0 检索程序：把用户问题转换成向量，并从 Qdrant 召回 Top-K 证据。

一、程序目标
1. 接收自然语言问题。
2. 使用与入库相同的 embedding 模型生成 query 向量。
3. 在 Qdrant 中查找最相近的若干个 chunk。
4. 将 Qdrant 返回结果整理成统一字典格式，供 answer/eval/api 复用。

二、程序入口与运行顺序
1. 命令入口：`python -m src.kb.retrieve "你的问题"`
2. Python 入口：
   - 命令行模式下执行模块底部的 `if __name__ == "__main__":`
   - 然后调用 `retrieve(question)`
3. `warmup_retrieval_stack()` 可在服务启动阶段预热 embedding 模型。
4. `retrieve(question)` 内部顺序如下：
   3.1 `load_dotenv()`：加载 `.env`
   3.2 `load_level_config(level)`：读取 `top_k`
   3.3 `_get_embedding_model(embed_model)`：按模型名复用已加载的 embedding 模型
   3.4 把 `query` 编码成向量 `qvec`
   3.5 创建 `QdrantClient`
   3.6 调用 Qdrant 查询接口
   3.7 对每个命中项调用 `_format_hit(...)`
   3.8 返回标准化证据列表

三、主要函数的输入输出
1. `load_level_config(level: str) -> dict[str, Any]`
   - 输入：level 名称，例如 `"l0"`
   - 输出：配置字典

2. `_format_hit(hit: Any) -> dict[str, Any]`
   - 输入：Qdrant 返回的单条命中对象
   - 输出：项目统一证据格式：
     {
       "score": float,
       "doc_id": str | None,
       "page": int | None,
       "snippet": str,
       "text": str
     }

3. `warmup_retrieval_stack() -> dict[str, Any]`
   - 输入：无
   - 输出：预热结果摘要，例如模型名和一次探测向量维度
   - 作用：将 embedding 冷启动前移到服务启动阶段，减少首问等待

4. `retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]`
   - 输入：
     - `query`: 用户问题字符串
     - `top_k`: 可选，若不传则从配置读取
   - 输出：证据列表，每项都是 `_format_hit(...)` 的结果
   - 无副作用写文件，但会访问 Qdrant

四、核心数据格式
1. query 输入：
   - 纯字符串，例如：`"学生的权利与义务主要包括哪些？"`
2. retrieve 输出：
   [
     {
       "score": 0.63,
       "doc_id": "moe_student_management",
       "page": 3,
       "snippet": "……",
       "text": "……"
     }
   ]

五、程序可以理解成的伪代码
1. 读环境变量和 level 配置
2. 取出 top_k
3. 用 embedding 模型把 query 转成向量
4. 调 Qdrant 做相似度检索
5. 把底层结果转换成统一格式
6. 返回给上层模块
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from functools import lru_cache
from typing import Any

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional runtime dependency guard
    BM25Okapi = None


def load_level_config(level: str) -> dict[str, Any]:
    """读取当前 level 的检索配置，例如 top_k。"""
    with open(f"configs/levels/{level}.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@lru_cache(maxsize=4)
def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """按模型名缓存 embedding 模型，避免同一进程里重复加载。"""
    """
    L1 回归会在一个 Python 进程里连续执行多次检索。
    如果每一道题都重新初始化 SentenceTransformer，CPU 与内存会被重复模型加载拖垮，
    导致 24 题评测耗时被放大很多倍。
    这里使用进程内缓存，保证同一个模型名在单次运行里只加载一次。
    """
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _get_rerank_model(model_name: str) -> CrossEncoder:
    """按模型名缓存 CrossEncoder reranker，避免每次检索重复加载模型。"""
    return CrossEncoder(model_name)


def _format_hit(hit: Any) -> dict[str, Any]:
    """把 Qdrant 返回对象整理成项目统一的证据字典格式。"""
    payload = getattr(hit, "payload", {}) or {}
    text = payload.get("text", "") or ""
    snippet = payload.get("snippet") or text[:240]
    return {
        "score": float(getattr(hit, "score", 0.0) or 0.0),
        "doc_id": payload.get("doc_id"),
        "page": payload.get("page"),
        "snippet": snippet,
        "text": text,
    }


def _format_payload_hit(payload: dict[str, Any], *, score: float = 0.0) -> dict[str, Any]:
    """把 Qdrant payload 整理成和向量检索一致的命中格式。"""
    text = str(payload.get("text") or "")
    snippet = str(payload.get("snippet") or "") or text[:240]
    return {
        "score": float(score or 0.0),
        "doc_id": payload.get("doc_id"),
        "page": payload.get("page"),
        "snippet": snippet,
        "text": text,
    }


def _hit_key(hit: dict[str, Any]) -> str:
    """
    为 dense/BM25 命中生成融合去重键。
    优先使用 doc_id + page + text 摘要，避免同页多个 chunk 被错误合并。
    """
    doc_id = str(hit.get("doc_id") or "")
    page = str(hit.get("page") or "")
    text = str(hit.get("text") or hit.get("snippet") or "")
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{doc_id}|{page}|{digest}"


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    面向中文制度文档的轻量 BM25 分词。
    - 英文/数字保留完整 token，照顾 VPN、IT-03、TCK 等精确词。
    - 中文连续串切 2/3-gram，避免额外引入 jieba 依赖。
    """
    normalized = str(text or "").lower()
    raw_tokens = re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]+", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[0-9a-z]+", token):
            tokens.append(token)
            continue
        if len(token) <= 3:
            tokens.append(token)
            continue
        for size in (2, 3):
            for index in range(0, len(token) - size + 1):
                tokens.append(token[index : index + size])
    return tokens


def _get_qdrant_client(qdrant_url: str) -> QdrantClient:
    """集中创建 QdrantClient，便于 dense 与 BM25 共用。"""
    return QdrantClient(url=qdrant_url)


def _query_qdrant_dense(
    client: QdrantClient,
    *,
    collection: str,
    query_vector: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """执行向量召回，并返回统一 hit 结构。"""
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        hits = getattr(response, "points", response)
    elif hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
    else:
        raise RuntimeError("Current qdrant-client exposes neither query_points nor search.")
    return [_format_hit(hit) for hit in hits]


def _scroll_collection_payloads(client: QdrantClient, collection: str) -> list[dict[str, Any]]:
    """从 Qdrant 拉取当前 collection 的全部 payload，用于构建轻量 BM25 内存索引。"""
    payloads: list[dict[str, Any]] = []
    offset: Any | None = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            if payload.get("text") or payload.get("snippet"):
                payloads.append(dict(payload))
        if next_offset is None:
            break
        offset = next_offset
    return payloads


@lru_cache(maxsize=8)
def _load_bm25_index(qdrant_url: str, collection: str) -> dict[str, Any]:
    """
    构建并缓存当前 collection 的 BM25 内存索引。
    Qdrant 仍是主存储；BM25 索引可由 payload 重建，适合个人项目和演示环境。
    """
    if BM25Okapi is None:
        raise RuntimeError("rank-bm25 is not installed; please install requirements.txt")

    client = _get_qdrant_client(qdrant_url)
    payloads = _scroll_collection_payloads(client, collection)
    documents = [_format_payload_hit(payload) for payload in payloads]
    tokenized_corpus = [_tokenize_for_bm25(str(doc.get("text") or doc.get("snippet") or "")) for doc in documents]
    non_empty_pairs = [
        (doc, tokens)
        for doc, tokens in zip(documents, tokenized_corpus)
        if tokens
    ]
    if not non_empty_pairs:
        raise RuntimeError(f"no tokenizable payloads found in collection: {collection}")
    filtered_documents = [item[0] for item in non_empty_pairs]
    filtered_tokens = [item[1] for item in non_empty_pairs]
    return {
        "bm25": BM25Okapi(filtered_tokens),
        "documents": filtered_documents,
    }


def _bm25_retrieve(
    query: str,
    *,
    qdrant_url: str,
    collection: str,
    limit: int,
) -> list[dict[str, Any]]:
    """使用 BM25 从 Qdrant payload 内存索引中召回候选。"""
    index = _load_bm25_index(qdrant_url, collection)
    bm25 = index["bm25"]
    documents: list[dict[str, Any]] = index["documents"]
    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
    hits: list[dict[str, Any]] = []
    for idx in ranked_indices[: max(0, limit)]:
        score = float(scores[idx])
        if score <= 0:
            continue
        hit = dict(documents[idx])
        hit["score"] = score
        hit["retrieval_source"] = "bm25"
        hits.append(hit)
    return hits


def _rrf_fuse(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    *,
    top_k: int,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> list[dict[str, Any]]:
    """
    使用 Reciprocal Rank Fusion 融合 dense 与 BM25 排名。
    RRF 不要求两路分数同尺度，比直接加权原始分数更稳。
    """
    fused: dict[str, dict[str, Any]] = {}

    def add_hits(hits: list[dict[str, Any]], *, source: str, weight: float) -> None:
        for rank, hit in enumerate(hits, start=1):
            key = _hit_key(hit)
            item = fused.setdefault(
                key,
                {
                    **hit,
                    "score": 0.0,
                    "retrieval_source": source,
                    "retrieval_scores": {},
                },
            )
            item["score"] = float(item.get("score") or 0.0) + (float(weight) / (float(rrf_k) + rank))
            item.setdefault("retrieval_scores", {})[source] = float(hit.get("score") or 0.0)
            sources = set(str(item.get("retrieval_source") or "").split("+"))
            sources.discard("")
            sources.add(source)
            item["retrieval_source"] = "+".join(sorted(sources))

    add_hits(dense_hits, source="dense", weight=dense_weight)
    add_hits(bm25_hits, source="bm25", weight=bm25_weight)
    return sorted(fused.values(), key=lambda hit: float(hit.get("score") or 0.0), reverse=True)[:top_k]


def _is_enabled(value: Any, default: bool = False) -> bool:
    """解析配置或环境变量里的布尔开关。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "y", "on"}


def _rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    model_name: str,
    top_k: int,
    candidate_k: int,
    batch_size: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """
    使用 CrossEncoder 对候选片段做二阶段重排。
    输入候选通常来自 RRF 融合；输出保持原有 hit 结构，并把最终 score 设为 rerank 分。
    """
    candidates = hits[: max(top_k, candidate_k)]
    if not candidates:
        return []

    pairs: list[tuple[str, str]] = []
    for hit in candidates:
        text = str(hit.get("text") or hit.get("snippet") or "")
        pairs.append((query, text[:max_chars]))

    model = _get_rerank_model(model_name)
    raw_scores = model.predict(pairs, batch_size=max(1, batch_size))
    scored_hits: list[dict[str, Any]] = []
    for hit, score in zip(candidates, raw_scores):
        item = dict(hit)
        original_score = float(item.get("score") or 0.0)
        item.setdefault("retrieval_scores", {})["pre_rerank"] = original_score
        item["rerank_score"] = float(score)
        item["score"] = float(score)
        sources = set(str(item.get("retrieval_source") or "retrieval").split("+"))
        sources.discard("")
        sources.add("rerank")
        item["retrieval_source"] = "+".join(sorted(sources))
        scored_hits.append(item)

    return sorted(scored_hits, key=lambda hit: float(hit.get("rerank_score") or 0.0), reverse=True)[:top_k]


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, parsed)


@lru_cache(maxsize=1)
def warmup_retrieval_stack() -> dict[str, Any]:
    """
    预热检索链路中最容易导致首问变慢的部分。
    当前主要预热的是 embedding 模型加载与第一次 encode。
    """
    load_dotenv()
    embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
    model = _get_embedding_model(embed_model)

    """
    这里做一次极小的探测编码。
    目的不是拿业务结果，而是把模型权重加载、首次算子初始化等成本前移。
    """
    probe_vector = model.encode(["检索预热"], normalize_embeddings=True)
    probe_dim = int(probe_vector.shape[1])

    return {
        "embed_model": embed_model,
        "probe_dim": probe_dim,
    }


def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    从 Qdrant 召回与问题最相关的证据块。
    返回统一的 list[dict]，这样 answer/eval/api 就不需要感知底层向量库对象格式。
    """
    """先加载 .env，确保本地开发、面试机器、未来部署环境都能共用同一套配置入口。"""
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "policy_kb_l0")
    embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
    level = os.getenv("APP_LEVEL", "l0")
    cfg = load_level_config(level)
    retrieval_cfg = dict(cfg.get("retrieval") or {})

    if top_k is None:
        top_k = int(retrieval_cfg["top_k"])

    retrieval_mode = str(os.getenv("RETRIEVAL_MODE") or retrieval_cfg.get("mode") or "dense").strip().lower()
    rerank_cfg = dict(cfg.get("rerank") or {})
    dense_candidate_k = _positive_int(
        os.getenv("DENSE_CANDIDATE_K") or retrieval_cfg.get("dense_candidate_k"),
        max(top_k, 30),
    )
    bm25_candidate_k = _positive_int(
        os.getenv("BM25_CANDIDATE_K") or retrieval_cfg.get("bm25_candidate_k"),
        max(top_k, 30),
    )
    rrf_k = _positive_int(os.getenv("RRF_K") or retrieval_cfg.get("rrf_k"), 60)
    dense_weight = _positive_float(os.getenv("DENSE_WEIGHT") or retrieval_cfg.get("dense_weight"), 1.0)
    bm25_weight = _positive_float(os.getenv("BM25_WEIGHT") or retrieval_cfg.get("bm25_weight"), 0.8)
    rerank_enabled = _is_enabled(os.getenv("RERANK_ENABLED"), _is_enabled(rerank_cfg.get("enabled"), False))
    rerank_model = str(
        os.getenv("RERANK_MODEL") or rerank_cfg.get("model") or "BAAI/bge-reranker-base"
    ).strip()
    rerank_candidate_k = _positive_int(
        os.getenv("RERANK_CANDIDATE_K") or rerank_cfg.get("candidate_k"),
        min(max(top_k, 30), dense_candidate_k + bm25_candidate_k),
    )
    rerank_batch_size = _positive_int(os.getenv("RERANK_BATCH_SIZE") or rerank_cfg.get("batch_size"), 8)
    rerank_max_chars = _positive_int(os.getenv("RERANK_MAX_CHARS") or rerank_cfg.get("max_chars"), 1200)

    """检索阶段必须和入库阶段使用同一 embedding 模型，否则向量空间不一致，结果会失真。"""
    """这里复用进程内缓存模型，避免回归集逐题重复加载。"""
    model = _get_embedding_model(embed_model)
    qvec = model.encode([query], normalize_embeddings=True).tolist()[0]

    client = _get_qdrant_client(qdrant_url)

    dense_limit = dense_candidate_k if retrieval_mode == "hybrid" else top_k
    dense_hits = _query_qdrant_dense(
        client,
        collection=collection,
        query_vector=qvec,
        limit=dense_limit,
    )
    for hit in dense_hits:
        hit["retrieval_source"] = "dense"

    if retrieval_mode != "hybrid":
        if rerank_enabled:
            try:
                return _rerank_hits(
                    query,
                    dense_hits,
                    model_name=rerank_model,
                    top_k=top_k,
                    candidate_k=rerank_candidate_k,
                    batch_size=rerank_batch_size,
                    max_chars=rerank_max_chars,
                )
            except Exception:
                return dense_hits[:top_k]
        return dense_hits[:top_k]

    try:
        bm25_hits = _bm25_retrieve(
            query,
            qdrant_url=qdrant_url,
            collection=collection,
            limit=bm25_candidate_k,
        )
    except Exception:
        return dense_hits[:top_k]

    fused_hits = _rrf_fuse(
        dense_hits,
        bm25_hits,
        top_k=max(top_k, rerank_candidate_k if rerank_enabled else top_k),
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    if not rerank_enabled:
        return fused_hits[:top_k]

    try:
        return _rerank_hits(
            query,
            fused_hits,
            model_name=rerank_model,
            top_k=top_k,
            candidate_k=rerank_candidate_k,
            batch_size=rerank_batch_size,
            max_chars=rerank_max_chars,
        )
    except Exception:
        return fused_hits[:top_k]


async def retrieve_async(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """异步检索入口：在线程池中复用现有同步检索逻辑。"""
    return await asyncio.to_thread(retrieve, query, top_k)


if __name__ == "__main__":
    import json
    import sys

    """允许直接命令行试检索，方便你在开发阶段快速观察命中片段。"""
    question = " ".join(sys.argv[1:]).strip() or "学生的权利与义务主要包括哪些？"
    results = retrieve(question)
    print(json.dumps(results, ensure_ascii=False, indent=2))
