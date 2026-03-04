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

import os
from functools import lru_cache
from typing import Any

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


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

    if top_k is None:
        top_k = int(cfg["retrieval"]["top_k"])

    """检索阶段必须和入库阶段使用同一 embedding 模型，否则向量空间不一致，结果会失真。"""
    """这里复用进程内缓存模型，避免回归集逐题重复加载。"""
    model = _get_embedding_model(embed_model)
    qvec = model.encode([query], normalize_embeddings=True).tolist()[0]

    client = QdrantClient(url=qdrant_url)

    """
    这里只在“客户端缺少某个方法”时做接口兼容回退。
    如果是连接失败、服务没启动、请求超时等运行时错误，应直接抛出，
    这样问题会暴露为基础设施故障，而不是被错误地伪装成接口兼容问题。
    """
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )
        hits = getattr(response, "points", response)
    elif hasattr(client, "search"):
        hits = client.search(
            collection_name=collection,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
        )
    else:
        raise RuntimeError("Current qdrant-client exposes neither query_points nor search.")

    return [_format_hit(hit) for hit in hits]


if __name__ == "__main__":
    import json
    import sys

    """允许直接命令行试检索，方便你在开发阶段快速观察命中片段。"""
    question = " ".join(sys.argv[1:]).strip() or "学生的权利与义务主要包括哪些？"
    results = retrieve(question)
    print(json.dumps(results, ensure_ascii=False, indent=2))
