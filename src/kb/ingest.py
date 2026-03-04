"""
L0 入库程序：把本地 PDF 文档转成可检索的向量索引。

一、程序目标
1. 读取 `data/demo/*.pdf` 与 `data/raw/*.pdf` 中的政策 PDF。
2. 按页抽取文本，并按配置切分成 chunk。
3. 用 embedding 模型把每个 chunk 编码成向量。
4. 把向量和 payload 一起写入 Qdrant。
5. 产出 `outputs/ingest_report.json`，记录入库统计。

二、程序入口与运行顺序
1. 命令入口：`python -m src.kb.ingest`
2. Python 入口：执行 `main()`
3. `main()` 内部顺序如下：
   3.1 `load_dotenv()`：加载 `.env`
   3.2 `load_level_config(level)`：读取 `configs/levels/<level>.yaml`
   3.3 扫描 `data/demo/*.pdf` 与 `data/raw/*.pdf`
   3.4 初始化 `RecursiveCharacterTextSplitter`
   3.5 初始化 `SentenceTransformer`
   3.6 创建 `QdrantClient`
   3.7 `ensure_collection(...)`：重建 collection
   3.8 遍历每个 PDF、每一页、每一个 chunk
   3.9 对达到批大小的 chunk 调用 `_flush_batch(...)`
   3.10 全部处理完后再次 `_flush_batch(...)` 清空尾批
   3.11 统计真实在库点数，写入 `outputs/ingest_report.json`
   3.12 打印入库完成信息

三、主要函数的输入输出
1. `clean_text(text: str) -> str`
   - 输入：原始文本字符串
   - 输出：压缩空白后的文本字符串
   - 用途：降低 PDF 抽取噪声，提升分块和检索稳定性

2. `sha1_id(*parts: str) -> str`
   - 输入：多个字符串片段（如 doc_id、页码、chunk 内容）
   - 输出：稳定哈希字符串
   - 用途：先生成稳定内容摘要，供 UUID 再加工

3. `stable_uuid(*parts: str) -> str`
   - 输入：多个字符串片段
   - 输出：确定性 UUID 字符串
   - 用途：生成 Qdrant 可接受的 point id

4. `load_level_config(level: str) -> dict[str, Any]`
   - 输入：level 名称，例如 `"l0"`
   - 输出：配置字典，例如：
     `{"chunk": {"size": 800, "overlap": 120}, "retrieval": {...}}`

5. `ensure_collection(client, collection, dim) -> None`
   - 输入：Qdrant 客户端、collection 名、向量维度
   - 输出：无返回值
   - 副作用：删除旧 collection 并重建新 collection

6. `_flush_batch(client, collection, model, pending) -> None`
   - 输入：
     - `client`: `QdrantClient`
     - `collection`: `str`
     - `model`: `SentenceTransformer`
     - `pending`: `list[dict]`
   - `pending` 中每项格式：
     `{"id": str, "text": str, "payload": dict[str, Any]}`
   - 输出：无返回值
   - 副作用：批量生成向量并写入 Qdrant

7. `main() -> None`
   - 输入：无显式参数，依赖 `.env`、配置文件、`data/demo/*.pdf` 与 `data/raw/*.pdf`
   - 输出：无返回值
   - 副作用：
     - 写入 Qdrant collection
     - 写入 `outputs/ingest_report.json`
     - 向终端打印日志

四、核心数据格式
1. 单个 chunk 的 payload 格式：
   {
     "doc_id": str,
     "page": int,
     "chunk_index": int,
     "text": str,
     "snippet": str,
     "source_file": str
   }
2. 入库报告 `outputs/ingest_report.json` 格式：
   {
     "generated_at": str,
     "collection": str,
     "documents": list[{
       "doc_id": str,
       "source_file": str,
       "pages": int,
       "chunk_count": int,
       "avg_chunk_chars": float,
       "failed_pages": int,
       "failed_page_numbers": list[int]
     }],
     "summary": {
       "documents": int,
       "total_chunks": int,
       "total_points_in_qdrant": int,
       "total_failed_pages": int
     }
   }

五、程序可以理解成的伪代码
1. 读配置
2. 找到 `data/demo/` 与 `data/raw/` 中的所有 PDF
3. 初始化切分器、embedding 模型、Qdrant 客户端
4. 重建 collection
5. 对每个 PDF：
   5.1 逐页抽取文本
   5.2 清洗文本
   5.3 跳过无效页
   5.4 分块
   5.5 给每个 chunk 生成 UUID 和 payload
   5.6 加入待写入批次
   5.7 到达批大小就写库
6. 处理尾批
7. 统计真实在库点数
8. 生成并保存入库报告
9. 打印结束日志
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer


def clean_text(text: str) -> str:
    """压缩连续空白字符，减少 PDF 抽取带来的检索噪声。"""
    return " ".join((text or "").split())


def sha1_id(*parts: str) -> str:
    """根据文档元信息和 chunk 内容生成稳定哈希。"""
    hasher = hashlib.sha1()
    for part in parts:
        hasher.update(part.encode("utf-8", errors="ignore"))
        hasher.update(b"|")
    return hasher.hexdigest()


def stable_uuid(*parts: str) -> str:
    """
    生成确定性的 UUID 作为 Qdrant 点 ID。
    这里不能直接用普通字符串，因为你当前这版 Qdrant 只接受 uint 或 UUID。
    """
    digest = sha1_id(*parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def load_level_config(level: str) -> dict[str, Any]:
    """读取当前 level 的配置文件，例如 chunk 大小、重叠长度。"""
    config_path = Path("configs/levels") / f"{level}.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    """
    为了保证 demo 可复现，每次入库都重建目标 collection。
    这样做的代价是全量覆盖旧数据，但好处是回归和演示结果更稳定。
    """
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


def _flush_batch(
    client: QdrantClient,
    collection: str,
    model: SentenceTransformer,
    pending: list[dict[str, Any]],
) -> None:
    """把积攒到一批的 chunk 一次性向量化并写入 Qdrant。"""
    if not pending:
        return

    """批量 encode 可以明显减少模型调用次数，比逐条向量化更适合入库阶段。"""
    texts = [item["text"] for item in pending]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    points = [
        qm.PointStruct(id=item["id"], vector=vectors[idx], payload=item["payload"])
        for idx, item in enumerate(pending)
    ]
    """一次 upsert 一整批点，既减少请求数，也方便后续统计入库总量。"""
    client.upsert(collection_name=collection, points=points)


def discover_pdf_paths() -> list[Path]:
    """收集公开 demo 文档与本地原始文档，去重后按目录顺序返回。"""
    discovered: list[Path] = []
    seen: set[Path] = set()
    for pdf_dir in (Path("data/demo"), Path("data/raw")):
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            resolved_path = pdf_path.resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            discovered.append(pdf_path)
    return discovered


def main() -> None:
    """先加载 .env，保证 Qdrant 地址、collection 名、embedding 模型等参数可配置。"""
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "policy_kb_l0")
    embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
    level = os.getenv("APP_LEVEL", "l0")

    cfg = load_level_config(level)
    chunk_size = int(cfg["chunk"]["size"])
    chunk_overlap = int(cfg["chunk"]["overlap"])

    pdf_paths = discover_pdf_paths()
    if not pdf_paths:
        raise SystemExit(
            "No PDFs found in data/demo or data/raw. "
            "Use the bundled demo PDF, add your own files to data/raw, "
            "or run scripts/download_pdfs.sh first."
        )

    print(f"[INGEST] QDRANT_URL={qdrant_url}")
    print(f"[INGEST] COLLECTION={collection}")
    print(f"[INGEST] EMBED_MODEL={embed_model}")
    print(f"[INGEST] APP_LEVEL={level}")
    print(f"[INGEST] chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"[INGEST] found_pdfs={len(pdf_paths)}")

    """
    这里显式加入中文分隔符，是因为当前语料是中文政策 PDF。
    如果不加，默认按英文/空白切分，中文句子边界会更差，影响引用可读性。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    model = SentenceTransformer(embed_model)
    dim = model.get_sentence_embedding_dimension()
    if dim is None or dim <= 0:
        """某些模型不会直接暴露向量维度，这里用一次最小 encode 做兜底探测。"""
        dim = int(model.encode(["维度探测"], normalize_embeddings=True).shape[1])

    client = QdrantClient(url=qdrant_url)
    ensure_collection(client, collection, dim)

    batch_size = 64
    pending_points: list[dict[str, Any]] = []
    total_chunks = 0
    per_doc_stats: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        doc_chunks = 0
        doc_chunk_chars = 0
        failed_pages: list[int] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                """
                企业里常会要求“失败页统计”。
                这里把抽取异常单独记下来，方便后续定位是 PDF 本身问题还是解析器问题。
                """
                failed_pages.append(page_idx)
                continue

            page_text = clean_text(raw_text)
            if len(page_text) < 50:
                """
                过短页面通常是封面、分隔页、扫描空页或无正文价值内容。
                这里跳过是为了减少无效向量，提升检索命中率。
                """
                continue

            chunks = splitter.split_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                chunk = clean_text(chunk)
                if len(chunk) < 30:
                    """过短 chunk 往往缺乏语义完整性，保留会增加噪声，直接丢弃。"""
                    continue

                point_id = stable_uuid(doc_id, str(page_idx), str(chunk_idx), chunk)
                """
                payload 是后续“可解释检索”和“引用展示”的关键。
                retrieve/answer/eval 都依赖这里的 doc_id/page/snippet 元信息。
                """
                payload = {
                    "doc_id": doc_id,
                    "page": page_idx,
                    "chunk_index": chunk_idx,
                    "text": chunk,
                    "snippet": chunk[:240],
                    "source_file": pdf_path.name,
                }

                pending_points.append({"id": point_id, "text": chunk, "payload": payload})
                total_chunks += 1
                doc_chunks += 1
                doc_chunk_chars += len(chunk)

                if len(pending_points) >= batch_size:
                    """达到批大小就立刻刷入，避免待处理列表无限增长占内存。"""
                    _flush_batch(client, collection, model, pending_points)
                    pending_points.clear()

        avg_chunk_chars = (doc_chunk_chars / doc_chunks) if doc_chunks else 0.0
        """
        每个文档单独统计，是为了形成企业常见的“入库小报告”。
        这类统计能帮助你解释：哪些文档切得更碎、哪些文档可能抽取质量更差。
        """
        per_doc_stats.append(
            {
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "pages": page_count,
                "chunk_count": doc_chunks,
                "avg_chunk_chars": round(avg_chunk_chars, 2),
                "failed_pages": len(failed_pages),
                "failed_page_numbers": failed_pages,
            }
        )
        print(
            "[INGEST] "
            f"{doc_id}: pages={page_count}, chunks={doc_chunks}, "
            f"avg_chunk_chars={avg_chunk_chars:.2f}, failed_pages={len(failed_pages)}"
        )

    _flush_batch(client, collection, model, pending_points)

    """
    再向 Qdrant 查一次 count，而不是直接假设 total_chunks == 写入成功数。
    这样统计的是“真实在库点数”，能避免因为写入失败造成的假乐观。
    """
    points_in_qdrant = int(client.count(collection_name=collection, exact=True).count)
    total_failed_pages = sum(item["failed_pages"] for item in per_doc_stats)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "documents": per_doc_stats,
        "summary": {
            "documents": len(per_doc_stats),
            "total_chunks": total_chunks,
            "total_points_in_qdrant": points_in_qdrant,
            "total_failed_pages": total_failed_pages,
        },
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "ingest_report.json"
    """把统计单独落盘，便于 README 引用、面试展示和后续回归比较。"""
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INGEST] DONE total_chunks={total_chunks}")
    print(f"[INGEST] total_points_in_qdrant={points_in_qdrant}")
    print(f"[INGEST] total_failed_pages={total_failed_pages}")
    print(f"[INGEST] wrote report: {report_path}")
    print("[INGEST] Next: python -m src.kb.retrieve '<question>'")


if __name__ == "__main__":
    main()
