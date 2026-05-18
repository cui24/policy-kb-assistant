"""
L0 入库程序：把本地 PDF 文档转成可检索的向量索引。

一、程序目标
1. 默认读取企业政策目录：`data/raw/人事|行政|财务|IT` 下的 PDF。
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
   3.3 扫描企业政策目录（可用 `KB_POLICY_DIRS` 覆盖）
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
   - 输入：无显式参数，依赖 `.env`、配置文件、企业政策目录 PDF
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
2. 找到企业政策目录中的所有 PDF
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
import re
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


DEFAULT_POLICY_DIRS = [
    Path("data/raw/人事"),
    Path("data/raw/行政"),
    Path("data/raw/财务"),
    Path("data/raw/IT"),
]

CHUNK_STRATEGIES = {"fixed", "overlap", "structured_hybrid"}

# 结构化切分时的默认参数（可通过环境变量覆盖）
DEFAULT_FIXED_CHUNK_SIZE = 800
DEFAULT_FIXED_OVERLAP = 0
DEFAULT_OVERLAP_CHUNK_SIZE = 800
DEFAULT_OVERLAP_OVERLAP = 150
DEFAULT_STRUCTURED_CHUNK_SIZE = 800
DEFAULT_STRUCTURED_SECONDARY_OVERLAP = 100
DEFAULT_STRUCTURED_MERGE_SHORT = 260
DEFAULT_STRUCTURED_MIN_CHUNK = 160
DEFAULT_STRUCTURED_SOFT_MAX_CHARS = 520
DEFAULT_STRUCTURED_SHORT_AUGMENT_UNDER = 180
DEFAULT_STRUCTURED_ALLOW_CROSS_ARTICLE_MERGE = 1
DEFAULT_WINDOW_PAGES = 1
DEFAULT_WINDOW_OVERLAP_PAGES = 0
DEFAULT_WINDOW_MAX_CHARS = 30000


_MAJOR_HEADING_RE = re.compile(r"^\s*第[一二三四五六七八九十百千万0-9]+[章节条款]\s*")
_SUB_HEADING_RE = re.compile(r"^\s*[（(][一二三四五六七八九十百千万0-9]+[）)]\s*")
_DECIMAL_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+){1,3}\s+")
# 常见中文制度文档编号样式补充：
# - 一、二、三、
# - 1、2、3、
# - 1. 2. 3.（避免误伤 39.3 这类小数编号，故限制点号后不是数字）
_CN_ENUM_HEADING_RE = re.compile(r"^\s*[一二三四五六七八九十百千万]+、\s*")
_ARABIC_ENUM_DUNHAO_HEADING_RE = re.compile(r"^\s*\d+、\s*")
_ARABIC_ENUM_DOT_HEADING_RE = re.compile(r"^\s*\d+[\.．]\s*(?!\d)")
_ARTICLE_HEADING_RE = re.compile(r"^\s*第([一二三四五六七八九十百千万零〇0-9]+)条")
_CHAPTER_HEADING_RE = re.compile(r"^\s*第([一二三四五六七八九十百千万零〇0-9]+)章")
_HEADING_PREFIX_TRIM_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百千万零〇0-9]+条|第[一二三四五六七八九十百千万零〇0-9]+章|"
    r"\d+(?:\.\d+){0,3}|[一二三四五六七八九十百千万]+、|\d+、|[（(]\d+[）)])\s*"
)
_CJK_NUM_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


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


def _read_int_env(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def _read_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _resolve_chunk_strategy() -> str:
    raw = str(os.getenv("KB_CHUNK_STRATEGY") or "fixed").strip().lower()
    if raw not in CHUNK_STRATEGIES:
        return "fixed"
    return raw


def _normalize_page_text(raw_text: str) -> str:
    """
    保留换行层级的轻清洗：
    - 行内压缩空白（便于模型编码）
    - 保留行边界（便于结构化切分识别标题）
    """
    lines = [clean_text(line) for line in str(raw_text or "").splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty).strip()


def _detect_heading_level(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    if _MAJOR_HEADING_RE.match(text):
        return "major"
    if _SUB_HEADING_RE.match(text):
        return "sub"
    if _DECIMAL_HEADING_RE.match(text):
        return "sub"
    if _CN_ENUM_HEADING_RE.match(text):
        return "sub"
    if _ARABIC_ENUM_DUNHAO_HEADING_RE.match(text):
        return "sub"
    if _ARABIC_ENUM_DOT_HEADING_RE.match(text):
        return "sub"
    return ""


def _parse_cn_number(text: str) -> int | None:
    s = str(text or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if all(ch in _CJK_NUM_MAP for ch in s):
        if len(s) == 1:
            return _CJK_NUM_MAP[s]
    if "十" in s:
        left, _, right = s.partition("十")
        left_num = _CJK_NUM_MAP.get(left, 1) if left else 1
        if right:
            if not all(ch in _CJK_NUM_MAP for ch in right):
                return None
            right_num = int("".join(str(_CJK_NUM_MAP[ch]) for ch in right))
        else:
            right_num = 0
        return left_num * 10 + right_num
    if all(ch in _CJK_NUM_MAP for ch in s):
        return int("".join(str(_CJK_NUM_MAP[ch]) for ch in s))
    return None


def _extract_chapter_and_article(section_path: str, text: str) -> tuple[str, str, int | None]:
    chapter = ""
    article = ""
    article_num: int | None = None
    for part in str(section_path or "").split(" > "):
        token = part.strip()
        if not token:
            continue
        if not chapter and _CHAPTER_HEADING_RE.match(token):
            chapter = token
        if not article:
            m = _ARTICLE_HEADING_RE.match(token)
            if m:
                article = token
                article_num = _parse_cn_number(m.group(1))
    if not article:
        m = _ARTICLE_HEADING_RE.match(str(text or ""))
        if m:
            article = m.group(0)
            article_num = _parse_cn_number(m.group(1))
    return chapter, article, article_num


def _topic_key(section_path: str, text: str) -> str:
    parts = [p.strip() for p in str(section_path or "").split(" > ") if p.strip()]
    source = parts[-1] if parts else str(text or "").strip()
    if " > " in source:
        source = source.split(" > ")[-1]
    source = _HEADING_PREFIX_TRIM_RE.sub("", source)
    source = re.sub(r"[：:；;，,.。、“”\"'（）()【】\[\]《》\s]", "", source)
    return source[:24]


def _same_theme(left: str, right: str) -> bool:
    a = str(left or "")
    b = str(right or "")
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 2 and (a in b or b in a):
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    grams_a = {a[i : i + 2] for i in range(len(a) - 1)}
    grams_b = {b[i : i + 2] for i in range(len(b) - 1)}
    return len(grams_a & grams_b) >= 1


def _strip_heading_prefix(title: str) -> str:
    trimmed = _HEADING_PREFIX_TRIM_RE.sub("", str(title or "")).strip()
    trimmed = re.sub(r"^[：:\-—\s]+", "", trimmed)
    return trimmed[:20] if trimmed else str(title or "").strip()


def _augment_short_chunk_text(section_path: str, text: str) -> str:
    raw_text = clean_text(text)
    if raw_text.startswith("【所属章节："):
        return raw_text
    section = str(section_path or "").strip()
    if not section:
        return raw_text
    return clean_text(f"【所属章节：{section}】 {raw_text}")


def _split_fixed_style(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    section_path: str = "",
) -> list[dict[str, str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(50, int(chunk_size)),
        chunk_overlap=max(0, int(chunk_overlap)),
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = []
    for part in splitter.split_text(str(text or "")):
        normalized = clean_text(part)
        if len(normalized) < 30:
            continue
        chunks.append(
            {
                "text": normalized,
                "section_path": str(section_path or ""),
            }
        )
    return chunks


def _split_structured_hybrid(
    text: str,
    *,
    initial_context: tuple[str, str, str] | None = None,
    return_context: bool = False,
) -> list[dict[str, str]] | tuple[list[dict[str, str]], tuple[str, str, str]]:
    """
    结构化混合切分（structured_hybrid）：
    1) 标题/条款优先切段；
    2) 短片段同父级合并；
    3) 超长片段按句子二次切分（overlap=100）；
    4) 保留 section_path。
    """
    chunk_size = _read_int_env("KB_STRUCTURED_CHUNK_SIZE", DEFAULT_STRUCTURED_CHUNK_SIZE)
    secondary_overlap = _read_int_env("KB_STRUCTURED_SECONDARY_OVERLAP", DEFAULT_STRUCTURED_SECONDARY_OVERLAP)
    merge_short = _read_int_env("KB_STRUCTURED_MERGE_SHORT", DEFAULT_STRUCTURED_MERGE_SHORT)
    min_chunk = _read_int_env("KB_STRUCTURED_MIN_CHUNK", DEFAULT_STRUCTURED_MIN_CHUNK)
    soft_max_chars = _read_int_env("KB_STRUCTURED_SOFT_MAX_CHARS", DEFAULT_STRUCTURED_SOFT_MAX_CHARS)
    short_augment_under = _read_int_env(
        "KB_STRUCTURED_SHORT_AUGMENT_UNDER", DEFAULT_STRUCTURED_SHORT_AUGMENT_UNDER
    )
    allow_cross_article_merge = _read_bool_env(
        "KB_STRUCTURED_ALLOW_CROSS_ARTICLE_MERGE", bool(DEFAULT_STRUCTURED_ALLOW_CROSS_ARTICLE_MERGE)
    )

    lines = [clean_text(line) for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    segments: list[dict[str, str]] = []
    current_chapter = str(initial_context[0]) if initial_context else ""
    current_article = str(initial_context[1]) if initial_context else ""
    current_section = str(initial_context[2]) if initial_context else ""
    buffer: list[str] = []
    buffer_section = ""

    def flush_buffer() -> None:
        nonlocal buffer, buffer_section
        if not buffer:
            return
        seg_text = clean_text(" ".join(buffer))
        if seg_text:
            segments.append({"section_path": str(buffer_section or current_section or ""), "text": seg_text})
        buffer = []
        buffer_section = ""

    for line in lines:
        level = _detect_heading_level(line)
        if level:
            flush_buffer()
            if level == "major" and _CHAPTER_HEADING_RE.match(line):
                current_chapter = line
                current_article = ""
                current_section = line
            elif level == "major" and _ARTICLE_HEADING_RE.match(line):
                current_article = line
                if current_chapter:
                    current_section = f"{current_chapter} > {current_article}"
                else:
                    current_section = current_article
            elif level == "major":
                current_section = line
            else:
                if current_article and current_chapter:
                    current_section = f"{current_chapter} > {current_article} > {line}"
                elif current_article:
                    current_section = f"{current_article} > {line}"
                elif current_chapter:
                    current_section = f"{current_chapter} > {line}"
                else:
                    current_section = line
            buffer_section = current_section
            buffer.append(line)
        else:
            if not buffer_section:
                buffer_section = current_section or current_article or current_chapter
            buffer.append(line)
    flush_buffer()

    if not segments:
        return _split_fixed_style(
            clean_text(text),
            chunk_size=chunk_size,
            chunk_overlap=secondary_overlap,
            section_path="",
        )

    # 短片段聚合：
    # 1) 优先同条内（同一 article）合并；
    # 2) 同条仍过短时，允许同章相邻条、同主题合并。
    merged: list[dict[str, str]] = []
    pending: dict[str, str] | None = None

    def _parent_path(section_path: str) -> str:
        path = str(section_path or "")
        parts = [p.strip() for p in path.split(" > ") if p.strip()]
        return parts[0] if parts else ""

    def _merge_path_for_cross_article(chapter: str, left_article: str, right_article: str) -> str:
        left_title = _strip_heading_prefix(left_article)
        right_title = _strip_heading_prefix(right_article)
        if chapter and left_title and right_title:
            return f"{chapter} > {left_title}与{right_title}"
        if chapter:
            return chapter
        return left_article or right_article

    for raw_seg in segments:
        seg = dict(raw_seg)
        seg_text = clean_text(seg.get("text", ""))
        if not seg_text:
            continue
        seg["text"] = seg_text
        if pending is None:
            pending = seg
            continue

        pending_text = clean_text(pending.get("text", ""))
        pending["text"] = pending_text
        p_section = pending.get("section_path", "")
        s_section = seg.get("section_path", "")
        p_chapter, p_article, p_article_num = _extract_chapter_and_article(p_section, pending_text)
        s_chapter, s_article, s_article_num = _extract_chapter_and_article(s_section, seg_text)
        p_topic = _topic_key(p_section, pending_text)
        s_topic = _topic_key(s_section, seg_text)
        same_parent = bool(_parent_path(p_section) and _parent_path(p_section) == _parent_path(s_section))
        same_article = bool(p_article and s_article and p_article == s_article)
        short_enough = (
            len(pending_text) < min_chunk
            or len(pending_text) < merge_short
            or len(seg_text) < merge_short
        )
        adjacent_article = (
            p_article_num is not None and s_article_num is not None and abs(s_article_num - p_article_num) == 1
        )
        can_cross_article = (
            allow_cross_article_merge
            and len(pending_text) < min_chunk
            and len(seg_text) < min_chunk
            and bool(p_chapter and s_chapter and p_chapter == s_chapter)
            and adjacent_article
            and _same_theme(p_topic, s_topic)
        )

        if short_enough and (same_article or (same_parent and len(seg_text) < merge_short) or can_cross_article):
            combined = clean_text(pending_text + " " + seg_text)
            max_target = min(chunk_size, soft_max_chars) if soft_max_chars > 0 else chunk_size
            if len(combined) <= max_target:
                merged_path = (
                    _merge_path_for_cross_article(p_chapter, p_article, s_article) if can_cross_article else (
                        seg.get("section_path", "") or pending.get("section_path", "")
                    )
                )
                pending = {
                    "section_path": merged_path,
                    "text": combined,
                }
                continue

        merged.append(pending)
        pending = seg

    if pending is not None:
        merged.append(pending)

    # 超长片段二次切分（保留 section_path）
    final_chunks: list[dict[str, str]] = []
    for seg in merged:
        seg_text = clean_text(seg.get("text", ""))
        seg_section = str(seg.get("section_path", "") or "")
        if len(seg_text) <= chunk_size:
            if len(seg_text) >= 30:
                final_chunks.append({"section_path": seg_section, "text": seg_text})
            continue
        split_parts = _split_fixed_style(
            seg_text,
            chunk_size=chunk_size,
            chunk_overlap=secondary_overlap,
            section_path=seg_section,
        )
        final_chunks.extend(split_parts)
    # 收尾：对仍然很短的 chunk 再做一次同条/同章相邻条聚合
    coalesced: list[dict[str, str]] = []
    pending_chunk: dict[str, str] | None = None
    for chunk in final_chunks:
        chunk_text = clean_text(chunk.get("text", ""))
        if not chunk_text:
            continue
        cur_chunk = {
            "section_path": str(chunk.get("section_path", "") or ""),
            "text": chunk_text,
        }
        if pending_chunk is None:
            pending_chunk = cur_chunk
            continue
        pending_text = clean_text(pending_chunk.get("text", ""))
        pending_chunk["text"] = pending_text
        p_section = pending_chunk.get("section_path", "")
        c_section = cur_chunk.get("section_path", "")
        p_chapter, p_article, p_article_num = _extract_chapter_and_article(p_section, pending_text)
        c_chapter, c_article, c_article_num = _extract_chapter_and_article(c_section, cur_chunk["text"])
        p_topic = _topic_key(p_section, pending_text)
        c_topic = _topic_key(c_section, cur_chunk["text"])
        same_article = bool(p_article and c_article and p_article == c_article)
        adjacent_article = (
            p_article_num is not None and c_article_num is not None and abs(c_article_num - p_article_num) == 1
        )
        can_cross_article = (
            allow_cross_article_merge
            and len(pending_text) < min_chunk
            and len(cur_chunk["text"]) < min_chunk
            and bool(p_chapter and c_chapter and p_chapter == c_chapter)
            and adjacent_article
            and _same_theme(p_topic, c_topic)
        )
        if same_article or can_cross_article:
            combined = clean_text(pending_text + " " + cur_chunk["text"])
            max_target = min(chunk_size, soft_max_chars) if soft_max_chars > 0 else chunk_size
            if len(combined) <= max_target:
                merged_path = (
                    _merge_path_for_cross_article(p_chapter, p_article, c_article)
                    if can_cross_article
                    else (cur_chunk.get("section_path", "") or pending_chunk.get("section_path", ""))
                )
                pending_chunk = {
                    "section_path": merged_path,
                    "text": combined,
                }
                continue
        coalesced.append(pending_chunk)
        pending_chunk = cur_chunk
    if pending_chunk is not None:
        coalesced.append(pending_chunk)

    # 对保留下来的短 chunk 增加标题增强，避免语义过薄
    enhanced: list[dict[str, str]] = []
    for chunk in coalesced:
        raw_text = clean_text(chunk.get("text", ""))
        if not raw_text:
            continue
        if short_augment_under > 0 and len(raw_text) < short_augment_under:
            raw_text = _augment_short_chunk_text(chunk.get("section_path", ""), raw_text)
        if len(raw_text) >= 30:
            enhanced.append(
                {
                    "section_path": str(chunk.get("section_path", "") or ""),
                    "text": raw_text,
                }
            )
    final_context = (current_chapter, current_article, current_section)
    if return_context:
        return enhanced, final_context
    return enhanced


def _build_page_windows(
    pages: list[tuple[int, str]],
    *,
    window_pages: int,
    overlap_pages: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """
    把“按页文本”组装成滑动窗口。
    - window_pages: 每个窗口最多包含的页数
    - overlap_pages: 相邻窗口页重叠数
    - max_chars: 单窗口字符上限（超限则收缩窗口尾页）
    """
    if not pages:
        return []
    wp = max(1, int(window_pages))
    ov = max(0, int(overlap_pages))
    if ov >= wp:
        ov = wp - 1

    windows: list[dict[str, Any]] = []
    idx = 0
    while idx < len(pages):
        end = min(idx + wp, len(pages))
        frame = list(pages[idx:end])
        if max_chars > 0:
            while len(frame) > 1 and len("\n".join(item[1] for item in frame)) > max_chars:
                frame = frame[:-1]
                end -= 1
        if not frame:
            idx += 1
            continue
        windows.append(
            {
                "page_start": int(frame[0][0]),
                "page_end": int(frame[-1][0]),
                "text": "\n".join(item[1] for item in frame),
            }
        )
        if end >= len(pages):
            break
        next_idx = end - ov
        if next_idx <= idx:
            next_idx = idx + 1
        idx = next_idx
    return windows


def split_text_with_strategy(text: str, strategy: str) -> list[dict[str, str]]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return []
    strategy = str(strategy or "fixed").strip().lower()
    if strategy == "fixed":
        return _split_fixed_style(
            normalized_text,
            chunk_size=_read_int_env("KB_FIXED_CHUNK_SIZE", DEFAULT_FIXED_CHUNK_SIZE),
            chunk_overlap=_read_int_env("KB_FIXED_OVERLAP", DEFAULT_FIXED_OVERLAP),
        )
    if strategy == "overlap":
        return _split_fixed_style(
            normalized_text,
            chunk_size=_read_int_env("KB_OVERLAP_CHUNK_SIZE", DEFAULT_OVERLAP_CHUNK_SIZE),
            chunk_overlap=_read_int_env("KB_OVERLAP_OVERLAP", DEFAULT_OVERLAP_OVERLAP),
        )
    if strategy == "structured_hybrid":
        return _split_structured_hybrid(normalized_text)
    # 理论上不会走到这里，兜底为 fixed
    return _split_fixed_style(
        normalized_text,
        chunk_size=_read_int_env("KB_FIXED_CHUNK_SIZE", DEFAULT_FIXED_CHUNK_SIZE),
        chunk_overlap=_read_int_env("KB_FIXED_OVERLAP", DEFAULT_FIXED_OVERLAP),
    )


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


def _resolve_policy_dirs() -> list[Path]:
    """
    解析政策文档目录。

    规则：
    1. 若设置 `KB_POLICY_DIRS`，按逗号分隔读取。
       例如：`KB_POLICY_DIRS=data/raw/人事,data/raw/行政`
    2. 未设置时，默认使用四个业务目录（人事/行政/财务/IT）。
    """
    raw = str(os.getenv("KB_POLICY_DIRS") or "").strip()
    if not raw:
        return list(DEFAULT_POLICY_DIRS)

    dirs: list[Path] = []
    for part in raw.split(","):
        normalized = part.strip()
        if not normalized:
            continue
        dirs.append(Path(normalized))
    return dirs or list(DEFAULT_POLICY_DIRS)


def discover_pdf_paths() -> list[Path]:
    """收集企业政策目录下的 PDF，去重后按目录顺序返回。"""
    discovered: list[Path] = []
    seen: set[Path] = set()
    for pdf_dir in _resolve_policy_dirs():
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
    chunk_strategy = _resolve_chunk_strategy()
    window_pages = _read_int_env("KB_WINDOW_PAGES", DEFAULT_WINDOW_PAGES)
    window_overlap_pages = _read_int_env("KB_WINDOW_OVERLAP_PAGES", DEFAULT_WINDOW_OVERLAP_PAGES)
    window_max_chars = _read_int_env("KB_WINDOW_MAX_CHARS", DEFAULT_WINDOW_MAX_CHARS)

    cfg = load_level_config(level)
    chunk_size = int(cfg["chunk"]["size"])
    chunk_overlap = int(cfg["chunk"]["overlap"])

    pdf_paths = discover_pdf_paths()
    if not pdf_paths:
        configured_dirs = ", ".join(str(path) for path in _resolve_policy_dirs())
        raise SystemExit(
            "No policy PDFs found. "
            f"Checked dirs: {configured_dirs}. "
            "Please place policy files under these directories or set KB_POLICY_DIRS."
        )

    scanned_dirs = ", ".join(str(path) for path in _resolve_policy_dirs())
    print(f"[INGEST] QDRANT_URL={qdrant_url}")
    print(f"[INGEST] COLLECTION={collection}")
    print(f"[INGEST] EMBED_MODEL={embed_model}")
    print(f"[INGEST] APP_LEVEL={level}")
    print(f"[INGEST] CHUNK_STRATEGY={chunk_strategy}")
    print(
        f"[INGEST] WINDOW: pages={window_pages}, overlap_pages={window_overlap_pages}, max_chars={window_max_chars}"
    )
    print(f"[INGEST] KB_POLICY_DIRS={scanned_dirs}")
    print(f"[INGEST] config_chunk_size={chunk_size}, config_chunk_overlap={chunk_overlap}")
    print(f"[INGEST] found_pdfs={len(pdf_paths)}")

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
        page_entries: list[tuple[int, str]] = []

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

            page_text = _normalize_page_text(raw_text)
            if len(page_text) < 50:
                """
                过短页面通常是封面、分隔页、扫描空页或无正文价值内容。
                这里跳过是为了减少无效向量，提升检索命中率。
                """
                continue
            page_entries.append((page_idx, page_text))

        windows = _build_page_windows(
            page_entries,
            window_pages=window_pages,
            overlap_pages=window_overlap_pages,
            max_chars=window_max_chars,
        )
        structured_context: tuple[str, str, str] | None = None
        seen_chunk_signatures: set[str] = set()

        for window_idx, window in enumerate(windows):
            if chunk_strategy == "structured_hybrid":
                chunks_result = _split_structured_hybrid(
                    str(window.get("text", "")),
                    initial_context=structured_context,
                    return_context=True,
                )
                chunks, structured_context = chunks_result
            else:
                chunks = split_text_with_strategy(str(window.get("text", "")), chunk_strategy)

            for chunk_idx, chunk_info in enumerate(chunks):
                chunk = clean_text(chunk_info.get("text", ""))
                section_path = str(chunk_info.get("section_path", "") or "")
                if len(chunk) < 30:
                    """过短 chunk 往往缺乏语义完整性，保留会增加噪声，直接丢弃。"""
                    continue
                page_start = int(window.get("page_start") or 0)
                page_end = int(window.get("page_end") or page_start)
                dedupe_key = sha1_id(doc_id, str(page_start), str(page_end), section_path, chunk)
                if dedupe_key in seen_chunk_signatures:
                    continue
                seen_chunk_signatures.add(dedupe_key)

                point_id = stable_uuid(
                    doc_id,
                    str(page_start),
                    str(page_end),
                    str(window_idx),
                    str(chunk_idx),
                    chunk_strategy,
                    section_path,
                    chunk,
                )
                """
                payload 是后续“可解释检索”和“引用展示”的关键。
                retrieve/answer/eval 都依赖这里的 doc_id/page/snippet 元信息。
                """
                payload = {
                    "doc_id": doc_id,
                    "page": page_start,
                    "page_start": page_start,
                    "page_end": page_end,
                    "window_index": window_idx,
                    "chunk_index": chunk_idx,
                    "section_path": section_path,
                    "chunk_strategy": chunk_strategy,
                    "chunk_chars": len(chunk),
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
                "windows": len(windows),
                "chunk_count": doc_chunks,
                "avg_chunk_chars": round(avg_chunk_chars, 2),
                "failed_pages": len(failed_pages),
                "failed_page_numbers": failed_pages,
            }
        )
        print(
            "[INGEST] "
            f"{doc_id}: pages={page_count}, windows={len(windows)}, chunks={doc_chunks}, "
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
        "chunk_strategy": chunk_strategy,
        "embed_model": embed_model,
        "documents": per_doc_stats,
        "summary": {
            "documents": len(per_doc_stats),
            "total_chunks": total_chunks,
            "total_points_in_qdrant": points_in_qdrant,
            "total_failed_pages": total_failed_pages,
            "window_pages": max(1, int(window_pages)),
            "window_overlap_pages": max(0, int(window_overlap_pages)),
            "window_max_chars": max(0, int(window_max_chars)),
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
