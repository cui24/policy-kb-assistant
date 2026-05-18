#!/usr/bin/env python3
"""基于抽取片段自动构建 hard 评测题集。

设计目标
--------
1. 复用 `data/extracted/**/*.jsonl` 中的结构化片段，不从原始 PDF 直接抽题。
2. 聚焦高区分问题：
   - 跨段流程题（flow_cross_segment）
   - 条件+例外题（condition_exception）
   - 相似制度干扰题（similar_policy_interference）
3. 自动生成题目骨架（query_id/category/difficulty/query_type/class_tag），再调用 LLM 填充。
4. 生成后做强校验，尽量保证“可评测、可追溯、可复现”。

输出 CSV 字段与现有评测集一致：
- query_id,query,category,difficulty,query_type,gold_doc_id,gold_section,answer_points,as_of_date,conflict_case,notes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# 允许直接运行 scripts/*.py 时互相导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 复用已有脚本中的常量与加载函数，避免重复维护。
from scripts.generate_rag_eval_set_with_llm import CATEGORY_TO_PREFIX, _load_documents, _load_chunks_by_category


DEFAULT_CHUNKS_DIR = Path("data/extracted")
DEFAULT_DOCS = Path("data/eval/documents.csv")
DEFAULT_PROMPT = Path("data/eval/LLM_HARD_QUESTION_GENERATION_PROMPT.zh-CN.md")
DEFAULT_OUT = Path("data/eval/policy_eval_hard_v1.csv")

FIELDNAMES = [
    "query_id",
    "query",
    "category",
    "difficulty",
    "query_type",
    "gold_doc_id",
    "gold_section",
    "answer_points",
    "as_of_date",
    "conflict_case",
    "notes",
]

# 为每个业务类别生成多少题。默认每类 12（3 类题型各 4）=> 总计 48。
DEFAULT_PER_CAT = 12
DEFAULT_FLOW_PER_CAT = 4
DEFAULT_EXCEPTION_PER_CAT = 4
DEFAULT_COMPARISON_PER_CAT = 4

DOC_ID_RE = re.compile(r"\b(?:HR|ADM|FIN|IT)-\d+\b", re.IGNORECASE)
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# 条款中常见的“例外/禁止”提示词，用于候选筛选。
EXCEPTION_HINTS = ["例外", "特殊", "除", "除非", "不得", "严禁", "不予", "可以调整", "不允许"]


@dataclass
class Task:
    """单道待生成题任务骨架。"""

    query_id: str
    category: str
    difficulty: str
    query_type: str
    class_tag: str
    recommended_chunk_ids: list[str]


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate hard eval set with LLM from extracted chunks")
    parser.add_argument("--chunks-dir", default=str(DEFAULT_CHUNKS_DIR))
    parser.add_argument("--documents", default=str(DEFAULT_DOCS))
    parser.add_argument("--prompt-template", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260513)

    # 配比控制（可按实验需要调整）
    parser.add_argument("--per-category", type=int, default=DEFAULT_PER_CAT)
    parser.add_argument("--flow-per-category", type=int, default=DEFAULT_FLOW_PER_CAT)
    parser.add_argument("--exception-per-category", type=int, default=DEFAULT_EXCEPTION_PER_CAT)
    parser.add_argument("--comparison-per-category", type=int, default=DEFAULT_COMPARISON_PER_CAT)

    return parser.parse_args()


def _chunk_num(chunk_id: str) -> int:
    """从 `XXX_chunk_0007` 提取数字序号，无法解析时返回大数。"""

    m = re.search(r"_chunk_(\d+)$", chunk_id)
    if not m:
        return 10**9
    return int(m.group(1))


def _normalize_text_for_tokens(text: str) -> list[str]:
    """做轻量分词（无需第三方分词器）。

    规则：
    - 以中文/英文/数字连续串为 token；
    - 去掉单字与纯数字短噪音；
    """

    raw = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", (text or "").lower())
    out: list[str] = []
    for tok in raw:
        if tok.isdigit():
            continue
        if len(tok) <= 1:
            continue
        out.append(tok)
    return out


def _chunk_token_set(chunk: dict[str, Any]) -> set[str]:
    """为相似度比较构造 token 集。

    优先使用抽取时给出的 `keywords`；若缺失再回退 section_path/text。
    """

    toks: set[str] = set()
    keywords = chunk.get("keywords")
    if isinstance(keywords, list):
        for k in keywords:
            toks.update(_normalize_text_for_tokens(str(k)))
    toks.update(_normalize_text_for_tokens(str(chunk.get("section_path", ""))))
    # 只取文本前 200 字降低噪声与计算量
    toks.update(_normalize_text_for_tokens(str(chunk.get("text", ""))[:200]))
    return toks


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard 相似度。"""

    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _make_query_id(cat: str, idx: int) -> str:
    """生成 hard 题 query_id，如 `QH_FIN_003`。"""

    prefix = {
        "hr": "HR",
        "admin": "ADM",
        "finance": "FIN",
        "it": "IT",
    }[cat]
    return f"QH_{prefix}_{idx:03d}"


def _pick_flow_tasks(cat: str, chunks: list[dict[str, Any]], n: int) -> list[Task]:
    """选“跨段流程题”候选任务。

    策略：
    1) 优先同文档内多个 `procedure` 片段组合，形成跨段流程。
    2) 若不足，回退到 `procedure + constraint/exception` 组合。
    """

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)
    for did in by_doc:
        by_doc[did].sort(key=lambda x: _chunk_num(x["chunk_id"]))

    pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for did, arr in by_doc.items():
        proc = [c for c in arr if c.get("rule_type") == "procedure"]
        # A: procedure 邻近对
        for i in range(len(proc) - 1):
            c1, c2 = proc[i], proc[i + 1]
            score = 2.0  # 流程对优先级最高
            pairs.append((c1, c2, score))

        # B: procedure + (constraint/exception)
        others = [c for c in arr if c.get("rule_type") in {"constraint", "exception"}]
        for p in proc:
            for o in others:
                if p["chunk_id"] == o["chunk_id"]:
                    continue
                dist = abs(_chunk_num(p["chunk_id"]) - _chunk_num(o["chunk_id"]))
                score = 1.2 - min(dist, 10) * 0.03
                pairs.append((p, o, score))

    # 分数高优先，尽量覆盖不同文档
    pairs.sort(key=lambda x: x[2], reverse=True)

    out: list[Task] = []
    used_pairs: set[tuple[str, str]] = set()
    used_docs: set[str] = set()
    idx = 1

    # 第一轮：先尽量文档去重
    for c1, c2, _ in pairs:
        if len(out) >= n:
            break
        did = c1["doc_id"]
        if did in used_docs:
            continue
        key = tuple(sorted([c1["chunk_id"], c2["chunk_id"]]))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        used_docs.add(did)
        out.append(
            Task(
                query_id=_make_query_id(cat, idx),
                category=cat,
                difficulty="hard",
                query_type="multi_hop",
                class_tag="flow_cross_segment",
                recommended_chunk_ids=[c1["chunk_id"], c2["chunk_id"]],
            )
        )
        idx += 1

    # 第二轮：补齐数量
    if len(out) < n:
        for c1, c2, _ in pairs:
            if len(out) >= n:
                break
            key = tuple(sorted([c1["chunk_id"], c2["chunk_id"]]))
            if key in used_pairs:
                continue
            used_pairs.add(key)
            out.append(
                Task(
                    query_id=_make_query_id(cat, idx),
                    category=cat,
                    difficulty="hard",
                    query_type="multi_hop",
                    class_tag="flow_cross_segment",
                    recommended_chunk_ids=[c1["chunk_id"], c2["chunk_id"]],
                )
            )
            idx += 1

    return out[:n]


def _pick_exception_tasks(cat: str, chunks: list[dict[str, Any]], n: int, start_idx: int) -> list[Task]:
    """选“条件+例外题”候选任务。

    策略：
    - 主片段优先 rule_type=exception，其次 constraint 且文本含例外提示词。
    - 为增强 hard 难度，再配一个同文档补充片段（procedure/constraint/definition）。
    """

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    def has_hint(c: dict[str, Any]) -> bool:
        text = f"{c.get('section_path','')} {c.get('text','')}"
        return any(h in text for h in EXCEPTION_HINTS)

    primary: list[dict[str, Any]] = []
    for c in chunks:
        rt = str(c.get("rule_type") or "")
        if rt == "exception":
            primary.append(c)
        elif rt == "constraint" and has_hint(c):
            primary.append(c)

    # 若异常片段太少，降级加入所有 constraint
    if len(primary) < n:
        more = [c for c in chunks if c.get("rule_type") == "constraint" and c not in primary]
        primary.extend(more)

    out: list[Task] = []
    seen: set[str] = set()
    idx = start_idx

    for p in primary:
        if len(out) >= n:
            break
        cid = p["chunk_id"]
        if cid in seen:
            continue
        did = p["doc_id"]
        mates = [
            c
            for c in by_doc[did]
            if c["chunk_id"] != cid and c.get("rule_type") in {"procedure", "constraint", "definition", "fact"}
        ]
        mates.sort(key=lambda x: _chunk_num(x["chunk_id"]))

        rec = [cid]
        if mates:
            rec.append(mates[0]["chunk_id"])
        out.append(
            Task(
                query_id=_make_query_id(cat, idx),
                category=cat,
                difficulty="hard",
                query_type="exception",
                class_tag="condition_exception",
                recommended_chunk_ids=rec,
            )
        )
        idx += 1
        seen.add(cid)

    return out[:n]


def _pick_comparison_tasks(cat: str, chunks: list[dict[str, Any]], n: int, start_idx: int) -> list[Task]:
    """选“相似制度干扰题”候选任务。

    策略：
    - 在不同 doc 之间找“语义重叠但不完全相同”的片段对；
    - 用关键词/标题/token 的 Jaccard 近似衡量相似度；
    - 过滤掉同文档配对，突出跨制度辨析。
    """

    items: list[tuple[dict[str, Any], set[str]]] = []
    for c in chunks:
        items.append((c, _chunk_token_set(c)))

    pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for i in range(len(items)):
        c1, t1 = items[i]
        for j in range(i + 1, len(items)):
            c2, t2 = items[j]
            if c1["doc_id"] == c2["doc_id"]:
                continue
            sim = _jaccard(t1, t2)
            # 太低通常毫无关系；太高可能重复条文。取中间段更有比较价值。
            if sim < 0.08:
                continue
            if sim > 0.90:
                continue
            rt_bonus = 0.0
            if c1.get("rule_type") == c2.get("rule_type"):
                rt_bonus = 0.05
            pairs.append((c1, c2, sim + rt_bonus))

    # 如果候选太少，放宽阈值
    if len(pairs) < n:
        for i in range(len(items)):
            c1, t1 = items[i]
            for j in range(i + 1, len(items)):
                c2, t2 = items[j]
                if c1["doc_id"] == c2["doc_id"]:
                    continue
                sim = _jaccard(t1, t2)
                if sim < 0.03:
                    continue
                pairs.append((c1, c2, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)

    out: list[Task] = []
    used_docs_pairs: set[tuple[str, str]] = set()
    used_chunk_pairs: set[tuple[str, str]] = set()
    idx = start_idx

    for c1, c2, _ in pairs:
        if len(out) >= n:
            break
        did_pair = tuple(sorted([c1["doc_id"], c2["doc_id"]]))
        ch_pair = tuple(sorted([c1["chunk_id"], c2["chunk_id"]]))
        if ch_pair in used_chunk_pairs:
            continue

        # 先优先文档对去重，提高题面多样性。
        if did_pair in used_docs_pairs and len(out) < max(1, n - 2):
            continue

        used_docs_pairs.add(did_pair)
        used_chunk_pairs.add(ch_pair)
        out.append(
            Task(
                query_id=_make_query_id(cat, idx),
                category=cat,
                difficulty="hard",
                query_type="comparison",
                class_tag="similar_policy_interference",
                recommended_chunk_ids=[c1["chunk_id"], c2["chunk_id"]],
            )
        )
        idx += 1

    return out[:n]


def _build_tasks_for_category(
    *,
    cat: str,
    chunks: list[dict[str, Any]],
    flow_n: int,
    exception_n: int,
    comparison_n: int,
) -> list[Task]:
    """按类别构建 hard 题任务骨架。"""

    flow_tasks = _pick_flow_tasks(cat, chunks, flow_n)
    exception_tasks = _pick_exception_tasks(cat, chunks, exception_n, start_idx=len(flow_tasks) + 1)
    comparison_tasks = _pick_comparison_tasks(
        cat,
        chunks,
        comparison_n,
        start_idx=len(flow_tasks) + len(exception_tasks) + 1,
    )

    tasks = flow_tasks + exception_tasks + comparison_tasks

    # 如果有短缺，降级用流程候选补足（仍保持 hard + multi_hop）
    target = flow_n + exception_n + comparison_n
    if len(tasks) < target:
        fallback = _pick_flow_tasks(cat, chunks, target)
        used = {t.query_id for t in tasks}
        for ft in fallback:
            if len(tasks) >= target:
                break
            if ft.query_id in used:
                continue
            ft.query_id = _make_query_id(cat, len(tasks) + 1)
            tasks.append(ft)

    # 重新编号，确保 query_id 连续
    for i, t in enumerate(tasks, start=1):
        t.query_id = _make_query_id(cat, i)

    return tasks[:target]


def _extract_json_array_text(raw: str) -> str:
    """从模型输出中截取最外层 JSON 数组。"""

    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return raw
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    raise ValueError("no_json_array")


def _normalize_item(item: dict[str, Any], task: Task) -> dict[str, str]:
    """规范化单题结构，并锁定骨架字段。"""

    answer_points = item.get("answer_points", "")
    if isinstance(answer_points, list):
        raw_points = [str(x).strip() for x in answer_points if str(x).strip()]
    else:
        raw_text = str(answer_points or "").strip()
        # 第一层：按中英文分号、换行切点
        raw_points = [p.strip() for p in re.split(r"[；;\n]+", raw_text) if p.strip()]
        # 若模型只给了 1-2 大句，尝试按逗号/顿号再切一层
        if len(raw_points) < 3:
            expanded: list[str] = []
            for rp in raw_points:
                expanded.extend([x.strip() for x in re.split(r"[，,、]+", rp) if x.strip()])
            if len(expanded) >= len(raw_points):
                raw_points = expanded

    # 去重保序，避免模型重复点
    uniq: list[str] = []
    seen: set[str] = set()
    for p in raw_points:
        key = p.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)

    # 约束到评测可接受范围：最多 6 条，超出时保留前 6 条核心点
    answer_points = "；".join(uniq[:6])

    conflict_case = str(item.get("conflict_case", "no") or "no").strip().lower()
    if conflict_case not in {"yes", "no"}:
        conflict_case = "no"

    return {
        "query_id": task.query_id,
        "query": str(item.get("query", "") or "").strip(),
        "category": task.category,
        "difficulty": task.difficulty,
        "query_type": task.query_type,
        "gold_doc_id": str(item.get("gold_doc_id", "") or "").strip(),
        "gold_section": str(item.get("gold_section", "") or "").strip(),
        "answer_points": answer_points,
        "as_of_date": str(item.get("as_of_date", "") or "").strip(),
        "conflict_case": conflict_case,
        "notes": str(item.get("notes", "") or "").strip(),
    }


def _split_points(text: str) -> list[str]:
    """拆分 answer_points（支持中英文分号）。"""

    return [p.strip() for p in re.split(r"[；;]", text or "") if p.strip()]


def _validate_row(
    row: dict[str, str],
    *,
    task: Task,
    chunks_map: dict[str, dict[str, Any]],
    all_chunks_by_doc: dict[str, list[dict[str, Any]]],
    documents: dict[str, dict[str, str]],
) -> str | None:
    """校验单题质量，返回 None 表示通过，否则返回失败原因。"""

    if row["difficulty"] != "hard":
        return "difficulty_not_hard"
    if row["category"] != task.category:
        return "category_changed"
    if row["query_type"] != task.query_type:
        return "query_type_changed"

    query = row["query"].strip()
    if not query:
        return "empty_query"
    if DOC_ID_RE.search(query):
        return "query_contains_doc_id"
    if len(query) < 16:
        return "query_too_short"

    did = row["gold_doc_id"].strip()
    if not did:
        return "empty_gold_doc_id"
    if did not in documents:
        return "unknown_gold_doc_id"
    prefix = CATEGORY_TO_PREFIX.get(task.category, "")
    if prefix and not did.startswith(prefix):
        return "gold_doc_cross_category"

    section = row["gold_section"].strip()
    if not section:
        return "empty_gold_section"

    # gold_section 优先在任务推荐证据中追溯；若未命中，则允许在同 gold_doc 的其他片段中追溯。
    sec_pool: list[str] = []
    for cid in task.recommended_chunk_ids:
        c = chunks_map.get(cid)
        if c:
            sec_pool.append(str(c.get("section_path", "")))
    in_recommended = any((section in s) or (s in section) for s in sec_pool) if sec_pool else False
    if not in_recommended:
        doc_secs = [str(c.get("section_path", "")) for c in all_chunks_by_doc.get(did, [])]
        in_gold_doc = any((section in s) or (s in section) for s in doc_secs)
        if not in_gold_doc:
            return "gold_section_not_traceable"

    points = _split_points(row["answer_points"])
    if len(points) < 3 or len(points) > 6:
        return "invalid_answer_points_count"

    cc = row["conflict_case"].strip().lower()
    as_of = row["as_of_date"].strip()
    if cc not in {"yes", "no"}:
        return "invalid_conflict_case"
    if cc == "yes":
        if not DATE_RE.fullmatch(as_of):
            return "missing_as_of_date_for_conflict"
    else:
        if as_of:
            return "unexpected_as_of_date_when_no_conflict"

    return None


def _generate_for_category(
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    category: str,
    tasks: list[Task],
    chunks: list[dict[str, Any]],
    documents: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """按类别一次调用模型，回填该类别全部 hard 题。

    注：为了控制 token，传入的是“任务相关子集片段”，而非该类别全量 chunks。
    """

    task_chunk_ids: set[str] = set()
    for t in tasks:
        task_chunk_ids.update(t.recommended_chunk_ids)

    focused_chunks = [c for c in chunks if c["chunk_id"] in task_chunk_ids]
    focused_chunks.sort(key=lambda x: x["chunk_id"])

    # documents payload 只传任务涉及 doc_id，减少上下文。
    docs_payload: list[dict[str, str]] = []
    seen: set[str] = set()
    for c in focused_chunks:
        did = c["doc_id"]
        if did in seen:
            continue
        seen.add(did)
        meta = documents.get(did, {})
        docs_payload.append(
            {
                "doc_id": did,
                "effective_date": meta.get("effective_date", ""),
                "status": meta.get("status", ""),
                "version": meta.get("version", ""),
                "family_id": meta.get("family_id", ""),
                "authority_level": meta.get("authority_level", ""),
            }
        )

    user_payload = {
        "category": category,
        "tasks": [
            {
                "query_id": t.query_id,
                "category": t.category,
                "difficulty": t.difficulty,
                "query_type": t.query_type,
                "class_tag": t.class_tag,
                "recommended_chunk_ids": t.recommended_chunk_ids,
            }
            for t in tasks
        ],
        "chunks": focused_chunks,
        "documents": docs_payload,
        "output_requirements": {
            "count": len(tasks),
            "must_cover_query_ids": [t.query_id for t in tasks],
            "format": "json_array_only",
        },
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "输入数据如下（JSON）：\n" + json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = str(resp.choices[0].message.content or "").strip()
    arr_text = _extract_json_array_text(content)
    arr = json.loads(arr_text)
    if not isinstance(arr, list):
        raise ValueError("model_output_not_array")

    by_id: dict[str, dict[str, Any]] = {}
    for it in arr:
        if isinstance(it, dict):
            qid = str(it.get("query_id") or "").strip()
            if qid:
                by_id[qid] = it

    chunks_map = {c["chunk_id"]: c for c in focused_chunks}
    all_chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        all_chunks_by_doc[c["doc_id"]].append(c)

    out: list[dict[str, str]] = []
    errors: list[str] = []
    for t in tasks:
        raw = by_id.get(t.query_id)
        if raw is None:
            errors.append(f"missing_query_id:{t.query_id}")
            continue
        row = _normalize_item(raw, t)
        err = _validate_row(
            row,
            task=t,
            chunks_map=chunks_map,
            all_chunks_by_doc=all_chunks_by_doc,
            documents=documents,
        )
        if err:
            errors.append(f"{t.query_id}:{err}")
            continue
        out.append(row)

    if errors:
        raise ValueError("category_generation_validation_failed:" + ",".join(errors))

    return out


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """按固定字段顺序写出 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """入口：构建任务 -> LLM 生成 -> 校验 -> 输出。"""

    args = _parse_args()
    if args.flow_per_category + args.exception_per_category + args.comparison_per_category != args.per_category:
        raise SystemExit("[ERROR] flow/exception/comparison per-category sum must equal --per-category")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] missing OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)

    random.seed(int(args.seed))

    chunks_by_cat = _load_chunks_by_category(Path(args.chunks_dir))
    documents = _load_documents(Path(args.documents))
    system_prompt = Path(args.prompt_template).read_text(encoding="utf-8")

    all_rows: list[dict[str, str]] = []

    for cat in ["hr", "admin", "finance", "it"]:
        chunks = list(chunks_by_cat.get(cat, []))
        if not chunks:
            raise SystemExit(f"[ERROR] no chunks for category={cat}")

        tasks = _build_tasks_for_category(
            cat=cat,
            chunks=chunks,
            flow_n=int(args.flow_per_category),
            exception_n=int(args.exception_per_category),
            comparison_n=int(args.comparison_per_category),
        )
        if len(tasks) != int(args.per_category):
            raise SystemExit(f"[ERROR] category={cat} task_count={len(tasks)} expected={args.per_category}")

        # 允许类别级最多重试 2 次，提升一次性成功率。
        ok_rows: list[dict[str, str]] | None = None
        last_err = ""
        for attempt in range(1, 4):
            try:
                print(f"[INFO] generating category={cat} tasks={len(tasks)} attempt={attempt}")
                ok_rows = _generate_for_category(
                    client=client,
                    model=str(args.model),
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    system_prompt=system_prompt,
                    category=cat,
                    tasks=tasks,
                    chunks=chunks,
                    documents=documents,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                print(f"[WARN] category={cat} attempt={attempt} failed: {last_err}")

        if ok_rows is None:
            raise SystemExit(f"[ERROR] category={cat} generation failed after retries: {last_err}")

        all_rows.extend(ok_rows)

    # 统一排序，保证同类结果稳定。
    def sort_key(r: dict[str, str]) -> tuple[int, str]:
        qid = r["query_id"]
        cat_order = {"QH_HR_": 1, "QH_ADM_": 2, "QH_FIN_": 3, "QH_IT_": 4}
        order = 99
        for p, v in cat_order.items():
            if qid.startswith(p):
                order = v
                break
        return (order, qid)

    all_rows.sort(key=sort_key)

    _write_csv(Path(args.output), all_rows)
    print(f"[OK] generated hard eval set: {args.output} rows={len(all_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
