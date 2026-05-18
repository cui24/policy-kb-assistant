#!/usr/bin/env python3
"""基于已抽取片段调用 LLM 生成 RAG 评测集并回填 CSV。

设计说明（高层）：
1. 输入是“题目骨架 + 证据片段 + 文档元数据”三件套：
   - 题目骨架：限定 query_id/category/difficulty/query_type，不允许模型改配比。
   - 证据片段：作为可引用事实池，减少模型编造。
   - 文档元数据：用于冲突题与时效性判断（status/version/effective_date）。
2. 每个 category（hr/admin/finance/it）分批调用一次模型，避免一次请求过大。
3. 模型只负责补齐内容字段（query/gold_doc_id/...），最终由脚本做结构归一化与必填校验。
4. 任何一题缺关键字段会直接失败退出，防止“静默坏数据”进入评测集。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_TEMPLATE = Path("data/eval/rag_eval_set_v2_100_template.csv")
DEFAULT_CHUNKS_DIR = Path("data/extracted")
DEFAULT_DOCS = Path("data/eval/documents.csv")
DEFAULT_PROMPT = Path("data/eval/LLM_QUESTION_GENERATION_PROMPT_TEMPLATE.zh-CN.md")
DEFAULT_OUT = Path("data/eval/rag_eval_set_v2_100_llm.csv")

CATEGORY_TO_PREFIX = {
    "hr": "HR-",
    "admin": "ADM-",
    "finance": "FIN-",
    "it": "IT-",
}

QUERY_TYPE_TO_RULE_TYPES = {
    "fact": ["definition", "fact"],
    "procedure": ["procedure"],
    "constraint": ["constraint"],
    "comparison": ["fact", "definition", "constraint", "procedure"],
    "exception": ["exception", "constraint"],
    "multi_hop": ["procedure", "constraint", "fact", "definition", "exception"],
}

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


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    参数策略：
    - template/chunks/documents/prompt-template：四类输入来源。
    - output：最终回填结果。
    - model/temperature/max-tokens：模型调用控制项，方便不同平台切换。
    """
    parser = argparse.ArgumentParser(description="Generate RAG eval set with LLM")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--chunks-dir", default=str(DEFAULT_CHUNKS_DIR))
    parser.add_argument("--documents", default=str(DEFAULT_DOCS))
    parser.add_argument("--prompt-template", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=12000)
    return parser.parse_args()


def _load_template_rows(path: Path) -> list[dict[str, str]]:
    """读取题目骨架 CSV，并统一做 strip 清洗。"""
    with path.open("r", encoding="utf-8", newline="") as rf:
        rows = list(csv.DictReader(rf))
    return [{k: str(v or "").strip() for k, v in row.items()} for row in rows]


def _load_documents(path: Path) -> dict[str, dict[str, str]]:
    """按 doc_id 加载文档元数据，便于后续快速索引。"""
    with path.open("r", encoding="utf-8", newline="") as rf:
        rows = list(csv.DictReader(rf))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        result[doc_id] = {k: str(v or "").strip() for k, v in row.items()}
    return result


def _iter_jsonl_files(base_dir: Path) -> list[Path]:
    """遍历抽取结果目录，忽略 sample 文件，避免污染正式数据集。"""
    files: list[Path] = []
    for p in sorted(base_dir.rglob("*.jsonl")):
        name = p.name.lower()
        if ".sample." in name:
            continue
        files.append(p)
    return files


def _category_from_doc_id(doc_id: str) -> str:
    """通过 doc_id 前缀映射业务类别（HR-/ADM-/FIN-/IT-）。"""
    for cat, prefix in CATEGORY_TO_PREFIX.items():
        if doc_id.startswith(prefix):
            return cat
    return ""


def _load_chunks_by_category(base_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """加载并按 category 分组证据片段。

    注意：
    - 只抽取生成题目真正需要的字段，控制 payload 体积。
    - 对每行 JSONL 做最小清洗，保证后续拼 Prompt 时字段稳定。
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in _iter_jsonl_files(base_dir):
        with path.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc_id = str(row.get("doc_id") or "").strip()
                cat = _category_from_doc_id(doc_id)
                if not cat:
                    continue
                grouped[cat].append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": str(row.get("chunk_id") or "").strip(),
                        "section_path": str(row.get("section_path") or "").strip(),
                        "page_start": row.get("page_start"),
                        "page_end": row.get("page_end"),
                        "rule_type": str(row.get("rule_type") or "").strip(),
                        "text": str(row.get("text") or "").strip(),
                        "evidence_span": str(row.get("evidence_span") or "").strip(),
                    }
                )
    return grouped


def _assign_chunk_ids_for_tasks(
    tasks: list[dict[str, str]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """为每道题预分配“推荐证据 chunk”。

    核心思路：
    - 按 query_type -> rule_type 偏好选择 chunk，尽量让题型与证据语义对齐。
    - 使用轮转队列（deque.rotate）做“均匀取样”，避免总是使用头部片段。
    - comparison/multi_hop 额外给第二个 chunk，鼓励跨段问题生成。
    """
    by_rule: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for c in chunks:
        by_rule[c["rule_type"]].append(c)
    all_q = deque(chunks)

    def take_one(preferred_rules: list[str]) -> dict[str, Any]:
        """优先按偏好规则取片段，取不到再回退到全量轮转池。"""
        for rt in preferred_rules:
            if by_rule.get(rt):
                c = by_rule[rt][0]
                by_rule[rt].rotate(-1)
                return c
        c = all_q[0]
        all_q.rotate(-1)
        return c

    assigned: list[dict[str, Any]] = []
    for t in tasks:
        qtype = t.get("query_type", "")
        preferred = QUERY_TYPE_TO_RULE_TYPES.get(qtype, ["fact", "definition", "procedure", "constraint"])
        first = take_one(preferred)
        chunk_ids = [first["chunk_id"]]
        if qtype in {"comparison", "multi_hop"} and len(chunks) > 1:
            second = take_one(preferred)
            if second["chunk_id"] != first["chunk_id"]:
                chunk_ids.append(second["chunk_id"])
        item = dict(t)
        item["recommended_chunk_ids"] = chunk_ids
        assigned.append(item)
    return assigned


def _extract_json_array_text(raw: str) -> str:
    """从模型返回文本中提取 JSON 数组主体。

    兼容场景：
    - 理想情况：模型只回 JSON 数组。
    - 非理想情况：前后夹杂说明文字；此时截取最外层 []。
    """
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return raw
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    raise ValueError("no_json_array")


def _normalize_item(item: dict[str, Any], task_row: dict[str, str]) -> dict[str, str]:
    """把模型返回单题归一化到标准 CSV 行结构。

    关键规则：
    - query_id/category/difficulty/query_type 以骨架为准，不信任模型重写。
    - answer_points 统一为中文分号分隔。
    - conflict_case 非 yes/no 时强制回退 no，减少脏值。
    """
    answer_points = item.get("answer_points", "")
    if isinstance(answer_points, list):
        answer_points = "；".join(str(x).strip() for x in answer_points if str(x).strip())
    answer_points = str(answer_points or "").strip().replace(";", "；")

    conflict_case = str(item.get("conflict_case", "no") or "no").strip().lower()
    if conflict_case not in {"yes", "no"}:
        conflict_case = "no"

    out = {
        "query_id": task_row["query_id"],
        "query": str(item.get("query", "") or "").strip(),
        "category": task_row["category"],
        "difficulty": task_row["difficulty"],
        "query_type": task_row["query_type"],
        "gold_doc_id": str(item.get("gold_doc_id", "") or "").strip(),
        "gold_section": str(item.get("gold_section", "") or "").strip(),
        "answer_points": answer_points,
        "as_of_date": str(item.get("as_of_date", "") or "").strip(),
        "conflict_case": conflict_case,
        "notes": str(item.get("notes", "") or "").strip(),
    }
    return out


def _generate_for_category(
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    category: str,
    tasks: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    doc_meta: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """按类别批量调用模型并回填该类别题目。

    执行流程：
    1) 为任务分配推荐 chunk；
    2) 构造紧凑 payload（tasks/chunks/documents）；
    3) 发起一次 LLM 调用；
    4) 解析 JSON 数组并按 query_id 对齐；
    5) 对每条题目做必填校验（缺字段立即失败）。
    """
    tasks_assigned = _assign_chunk_ids_for_tasks(tasks, chunks)
    chunk_id_set: set[str] = set()
    for t in tasks_assigned:
        for cid in t["recommended_chunk_ids"]:
            chunk_id_set.add(cid)

    focused_chunks = [c for c in chunks if c["chunk_id"] in chunk_id_set]
    focused_chunks.sort(key=lambda x: x["chunk_id"])

    docs_payload: list[dict[str, str]] = []
    seen_doc_ids: set[str] = set()
    for c in focused_chunks:
        did = c["doc_id"]
        if did in seen_doc_ids:
            continue
        seen_doc_ids.add(did)
        meta = doc_meta.get(did, {})
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

    # 发送给模型的结构化输入。尽量显式，减少模型“自由发挥”。
    user_payload = {
        "category": category,
        "tasks": [
            {
                "query_id": t["query_id"],
                "category": t["category"],
                "difficulty": t["difficulty"],
                "query_type": t["query_type"],
                "recommended_chunk_ids": t["recommended_chunk_ids"],
            }
            for t in tasks_assigned
        ],
        "chunks": focused_chunks,
        "documents": docs_payload,
        "output_requirements": {
            "count": len(tasks),
            "must_cover_query_ids": [t["query_id"] for t in tasks],
            "format": "json_array_only",
        },
    }

    user_prompt = "输入数据如下（JSON）：\n" + json.dumps(user_payload, ensure_ascii=False)
    # 统一使用 chat.completions，兼容当前项目的 OpenAI-compatible 后端。
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = str(resp.choices[0].message.content or "").strip()
    arr_text = _extract_json_array_text(content)
    arr = json.loads(arr_text)
    if not isinstance(arr, list):
        raise ValueError("model_output_not_array")

    # 将模型输出按 query_id 建立索引，便于与任务骨架对齐。
    by_id: dict[str, dict[str, Any]] = {}
    for it in arr:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("query_id") or "").strip()
        if qid:
            by_id[qid] = it

    results: list[dict[str, str]] = []
    for task in tasks:
        raw = by_id.get(task["query_id"], {})
        normalized = _normalize_item(raw, task)
        # 核心字段缺失即报错，避免写出“表面可读但无法评测”的数据。
        if not normalized["query"]:
            raise ValueError(f"missing_query_for_{task['query_id']}")
        if not normalized["gold_doc_id"]:
            raise ValueError(f"missing_gold_doc_id_for_{task['query_id']}")
        if not normalized["gold_section"]:
            raise ValueError(f"missing_gold_section_for_{task['query_id']}")
        if not normalized["answer_points"]:
            raise ValueError(f"missing_answer_points_for_{task['query_id']}")
        results.append(normalized)
    return results


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """将最终题集按固定列顺序写入 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """程序入口：加载输入 -> 分类别生成 -> 合并输出。

    失败策略：
    - 缺 API Key、缺类别片段、缺 query_id 输出等场景均直接 fail-fast。
    - 这样可以尽早暴露问题，不会生成“半残评测集”。
    """
    args = _parse_args()
    template_path = Path(args.template)
    chunks_dir = Path(args.chunks_dir)
    docs_path = Path(args.documents)
    prompt_path = Path(args.prompt_template)
    output_path = Path(args.output)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] missing OPENAI_API_KEY (not found in env/.env)")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    # timeout 稍大，给 25 题级别的批量生成预留时间。
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)

    template_rows = _load_template_rows(template_path)
    documents = _load_documents(docs_path)
    chunks_by_category = _load_chunks_by_category(chunks_dir)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    tasks_by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in template_rows:
        cat = row.get("category", "")
        tasks_by_category[cat].append(row)

    generated_by_qid: dict[str, dict[str, str]] = {}
    # 固定顺序执行，便于日志定位和重跑复现。
    for cat in ["hr", "admin", "finance", "it"]:
        tasks = tasks_by_category.get(cat, [])
        chunks = chunks_by_category.get(cat, [])
        if not tasks:
            continue
        if not chunks:
            raise SystemExit(f"[ERROR] no chunks found for category={cat}")
        print(f"[INFO] generating category={cat} tasks={len(tasks)} chunks={len(chunks)}")
        results = _generate_for_category(
            client=client,
            model=args.model,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            system_prompt=system_prompt,
            category=cat,
            tasks=tasks,
            chunks=chunks,
            doc_meta=documents,
        )
        for r in results:
            generated_by_qid[r["query_id"]] = r

    output_rows: list[dict[str, str]] = []
    for t in template_rows:
        qid = t["query_id"]
        filled = generated_by_qid.get(qid)
        if not filled:
            raise SystemExit(f"[ERROR] missing generated row for {qid}")
        output_rows.append(filled)

    _write_csv(output_path, output_rows)
    print(f"[OK] generated file: {output_path} rows={len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
