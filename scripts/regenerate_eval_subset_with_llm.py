#!/usr/bin/env python3
"""定点重生成评测题并回写 CSV。

用途：
1. 只重做少量“坏样本”（如标注腔、文档编号入题、冲突题字段缺失）。
2. 保持全量 100 题配比不变（仅替换指定 query_id）。
3. 在重生阶段加入更严格校验，避免把问题带入最终评测集。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# 允许直接运行 scripts/*.py 时互相导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_rag_eval_set_with_llm import (
    CATEGORY_TO_PREFIX,
    QUERY_TYPE_TO_RULE_TYPES,
    _load_chunks_by_category,
    _load_documents,
    _load_template_rows,
)


DEFAULT_INPUT = Path("data/eval/policy_eval_100.csv")
DEFAULT_OUTPUT = Path("data/eval/policy_eval_100_v2.csv")
DEFAULT_CHUNKS_DIR = Path("data/extracted")
DEFAULT_DOCS = Path("data/eval/documents.csv")
DEFAULT_PROMPT = Path("data/eval/LLM_QUESTION_REGEN_PROMPT_STRICT.zh-CN.md")

TARGET_IDS = [
    "Q_HR_020",
    "Q_HR_021",
    "Q_HR_022",
    "Q_ADM_022",
    "Q_ADM_023",
    "Q_FIN_021",
    "Q_FIN_022",
    "Q_FIN_023",
    "Q_FIN_025",
    "Q_IT_021",
    "Q_IT_025",
]

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

DOC_ID_RE = re.compile(r"\b(?:HR|ADM|FIN|IT)-\d+\b")


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    典型调用：
    - 默认重生 TARGET_IDS。
    - 也可通过 --ids 指定单题或子集重生（便于迭代修复）。
    """
    parser = argparse.ArgumentParser(description="Regenerate subset of eval rows with LLM")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--chunks-dir", default=str(DEFAULT_CHUNKS_DIR))
    parser.add_argument("--documents", default=str(DEFAULT_DOCS))
    parser.add_argument("--prompt-template", default=str(DEFAULT_PROMPT))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument(
        "--ids",
        default=",".join(TARGET_IDS),
        help="Comma-separated query_ids to regenerate",
    )
    return parser.parse_args()


def _extract_json(raw: str) -> dict[str, Any]:
    """从模型返回中提取单题 JSON。

    兼容三种返回形态：
    1) 直接 dict；
    2) 长度为 1 的 list[dict]；
    3) 前后夹杂文本，需截取 [] 或 {}。
    """
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        start_arr, end_arr = raw.find("["), raw.rfind("]")
        if start_arr >= 0 and end_arr > start_arr:
            data = json.loads(raw[start_arr : end_arr + 1])
        else:
            start_obj, end_obj = raw.find("{"), raw.rfind("}")
            if start_obj >= 0 and end_obj > start_obj:
                data = json.loads(raw[start_obj : end_obj + 1])
            else:
                raise ValueError("no_json_found")
    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            raise ValueError("expect_single_item_array")
        return data[0]
    if isinstance(data, dict):
        return data
    raise ValueError("invalid_json_payload")


def _split_points(points: str) -> list[str]:
    """拆分 answer_points（兼容中文/英文分号）。"""
    return [p.strip() for p in re.split(r"[；;]", points) if p.strip()]


def _validate_generated(
    *,
    current: dict[str, str],
    generated: dict[str, str],
    chunks: list[dict[str, Any]],
    documents: dict[str, dict[str, str]],
) -> str | None:
    """对重生结果做强约束校验。

    返回值：
    - None：通过
    - str：失败原因（用于反馈给模型重试）
    """
    # fixed fields must match
    # 关键元字段必须锁死，避免模型把题目迁移到别的类别或难度。
    for f in ["query_id", "category", "difficulty", "query_type"]:
        if generated.get(f, "").strip() != current.get(f, "").strip():
            return f"field_changed:{f}"

    query = generated.get("query", "").strip()
    if not query:
        return "empty_query"
    if DOC_ID_RE.search(query):
        return "query_contains_doc_id"
    # 针对已知“标注腔”模式的额外拦截。
    if "手册中" in query and "有何不同" in query:
        return "annotation_tone_query"

    cat = generated.get("category", "").strip()
    gold_doc_id = generated.get("gold_doc_id", "").strip()
    if not gold_doc_id:
        return "empty_gold_doc_id"
    prefix = CATEGORY_TO_PREFIX.get(cat, "")
    if prefix and not gold_doc_id.startswith(prefix):
        return "gold_doc_cross_category"
    if gold_doc_id not in documents:
        return "gold_doc_unknown"

    gold_section = generated.get("gold_section", "").strip()
    if not gold_section:
        return "empty_gold_section"
    section_set = {str(c.get("section_path", "")).strip() for c in chunks}
    # 允许“包含关系”匹配，兼容模型输出更长/更短 section 文本。
    if not any((gold_section in s) or (s in gold_section) for s in section_set):
        return "gold_section_not_in_chunks"

    points = generated.get("answer_points", "").strip()
    point_list = _split_points(points)
    if len(point_list) < 3 or len(point_list) > 6:
        return "invalid_answer_points_count"

    conflict_case = generated.get("conflict_case", "").strip().lower()
    if conflict_case not in {"yes", "no"}:
        return "invalid_conflict_case"
    as_of_date = generated.get("as_of_date", "").strip()
    if conflict_case == "yes":
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of_date):
            return "conflict_missing_as_of_date"
    else:
        if as_of_date:
            return "as_of_date_should_be_empty_when_no_conflict"

    difficulty = generated.get("difficulty", "").strip()
    if difficulty == "hard":
        if len(query) < 22:
            return "hard_query_too_short"

    return None


def _choose_chunks_for_row(
    *,
    row: dict[str, str],
    category_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """为单题重生选择证据片段。

    当前策略：
    - 默认只用该题 gold_doc 的片段，保证证据闭环。
    - 若 conflict_case=yes，再补充 notes 中提到的冲突文档片段。
    """
    gold_doc_id = row["gold_doc_id"]

    selected: list[dict[str, Any]] = []

    # 1) gold_doc all chunks first
    selected.extend([c for c in category_chunks if c["doc_id"] == gold_doc_id])

    # 2) if conflict case, add note-mentioned docs
    if row.get("conflict_case", "").lower() == "yes":
        note = row.get("notes", "")
        doc_ids = re.findall(r"\b(?:HR|ADM|FIN|IT)-\d+\b", note)
        for did in doc_ids:
            selected.extend([c for c in category_chunks if c["doc_id"] == did])

    # dedup by chunk_id, keep first occurrence
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for c in selected:
        cid = str(c["chunk_id"])
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(c)

    # 控制 token 预算，避免单题 payload 过大。
    return uniq[:10]


def _regen_one(
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    row: dict[str, str],
    row_reason: str,
    category_chunks: list[dict[str, Any]],
    documents: dict[str, dict[str, str]],
) -> dict[str, str]:
    """重生单题，带最多 N 次自动重试。

    重试机制：
    - 若 JSON 不合法或字段校验失败，则把失败原因写入 feedback。
    - 下一轮模型会看到上轮失败原因，定向修复。
    """
    chosen_chunks = _choose_chunks_for_row(row=row, category_chunks=category_chunks)

    docs_payload: list[dict[str, str]] = []
    seen_doc: set[str] = set()
    for c in chosen_chunks:
        did = c["doc_id"]
        if did in seen_doc:
            continue
        seen_doc.add(did)
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

    feedback = ""  # 累积式反馈，让模型知道“上次错在哪”
    for attempt in range(1, 7):
        payload = {
            "current_row": row,
            "regeneration_reason": row_reason,
            "chunks": chosen_chunks,
            "documents": docs_payload,
            "feedback_from_previous_attempt": feedback,
            "output_requirements": {
                "only_one_item": True,
                "array_length": 1,
            },
        }
        # 单题场景 token 较小，稳定性优先，不做流式。
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "输入数据如下（JSON）：\n" + json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = str(resp.choices[0].message.content or "").strip()
        try:
            item = _extract_json(raw)
        except Exception as exc:
            feedback = f"输出不是合法单条JSON：{exc}"
            continue

        normalized = {
            "query_id": str(item.get("query_id", "")).strip(),
            "query": str(item.get("query", "")).strip(),
            "category": str(item.get("category", "")).strip(),
            "difficulty": str(item.get("difficulty", "")).strip(),
            "query_type": str(item.get("query_type", "")).strip(),
            "gold_doc_id": str(item.get("gold_doc_id", "")).strip(),
            "gold_section": str(item.get("gold_section", "")).strip(),
            "answer_points": str(item.get("answer_points", "")).strip().replace(";", "；"),
            "as_of_date": str(item.get("as_of_date", "")).strip(),
            "conflict_case": str(item.get("conflict_case", "")).strip().lower(),
            "notes": str(item.get("notes", "")).strip(),
        }
        err = _validate_generated(
            current=row,
            generated=normalized,
            chunks=chosen_chunks,
            documents=documents,
        )
        if not err:
            return normalized
        print(f"[WARN] {row['query_id']} attempt={attempt} invalid={err}")
        feedback = f"上次输出未通过校验：{err}。请严格修正。"

    raise RuntimeError(f"failed_regen_after_retries:{row['query_id']}")


def _reason_for_row(row: dict[str, str]) -> str:
    """根据当前旧题目自动生成重生原因标签（用于 prompt 提示）。"""
    q = row.get("query", "")
    reasons: list[str] = []
    if DOC_ID_RE.search(q):
        reasons.append("query_contains_doc_id")
    if row.get("difficulty") == "hard":
        reasons.append("hard_should_be_more_user_like_and_complex")
    if row.get("conflict_case", "").lower() == "yes" and not row.get("as_of_date", "").strip():
        reasons.append("conflict_case_missing_as_of_date")
    if row.get("query_id") == "Q_IT_021":
        reasons.append("answer_points_cross_doc_risk")
    return ",".join(reasons) or "quality_improvement"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """按标准列写出 CSV。"""
    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """程序入口：读取输入表 -> 重生指定题 -> 合并输出。

    注意：
    - 仅替换 id_set 命中的行，未命中行原样保留。
    - 这样可以保证配比、顺序、query_id 稳定不变。
    """
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    chunks_dir = Path(args.chunks_dir)
    docs_path = Path(args.documents)
    prompt_path = Path(args.prompt_template)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] missing OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    # timeout 适配远端兼容模型，减少偶发超时重跑。
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)

    ids = [x.strip() for x in str(args.ids).split(",") if x.strip()]
    id_set = set(ids)
    src_rows = _load_template_rows(input_path)
    documents = _load_documents(docs_path)
    chunks_by_category = _load_chunks_by_category(chunks_dir)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    by_qid = {r["query_id"]: r for r in src_rows}
    # 预检查：防止传了不存在的 query_id 造成静默漏改。
    missing = [qid for qid in ids if qid not in by_qid]
    if missing:
        raise SystemExit(f"[ERROR] query_id not found in input: {missing}")

    regenerated: dict[str, dict[str, str]] = {}
    for qid in ids:
        row = by_qid[qid]
        cat = row["category"]
        cat_chunks = chunks_by_category.get(cat, [])
        if not cat_chunks:
            raise SystemExit(f"[ERROR] no chunks for category={cat}")
        reason = _reason_for_row(row)
        print(f"[INFO] regenerating {qid} category={cat} reason={reason}")
        new_row = _regen_one(
            client=client,
            model=args.model,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            system_prompt=system_prompt,
            row=row,
            row_reason=reason,
            category_chunks=cat_chunks,
            documents=documents,
        )
        regenerated[qid] = new_row

    out_rows: list[dict[str, str]] = []
    for row in src_rows:
        qid = row["query_id"]
        if qid in id_set:
            out_rows.append(regenerated[qid])
        else:
            out_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, out_rows)
    print(f"[OK] regenerated={len(regenerated)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
