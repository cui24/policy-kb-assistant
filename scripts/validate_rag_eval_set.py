#!/usr/bin/env python3
"""校验 RAG 评测集结构与配比。"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FILE_PATH = Path("data/eval/rag_eval_set_v2_100_template.csv")
REQUIRED_COLUMNS = [
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

CATEGORY_ENUM = {"hr", "admin", "finance", "it"}
DIFF_ENUM = {"easy", "medium", "hard"}
TYPE_ENUM = {"fact", "procedure", "constraint", "comparison", "exception", "multi_hop"}
CONFLICT_ENUM = {"yes", "no"}

EXPECTED_TOTAL = 100
EXPECTED_PER_CATEGORY = 25
EXPECTED_DIFF = {"easy": 10, "medium": 10, "hard": 5}


def _fail(message: str) -> None:
    raise SystemExit(f"[ERROR] {message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RAG eval CSV")
    parser.add_argument(
        "--file",
        default=str(DEFAULT_FILE_PATH),
        help="Path to eval CSV file",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    file_path = Path(args.file)
    if not file_path.exists():
        _fail(f"missing file: {file_path}")

    with file_path.open("r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        header = reader.fieldnames or []
        if header != REQUIRED_COLUMNS:
            _fail(f"invalid header: {header}")
        rows = list(reader)

    if len(rows) != EXPECTED_TOTAL:
        _fail(f"expected {EXPECTED_TOTAL} rows, got {len(rows)}")

    id_set: set[str] = set()
    cat_counter = Counter()
    diff_counter_by_cat: dict[str, Counter] = defaultdict(Counter)

    for idx, row in enumerate(rows, start=2):
        qid = str(row.get("query_id") or "").strip()
        if not qid:
            _fail(f"line {idx}: empty query_id")
        if qid in id_set:
            _fail(f"line {idx}: duplicate query_id={qid}")
        id_set.add(qid)

        category = str(row.get("category") or "").strip()
        difficulty = str(row.get("difficulty") or "").strip()
        query_type = str(row.get("query_type") or "").strip()
        conflict_case = str(row.get("conflict_case") or "").strip().lower()

        if category not in CATEGORY_ENUM:
            _fail(f"line {idx}: invalid category={category}")
        if difficulty not in DIFF_ENUM:
            _fail(f"line {idx}: invalid difficulty={difficulty}")
        if query_type not in TYPE_ENUM:
            _fail(f"line {idx}: invalid query_type={query_type}")
        if conflict_case not in CONFLICT_ENUM:
            _fail(f"line {idx}: invalid conflict_case={conflict_case}")

        cat_counter[category] += 1
        diff_counter_by_cat[category][difficulty] += 1

    for cat in sorted(CATEGORY_ENUM):
        if cat_counter[cat] != EXPECTED_PER_CATEGORY:
            _fail(f"category={cat} expected {EXPECTED_PER_CATEGORY}, got {cat_counter[cat]}")
        for diff, expected in EXPECTED_DIFF.items():
            got = diff_counter_by_cat[cat][diff]
            if got != expected:
                _fail(f"category={cat} difficulty={diff} expected {expected}, got {got}")

    print(f"[OK] {file_path} rows={len(rows)}")
    print(f"[OK] category distribution: {dict(cat_counter)}")
    print(f"[OK] difficulty by category: { {k: dict(v) for k, v in diff_counter_by_cat.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
