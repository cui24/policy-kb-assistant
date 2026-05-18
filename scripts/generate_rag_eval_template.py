#!/usr/bin/env python3
"""生成 100 条 RAG 评测模板（4 类各 25 条，含难度和题型配比）。"""

from __future__ import annotations

import csv
from pathlib import Path


OUTPUT = Path("data/eval/rag_eval_set_v2_100_template.csv")

# 每个类别固定 25 条，题型配比总计 25
TYPE_PLAN = [
    ("fact", 8),
    ("procedure", 6),
    ("constraint", 4),
    ("comparison", 3),
    ("exception", 2),
    ("multi_hop", 2),
]

# 每个类别固定难度配比总计 25
DIFF_PLAN = [
    ("easy", 10),
    ("medium", 10),
    ("hard", 5),
]

CATEGORY_CONFIG = [
    ("HR", "hr"),
    ("ADM", "admin"),
    ("FIN", "finance"),
    ("IT", "it"),
]


def _expand_plan(plan: list[tuple[str, int]]) -> list[str]:
    result: list[str] = []
    for label, n in plan:
        result.extend([label] * n)
    return result


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    query_types = _expand_plan(TYPE_PLAN)
    difficulties = _expand_plan(DIFF_PLAN)
    if len(query_types) != 25 or len(difficulties) != 25:
        raise SystemExit("invalid plan length, must be 25 per category")

    rows: list[dict[str, str]] = []
    for short_code, category in CATEGORY_CONFIG:
        for i in range(25):
            idx = i + 1
            row = {
                "query_id": f"Q_{short_code}_{idx:03d}",
                "query": "",
                "category": category,
                "difficulty": difficulties[i],
                "query_type": query_types[i],
                "gold_doc_id": "",
                "gold_section": "",
                "answer_points": "",
                "as_of_date": "",
                "conflict_case": "no",
                "notes": "",
            }
            rows.append(row)

    with OUTPUT.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(
            wf,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] generated {OUTPUT} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
