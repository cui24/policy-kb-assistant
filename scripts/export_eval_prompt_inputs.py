#!/usr/bin/env python3
"""导出四类评测题生成输入包（元数据 + 快速扫描文本）。"""

from __future__ import annotations

import csv
from pathlib import Path


CATEGORY_MAP = {
    "finance": "fin",
    "hr": "hr",
    "admin": "adm",
    "it": "it",
}


def main() -> int:
    docs_path = Path("data/eval/documents.csv")
    scan_path = Path("data/eval/document_quick_scan.csv")
    out_dir = Path("data/eval/prompt_inputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    with docs_path.open("r", encoding="utf-8", newline="") as f:
        docs_rows = list(csv.DictReader(f))
    with scan_path.open("r", encoding="utf-8", newline="") as f:
        scan_rows = list(csv.DictReader(f))

    scan_by_doc = {row["doc_id"]: row for row in scan_rows}

    # docs-only metadata export per category
    for category in ("finance", "hr", "admin", "it"):
        rows = [r for r in docs_rows if r.get("category") == category]
        rows.sort(key=lambda r: r.get("doc_id", ""))
        docs_out = out_dir / f"{category}_documents.csv"
        with docs_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "doc_id",
                    "title",
                    "category",
                    "policy_type",
                    "doc_family",
                    "status",
                    "version",
                    "effective_date",
                    "family_id",
                    "authority_level",
                    "source_file",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        # merged prompt input export: metadata + scan snippets
        merged_out = out_dir / f"{category}_prompt_source.csv"
        merged_rows = []
        for r in rows:
            scan = scan_by_doc.get(r["doc_id"], {})
            merged_rows.append(
                {
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "status": r["status"],
                    "version": r["version"],
                    "effective_date": r["effective_date"],
                    "family_id": r["family_id"],
                    "policy_type": r["policy_type"],
                    "doc_family": r["doc_family"],
                    "source_file": r["source_file"],
                    "heading_candidates": str(scan.get("heading_candidates") or ""),
                    "first_page_preview": str(scan.get("first_page_preview") or ""),
                }
            )
        with merged_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "doc_id",
                    "title",
                    "status",
                    "version",
                    "effective_date",
                    "family_id",
                    "policy_type",
                    "doc_family",
                    "source_file",
                    "heading_candidates",
                    "first_page_preview",
                ],
            )
            writer.writeheader()
            writer.writerows(merged_rows)

    print(f"[OK] exported prompt input files to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
