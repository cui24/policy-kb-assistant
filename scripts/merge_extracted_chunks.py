#!/usr/bin/env python3
"""合并 data/extracted 下所有 JSONL 为单个 CSV（不做严格校验）。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge extracted JSONL to CSV")
    parser.add_argument(
        "--input-dir",
        default="data/extracted",
        help="Directory containing *.jsonl files",
    )
    parser.add_argument(
        "--out-csv",
        default="data/extracted/all_chunks.csv",
        help="Output merged CSV path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(p for p in input_dir.rglob("*.jsonl") if p.is_file())
    if not files:
        raise SystemExit(f"[ERROR] no jsonl found under: {input_dir}")

    rows: list[dict[str, str]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as rf:
            for line_no, raw_line in enumerate(rf, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] skip invalid json {file_path}:{line_no}")
                    continue
                if not isinstance(record, dict):
                    continue
                rows.append(
                    {
                        "doc_id": str(record.get("doc_id") or ""),
                        "chunk_id": str(record.get("chunk_id") or ""),
                        "section_path": str(record.get("section_path") or ""),
                        "page_start": str(record.get("page_start") or ""),
                        "page_end": str(record.get("page_end") or ""),
                        "rule_type": str(record.get("rule_type") or ""),
                        "effective_date": str(record.get("effective_date") or ""),
                        "keywords": ";".join(
                            str(k).strip() for k in (record.get("keywords") or [])
                        ),
                        "evidence_span": str(record.get("evidence_span") or ""),
                        "text": str(record.get("text") or ""),
                        "source_jsonl": str(file_path),
                    }
                )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(
            wf,
            fieldnames=[
                "doc_id",
                "chunk_id",
                "section_path",
                "page_start",
                "page_end",
                "rule_type",
                "effective_date",
                "keywords",
                "evidence_span",
                "text",
                "source_jsonl",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] merged files={len(files)} rows={len(rows)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
