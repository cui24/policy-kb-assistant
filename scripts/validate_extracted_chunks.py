#!/usr/bin/env python3
"""校验 LLM 抽取的结构化片段 JSONL。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ALLOWED_RULE_TYPES = {
    "fact",
    "procedure",
    "constraint",
    "comparison",
    "exception",
    "multi_hop",
    "definition",
    "checklist",
}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class DocMeta:
    category: str
    title: str
    page_count: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extracted JSONL chunks")
    parser.add_argument(
        "--input",
        required=True,
        help="Input jsonl file or directory (recursive *.jsonl)",
    )
    parser.add_argument(
        "--documents",
        default="data/eval/documents.csv",
        help="documents.csv path",
    )
    parser.add_argument(
        "--quickscan",
        default="data/eval/document_quick_scan.csv",
        help="quick scan csv path (for page_count check)",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=30,
        help="Minimum text length",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=500,
        help="Maximum text length (0 to disable)",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional merged CSV output path",
    )
    return parser.parse_args()


def load_doc_meta(documents_path: Path, quickscan_path: Path) -> dict[str, DocMeta]:
    page_count_by_doc: dict[str, int] = {}
    if quickscan_path.exists():
        with quickscan_path.open("r", encoding="utf-8", newline="") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                doc_id = str(row.get("doc_id") or "").strip()
                raw_page_count = str(row.get("page_count") or "").strip()
                if not doc_id or not raw_page_count.isdigit():
                    continue
                page_count_by_doc[doc_id] = int(raw_page_count)

    doc_meta: dict[str, DocMeta] = {}
    with documents_path.open("r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            doc_id = str(row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            doc_meta[doc_id] = DocMeta(
                category=str(row.get("category") or "").strip(),
                title=str(row.get("title") or "").strip(),
                page_count=page_count_by_doc.get(doc_id),
            )
    return doc_meta


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.jsonl") if p.is_file())
    return []


def validate_record(
    *,
    record: dict,
    line_no: int,
    file_path: Path,
    doc_meta: dict[str, DocMeta],
    min_text_len: int,
    max_text_len: int,
    seen_chunk_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    required = [
        "doc_id",
        "chunk_id",
        "section_path",
        "page_start",
        "page_end",
        "text",
        "keywords",
        "rule_type",
        "effective_date",
        "evidence_span",
    ]
    for key in required:
        if key not in record:
            errors.append(f"{file_path}:{line_no} missing field: {key}")

    doc_id = str(record.get("doc_id") or "").strip()
    if not doc_id:
        errors.append(f"{file_path}:{line_no} empty doc_id")
    elif doc_id not in doc_meta:
        errors.append(f"{file_path}:{line_no} unknown doc_id={doc_id}")

    chunk_id = str(record.get("chunk_id") or "").strip()
    if not chunk_id:
        errors.append(f"{file_path}:{line_no} empty chunk_id")
    elif chunk_id in seen_chunk_ids:
        errors.append(f"{file_path}:{line_no} duplicate chunk_id={chunk_id}")
    else:
        seen_chunk_ids.add(chunk_id)

    section_path = str(record.get("section_path") or "").strip()
    if not section_path:
        errors.append(f"{file_path}:{line_no} empty section_path")

    def _to_int(value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    page_start = _to_int(record.get("page_start"))
    page_end = _to_int(record.get("page_end"))
    if page_start is None or page_start < 1:
        errors.append(f"{file_path}:{line_no} invalid page_start={record.get('page_start')}")
    if page_end is None or page_end < 1:
        errors.append(f"{file_path}:{line_no} invalid page_end={record.get('page_end')}")
    if page_start is not None and page_end is not None and page_end < page_start:
        errors.append(f"{file_path}:{line_no} page_end < page_start")

    if doc_id in doc_meta and page_end is not None:
        max_page = doc_meta[doc_id].page_count
        if max_page is not None and page_end > max_page:
            errors.append(
                f"{file_path}:{line_no} page_end={page_end} exceed page_count={max_page}"
            )

    text = str(record.get("text") or "").strip()
    if len(text) < min_text_len:
        errors.append(
            f"{file_path}:{line_no} text too short len={len(text)} (<{min_text_len})"
        )
    if max_text_len > 0 and len(text) > max_text_len:
        errors.append(
            f"{file_path}:{line_no} text too long len={len(text)} (>{max_text_len})"
        )

    keywords = record.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        errors.append(f"{file_path}:{line_no} keywords must be non-empty list")
    elif any(not str(item).strip() for item in keywords):
        errors.append(f"{file_path}:{line_no} keywords contains empty item")
    else:
        for kw in keywords:
            kw_text = str(kw).strip()
            if kw_text and kw_text not in text:
                errors.append(
                    f"{file_path}:{line_no} keyword not found in text: {kw_text}"
                )

    rule_type = str(record.get("rule_type") or "").strip()
    if rule_type not in ALLOWED_RULE_TYPES:
        errors.append(f"{file_path}:{line_no} invalid rule_type={rule_type}")

    effective_date = str(record.get("effective_date") or "").strip()
    if effective_date and not DATE_PATTERN.fullmatch(effective_date):
        errors.append(
            f"{file_path}:{line_no} invalid effective_date={effective_date} (expect YYYY-MM-DD or empty)"
        )

    evidence_span = str(record.get("evidence_span") or "").strip()
    if len(evidence_span) < 8:
        errors.append(f"{file_path}:{line_no} evidence_span too short")
    elif evidence_span not in text:
        errors.append(f"{file_path}:{line_no} evidence_span not found in text")

    return errors


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    documents_path = Path(args.documents)
    quickscan_path = Path(args.quickscan)

    if not documents_path.exists():
        raise SystemExit(f"[ERROR] missing documents csv: {documents_path}")

    files = iter_input_files(input_path)
    if not files:
        raise SystemExit(f"[ERROR] no jsonl files found from: {input_path}")

    doc_meta = load_doc_meta(documents_path, quickscan_path)
    errors: list[str] = []
    seen_chunk_ids: set[str] = set()
    merged_rows: list[dict[str, str]] = []

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as rf:
            for line_no, raw_line in enumerate(rf, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{file_path}:{line_no} invalid json: {exc}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{file_path}:{line_no} record must be object")
                    continue

                row_errors = validate_record(
                    record=record,
                    line_no=line_no,
                    file_path=file_path,
                    doc_meta=doc_meta,
                    min_text_len=args.min_text_len,
                    max_text_len=args.max_text_len,
                    seen_chunk_ids=seen_chunk_ids,
                )
                errors.extend(row_errors)

                if not row_errors:
                    keywords = record.get("keywords") or []
                    merged_rows.append(
                        {
                            "doc_id": str(record.get("doc_id") or ""),
                            "chunk_id": str(record.get("chunk_id") or ""),
                            "section_path": str(record.get("section_path") or ""),
                            "page_start": str(record.get("page_start") or ""),
                            "page_end": str(record.get("page_end") or ""),
                            "rule_type": str(record.get("rule_type") or ""),
                            "effective_date": str(record.get("effective_date") or ""),
                            "keywords": ";".join(str(k).strip() for k in keywords),
                            "evidence_span": str(record.get("evidence_span") or ""),
                            "text": str(record.get("text") or ""),
                        }
                    )

    if errors:
        print(f"[ERROR] validation failed, errors={len(errors)}")
        for err in errors[:200]:
            print(f"- {err}")
        if len(errors) > 200:
            print(f"... truncated {len(errors)-200} more errors")
        raise SystemExit(1)

    print(
        f"[OK] validated files={len(files)} records={len(merged_rows)} unique_chunk_ids={len(seen_chunk_ids)}"
    )

    if args.out_csv:
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
                ],
            )
            writer.writeheader()
            writer.writerows(merged_rows)
        print(f"[OK] merged csv written: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
