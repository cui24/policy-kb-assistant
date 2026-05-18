#!/usr/bin/env python3
"""对文档做粗读扫描：抽取页数、前几页文本和疑似目录/章节线索。"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pypdf import PdfReader


HEADING_PATTERNS = [
    re.compile(r"^\s*第[一二三四五六七八九十百零\d]+[章节条款]\s*"),
    re.compile(r"^\s*[0-9]+(\.[0-9]+){0,3}\s+"),
    re.compile(r"^\s*目录\s*$"),
]


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _extract_heading_candidates(text: str, limit: int = 12) -> list[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    picked: list[str] = []
    for line in lines:
        if len(line) < 4:
            continue
        if any(pattern.search(line) for pattern in HEADING_PATTERNS):
            picked.append(line[:120])
        if len(picked) >= limit:
            break
    return picked


def _safe_csv_value(value: object) -> str:
    text = str(value or "")
    return text.replace("\x00", " ").replace("\r", " ").replace("\n", " ")


def main() -> int:
    docs_csv = Path("data/eval/documents.csv")
    out_csv = Path("data/eval/document_quick_scan.csv")
    if not docs_csv.exists():
        raise SystemExit("missing data/eval/documents.csv")

    rows: list[dict[str, str]] = []
    with docs_csv.open("r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            source_file = str(row.get("source_file") or "").strip()
            source = Path(source_file)
            if not source.exists():
                rows.append(
                    {
                        "doc_id": str(row.get("doc_id") or ""),
                        "source_file": source_file,
                        "page_count": "",
                        "toc_like": "",
                        "heading_candidates": "",
                        "first_page_preview": "",
                        "scan_error": "file_not_found",
                    }
                )
                continue

            try:
                reader_pdf = PdfReader(str(source))
                page_count = len(reader_pdf.pages)
                joined_preview_pages: list[str] = []
                for idx in range(min(3, page_count)):
                    page_text = _clean(reader_pdf.pages[idx].extract_text() or "")
                    if page_text:
                        joined_preview_pages.append(page_text)

                joined_preview = "\n".join(joined_preview_pages)
                toc_like = "yes" if "目录" in joined_preview else "no"
                headings = _extract_heading_candidates(joined_preview)

                rows.append(
                    {
                        "doc_id": str(row.get("doc_id") or ""),
                        "source_file": source_file,
                        "page_count": str(page_count),
                        "toc_like": toc_like,
                        "heading_candidates": " | ".join(headings),
                        "first_page_preview": _clean(joined_preview[:500]),
                        "scan_error": "",
                    }
                )
            except Exception as exc:  # pragma: no cover
                rows.append(
                    {
                        "doc_id": str(row.get("doc_id") or ""),
                        "source_file": source_file,
                        "page_count": "",
                        "toc_like": "",
                        "heading_candidates": "",
                        "first_page_preview": "",
                        "scan_error": exc.__class__.__name__,
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as wf:
        fieldnames = [
            "doc_id",
            "source_file",
            "page_count",
            "toc_like",
            "heading_candidates",
            "first_page_preview",
            "scan_error",
        ]
        writer = csv.DictWriter(
            wf,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            doublequote=True,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe_csv_value(value) for key, value in row.items()})

    print(f"[OK] wrote {out_csv} ({len(rows)} docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
