#!/usr/bin/env python3
"""校验 documents.csv 列名与枚举值。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


REQUIRED_COLUMNS = [
    "doc_id",
    "title",
    "category",
    "policy_type",
    "doc_family",
    "source_file",
    "status",
    "version",
    "effective_date",
    "family_id",
    "authority_level",
    "notes",
]

CATEGORY_ENUM = {"hr", "admin", "finance", "it"}
STATUS_ENUM = {"active", "archived", "unknown"}
AUTHORITY_ENUM = {"company_policy"}

POLICY_TYPE_ENUM = {
    "employee_handbook",
    "hr_general_policy",
    "attendance_policy",
    "performance_policy",
    "compensation_policy",
    "admin_general_policy",
    "seal_management",
    "confidentiality_policy",
    "archive_management_policy",
    "asset_management",
    "procurement_policy",
    "contract_management",
    "travel_policy",
    "budget_policy",
    "financial_management",
    "it_general_policy",
    "account_permission_policy",
    "it_security_policy",
    "operation_maintenance_policy",
}

DOC_FAMILY_ENUM = {
    "employee_handbook",
    "employee_management",
    "attendance_management",
    "performance_management",
    "compensation_management",
    "administration",
    "seal_governance",
    "confidentiality_governance",
    "archive_governance",
    "asset_lifecycle",
    "procurement_governance",
    "contract_lifecycle",
    "travel_reimbursement",
    "budget_management",
    "financial_management",
    "it_governance",
    "account_permission",
    "it_security",
    "it_ops",
}


def _fail(message: str) -> None:
    raise SystemExit(f"[ERROR] {message}")


def main() -> int:
    path = Path("data/eval/documents.csv")
    if not path.exists():
        _fail(f"missing file: {path}")

    with path.open("r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        header = reader.fieldnames or []
        if header != REQUIRED_COLUMNS:
            _fail(f"invalid header: {header}")

        seen_doc_ids: set[str] = set()
        rows = list(reader)

    for idx, row in enumerate(rows, start=2):
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            _fail(f"line {idx}: empty doc_id")
        if doc_id in seen_doc_ids:
            _fail(f"line {idx}: duplicate doc_id={doc_id}")
        seen_doc_ids.add(doc_id)

        category = str(row.get("category") or "").strip()
        policy_type = str(row.get("policy_type") or "").strip()
        doc_family = str(row.get("doc_family") or "").strip()
        status = str(row.get("status") or "").strip()
        authority_level = str(row.get("authority_level") or "").strip()
        source_file = str(row.get("source_file") or "").strip()
        version = str(row.get("version") or "").strip()
        effective_date = str(row.get("effective_date") or "").strip()
        family_id = str(row.get("family_id") or "").strip()
        title = str(row.get("title") or "").strip()

        if category not in CATEGORY_ENUM:
            _fail(f"line {idx}: invalid category={category}")
        if policy_type not in POLICY_TYPE_ENUM:
            _fail(f"line {idx}: invalid policy_type={policy_type}")
        if doc_family not in DOC_FAMILY_ENUM:
            _fail(f"line {idx}: invalid doc_family={doc_family}")
        if status not in STATUS_ENUM:
            _fail(f"line {idx}: invalid status={status}")
        if authority_level not in AUTHORITY_ENUM:
            _fail(f"line {idx}: invalid authority_level={authority_level}")
        if not version:
            _fail(f"line {idx}: empty version")
        if effective_date:
            try:
                datetime.strptime(effective_date, "%Y-%m-%d")
            except ValueError:
                _fail(
                    f"line {idx}: invalid effective_date={effective_date}, expect YYYY-MM-DD"
                )
        if not family_id:
            _fail(f"line {idx}: empty family_id")
        if not title:
            _fail(f"line {idx}: empty title")
        if not source_file:
            _fail(f"line {idx}: empty source_file")

    print(f"[OK] {path} validated rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
