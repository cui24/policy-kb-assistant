#!/usr/bin/env python3
"""规范化 documents.csv：冻结列名并填充 policy_type/doc_family/family_id。"""

from __future__ import annotations

import csv
from pathlib import Path


TARGET_COLUMNS = [
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


DOC_ID_OVERRIDES: dict[str, tuple[str, str]] = {
    "HR-01": ("employee_handbook", "employee_handbook"),
    "HR-02": ("employee_handbook", "employee_handbook"),
    "HR-03": ("employee_handbook", "employee_handbook"),
    "HR-04": ("hr_general_policy", "employee_management"),
    "HR-05": ("hr_general_policy", "employee_management"),
    "HR-06": ("attendance_policy", "attendance_management"),
    "HR-07": ("performance_policy", "performance_management"),
    "HR-08": ("compensation_policy", "compensation_management"),
    "ADM-01": ("admin_general_policy", "administration"),
    "ADM-02": ("seal_management", "seal_governance"),
    "ADM-03": ("confidentiality_policy", "confidentiality_governance"),
    "ADM-04": ("archive_management_policy", "archive_governance"),
    "ADM-05": ("asset_management", "asset_lifecycle"),
    "ADM-06": ("admin_general_policy", "administration"),
    "ADM-07": ("procurement_policy", "procurement_governance"),
    "ADM-08": ("contract_management", "contract_lifecycle"),
    "FIN-01": ("financial_management", "financial_management"),
    "FIN-02": ("travel_policy", "travel_reimbursement"),
    "FIN-03": ("financial_management", "financial_management"),
    "FIN-04": ("financial_management", "financial_management"),
    "FIN-05": ("financial_management", "financial_management"),
    "FIN-06": ("financial_management", "financial_management"),
    "FIN-07": ("budget_policy", "budget_management"),
    "IT-01": ("it_general_policy", "it_governance"),
    "IT-02": ("it_general_policy", "it_governance"),
    "IT-03": ("account_permission_policy", "account_permission"),
    "IT-05": ("it_security_policy", "it_security"),
    "IT-06": ("it_security_policy", "it_security"),
    "IT-07": ("operation_maintenance_policy", "it_ops"),
}


def _fallback_mapping(*, category: str, title: str) -> tuple[str, str]:
    normalized_title = str(title or "")
    if category == "hr":
        if "手册" in normalized_title:
            return "employee_handbook", "employee_handbook"
        if "考勤" in normalized_title:
            return "attendance_policy", "attendance_management"
        if "绩效" in normalized_title:
            return "performance_policy", "performance_management"
        if "薪酬" in normalized_title:
            return "compensation_policy", "compensation_management"
        return "hr_general_policy", "employee_management"
    if category == "admin":
        if "印章" in normalized_title:
            return "seal_management", "seal_governance"
        if "保密" in normalized_title:
            return "confidentiality_policy", "confidentiality_governance"
        if "档案" in normalized_title:
            return "archive_management_policy", "archive_governance"
        if "固定资产" in normalized_title:
            return "asset_management", "asset_lifecycle"
        if "采购" in normalized_title or "招标" in normalized_title:
            return "procurement_policy", "procurement_governance"
        if "合同" in normalized_title:
            return "contract_management", "contract_lifecycle"
        return "admin_general_policy", "administration"
    if category == "finance":
        if "出差" in normalized_title or "差旅" in normalized_title:
            return "travel_policy", "travel_reimbursement"
        if "预算" in normalized_title:
            return "budget_policy", "budget_management"
        return "financial_management", "financial_management"
    if category == "it":
        if "账户" in normalized_title or "权限" in normalized_title:
            return "account_permission_policy", "account_permission"
        if "运维" in normalized_title:
            return "operation_maintenance_policy", "it_ops"
        if "安全" in normalized_title or "隐私" in normalized_title:
            return "it_security_policy", "it_security"
        return "it_general_policy", "it_governance"
    return "general_policy", "general"


def _build_default_family_id(*, category: str, doc_family: str, doc_id: str) -> str:
    category_prefix = {
        "hr": "HR",
        "admin": "ADM",
        "finance": "FIN",
        "it": "IT",
    }.get(category, "GEN")
    normalized_family = (
        doc_family.strip().upper().replace("-", "_").replace(" ", "_")
    )
    if normalized_family:
        return f"{category_prefix}_{normalized_family}"
    return f"{category_prefix}_{doc_id.replace('-', '_')}"


def main() -> int:
    path = Path("data/eval/documents.csv")
    if not path.exists():
        raise SystemExit(f"missing file: {path}")

    with path.open("r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        source_rows = list(reader)

    normalized_rows: list[dict[str, str]] = []
    for row in source_rows:
        doc_id = str(row.get("doc_id") or "").strip()
        title = str(row.get("title") or "").strip()
        category = str(row.get("category") or "").strip()

        mapped = DOC_ID_OVERRIDES.get(doc_id)
        if mapped is None:
            mapped = _fallback_mapping(category=category, title=title)
        policy_type, doc_family = mapped

        normalized_rows.append(
            {
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "policy_type": policy_type,
                "doc_family": doc_family,
                "source_file": str(row.get("source_file") or "").strip(),
                "status": str(row.get("status") or "active").strip() or "active",
                "version": str(row.get("version") or "v1.0").strip() or "v1.0",
                "effective_date": str(row.get("effective_date") or "").strip(),
                "family_id": (
                    str(row.get("family_id") or "").strip()
                    or _build_default_family_id(
                        category=category, doc_family=doc_family, doc_id=doc_id
                    )
                ),
                "authority_level": str(row.get("authority_level") or "company_policy").strip() or "company_policy",
                "notes": str(row.get("notes") or "").strip(),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"[OK] normalized {path} rows={len(normalized_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
