#!/usr/bin/env python3
"""Lightweight CI quality gate for RAG metrics and Agent route evaluation.

This script is intentionally deterministic:
- RAG uses fixed in-memory fixtures to verify metric calculation and thresholds.
- Agent uses the existing workflow evaluator with patched services and rules mode.

The full end-to-end RAG evaluation remains in scripts/evaluate_policy_eval_set.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import evaluate_policy_eval_set as rag_eval
from src.api import planner_eval


RAG_FIXTURE_ROWS: list[dict[str, Any]] = [
    {
        "query_id": "ci-rag-001",
        "category": "it",
        "difficulty": "easy",
        "query_type": "fact",
        "query": "统一身份认证登录失败怎么办？",
        "gold_doc_id": "IT-01",
        "answer_points": "检查账号密码;联系 IT 服务台",
        "hits": [
            {"doc_id": "IT-01", "score": 0.96},
            {"doc_id": "HR-03", "score": 0.41},
            {"doc_id": "ADM-02", "score": 0.30},
        ],
        "answer": "请先检查账号密码是否正确，如仍无法登录，请联系 IT 服务台处理。",
        "citations": [{"doc_id": "IT-01"}],
        "retrieve_ms": 21,
        "answer_ms": 36,
    },
    {
        "query_id": "ci-rag-002",
        "category": "hr",
        "difficulty": "medium",
        "query_type": "policy",
        "query": "未经审批可以算加班吗？",
        "gold_doc_id": "HR-03",
        "answer_points": "事前审批;未经审批不计加班",
        "hits": [
            {"doc_id": "FIN-01", "score": 0.72},
            {"doc_id": "HR-03", "score": 0.68},
            {"doc_id": "ADM-06", "score": 0.35},
        ],
        "answer": "加班通常需要事前审批，未经审批不计加班。",
        "citations": [{"doc_id": "HR-03"}],
        "retrieve_ms": 25,
        "answer_ms": 42,
    },
    {
        "query_id": "ci-rag-003",
        "category": "finance",
        "difficulty": "medium",
        "query_type": "procedure",
        "query": "差旅报销需要哪些材料？",
        "gold_doc_id": "FIN-02",
        "answer_points": "发票;审批单",
        "hits": [
            {"doc_id": "FIN-02", "score": 0.91},
            {"doc_id": "FIN-05", "score": 0.59},
            {"doc_id": "ADM-01", "score": 0.20},
        ],
        "answer": "差旅报销通常需要发票和审批单，并按财务流程提交。",
        "citations": [{"doc_id": "FIN-02"}],
        "retrieve_ms": 18,
        "answer_ms": 40,
    },
    {
        "query_id": "ci-rag-004",
        "category": "admin",
        "difficulty": "hard",
        "query_type": "boundary",
        "query": "公司有没有火星出差补贴？",
        "gold_doc_id": "ADM-99",
        "answer_points": "证据不足",
        "hits": [
            {"doc_id": "FIN-07", "score": 0.44},
            {"doc_id": "ADM-03", "score": 0.39},
            {"doc_id": "HR-01", "score": 0.22},
        ],
        "answer": "证据不足：当前制度材料中未找到火星出差补贴规定。",
        "citations": [],
        "retrieve_ms": 17,
        "answer_ms": 34,
    },
]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _evaluate_rag_fixture() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in RAG_FIXTURE_ROWS:
        hits = list(item["hits"])
        rank = rag_eval._rank_of_gold(hits, str(item["gold_doc_id"]))
        answer_points = rag_eval._split_points(str(item.get("answer_points") or ""))
        answer = str(item.get("answer") or "")
        hit_n = sum(1 for point in answer_points if rag_eval._point_hit(answer, point))
        rows.append(
            {
                "query_id": item["query_id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "query_type": item["query_type"],
                "query": item["query"],
                "gold_doc_id": item["gold_doc_id"],
                "rank": rank,
                "doc_hit_at_3": bool(rank is not None and rank <= 3),
                "doc_hit_at_5": bool(rank is not None and rank <= 5),
                "top_docs": [str(hit.get("doc_id") or "") for hit in hits[:5]],
                "retrieve_ms": int(item["retrieve_ms"]),
                "answer": answer,
                "answer_point_total": len(answer_points),
                "answer_point_hit": hit_n,
                "answer_point_coverage": _safe_div(hit_n, len(answer_points)),
                "citations_n": len(item.get("citations") or []),
                "refused": answer.startswith("证据不足"),
                "answer_ms": int(item["answer_ms"]),
                "meta": {"fixture": True},
            }
        )

    summary = rag_eval._summarize_rows(rows, skip_answer=False)
    summary["metric_scope"] = {
        "gold_doc_recall": "document-level hit, not strict clause-level evidence hit",
        "auto_apc": "automatic answer point coverage, not final human accuracy",
        "citation_output": "citation presence rate, not citation correctness",
    }
    return {"summary": summary, "results": rows}


def _evaluate_agent(limit: int) -> dict[str, Any]:
    cases = planner_eval.planner.load_global_planner_regression_cases()
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise RuntimeError("missing agent regression cases")
    return planner_eval.evaluate_agent_workflow_cases(cases, strategy="rules")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, rag_report: dict[str, Any], agent_report: dict[str, Any], passed: bool) -> None:
    rag = rag_report["summary"]
    agent = agent_report["summary"]
    lines = [
        "# CI Quality Gate Report",
        "",
        f"Status: {'PASS' if passed else 'FAIL'}",
        "",
        "## RAG Fixture Metrics",
        "",
        f"- GoldDoc Recall@3: {rag['recall_at_3']:.3f}",
        f"- GoldDoc Recall@5: {rag['recall_at_5']:.3f}",
        f"- GoldDoc MRR: {rag['mrr']:.3f}",
        f"- Auto APC: {rag['answer_point_coverage_avg']:.3f}",
        f"- Citation Output Rate: {rag['citation_rate']:.3f}",
        f"- Refusal Rate: {rag['refusal_rate']:.3f}",
        f"- Retrieve p95 ms: {rag['retrieve_ms_p95']}",
        "",
        "## Agent Workflow Metrics",
        "",
        f"- Strategy: {agent['strategy']}",
        f"- Total Cases: {agent['total_cases']}",
        f"- Executed Cases: {agent['executed_case_count']}",
        f"- Route Accuracy: {agent['route_accuracy']:.3f}",
        f"- Clarification Matches: {agent['clarification_match_count']}",
        f"- Error Count: {agent['error_count']}",
        "",
        "## Metric Scope",
        "",
        "- GoldDoc Recall/MRR are document-level metrics, not strict evidence-clause metrics.",
        "- Auto APC is automatic answer point coverage, not final human accuracy.",
        "- Citation Output Rate only measures whether citations are present.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run lightweight CI quality gates.")
    parser.add_argument("--out-dir", default="outputs/ci", help="Output directory.")
    parser.add_argument("--agent-limit", type=int, default=12, help="First N agent eval cases.")
    parser.add_argument("--min-rag-r3", type=float, default=0.75)
    parser.add_argument("--min-rag-mrr", type=float, default=0.60)
    parser.add_argument("--min-rag-apc", type=float, default=0.70)
    parser.add_argument("--min-citation-rate", type=float, default=0.70)
    parser.add_argument("--min-agent-route-accuracy", type=float, default=0.80)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    rag_report = _evaluate_rag_fixture()
    agent_report = _evaluate_agent(limit=int(args.agent_limit))

    rag = rag_report["summary"]
    agent = agent_report["summary"]
    failures: list[str] = []
    if float(rag["recall_at_3"]) < float(args.min_rag_r3):
        failures.append(f"rag_recall_at_3<{args.min_rag_r3}")
    if float(rag["mrr"]) < float(args.min_rag_mrr):
        failures.append(f"rag_mrr<{args.min_rag_mrr}")
    if float(rag["answer_point_coverage_avg"]) < float(args.min_rag_apc):
        failures.append(f"rag_auto_apc<{args.min_rag_apc}")
    if float(rag["citation_rate"]) < float(args.min_citation_rate):
        failures.append(f"rag_citation_rate<{args.min_citation_rate}")
    if float(agent["route_accuracy"]) < float(args.min_agent_route_accuracy):
        failures.append(f"agent_route_accuracy<{args.min_agent_route_accuracy}")
    if int(agent["error_count"]) > 0:
        failures.append("agent_error_count>0")

    combined = {
        "passed": not failures,
        "failures": failures,
        "rag": rag_report,
        "agent": agent_report,
    }
    _write_json(out_dir / "quality_gate_latest.json", combined)
    _write_json(out_dir / "rag_fixture_latest.json", rag_report)
    _write_json(out_dir / "agent_eval_latest.json", agent_report)
    _write_markdown(out_dir / "quality_gate_latest.md", rag_report, agent_report, passed=not failures)

    if failures:
        print("[FAIL] CI quality gate:", ", ".join(failures))
        return 1

    print("[PASS] CI quality gate")
    print(f"[OK] outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
