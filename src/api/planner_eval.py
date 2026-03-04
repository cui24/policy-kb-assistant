"""
Global Planner 评测：批量跑本地回归集，统计命中率与 fallback 率。

一、程序目标
1. 对 Global Planner 做离线批量回归，观察分支选择是否稳定。
2. 在 `rules / llm / hybrid` 三种策略下统计命中率、planner 有效率与 fallback 率。
3. 保持评测只作用于规划层，不改执行层。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import re
from typing import Any, Literal

from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import crud, planner, services
from src.api.db import Base
from src.api.planner import PlannerError
from src.api.schemas import (
    ContinueTicketDraftPlanArgs,
    CreateTicketPlanArgs,
    KBAnswerPlanArgs,
    TicketToolPlannerPlanArgs,
    ToolPlan,
)


EvalStrategy = Literal["rules", "llm", "hybrid"]
EvalLevel = Literal["planner", "workflow"]
_TICKET_TOOL_NAMES = {"lookup_ticket", "add_ticket_comment", "escalate_ticket", "cancel_ticket", "ticket_tool_planner"}
_DRAFT_ID_RE = re.compile(r"(DRF-\d{4}-[A-Z0-9]+|DRAFT-[A-Z0-9-]+)")


def _extract_first_match(pattern: re.Pattern[str], text: str) -> str:
    """从文本中取第一个匹配值；无命中则返回空串。"""
    matched = pattern.search(text or "")
    if matched is None:
        return ""
    return matched.group(0)


def build_eval_context(case: dict[str, Any]) -> dict[str, Any]:
    """根据回归样例构造 Global Planner 所需的最小上下文。"""
    utterance = str(case.get("utterance") or "")
    provided_ticket_id = str(case.get("provided_ticket_id") or "") or str(services._extract_ticket_public_id(utterance) or "")
    provided_draft_id = str(case.get("provided_draft_id") or "") or _extract_first_match(_DRAFT_ID_RE, utterance)

    has_ticket_id = bool(case.get("has_ticket_id")) if "has_ticket_id" in case else bool(provided_ticket_id)
    has_draft_id = bool(case.get("has_draft_id")) if "has_draft_id" in case else bool(provided_draft_id)
    draft_mode = bool(case.get("draft_mode")) if "draft_mode" in case else has_draft_id
    ticket_tool_mode = (
        bool(case.get("ticket_tool_mode")) if "ticket_tool_mode" in case else has_ticket_id
    )

    return {
        "actor_user_id": str(case.get("actor_user_id") or "eval_runner"),
        "provided_ticket_id": provided_ticket_id,
        "provided_draft_id": provided_draft_id,
        "has_ticket_id": has_ticket_id,
        "has_draft_id": has_draft_id,
        "confirm_token_present": False,
        "ticket_tool_mode": ticket_tool_mode,
        "draft_mode": draft_mode,
    }


def expected_global_tool(case: dict[str, Any]) -> str:
    """把样例中的期望工具归一化成 Global Planner 需要命中的分支工具。"""
    expected_tool = str(case.get("expected_tool") or "")
    if expected_tool in _TICKET_TOOL_NAMES:
        return "ticket_tool_planner"
    return expected_tool or "create_ticket"


def _rules_global_plan(utterance: str, context: dict[str, Any]) -> ToolPlan:
    """复用当前规则语义，生成 Global Planner 的基线 plan。"""
    if bool(context.get("draft_mode")) or bool(context.get("has_draft_id")):
        return ToolPlan(
            tool="continue_ticket_draft",
            args={"draft_id": str(context.get("provided_draft_id") or ""), "fields": {}},
            need_confirmation=False,
            missing_fields=[],
        )
    if bool(context.get("ticket_tool_mode")) or bool(context.get("has_ticket_id")):
        return ToolPlan(
            tool="ticket_tool_planner",
            args={
                "ticket_id": str(context.get("provided_ticket_id") or ""),
                "raw_text": utterance,
            },
            need_confirmation=False,
            missing_fields=[],
        )
    if not services._should_route_to_ticket(utterance):
        return ToolPlan(
            tool="kb_answer",
            args={"query": utterance},
            need_confirmation=False,
            missing_fields=[],
        )
    return ToolPlan(
        tool="create_ticket",
        args={"text": utterance, "fields": {}},
        need_confirmation=False,
        missing_fields=[],
    )


def _global_plan_args_model(tool_name: str):
    """返回 Global Planner 分支工具的参数模型。"""
    return {
        "continue_ticket_draft": ContinueTicketDraftPlanArgs,
        "ticket_tool_planner": TicketToolPlannerPlanArgs,
        "kb_answer": KBAnswerPlanArgs,
        "create_ticket": CreateTicketPlanArgs,
    }.get(tool_name)


def _dump_pydantic_model(model_instance) -> dict[str, Any]:
    """兼容 Pydantic v1/v2 的字典导出。"""
    if hasattr(model_instance, "model_dump"):
        return model_instance.model_dump(exclude_none=True)
    return model_instance.dict(exclude_none=True)


def validate_global_plan_for_eval(
    plan: ToolPlan,
    utterance: str,
    context: dict[str, Any],
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """用与运行时一致的最小规则校验全局分支 plan。"""
    tool_names = {item["name"] for item in services.list_global_planner_skill_contracts()}
    if plan.tool not in tool_names:
        return False, "tool_not_in_global_registry", None

    if plan.missing_fields:
        return False, "missing_fields_from_plan", None

    model_cls = _global_plan_args_model(plan.tool)
    if model_cls is None:
        return False, "unsupported_tool", None

    try:
        normalized_args = _dump_pydantic_model(services._validate_pydantic_model(model_cls, plan.args or {}))
    except ValidationError:
        return False, "schema_invalid", None

    provided_ticket_id = str(context.get("provided_ticket_id") or "")
    provided_draft_id = str(context.get("provided_draft_id") or "")

    if plan.tool == "ticket_tool_planner":
        if not provided_ticket_id:
            return False, "ticket_id_missing", None
        if str(normalized_args.get("ticket_id") or "") != provided_ticket_id:
            return False, "ticket_id_mismatch", None
        normalized_args["raw_text"] = utterance
    elif plan.tool == "continue_ticket_draft":
        if not provided_draft_id:
            return False, "draft_id_missing", None
        if str(normalized_args.get("draft_id") or "") != provided_draft_id:
            return False, "draft_id_mismatch", None
    elif plan.tool == "kb_answer":
        normalized_args["query"] = utterance
    elif plan.tool == "create_ticket":
        normalized_args["text"] = utterance

    return True, None, normalized_args


def _build_eval_session() -> Session:
    """创建评测专用内存数据库，避免污染开发库。"""
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return local_session()


@contextmanager
def _patched_eval_agent_dependencies():
    """给 workflow 评测装入稳定桩，避免真实检索/抽取依赖。"""
    original_retrieve = services.retrieve
    original_answer_with_citations = services.answer_with_citations
    original_extract_ticket_payload = services.extract_ticket_payload

    def _fake_extract_ticket_payload(text: str, user: str, department: str) -> dict[str, Any]:
        return {
            "creator": user,
            "department": department,
            "category": "network",
            "priority": "P1",
            "title": "评测工单",
            "description": text or "补充信息",
            "contact": None,
            "location": None,
            "missing_fields": ["location", "contact"],
            "extractor": "planner_eval_stub",
        }

    services.retrieve = lambda text: [  # type: ignore[assignment]
        {
            "doc_id": "planner_eval_stub_doc",
            "page": 1,
            "score": 0.9,
            "snippet": "评测桩：缺地点和联系方式时应继续追问。",
        }
    ]
    services.answer_with_citations = lambda text, hits: {  # type: ignore[assignment]
        "answer": "这是评测桩答案。",
        "citations": list(hits or []),
        "meta": {
            "attempt_stage": "planner_eval_stub",
            "json_ok": True,
            "repair_used": False,
            "failure_reason": None,
        },
    }
    services.extract_ticket_payload = _fake_extract_ticket_payload  # type: ignore[assignment]
    try:
        yield
    finally:
        services.retrieve = original_retrieve  # type: ignore[assignment]
        services.answer_with_citations = original_answer_with_citations  # type: ignore[assignment]
        services.extract_ticket_payload = original_extract_ticket_payload  # type: ignore[assignment]


@contextmanager
def _temporary_agent_planner_mode(mode: EvalStrategy):
    """在评测期间临时切换 Agent Planner 模式。"""
    previous_value = os.environ.get("AGENT_PLANNER_MODE")
    os.environ["AGENT_PLANNER_MODE"] = mode
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop("AGENT_PLANNER_MODE", None)
        else:
            os.environ["AGENT_PLANNER_MODE"] = previous_value


def _seed_eval_ticket(db: Session, ticket_id: str, actor: str, department: str) -> None:
    """按 public_id 预置一张评测工单。"""
    if not ticket_id or crud.get_ticket_by_public_id(db, ticket_id) is not None:
        return
    crud.create_ticket(
        db,
        {
            "public_id": ticket_id,
            "creator": actor,
            "department": department,
            "category": "network",
            "priority": "P1",
            "title": "评测工单",
            "description": "用于 workflow 评测。",
            "status": "open",
            "contact": "13800000000",
            "context_json": {"location": "3栋402"},
        },
    )


def _seed_eval_draft(db: Session, draft_id: str, actor: str, department: str) -> None:
    """按 draft_id 预置一张评测草稿。"""
    if not draft_id or crud.get_ticket_draft_by_draft_id(db, draft_id) is not None:
        return
    crud.create_ticket_draft(
        db,
        {
            "draft_id": draft_id,
            "creator": actor,
            "owner_user_id": actor,
            "department": department,
            "payload_json": {
                "creator": actor,
                "department": department,
                "category": "network",
                "priority": "P1",
                "title": "评测草稿",
                "description": "用于 workflow 评测。",
                "contact": None,
                "location": None,
            },
            "missing_fields_json": ["location", "contact"],
            "status": "open",
            "expires_at": services._draft_expiry(),
            "kb_request_id": "eval_draft_chain",
        },
    )


def _seed_eval_case_state(
    db: Session,
    case: dict[str, Any],
    context: dict[str, Any],
) -> None:
    """按样例中显式对象预置工单或草稿。"""
    actor = str(context.get("actor_user_id") or "eval_runner")
    department = str(case.get("department") or "IT")
    provided_ticket_id = str(context.get("provided_ticket_id") or "")
    provided_draft_id = str(context.get("provided_draft_id") or "")
    if provided_ticket_id:
        _seed_eval_ticket(db, provided_ticket_id, actor, department)
    if provided_draft_id:
        _seed_eval_draft(db, provided_draft_id, actor, department)


def _workflow_route_matches_case(
    case: dict[str, Any],
    context: dict[str, Any],
    response: dict[str, Any],
) -> tuple[bool, str]:
    """按最终 Agent route 判断是否满足该样例预期。"""
    expected_tool = str(case.get("expected_tool") or "")
    route = str(response.get("route") or "")
    has_object_reference = bool(context.get("provided_ticket_id")) or bool(context.get("provided_draft_id"))

    if expected_tool == "kb_answer":
        return route == "ASK", "kb_answer_route_match" if route == "ASK" else "kb_answer_route_mismatch"

    if expected_tool == "create_ticket":
        matched = route in {"CREATE_TICKET", "NEED_MORE_INFO"}
        return matched, "create_ticket_route_match" if matched else "create_ticket_route_mismatch"

    if expected_tool == "continue_ticket_draft":
        matched = route in {"CREATE_TICKET", "NEED_MORE_INFO", "DRAFT_EXPIRED"}
        if matched:
            if route == "NEED_MORE_INFO" and not has_object_reference:
                return True, "continue_draft_clarification_match"
            return True, "continue_draft_route_match"
        return False, "continue_draft_route_mismatch"

    if expected_tool == "lookup_ticket":
        if route == "LOOKUP_TICKET":
            return True, "lookup_route_match"
        if route == "NEED_MORE_INFO" and not has_object_reference:
            return True, "lookup_clarification_match"
        return False, "lookup_route_mismatch"

    if expected_tool == "add_ticket_comment":
        if route == "ADD_TICKET_COMMENT":
            return True, "comment_route_match"
        if route == "NEED_MORE_INFO" and not has_object_reference:
            return True, "comment_clarification_match"
        return False, "comment_route_mismatch"

    if expected_tool == "escalate_ticket":
        if route == "ESCALATE_TICKET":
            return True, "escalate_route_match"
        if route == "NEED_MORE_INFO" and not has_object_reference:
            return True, "escalate_clarification_match"
        return False, "escalate_route_mismatch"

    if expected_tool == "cancel_ticket":
        if route in {"NEED_CONFIRMATION", "CANCEL_TICKET"}:
            return True, "cancel_route_match"
        if route == "NEED_MORE_INFO" and not has_object_reference:
            return True, "cancel_clarification_match"
        return False, "cancel_route_mismatch"

    if expected_tool == "ticket_tool_planner":
        if route in {"LOOKUP_TICKET", "ADD_TICKET_COMMENT", "ESCALATE_TICKET", "CANCEL_TICKET", "NEED_CONFIRMATION"}:
            return True, "ticket_tool_route_match"
        if route == "NEED_MORE_INFO" and not has_object_reference:
            return True, "ticket_tool_clarification_match"
        return False, "ticket_tool_route_mismatch"

    return False, "unsupported_expected_tool"


def evaluate_agent_workflow_cases(
    cases: list[dict[str, Any]],
    strategy: EvalStrategy = "hybrid",
) -> dict[str, Any]:
    """批量评测 `run_agent_workflow(...)` 的最终用户态效果。"""
    rows: list[dict[str, Any]] = []
    executed_case_count = 0
    route_match_count = 0
    clarification_match_count = 0
    error_count = 0

    with _temporary_agent_planner_mode(strategy), _patched_eval_agent_dependencies():
        for index, case in enumerate(cases, start=1):
            utterance = str(case.get("utterance") or "")
            context = build_eval_context(case)
            db = _build_eval_session()
            response: dict[str, Any] | None = None
            response_route: str | None = None
            failure_reason: str | None = None
            matched = False
            match_reason = ""

            try:
                _seed_eval_case_state(db, case, context)
                response = services.run_agent_workflow(
                    db,
                    text=utterance,
                    user=str(context.get("actor_user_id") or "eval_runner"),
                    department=str(case.get("department") or "IT"),
                    draft_id=str(context.get("provided_draft_id") or "") or None,
                    fields=case.get("fields") if isinstance(case.get("fields"), dict) else None,
                    confirm_token=str(case.get("confirm_token") or "") or None,
                )
                executed_case_count += 1
                response_route = str(response.get("route") or "")
                matched, match_reason = _workflow_route_matches_case(case, context, response)
                if matched:
                    route_match_count += 1
                    if "clarification" in match_reason:
                        clarification_match_count += 1
            except Exception as exc:
                error_count += 1
                failure_reason = f"{type(exc).__name__}:{exc}"
                match_reason = "execution_error"
            finally:
                db.close()

            rows.append(
                {
                    "case_index": index,
                    "utterance": utterance,
                    "category": str(case.get("category") or ""),
                    "expected_tool": str(case.get("expected_tool") or ""),
                    "expected_global_tool": expected_global_tool(case),
                    "response_route": response_route,
                    "matched": matched,
                    "match_reason": match_reason,
                    "failure_reason": failure_reason,
                    "response_summary": {
                        "route": response_route,
                        "message": str((response or {}).get("message") or "") or None,
                        "ticket_id": str(((response or {}).get("ticket") or {}).get("ticket_id") or "") or None,
                        "draft_id": str(((response or {}).get("draft") or {}).get("draft_id") or "") or None,
                    },
                    "notes": str(case.get("notes") or ""),
                }
            )

    total_cases = len(cases)
    route_accuracy = (route_match_count / executed_case_count) if executed_case_count else 0.0

    return {
        "summary": {
            "evaluation_level": "workflow",
            "strategy": strategy,
            "total_cases": total_cases,
            "executed_case_count": executed_case_count,
            "route_match_count": route_match_count,
            "route_accuracy": round(route_accuracy, 4),
            "clarification_match_count": clarification_match_count,
            "error_count": error_count,
        },
        "results": rows,
    }


def evaluate_global_planner_cases(
    cases: list[dict[str, Any]],
    strategy: EvalStrategy = "hybrid",
) -> dict[str, Any]:
    """批量评测 Global Planner，并返回结构化结果与摘要。"""
    tools_json = services.list_global_planner_skill_contracts()
    rows: list[dict[str, Any]] = []

    planner_attempted_cases = 0
    planner_returned_plan_count = 0
    planner_valid_plan_count = 0
    planner_branch_match_count = 0
    effective_valid_plan_count = 0
    effective_branch_match_count = 0
    fallback_count = 0
    planner_error_count = 0
    validation_failure_count = 0

    for index, case in enumerate(cases, start=1):
        utterance = str(case.get("utterance") or "")
        context = build_eval_context(case)
        expected_tool_name = expected_global_tool(case)

        planner_plan: ToolPlan | None = None
        planner_valid = False
        planner_failure_reason: str | None = None
        normalized_planner_args: dict[str, Any] | None = None
        fallback_used = False
        final_plan: ToolPlan | None = None

        if strategy == "rules":
            final_plan = _rules_global_plan(utterance, context)
            planner_valid = True
        else:
            planner_attempted_cases += 1
            try:
                planner_plan = planner.run_global_planner(
                    user_text=utterance,
                    tools_json=tools_json,
                    context=context,
                )
                planner_returned_plan_count += 1
                planner_valid, planner_failure_reason, normalized_planner_args = validate_global_plan_for_eval(
                    planner_plan,
                    utterance,
                    context,
                )
            except PlannerError as exc:
                planner_error_count += 1
                planner_failure_reason = exc.code

            if planner_valid and planner_plan is not None:
                planner_valid_plan_count += 1
                if planner_plan.tool == expected_tool_name:
                    planner_branch_match_count += 1
                final_plan = ToolPlan(
                    tool=planner_plan.tool,
                    args=normalized_planner_args or dict(planner_plan.args or {}),
                    need_confirmation=planner_plan.need_confirmation,
                    missing_fields=list(planner_plan.missing_fields or []),
                )
            else:
                if planner_failure_reason is not None and planner_failure_reason != "":
                    validation_failure_count += 0 if planner_failure_reason in {"missing_api_key", "llm_call_failed", "global_repair_failed"} else 1
                if strategy == "hybrid":
                    fallback_used = True
                    fallback_count += 1
                    final_plan = _rules_global_plan(utterance, context)

        final_valid = False
        final_failure_reason: str | None = None
        normalized_final_args: dict[str, Any] | None = None
        if final_plan is not None:
            final_valid, final_failure_reason, normalized_final_args = validate_global_plan_for_eval(
                final_plan,
                utterance,
                context,
            )
            if final_valid:
                effective_valid_plan_count += 1
                if final_plan.tool == expected_tool_name:
                    effective_branch_match_count += 1

        rows.append(
            {
                "case_index": index,
                "utterance": utterance,
                "category": str(case.get("category") or ""),
                "expected_tool": str(case.get("expected_tool") or ""),
                "expected_global_tool": expected_tool_name,
                "planner_tool": planner_plan.tool if planner_plan is not None else None,
                "planner_valid": planner_valid,
                "planner_failure_reason": planner_failure_reason,
                "planner_matched": bool(planner_plan is not None and planner_valid and planner_plan.tool == expected_tool_name),
                "fallback_used": fallback_used,
                "final_tool": final_plan.tool if final_plan is not None else None,
                "final_valid": final_valid,
                "final_failure_reason": final_failure_reason,
                "final_matched": bool(final_plan is not None and final_valid and final_plan.tool == expected_tool_name),
                "normalized_final_args": normalized_final_args,
                "notes": str(case.get("notes") or ""),
            }
        )

    total_cases = len(cases)
    planner_branch_accuracy = (
        planner_branch_match_count / planner_valid_plan_count if planner_valid_plan_count else 0.0
    )
    effective_branch_accuracy = (
        effective_branch_match_count / effective_valid_plan_count if effective_valid_plan_count else 0.0
    )
    fallback_rate = (fallback_count / total_cases) if total_cases else 0.0

    return {
        "summary": {
            "evaluation_level": "planner",
            "strategy": strategy,
            "total_cases": total_cases,
            "planner_attempted_cases": planner_attempted_cases,
            "planner_returned_plan_count": planner_returned_plan_count,
            "planner_valid_plan_count": planner_valid_plan_count,
            "planner_branch_match_count": planner_branch_match_count,
            "planner_branch_accuracy": round(planner_branch_accuracy, 4),
            "effective_valid_plan_count": effective_valid_plan_count,
            "effective_branch_match_count": effective_branch_match_count,
            "effective_branch_accuracy": round(effective_branch_accuracy, 4),
            "fallback_count": fallback_count,
            "fallback_rate": round(fallback_rate, 4),
            "planner_error_count": planner_error_count,
            "validation_failure_count": validation_failure_count,
        },
        "results": rows,
    }


def _print_text_report(report: dict[str, Any], show_mismatches: int = 10) -> None:
    """打印紧凑文本版评测结果。"""
    summary = report.get("summary") or {}
    results = report.get("results") or []
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    evaluation_level = str(summary.get("evaluation_level") or "planner")
    mismatch_key = "matched" if evaluation_level == "workflow" else "final_matched"
    mismatches = [item for item in results if not item.get(mismatch_key)]
    if not mismatches:
        print("\nNo mismatches.")
        return

    print(f"\nMismatches (showing up to {show_mismatches}):")
    for item in mismatches[: max(0, show_mismatches)]:
        if evaluation_level == "workflow":
            payload = {
                "case_index": item.get("case_index"),
                "utterance": item.get("utterance"),
                "expected_tool": item.get("expected_tool"),
                "response_route": item.get("response_route"),
                "match_reason": item.get("match_reason"),
                "failure_reason": item.get("failure_reason"),
            }
        else:
            payload = {
                "case_index": item.get("case_index"),
                "utterance": item.get("utterance"),
                "expected_global_tool": item.get("expected_global_tool"),
                "planner_tool": item.get("planner_tool"),
                "final_tool": item.get("final_tool"),
                "fallback_used": item.get("fallback_used"),
                "planner_failure_reason": item.get("planner_failure_reason"),
            }
        print(json.dumps(payload, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="Evaluate Global Planner on local regression cases.")
    parser.add_argument(
        "--level",
        choices=("workflow", "planner"),
        default="workflow",
        help="Evaluation level. Default: workflow.",
    )
    parser.add_argument(
        "--strategy",
        choices=("rules", "llm", "hybrid"),
        default="hybrid",
        help="Evaluation strategy. Default: hybrid.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=10,
        help="How many mismatches to print in text mode. Default: 10.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save full JSON report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N cases. Default: 0 (all cases).",
    )
    args = parser.parse_args(argv)

    cases = planner.load_global_planner_regression_cases()
    if not cases:
        print("No regression cases found.")
        return 1
    if args.limit > 0:
        cases = cases[: args.limit]

    if args.level == "planner":
        report = evaluate_global_planner_cases(cases, strategy=args.strategy)
    else:
        report = evaluate_agent_workflow_cases(cases, strategy=args.strategy)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text_report(report, show_mismatches=args.show_mismatches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
