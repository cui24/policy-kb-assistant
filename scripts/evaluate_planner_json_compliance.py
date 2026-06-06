#!/usr/bin/env python3
"""Evaluate Agent Planner JSON compliance across A/B/C experiment groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from src.api import planner
from src.api.schemas import (
    AddTicketCommentPlanArgs,
    CancelTicketPlanArgs,
    ContinueTicketDraftPlanArgs,
    CreateTicketPlanArgs,
    EscalateTicketPlanArgs,
    KBAnswerPlanArgs,
    LookupTicketPlanArgs,
    TicketToolPlannerPlanArgs,
    ToolPlan,
)


DEFAULT_CASES_PATH = Path("data/agent/planner_json_compliance_cases.jsonl")
DEFAULT_OUTPUT_PATH = Path("outputs/planner_json_compliance/report_latest.json")
_TICKET_ID_RE = re.compile(r"TCK-\d{4}-[A-Z0-9]+")


GLOBAL_TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        "name": "continue_ticket_draft",
        "planner_scope": "global",
        "risk_level": "LOW",
        "input_schema": {
            "type": "object",
            "properties": {"draft_id": {"type": "string"}, "fields": {"type": "object"}},
            "required": ["draft_id"],
        },
    },
    {
        "name": "ticket_tool_planner",
        "planner_scope": "global",
        "risk_level": "LOW",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "raw_text": {"type": "string"}},
            "required": ["ticket_id", "raw_text"],
        },
    },
    {
        "name": "kb_answer",
        "planner_scope": "global",
        "risk_level": "LOW",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "create_ticket",
        "planner_scope": "global",
        "risk_level": "LOW",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "fields": {"type": "object"}},
            "required": ["text"],
        },
    },
]

TICKET_TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        "name": "lookup_ticket",
        "planner_scope": "ticket",
        "risk_level": "LOW",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
    },
    {
        "name": "add_ticket_comment",
        "planner_scope": "ticket",
        "risk_level": "LOW",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "comment": {"type": "string"}},
            "required": ["ticket_id", "comment"],
        },
    },
    {
        "name": "escalate_ticket",
        "planner_scope": "ticket",
        "risk_level": "LOW",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["ticket_id"],
        },
    },
    {
        "name": "cancel_ticket",
        "planner_scope": "ticket",
        "risk_level": "HIGH",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "reason": {"type": "string"},
                "confirm": {"type": "boolean"},
            },
            "required": ["ticket_id", "reason"],
        },
    },
]

ARGS_MODELS = {
    "continue_ticket_draft": ContinueTicketDraftPlanArgs,
    "ticket_tool_planner": TicketToolPlannerPlanArgs,
    "kb_answer": KBAnswerPlanArgs,
    "create_ticket": CreateTicketPlanArgs,
    "lookup_ticket": LookupTicketPlanArgs,
    "add_ticket_comment": AddTicketCommentPlanArgs,
    "escalate_ticket": EscalateTicketPlanArgs,
    "cancel_ticket": CancelTicketPlanArgs,
}

GROUPS: dict[str, dict[str, Any]] = {
    "a": {
        "label": "A Baseline",
        "description": "普通 Chat Completion + 普通 JSON Prompt + 默认温度，无 retry。",
        "temperature": 0.7,
        "response_format": "none",
        "prompt_style": "plain",
        "retry": False,
    },
    "b": {
        "label": "B API-level",
        "description": "低温度 + JSON response_format，无 retry。",
        "temperature": 0.0,
        "response_format": "json_object",
        "prompt_style": "plain",
        "retry": False,
    },
    "c": {
        "label": "C System-level",
        "description": "低温度 + JSON response_format + 项目 prompt + few-shot + 一次 retry repair。",
        "temperature": 0.0,
        "response_format": "json_object",
        "prompt_style": "project",
        "retry": True,
    },
}

GLOBAL_FEW_SHOT_SUFFIX = """

边界 few-shot 示例：
1) 用户问制度流程：
输入：报销交通费有什么规定？
输出：{"tool":"kb_answer","args":{"query":"报销交通费有什么规定？"},"need_confirmation":false,"missing_fields":[]}
2) 已有工单但缺 ticket_id：
输入：把上一单撤掉吧
输出：{"tool":"ticket_tool_planner","args":{},"need_confirmation":false,"missing_fields":["ticket_id"]}
3) 继续草稿优先使用 draft_id：
输入：system_state.provided_draft_id=DRF-2026-AB12；user_text=联系方式改成 13800000000
输出：{"tool":"continue_ticket_draft","args":{"draft_id":"DRF-2026-AB12","fields":{"contact":"13800000000"}},"need_confirmation":false,"missing_fields":[]}
4) 新报修进入建单分支：
输入：电脑蓝屏了，帮我报修，地点 3 楼，电话 13800000000
输出：{"tool":"create_ticket","args":{"text":"电脑蓝屏了，帮我报修，地点 3 楼，电话 13800000000","fields":{"location":"3 楼","contact":"13800000000"}},"need_confirmation":false,"missing_fields":[]}
"""

TICKET_FEW_SHOT_SUFFIX = """

边界 few-shot 示例：
1) 取消已有工单必须确认：
输入：取消 TCK-2026-AB12，因为已经好了
输出：{"tool":"cancel_ticket","args":{"ticket_id":"TCK-2026-AB12","reason":"因为已经好了"},"need_confirmation":true,"missing_fields":[]}
2) 工单备注不是建新单：
输入：TCK-2026-AB12 补充一下：交换机在走廊
输出：{"tool":"add_ticket_comment","args":{"ticket_id":"TCK-2026-AB12","comment":"补充一下：交换机在走廊"},"need_confirmation":false,"missing_fields":[]}
3) 催办不是查单：
输入：TCK-2026-AB12 太慢了，帮我催一下
输出：{"tool":"escalate_ticket","args":{"ticket_id":"TCK-2026-AB12","reason":"太慢了，帮我催一下"},"need_confirmation":false,"missing_fields":[]}
4) 意图不明确默认查单：
输入：TCK-2026-AB12 现在怎么样了？
输出：{"tool":"lookup_ticket","args":{"ticket_id":"TCK-2026-AB12"},"need_confirmation":false,"missing_fields":[]}
"""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: case must be a JSON object")
        records.append(payload)
    return records


def _tools_for_scope(scope: str) -> list[dict[str, Any]]:
    if scope == "global":
        return list(GLOBAL_TOOL_CONTRACTS)
    if scope == "ticket":
        return list(TICKET_TOOL_CONTRACTS)
    raise ValueError(f"unsupported scope: {scope}")


def _contract_by_name(scope: str, tool_name: str) -> dict[str, Any] | None:
    for item in _tools_for_scope(scope):
        if str(item.get("name") or "") == tool_name:
            return item
    return None


def _required_args(scope: str, tool_name: str) -> list[str]:
    contract = _contract_by_name(scope, tool_name)
    if contract is None:
        return []
    input_schema = contract.get("input_schema") if isinstance(contract.get("input_schema"), dict) else {}
    required = input_schema.get("required") if isinstance(input_schema, dict) else []
    return [str(item) for item in required if str(item or "").strip()]


def _risk_level(scope: str, tool_name: str) -> str:
    contract = _contract_by_name(scope, tool_name)
    if contract is None:
        return ""
    return str(contract.get("risk_level") or "").upper()


def _first_ticket_id(text: str) -> str:
    matched = _TICKET_ID_RE.search(text or "")
    return matched.group(0) if matched else ""


def _build_context(case: dict[str, Any]) -> dict[str, Any]:
    utterance = str(case.get("utterance") or "")
    provided_ticket_id = str(case.get("provided_ticket_id") or "") or _first_ticket_id(utterance)
    provided_draft_id = str(case.get("provided_draft_id") or "")
    return {
        "actor_user_id": str(case.get("actor_user_id") or "eval_runner"),
        "provided_ticket_id": provided_ticket_id,
        "provided_draft_id": provided_draft_id,
        "has_ticket_id": bool(provided_ticket_id),
        "has_draft_id": bool(provided_draft_id),
        "confirm_token_present": False,
        "ticket_tool_mode": bool(provided_ticket_id),
        "draft_mode": bool(provided_draft_id),
    }


def _plain_prompt(case: dict[str, Any]) -> tuple[str, str]:
    scope = str(case.get("scope") or "")
    tools_json = _tools_for_scope(scope)
    context = _build_context(case)
    system_prompt = (
        "你是企业内部助手的 Agent Planner。请根据用户输入选择一个工具，并输出 JSON。"
        "JSON 字段为 tool、args、need_confirmation、missing_fields。"
        "tool 必须来自 tools；args 是对象；need_confirmation 是布尔值；missing_fields 是数组。"
    )
    user_prompt = (
        f"scope:\n{scope}\n\n"
        f"tools:\n{json.dumps(tools_json, ensure_ascii=False, sort_keys=True)}\n\n"
        f"context:\n{json.dumps(context, ensure_ascii=False, sort_keys=True)}\n\n"
        f"user_text:\n{case.get('utterance')}\n\n"
        "请输出 JSON。"
    )
    return system_prompt, user_prompt


def _project_prompt(case: dict[str, Any]) -> tuple[str, str]:
    scope = str(case.get("scope") or "")
    utterance = str(case.get("utterance") or "")
    if scope == "global":
        system_prompt, user_prompt = planner.build_global_planner_prompt(
            user_text=utterance,
            tools_json=_tools_for_scope(scope),
            context=_build_context(case),
        )
        return system_prompt + GLOBAL_FEW_SHOT_SUFFIX, user_prompt

    provided_ticket_id = str(case.get("provided_ticket_id") or "") or _first_ticket_id(utterance)
    system_prompt, user_prompt = planner.build_ticket_subplanner_prompt(
        user_text=utterance,
        provided_ticket_id=provided_ticket_id,
        tools_json=_tools_for_scope(scope),
    )
    return system_prompt + TICKET_FEW_SHOT_SUFFIX, user_prompt


def _repair_prompt(case: dict[str, Any], invalid_output: str, failure_reason: str) -> tuple[str, str]:
    scope = str(case.get("scope") or "")
    utterance = str(case.get("utterance") or "")
    if scope == "global":
        system_prompt, user_prompt = planner.build_global_repair_prompt(
            user_text=utterance,
            tools_json=_tools_for_scope(scope),
            context=_build_context(case),
            invalid_output=invalid_output,
        )
    else:
        system_prompt, user_prompt = planner.build_repair_prompt(
            user_text=utterance,
            provided_ticket_id=str(case.get("provided_ticket_id") or "") or _first_ticket_id(utterance),
            tools_json=_tools_for_scope(scope),
            invalid_output=invalid_output,
        )
    user_prompt += f"\n\nvalidation_error:\n{failure_reason}\n"
    suffix = GLOBAL_FEW_SHOT_SUFFIX if scope == "global" else TICKET_FEW_SHOT_SUFFIX
    return system_prompt + suffix, user_prompt


def _response_format_payload(kind: str) -> dict[str, Any] | None:
    normalized = str(kind or "none").strip().lower()
    if normalized == "none":
        return None
    if normalized == "json_object":
        return {"type": "json_object"}
    if normalized == "json_schema":
        schema = ToolPlan.model_json_schema() if hasattr(ToolPlan, "model_json_schema") else {}
        return {"type": "json_schema", "json_schema": {"name": "tool_plan", "schema": schema, "strict": True}}
    raise ValueError(f"unsupported response_format: {kind}")


def _build_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing OPENAI_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=planner.planner_timeout_seconds(),
    )


def _model_name() -> str:
    return str(os.getenv("AGENT_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "deepseek-chat")))


def _call_chat(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    response_format: str,
    max_tokens: int,
) -> str:
    kwargs: dict[str, Any] = {
        "model": _model_name(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response_format_payload = _response_format_payload(response_format)
    if response_format_payload is not None:
        kwargs["response_format"] = response_format_payload
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _validate_args_model(tool_name: str, args: dict[str, Any]) -> tuple[bool, str | None]:
    model_cls = ARGS_MODELS.get(tool_name)
    if model_cls is None:
        return False, "args_model_missing"
    try:
        if hasattr(model_cls, "model_validate"):
            model_cls.model_validate(args)
        else:
            model_cls.parse_obj(args)
    except ValidationError as exc:
        missing = [str((item.get("loc") or [""])[-1]) for item in exc.errors() if item.get("type") == "missing"]
        if missing:
            return False, "schema_missing:" + ",".join(missing)
        return False, "args_schema_invalid"
    return True, None


def _parse_output(raw_output: str) -> dict[str, Any]:
    raw = (raw_output or "").strip()
    strict_payload: Any = None
    extracted_payload: Any = None
    strict_parse_ok = False
    extracted_parse_ok = False
    json_text = planner._extract_json_object_text(raw)

    try:
        strict_payload = json.loads(raw)
        strict_parse_ok = isinstance(strict_payload, dict)
    except Exception:
        strict_payload = None

    if json_text:
        try:
            extracted_payload = json.loads(json_text)
            extracted_parse_ok = isinstance(extracted_payload, dict)
        except Exception:
            extracted_payload = None

    payload = strict_payload if isinstance(strict_payload, dict) else extracted_payload
    schema_valid = False
    schema_error: str | None = None
    plan_dump: dict[str, Any] | None = None
    if isinstance(payload, dict):
        try:
            if hasattr(ToolPlan, "model_validate"):
                plan = ToolPlan.model_validate(payload)
                plan_dump = plan.model_dump()
            else:
                plan = ToolPlan.parse_obj(payload)
                plan_dump = plan.dict()
            schema_valid = True
        except Exception as exc:
            schema_error = type(exc).__name__

    return {
        "raw_output": raw,
        "json_text": json_text,
        "strict_json_parse_ok": strict_parse_ok,
        "extracted_json_parse_ok": extracted_parse_ok,
        "payload": payload if isinstance(payload, dict) else None,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "plan": plan_dump,
    }


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _has_required_args(scope: str, tool_name: str, args: dict[str, Any]) -> bool:
    for field_name in _required_args(scope, tool_name):
        if not _non_empty(args.get(field_name)):
            return False
    return True


def _missing_fields_cover_expected(case: dict[str, Any], missing_fields: list[Any]) -> bool:
    expected = [str(item) for item in case.get("expected_missing_fields") or []]
    actual = {str(item) for item in missing_fields or []}
    return all(item in actual for item in expected)


def _evaluate_parsed(case: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    scope = str(case.get("scope") or "")
    expected_tool = str(case.get("expected_tool") or "")
    expected_outcome = str(case.get("expected_outcome") or "execute")
    payload = parsed.get("payload") if isinstance(parsed.get("payload"), dict) else {}
    tool_name = str(payload.get("tool") or "")
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    missing_fields = payload.get("missing_fields") if isinstance(payload.get("missing_fields"), list) else []
    need_confirmation = bool(payload.get("need_confirmation"))

    scoped_tool_names = {str(item.get("name") or "") for item in _tools_for_scope(scope)}
    tool_valid = tool_name in scoped_tool_names
    intent_accurate = tool_name == expected_tool

    if expected_outcome == "clarify":
        required_args_complete = _missing_fields_cover_expected(case, missing_fields)
        args_schema_valid = True
        args_schema_error = None
    else:
        required_args_complete = _has_required_args(scope, tool_name, args)
        args_schema_valid, args_schema_error = _validate_args_model(tool_name, args) if tool_name else (False, "tool_missing")

    high_risk_expected = expected_outcome == "confirm" or _risk_level(scope, expected_tool) == "HIGH"
    unsafe_action_blocked = None
    if high_risk_expected:
        unsafe_action_blocked = tool_name == expected_tool and need_confirmation

    base_ok = (
        bool(parsed.get("extracted_json_parse_ok"))
        and bool(parsed.get("schema_valid"))
        and tool_valid
        and intent_accurate
    )
    executable_ok = (
        expected_outcome == "execute"
        and base_ok
        and required_args_complete
        and args_schema_valid
        and not missing_fields
    )
    clarify_ok = (
        expected_outcome == "clarify"
        and base_ok
        and _missing_fields_cover_expected(case, missing_fields)
    )
    confirm_ok = (
        expected_outcome == "confirm"
        and base_ok
        and required_args_complete
        and args_schema_valid
        and bool(unsafe_action_blocked)
    )
    business_acceptable = executable_ok or clarify_ok or confirm_ok

    failure_reason = None
    if not parsed.get("extracted_json_parse_ok"):
        failure_reason = "json_parse_failed"
    elif not parsed.get("schema_valid"):
        failure_reason = str(parsed.get("schema_error") or "schema_invalid")
    elif not tool_valid:
        failure_reason = "tool_invalid"
    elif not intent_accurate:
        failure_reason = f"intent_mismatch:{tool_name or 'missing'}"
    elif not required_args_complete:
        failure_reason = "required_args_incomplete"
    elif not args_schema_valid:
        failure_reason = str(args_schema_error or "args_schema_invalid")
    elif expected_outcome == "confirm" and not unsafe_action_blocked:
        failure_reason = "unsafe_action_not_blocked"
    elif expected_outcome == "clarify" and not clarify_ok:
        failure_reason = "clarification_missing_fields_mismatch"
    elif expected_outcome == "execute" and missing_fields:
        failure_reason = "unexpected_missing_fields"

    return {
        "tool_name": tool_name or None,
        "tool_valid": tool_valid,
        "intent_accurate": intent_accurate,
        "required_args_complete": required_args_complete,
        "args_schema_valid": args_schema_valid,
        "unsafe_action_blocked": unsafe_action_blocked,
        "executable_plan": executable_ok,
        "business_acceptable": business_acceptable,
        "failure_reason": failure_reason,
    }


def _prompt_for_group(case: dict[str, Any], group_config: dict[str, Any]) -> tuple[str, str]:
    if str(group_config.get("prompt_style") or "") == "project":
        return _project_prompt(case)
    return _plain_prompt(case)


def _evaluate_case(
    client: OpenAI,
    case: dict[str, Any],
    group_key: str,
    group_config: dict[str, Any],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    call_count = 0
    first_raw = ""
    repair_raw = ""
    first_eval: dict[str, Any] | None = None
    final_eval: dict[str, Any] | None = None
    retry_used = False
    api_error: str | None = None

    try:
        system_prompt, user_prompt = _prompt_for_group(case, group_config)
        first_raw = _call_chat(
            client,
            system_prompt,
            user_prompt,
            temperature=float(group_config["temperature"]),
            response_format=str(group_config["response_format"]),
            max_tokens=max_tokens,
        )
        call_count += 1
        first_parsed = _parse_output(first_raw)
        first_eval = {**first_parsed, **_evaluate_parsed(case, first_parsed)}
        final_eval = dict(first_eval)

        if bool(group_config.get("retry")) and not bool(first_eval.get("business_acceptable")):
            retry_used = True
            repair_system_prompt, repair_user_prompt = _repair_prompt(
                case,
                invalid_output=first_raw,
                failure_reason=str(first_eval.get("failure_reason") or "business_unacceptable"),
            )
            repair_raw = _call_chat(
                client,
                repair_system_prompt,
                repair_user_prompt,
                temperature=float(group_config["temperature"]),
                response_format=str(group_config["response_format"]),
                max_tokens=max_tokens,
            )
            call_count += 1
            repair_parsed = _parse_output(repair_raw)
            final_eval = {**repair_parsed, **_evaluate_parsed(case, repair_parsed)}
    except Exception as exc:
        api_error = f"{type(exc).__name__}:{exc}"
        first_eval = first_eval or {}
        final_eval = final_eval or {}

    latency_ms = int(round((time.perf_counter() - start) * 1000))
    first_eval = first_eval or {}
    final_eval = final_eval or {}
    return {
        "case_id": str(case.get("case_id") or ""),
        "category": str(case.get("category") or ""),
        "scope": str(case.get("scope") or ""),
        "utterance": str(case.get("utterance") or ""),
        "expected_tool": str(case.get("expected_tool") or ""),
        "expected_outcome": str(case.get("expected_outcome") or ""),
        "group": group_key,
        "api_error": api_error,
        "calls": call_count,
        "latency_ms": latency_ms,
        "retry_used": retry_used,
        "retry_repair_success": bool(retry_used and not first_eval.get("business_acceptable") and final_eval.get("business_acceptable")),
        "first": _compact_eval(first_eval, first_raw),
        "final": _compact_eval(final_eval, repair_raw or first_raw),
    }


def _compact_eval(eval_payload: dict[str, Any], raw_output: str) -> dict[str, Any]:
    return {
        "strict_json_parse_ok": bool(eval_payload.get("strict_json_parse_ok")),
        "extracted_json_parse_ok": bool(eval_payload.get("extracted_json_parse_ok")),
        "schema_valid": bool(eval_payload.get("schema_valid")),
        "tool_name": eval_payload.get("tool_name"),
        "tool_valid": bool(eval_payload.get("tool_valid")),
        "intent_accurate": bool(eval_payload.get("intent_accurate")),
        "required_args_complete": bool(eval_payload.get("required_args_complete")),
        "args_schema_valid": bool(eval_payload.get("args_schema_valid")),
        "unsafe_action_blocked": eval_payload.get("unsafe_action_blocked"),
        "executable_plan": bool(eval_payload.get("executable_plan")),
        "business_acceptable": bool(eval_payload.get("business_acceptable")),
        "failure_reason": eval_payload.get("failure_reason"),
        "raw_output": raw_output,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round((numerator / denominator), 4) if denominator else 0.0


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(round(len(sorted_values) * 0.95 + 0.5)) - 1))
    return int(sorted_values[index])


def _summarize_group(group_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    first_rows = [row.get("first") or {} for row in rows]
    final_rows = [row.get("final") or {} for row in rows]
    executable_case_count = sum(1 for row in rows if row.get("expected_outcome") == "execute")
    high_risk_case_count = sum(1 for row in rows if row.get("expected_outcome") == "confirm")
    retry_used_count = sum(1 for row in rows if row.get("retry_used"))
    retry_success_count = sum(1 for row in rows if row.get("retry_repair_success"))
    call_count = sum(int(row.get("calls") or 0) for row in rows)
    latencies = [int(row.get("latency_ms") or 0) for row in rows]

    def count(rows_to_count: list[dict[str, Any]], key: str) -> int:
        return sum(1 for item in rows_to_count if bool(item.get(key)))

    def executable_count(rows_to_count: list[dict[str, Any]]) -> int:
        return sum(
            1
            for row, item in zip(rows, rows_to_count, strict=False)
            if row.get("expected_outcome") == "execute" and bool(item.get("executable_plan"))
        )

    def high_risk_block_count(rows_to_count: list[dict[str, Any]]) -> int:
        return sum(
            1
            for row, item in zip(rows, rows_to_count, strict=False)
            if row.get("expected_outcome") == "confirm" and bool(item.get("unsafe_action_blocked"))
        )

    return {
        "group": group_key,
        "label": GROUPS[group_key]["label"],
        "total_cases": total,
        "executable_case_count": executable_case_count,
        "high_risk_case_count": high_risk_case_count,
        "json_parse_rate": _rate(count(first_rows, "strict_json_parse_ok"), total),
        "final_json_parse_rate": _rate(count(final_rows, "strict_json_parse_ok"), total),
        "extracted_json_parse_rate": _rate(count(first_rows, "extracted_json_parse_ok"), total),
        "schema_valid_rate": _rate(count(first_rows, "schema_valid"), total),
        "final_schema_valid_rate": _rate(count(final_rows, "schema_valid"), total),
        "tool_valid_rate": _rate(count(first_rows, "tool_valid"), total),
        "final_tool_valid_rate": _rate(count(final_rows, "tool_valid"), total),
        "required_args_complete_rate": _rate(count(first_rows, "required_args_complete"), total),
        "final_required_args_complete_rate": _rate(count(final_rows, "required_args_complete"), total),
        "intent_accuracy": _rate(count(first_rows, "intent_accurate"), total),
        "final_intent_accuracy": _rate(count(final_rows, "intent_accurate"), total),
        "first_pass_executable_plan_rate": _rate(executable_count(first_rows), executable_case_count),
        "final_executable_plan_rate": _rate(executable_count(final_rows), executable_case_count),
        "business_acceptable_rate": _rate(count(first_rows, "business_acceptable"), total),
        "final_business_acceptable_rate": _rate(count(final_rows, "business_acceptable"), total),
        "unsafe_action_block_rate": _rate(high_risk_block_count(final_rows), high_risk_case_count),
        "retry_trigger_rate": _rate(retry_used_count, total),
        "retry_repair_success_rate": _rate(retry_success_count, retry_used_count),
        "avg_calls_per_case": round(call_count / total, 4) if total else 0.0,
        "avg_latency_ms": int(round(statistics.mean(latencies))) if latencies else 0,
        "p50_latency_ms": int(round(statistics.median(latencies))) if latencies else 0,
        "p95_latency_ms": _p95(latencies),
        "api_error_count": sum(1 for row in rows if row.get("api_error")),
    }


def _validate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    seen_ids: set[str] = set()
    by_scope: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    errors: list[str] = []

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id") or "")
        scope = str(case.get("scope") or "")
        category = str(case.get("category") or "")
        outcome = str(case.get("expected_outcome") or "")
        expected_tool = str(case.get("expected_tool") or "")
        if not case_id:
            errors.append(f"case #{index}: missing case_id")
        elif case_id in seen_ids:
            errors.append(f"{case_id}: duplicate case_id")
        seen_ids.add(case_id)
        if scope not in {"global", "ticket"}:
            errors.append(f"{case_id}: invalid scope")
        elif expected_tool not in {str(item.get("name") or "") for item in _tools_for_scope(scope)}:
            errors.append(f"{case_id}: expected_tool not in scope tools")
        if outcome not in {"execute", "clarify", "confirm"}:
            errors.append(f"{case_id}: invalid expected_outcome")
        if outcome == "clarify" and not case.get("expected_missing_fields"):
            errors.append(f"{case_id}: clarify case missing expected_missing_fields")
        by_scope[scope] = by_scope.get(scope, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

    return {
        "total_cases": len(cases),
        "by_scope": dict(sorted(by_scope.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "errors": errors,
    }


def _print_text_report(report: dict[str, Any], show_failures: int) -> None:
    summaries = report.get("summaries") or {}
    group_order = report.get("group_order") or []
    print("Planner JSON Compliance Evaluation")
    print(json.dumps(report.get("case_validation") or {}, ensure_ascii=False, indent=2, sort_keys=True))
    print()
    if not summaries:
        print("No model results. Use without --dry-run to run A/B/C experiments.")
        return

    metrics = [
        ("JSON Parse", "json_parse_rate"),
        ("Schema Valid", "schema_valid_rate"),
        ("Tool Valid", "tool_valid_rate"),
        ("Required Args", "required_args_complete_rate"),
        ("Intent Accuracy", "intent_accuracy"),
        ("First Exec Plan", "first_pass_executable_plan_rate"),
        ("Final Exec Plan", "final_executable_plan_rate"),
        ("Final Acceptable", "final_business_acceptable_rate"),
        ("Unsafe Block", "unsafe_action_block_rate"),
        ("Avg Calls", "avg_calls_per_case"),
    ]
    header = ["Metric"] + [str(summaries.get(group, {}).get("label") or group) for group in group_order]
    widths = [max(len(item), 18) for item in header]
    rows: list[list[str]] = []
    for label, key in metrics:
        values = [label]
        for group in group_order:
            value = summaries.get(group, {}).get(key)
            values.append(str(value) if key == "avg_calls_per_case" else _percent(float(value or 0.0)))
        rows.append(values)
        widths = [max(width, len(value)) for width, value in zip(widths, values, strict=False)]

    print(" | ".join(value.ljust(width) for value, width in zip(header, widths, strict=False)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths, strict=False)))

    if show_failures <= 0:
        return
    print()
    for group in group_order:
        failures = [
            row
            for row in report.get("results", {}).get(group, [])
            if not (row.get("final") or {}).get("business_acceptable")
        ]
        if not failures:
            print(f"{group}: no final business failures.")
            continue
        print(f"{group}: final business failures, showing up to {show_failures}")
        for item in failures[:show_failures]:
            payload = {
                "case_id": item.get("case_id"),
                "expected_tool": item.get("expected_tool"),
                "expected_outcome": item.get("expected_outcome"),
                "actual_tool": (item.get("final") or {}).get("tool_name"),
                "failure_reason": (item.get("final") or {}).get("failure_reason") or item.get("api_error"),
            }
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Planner JSON compliance on local cases.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="JSONL case file path.")
    parser.add_argument("--groups", default="a,b,c", help="Comma-separated groups: a,b,c.")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N cases.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate cases; do not call LLM.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Report format.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to save full JSON report.")
    parser.add_argument("--show-failures", type=int, default=8, help="Text mode failure examples.")
    parser.add_argument("--max-tokens", type=int, default=320, help="Planner response max tokens.")
    parser.add_argument(
        "--b-response-format",
        choices=("json_object", "json_schema"),
        default="",
        help="Override group B response_format when provider supports it.",
    )
    parser.add_argument(
        "--c-response-format",
        choices=("json_object", "json_schema"),
        default="",
        help="Override group C response_format when provider supports it.",
    )
    args = parser.parse_args(argv)

    cases = _load_jsonl(Path(args.cases))
    if args.limit > 0:
        cases = cases[: args.limit]
    case_validation = _validate_cases(cases)
    if case_validation["errors"]:
        print(json.dumps(case_validation, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    group_order = [item.strip().lower() for item in str(args.groups or "").split(",") if item.strip()]
    unknown_groups = [group for group in group_order if group not in GROUPS]
    if unknown_groups:
        raise ValueError(f"unknown groups: {unknown_groups}")
    group_configs = {group: dict(GROUPS[group]) for group in group_order}
    if args.b_response_format and "b" in group_configs:
        group_configs["b"]["response_format"] = args.b_response_format
    if args.c_response_format and "c" in group_configs:
        group_configs["c"]["response_format"] = args.c_response_format

    report: dict[str, Any] = {
        "case_file": str(args.cases),
        "case_validation": case_validation,
        "group_order": group_order,
        "group_configs": group_configs,
        "summaries": {},
        "results": {},
    }

    if not args.dry_run:
        client = _build_openai_client()
        for group in group_order:
            rows = [
                _evaluate_case(client, case, group, group_configs[group], max_tokens=int(args.max_tokens))
                for case in cases
            ]
            report["results"][group] = rows
            report["summaries"][group] = _summarize_group(group, rows)

    output_path = Path(args.output)
    if args.output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text_report(report, show_failures=int(args.show_failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
