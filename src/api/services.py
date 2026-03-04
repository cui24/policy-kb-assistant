"""
L2/L4 业务编排层：把检索、回答、工单、审计与草稿串成完整业务动作。

一、程序目标
1. 避免把复杂逻辑直接写进 FastAPI 路由。
2. 让 `/ask`、`/tickets`、`/agent` 都复用同一套核心工作流。
3. 为 L3-3 提供统一的序列化函数，让问答记录和审计日志可直接返回给 API 与 UI。
4. 为 L4-1 提供工单草稿续办能力，让 Agent 支持多轮补全后再建单。

二、主要工作流
1. `run_ask_workflow(...)`
   - 调 `retrieve(...)`
   - 调 `answer_with_citations(...)`
   - 记录 `kb_queries`
   - 记录 `audit_logs`
2. `create_ticket_workflow(...)`
   - 创建 `tickets`
   - 记录 `audit_logs`
3. `run_agent_workflow(...)`
   - 新请求：规则路由 `ASK / CREATE_TICKET / NEED_MORE_INFO`
   - 草稿续办：读取 `ticket_drafts`，合并新信息，字段齐则建单
4. `serialize_*`
   - 把 ORM 对象转换成稳定响应结构，供追溯接口和 UI 直接展示

三、输入输出
1. 输入：路由层传入的请求参数和数据库会话。
2. 输出：适合直接返回给 API 的字典结果。
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.agent.ticket_extractor import extract_ticket_payload
from src.api import crud, models
from src.api import planner
from src.api.planner import PlannerError
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
from src.api.skills import get_ticket_tool_registry, list_global_planner_skills, list_ticket_tool_skills
from src.kb.answer import answer_with_citations
from src.kb.retrieve import retrieve


_TICKET_ROUTE_KEYWORDS = (
    "报修",
    "工单",
    "提交工单",
    "提交报修",
    "帮我提交",
    "帮我报修",
)

_OPTIONAL_ACTION_KEYWORDS = ("申请", "开通")
_OPTIONAL_ACTION_DOMAINS = ("网络", "账号", "权限", "校园网", "认证")
_TICKET_LOOKUP_KEYWORDS = ("查工单", "查一下", "查询", "进度", "状态")
_TICKET_COMMENT_KEYWORDS = ("补充", "补充说明", "追加", "备注", "留言")
_TICKET_ESCALATE_KEYWORDS = ("催办", "催一下", "加急", "升级")
_TICKET_CANCEL_KEYWORDS = ("取消", "撤销", "关闭工单")
_DRAFT_REQUIRED_FIELDS = ("location", "contact")
_DRAFT_ALLOWED_FIELDS = (
    "creator",
    "department",
    "category",
    "priority",
    "title",
    "description",
    "contact",
    "location",
    "source_text",
    "kb_attempt_stage",
)
_TICKET_PUBLIC_ID_RE = re.compile(r"TCK-\d{4}-[A-Z0-9]+")
_TICKET_COMMENT_FETCH_LIMIT = 20
_AUDIT_COMMENT_PREVIEW_LIMIT = 160
_ADMIN_ACTORS = {"admin", "service_admin"}
_SHORT_TERM_MEMORY_SUMMARY_LIMIT = 160
_SHORT_TERM_TICKET_REFERENCE_MARKERS = (
    "上一单",
    "上一张单",
    "上一个工单",
    "那个工单",
    "那张单",
    "之前那个",
    "上次那个",
    "刚建的单",
    "那单",
)
_SHORT_TERM_DRAFT_REFERENCE_MARKERS = (
    "刚才那个问题",
    "继续刚才",
    "继续那个",
    "补充地点",
    "补充联系方式",
    "地点在",
    "电话",
    "联系方式",
)
_LONG_TERM_MEMORY_SUMMARY_LIMIT = 120



def _utc_now() -> datetime:
    """生成统一的 UTC 当前时间。"""
    return datetime.now(timezone.utc)



def _utc_year() -> str:
    """生成当前 UTC 年份，用于工单号与草稿号。"""
    return _utc_now().strftime("%Y")



def _new_request_id() -> str:
    """生成一次问答或路由的请求标识。"""
    return f"req_{uuid4().hex[:12]}"



def _new_ticket_public_id() -> str:
    """生成对外展示的工单号。"""
    return f"TCK-{_utc_year()}-{uuid4().hex[:6].upper()}"



def _new_draft_public_id() -> str:
    """生成对外展示的草稿号。"""
    return f"DRF-{_utc_year()}-{uuid4().hex[:6].upper()}"



def _draft_ttl_minutes() -> int:
    """读取草稿有效期，默认 20 分钟。"""
    raw_value = os.getenv("TICKET_DRAFT_TTL_MINUTES", "20")
    try:
        minutes = int(raw_value)
    except ValueError:
        minutes = 20
    return max(5, min(minutes, 120))



def _draft_expiry() -> datetime:
    """计算新的草稿过期时间。"""
    return _utc_now() + timedelta(minutes=_draft_ttl_minutes())


def _pending_action_ttl_minutes() -> int:
    """读取确认态 token 的有效期，默认 15 分钟。"""
    raw_value = os.getenv("AGENT_CONFIRM_TTL_MINUTES", "15")
    try:
        minutes = int(raw_value)
    except ValueError:
        minutes = 15
    return max(3, min(minutes, 60))


def _pending_action_expiry() -> datetime:
    """计算新的确认态过期时间。"""
    return _utc_now() + timedelta(minutes=_pending_action_ttl_minutes())



def _normalize_datetime(value: datetime | None) -> datetime | None:
    """把数据库读出的时间统一转成带 UTC 时区的 datetime，兼容 SQLite 的 naive 值。"""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)



def _trim_hits_for_trace(hits: list[dict], limit: int = 6) -> list[dict]:
    """压缩检索证据，只保留 API trace 必需字段。"""
    trimmed: list[dict] = []
    for hit in hits[:limit]:
        trimmed.append(
            {
                "doc_id": hit.get("doc_id"),
                "page": hit.get("page"),
                "score": float(hit.get("score", 0.0) or 0.0),
                "snippet": str(hit.get("snippet") or "")[:220],
            }
        )
    return trimmed



def _model_name() -> str:
    """读取当前回答模型名，便于落库。"""
    return os.getenv("OPENAI_MODEL", "deepseek-chat")



def _should_route_to_ticket(text: str) -> bool:
    """用保守规则判断是否要走建单路径。"""
    normalized = text or ""
    if any(keyword in normalized for keyword in _TICKET_ROUTE_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in _OPTIONAL_ACTION_KEYWORDS):
        return any(domain in normalized for domain in _OPTIONAL_ACTION_DOMAINS)
    return False


def _extract_ticket_public_id(text: str) -> str | None:
    """从自然语言里提取工单号。"""
    matched = _TICKET_PUBLIC_ID_RE.search(text or "")
    if matched is None:
        return None
    return matched.group(0)


def _detect_ticket_tool_action(text: str) -> str | None:
    """判断是否是针对既有工单的查询或操作。"""
    normalized = text or ""
    if any(keyword in normalized for keyword in _TICKET_CANCEL_KEYWORDS):
        return "CANCEL_TICKET"
    if any(keyword in normalized for keyword in _TICKET_ESCALATE_KEYWORDS):
        return "ESCALATE_TICKET"
    if any(keyword in normalized for keyword in _TICKET_COMMENT_KEYWORDS):
        return "ADD_TICKET_COMMENT"
    if any(keyword in normalized for keyword in _TICKET_LOOKUP_KEYWORDS):
        return "LOOKUP_TICKET"
    return None


def _strip_ticket_reference(text: str, ticket_id: str) -> str:
    """从自然语言里去掉工单号，尽量保留剩余说明文本。"""
    cleaned = (text or "").replace(ticket_id, " ")
    return " ".join(cleaned.split()).strip("：:，,。 ")



def _clean_value(value):
    """清洗单个字段值：字符串去空白，空串视为缺失。"""
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return value



def _normalize_partial_fields(raw_fields: dict | None) -> dict:
    """把外部输入整理成可合并的部分字段，只保留草稿允许维护的键。"""
    if not isinstance(raw_fields, dict):
        return {}

    payload: dict = {}
    for key in _DRAFT_ALLOWED_FIELDS:
        if key in raw_fields:
            cleaned = _clean_value(raw_fields.get(key))
            if cleaned not in (None, "", []):
                payload[key] = cleaned
    return payload



def _build_ticket_payload(
    source_fields: dict | None,
    actor: str,
    actor_department: str,
    source_text: str,
    kb_attempt_stage: str | None = None,
) -> dict:
    """把一次新请求的抽取结果整理成完整草稿载荷。"""
    partial = _normalize_partial_fields(source_fields)
    text_value = (source_text or "").strip()
    payload = {
        "creator": partial.get("creator") or actor,
        "department": partial.get("department") or actor_department,
        "category": partial.get("category") or "other",
        "priority": partial.get("priority") or "P2",
        "title": partial.get("title") or "通用服务请求",
        "description": partial.get("description") or text_value or "用户未提供详细描述。",
        "contact": partial.get("contact"),
        "location": partial.get("location"),
        "source_text": partial.get("source_text") or text_value,
        "kb_attempt_stage": partial.get("kb_attempt_stage") or kb_attempt_stage,
    }
    return payload



def _merge_draft_payload(existing_payload: dict | None, incoming_fields: dict | None) -> dict:
    """合并草稿字段：旧值优先保留，只有旧值缺失时才接受新值。"""
    merged = dict(existing_payload or {})
    incoming = _normalize_partial_fields(incoming_fields)

    for key, value in incoming.items():
        old_value = _clean_value(merged.get(key))
        if old_value in (None, "", []):
            merged[key] = value
        elif key not in merged:
            merged[key] = value

    return merged



def _compute_missing_fields(payload: dict | None) -> list[str]:
    """根据当前草稿载荷计算仍然缺失的必填字段。"""
    normalized_payload = payload or {}
    missing_fields: list[str] = []
    for field_name in _DRAFT_REQUIRED_FIELDS:
        if _clean_value(normalized_payload.get(field_name)) in (None, "", []):
            missing_fields.append(field_name)
    return missing_fields



def _build_need_more_info_message(missing_fields: list[str]) -> str:
    """根据缺失字段生成统一提示。"""
    if not missing_fields:
        return "当前信息已齐全，可以继续处理。"
    return "为了创建报修工单，请补充：" + "、".join(missing_fields) + "。"



def _build_extraction_view(payload: dict | None, missing_fields: list[str], extractor: str) -> dict:
    """把当前草稿载荷整理成 UI 可展示的抽取结果结构。"""
    normalized_payload = payload or {}
    return {
        "creator": str(normalized_payload.get("creator") or "anonymous"),
        "department": str(normalized_payload.get("department") or "general"),
        "category": str(normalized_payload.get("category") or "other"),
        "priority": str(normalized_payload.get("priority") or "P2"),
        "title": str(normalized_payload.get("title") or "通用服务请求"),
        "description": str(normalized_payload.get("description") or ""),
        "contact": normalized_payload.get("contact"),
        "location": normalized_payload.get("location"),
        "missing_fields": list(missing_fields),
        "extractor": extractor,
    }



def run_ask_workflow(
    db: Session,
    question: str,
    user: str | None = None,
    department: str | None = None,
) -> dict:
    """执行一次完整的 L2 问答，并把问答轨迹落库。"""
    request_id = _new_request_id()
    actor = user or "anonymous"
    actor_department = department or "general"

    retrieve_start = time.perf_counter()
    hits = retrieve(question)
    retrieve_end = time.perf_counter()

    answer_start = time.perf_counter()
    output = answer_with_citations(question, hits)
    answer_end = time.perf_counter()

    retrieve_ms = int((retrieve_end - retrieve_start) * 1000)
    answer_ms = int((answer_end - answer_start) * 1000)

    citations = output.get("citations", []) or []
    output_meta = output.get("meta", {}) or {}
    trace_hits = _trim_hits_for_trace(hits)

    kb_query = crud.create_kb_query(
        db,
        {
            "request_id": request_id,
            "user_name": actor,
            "department": actor_department,
            "question": question,
            "answer": str(output.get("answer") or ""),
            "citations_json": citations,
            "retrieve_topk_json": trace_hits,
            "attempt_stage": str(output_meta.get("attempt_stage") or "unknown"),
            "latency_retrieve_ms": retrieve_ms,
            "latency_answer_ms": answer_ms,
            "model": _model_name(),
            "valid_json": bool(output_meta.get("json_ok", False)),
            "failure_reason": output_meta.get("failure_reason"),
        },
    )

    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": "ASK",
            "target_type": "KB_QUERY",
            "target_id": kb_query.id,
            "request_id": request_id,
            "payload_json": {
                "question": question,
                "department": actor_department,
                "attempt_stage": output_meta.get("attempt_stage"),
                "top_hit": trace_hits[0] if trace_hits else None,
                "latency_ms": {"retrieve": retrieve_ms, "answer": answer_ms},
                "failure_reason": output_meta.get("failure_reason"),
            },
        },
    )

    return {
        "request_id": request_id,
        "query_id": kb_query.id,
        "answer": str(output.get("answer") or ""),
        "citations": citations,
        "meta": {
            "attempt_stage": str(output_meta.get("attempt_stage") or "unknown"),
            "valid_json": bool(output_meta.get("json_ok", False)),
            "repair_used": bool(output_meta.get("repair_used", False)),
            "failure_reason": output_meta.get("failure_reason"),
            "retrieve_topk": trace_hits,
            "latency_ms": {"retrieve": retrieve_ms, "answer": answer_ms},
        },
    }



def public_kb_response(kb_result: dict) -> dict:
    """去掉仅供服务内部使用的字段，避免把内部主键直接暴露给外部调用方。"""
    visible = dict(kb_result)
    visible.pop("query_id", None)
    return visible



def create_ticket_workflow(
    db: Session,
    creator: str | None,
    department: str | None,
    category: str,
    priority: str,
    title: str,
    description: str,
    contact: str | None = None,
    context: dict | None = None,
    source_draft_id: str | None = None,
    request_id: str | None = None,
    audit_action: str = "CREATE_TICKET",
) -> dict:
    """创建工单并写审计日志。"""
    actor = creator or "anonymous"
    ticket = crud.create_ticket(
        db,
        {
            "public_id": _new_ticket_public_id(),
            "creator": actor,
            "assignee": None,
            "department": department or "IT",
            "category": category,
            "priority": priority,
            "title": title,
            "description": description,
            "status": "open",
            "contact": contact,
            "source_draft_id": source_draft_id,
            "context_json": context or {},
        },
    )

    audit_request_id = request_id or _new_request_id()
    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": audit_action,
            "target_type": "TICKET",
            "target_id": ticket.public_id,
            "request_id": audit_request_id,
            "payload_json": {
                "department": ticket.department,
                "category": ticket.category,
                "priority": ticket.priority,
                "status": ticket.status,
                "title": ticket.title,
                "context": ticket.context_json,
            },
        },
    )
    ticket_context = context or {}
    _update_user_memory_from_ticket_facts(
        db,
        actor,
        location=ticket_context.get("location"),
        contact=contact,
        source_ticket_id=ticket.public_id,
    )

    return {
        "ticket_id": ticket.public_id,
        "status": ticket.status,
    }



def _serialize_legacy_ticket_comment(raw_comment: dict) -> dict | None:
    """兼容读取旧版 `context_json.comments`，便于平滑过渡到独立评论表。"""
    if not isinstance(raw_comment, dict):
        return None

    content = str(raw_comment.get("content") or raw_comment.get("comment") or "").strip()
    if not content:
        return None

    return {
        "comment_id": None,
        "actor": str(raw_comment.get("actor") or raw_comment.get("actor_user_id") or "anonymous"),
        "content": content,
        "created_at": str(raw_comment.get("created_at") or ""),
    }


def serialize_ticket_comment(comment) -> dict:
    """把 ORM 评论对象转换成 API 可返回的字典。"""
    created_at = _normalize_datetime(comment.created_at)
    return {
        "comment_id": comment.comment_id,
        "actor": comment.actor_user_id,
        "content": comment.content,
        "created_at": created_at.isoformat() if created_at is not None else "",
    }


def _ticket_context_without_comments(ticket) -> dict:
    """从响应里的 context 去掉旧版评论字段，避免与顶层 comments 重复。"""
    context = dict(ticket.context_json or {})
    context.pop("comments", None)
    return context


def _load_ticket_comments(
    db: Session,
    ticket,
    limit: int = _TICKET_COMMENT_FETCH_LIMIT,
) -> list[dict]:
    """读取工单评论，优先从独立表取，并兼容旧版 context 里的历史评论。"""
    safe_limit = max(1, min(int(limit), 200))
    legacy_comments = [
        item
        for item in (
            _serialize_legacy_ticket_comment(raw_comment)
            for raw_comment in list((ticket.context_json or {}).get("comments") or [])
        )
        if item is not None
    ]
    persisted_comments = [
        serialize_ticket_comment(comment)
        for comment in reversed(list(crud.list_ticket_comments(db, ticket.id, limit=safe_limit)))
    ]

    merged_comments: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for item in legacy_comments + persisted_comments:
        dedupe_key = (
            str(item.get("actor") or ""),
            str(item.get("content") or ""),
            str(item.get("created_at") or ""),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        merged_comments.append(item)

    merged_comments.sort(key=lambda item: str(item.get("created_at") or ""))
    if len(merged_comments) > safe_limit:
        return merged_comments[-safe_limit:]
    return merged_comments


def serialize_ticket(ticket, comments: list[dict] | None = None) -> dict:
    """把 ORM 工单对象转换成 API 可返回的字典。"""
    created_at = _normalize_datetime(ticket.created_at)
    updated_at = _normalize_datetime(ticket.updated_at)
    return {
        "ticket_id": ticket.public_id,
        "creator": ticket.creator,
        "assignee": ticket.assignee,
        "department": ticket.department,
        "category": ticket.category,
        "priority": ticket.priority,
        "title": ticket.title,
        "description": ticket.description,
        "status": ticket.status,
        "contact": ticket.contact,
        "comments": list(comments or []),
        "context": _ticket_context_without_comments(ticket),
        "created_at": created_at.isoformat() if created_at is not None else "",
        "updated_at": updated_at.isoformat() if updated_at is not None else "",
    }


def serialize_ticket_detail(
    db: Session,
    ticket,
    comment_limit: int = _TICKET_COMMENT_FETCH_LIMIT,
) -> dict:
    """把工单和最近评论一起序列化成详情响应。"""
    return serialize_ticket(ticket, comments=_load_ticket_comments(db, ticket, limit=comment_limit))



def serialize_ticket_draft(draft) -> dict:
    """把工单草稿 ORM 对象转换成 API 可返回的字典。"""
    return {
        "draft_id": draft.draft_id,
        "status": draft.status,
        "missing_fields": list(draft.missing_fields_json or []),
        "expires_at": draft.expires_at.isoformat(),
        "payload": draft.payload_json or {},
        "kb_request_id": draft.kb_request_id,
    }



def serialize_kb_query(record) -> dict:
    """把问答记录 ORM 对象转换成 API 可返回的追溯结构。"""
    return {
        "request_id": record.request_id,
        "user": record.user_name,
        "department": record.department,
        "question": record.question,
        "answer": record.answer,
        "citations": record.citations_json or [],
        "retrieve_topk": record.retrieve_topk_json or [],
        "attempt_stage": record.attempt_stage,
        "latency_ms": {
            "retrieve": int(record.latency_retrieve_ms or 0),
            "answer": int(record.latency_answer_ms or 0),
        },
        "model": record.model,
        "valid_json": bool(record.valid_json),
        "failure_reason": record.failure_reason,
        "created_at": record.created_at.isoformat(),
    }



def serialize_audit_log(record) -> dict:
    """把审计日志 ORM 对象转换成 API 可返回的时间线结构。"""
    return {
        "id": record.id,
        "created_at": record.created_at.isoformat(),
        "actor": record.actor,
        "action_type": record.action_type,
        "target_type": record.target_type,
        "target_id": record.target_id,
        "request_id": record.request_id,
        "payload": record.payload_json or {},
    }


def list_ticket_tool_skill_contracts() -> list[dict]:
    """返回当前 `/agent` 既有工单工具的技能清单。"""
    return list_ticket_tool_skills()


def list_global_planner_skill_contracts() -> list[dict]:
    """返回 Global Planner 可用的分支工具清单。"""
    return list_global_planner_skills()


def _validate_pydantic_model(model_cls, payload):
    """兼容 Pydantic v1/v2 的模型校验入口。"""
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _agent_planner_mode() -> str:
    """读取 Agent Planner 模式。"""
    return planner.agent_planner_mode()


def _plan_args_summary(args: dict | None) -> dict:
    """压缩 plan 参数，避免审计里写入过长文本。"""
    normalized = dict(args or {})
    for key in ("comment", "reason"):
        if key in normalized:
            normalized[key] = str(normalized.get(key) or "")[:_AUDIT_COMMENT_PREVIEW_LIMIT]
    return normalized


def _with_audit_source(payload_json: dict | None, audit_source: str | None = None) -> dict:
    """按需把来源标签附加到审计 payload。"""
    normalized = dict(payload_json or {})
    if audit_source:
        normalized["source"] = str(audit_source)
    return normalized


def _audit_plan_event(
    db: Session,
    actor: str,
    action_type: str,
    ticket_id: str,
    request_id: str,
    payload_json: dict,
) -> None:
    """统一写入 Planner 相关审计事件。"""
    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": action_type,
            "target_type": "TICKET",
            "target_id": ticket_id,
            "request_id": request_id,
            "payload_json": payload_json,
        },
    )


def _audit_global_plan_event(
    db: Session,
    actor: str,
    action_type: str,
    target_type: str,
    target_id: str,
    request_id: str,
    payload_json: dict,
) -> None:
    """为 Global Planner 写入统一审计事件。"""
    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": action_type,
            "target_type": target_type,
            "target_id": target_id,
            "request_id": request_id,
            "payload_json": payload_json,
        },
    )


def _ticket_tool_plan_args_model(tool_name: str):
    """根据工具名返回对应的参数校验模型。"""
    return {
        "lookup_ticket": LookupTicketPlanArgs,
        "add_ticket_comment": AddTicketCommentPlanArgs,
        "escalate_ticket": EscalateTicketPlanArgs,
        "cancel_ticket": CancelTicketPlanArgs,
    }.get(tool_name)


def _global_plan_args_model(tool_name: str):
    """根据全局分支工具名返回对应的参数校验模型。"""
    return {
        "continue_ticket_draft": ContinueTicketDraftPlanArgs,
        "ticket_tool_planner": TicketToolPlannerPlanArgs,
        "kb_answer": KBAnswerPlanArgs,
        "create_ticket": CreateTicketPlanArgs,
    }.get(tool_name)


def _extract_missing_fields_from_validation_error(exc: ValidationError) -> list[str]:
    """从 Pydantic 校验错误中提取缺失字段名。"""
    missing_fields: list[str] = []
    for item in exc.errors():
        if str(item.get("type") or "") == "missing":
            loc = item.get("loc") or []
            if isinstance(loc, tuple):
                field_name = str(loc[-1]) if loc else ""
            elif isinstance(loc, list):
                field_name = str(loc[-1]) if loc else ""
            else:
                field_name = str(loc)
            if field_name and field_name not in missing_fields:
                missing_fields.append(field_name)
    return missing_fields


def _append_audit_log_uncommitted(
    db: Session,
    actor: str,
    action_type: str,
    target_type: str,
    target_id: str,
    request_id: str,
    payload_json: dict,
):
    """在当前事务里追加审计日志，但不立即提交，便于多步动作原子完成。"""
    record = models.AuditLog(
        actor=actor,
        action_type=action_type,
        target_type=target_type,
        target_id=target_id,
        request_id=request_id,
        payload_json=payload_json,
    )
    db.add(record)
    return record


def _memory_enabled_for_actor(actor: str) -> bool:
    """只有明确登录用户才启用短期记忆，避免 anonymous 共享状态。"""
    normalized_actor = (actor or "").strip()
    return bool(normalized_actor and normalized_actor != "anonymous")


def _serialize_agent_conversation_memory(record) -> dict:
    """把短期记忆 ORM 记录转换成可直接注入上下文的字典。"""
    if record is None:
        return {}
    return {
        "last_ticket_id": str(record.last_ticket_id or "") or None,
        "last_draft_id": str(record.last_draft_id or "") or None,
        "last_tool": str(record.last_tool or "") or None,
        "last_topic_summary": str(record.last_topic_summary or "") or None,
        "updated_at": record.updated_at.isoformat() if record.updated_at is not None else None,
    }


def _serialize_user_memory(record) -> dict:
    """把长期记忆 ORM 记录转换成可直接使用的字典。"""
    if record is None:
        return {}
    return {
        "default_location": str(record.default_location or "") or None,
        "default_contact": str(record.default_contact or "") or None,
        "source_ticket_id": str(record.source_ticket_id or "") or None,
        "updated_at": record.updated_at.isoformat() if record.updated_at is not None else None,
    }


def _load_short_term_memory(db: Session, actor: str) -> dict:
    """按当前用户读取短期对话记忆。"""
    if not _memory_enabled_for_actor(actor):
        return {}
    record = crud.get_agent_conversation_memory(db, actor)
    return _serialize_agent_conversation_memory(record)


def _load_user_memory(db: Session, actor: str) -> dict:
    """按当前用户读取长期默认资料。"""
    if not _memory_enabled_for_actor(actor):
        return {}
    record = crud.get_user_memory(db, actor)
    return _serialize_user_memory(record)


def _text_has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    """判断文本是否包含任一引用恢复触发词。"""
    normalized_text = text or ""
    return any(marker in normalized_text for marker in markers)


def _infer_ticket_id_from_memory(
    text: str,
    memory_snapshot: dict | None,
    explicit_draft_id: str | None = None,
) -> str | None:
    """在没有显式 ticket_id 时，尝试从短期记忆恢复最近一次工单。"""
    if explicit_draft_id:
        return None
    last_ticket_id = str((memory_snapshot or {}).get("last_ticket_id") or "").strip()
    if not last_ticket_id:
        return None
    if _extract_ticket_public_id(text):
        return None
    if _text_has_any_marker(text, _SHORT_TERM_TICKET_REFERENCE_MARKERS):
        return last_ticket_id
    return None


def _infer_draft_id_from_memory(
    text: str,
    memory_snapshot: dict | None,
    explicit_ticket_id: str | None = None,
) -> str | None:
    """在没有显式 draft_id 时，尝试从短期记忆恢复最近一次草稿。"""
    if explicit_ticket_id:
        return None
    last_draft_id = str((memory_snapshot or {}).get("last_draft_id") or "").strip()
    if not last_draft_id:
        return None
    if _text_has_any_marker(text, _SHORT_TERM_DRAFT_REFERENCE_MARKERS):
        return last_draft_id
    return None


def _update_short_term_memory_from_response(
    db: Session,
    actor: str,
    text: str,
    response: dict | None,
) -> None:
    """根据本轮响应回写短期记忆，保存最近的 ticket/draft 引用。"""
    if not _memory_enabled_for_actor(actor):
        return

    current = _serialize_agent_conversation_memory(crud.get_agent_conversation_memory(db, actor))
    normalized_response = response or {}
    route = str(normalized_response.get("route") or "")
    ticket_payload = normalized_response.get("ticket") or {}
    draft_payload = normalized_response.get("draft") or {}

    updates = {
        "last_ticket_id": current.get("last_ticket_id"),
        "last_draft_id": current.get("last_draft_id"),
        "last_tool": route or current.get("last_tool"),
        "last_topic_summary": (text or "")[:_SHORT_TERM_MEMORY_SUMMARY_LIMIT] or current.get("last_topic_summary"),
    }

    ticket_id = str(ticket_payload.get("ticket_id") or "").strip()
    if ticket_id:
        updates["last_ticket_id"] = ticket_id

    draft_id = str(draft_payload.get("draft_id") or "").strip()
    draft_status = str(draft_payload.get("status") or "").strip()
    if draft_id:
        if draft_status == "open":
            updates["last_draft_id"] = draft_id
        elif draft_status in {"consumed", "completed", "expired"}:
            updates["last_draft_id"] = None
    elif route == "DRAFT_EXPIRED":
        updates["last_draft_id"] = None

    crud.upsert_agent_conversation_memory(
        db,
        actor,
        **updates,
    )


def _update_user_memory_from_ticket_facts(
    db: Session,
    actor: str,
    *,
    location: str | None = None,
    contact: str | None = None,
    source_ticket_id: str | None = None,
) -> None:
    """根据最新建单事实更新用户长期默认资料。"""
    if not _memory_enabled_for_actor(actor):
        return

    normalized_location = _clean_value(location)
    normalized_contact = _clean_value(contact)
    if normalized_location in (None, "", []) and normalized_contact in (None, "", []):
        return

    current = _serialize_user_memory(crud.get_user_memory(db, actor))
    crud.upsert_user_memory(
        db,
        actor,
        default_location=normalized_location or current.get("default_location"),
        default_contact=normalized_contact or current.get("default_contact"),
        source_ticket_id=str(source_ticket_id or "") or current.get("source_ticket_id"),
    )


def _apply_user_memory_defaults(payload: dict, user_memory_snapshot: dict | None) -> tuple[dict, dict]:
    """在建单路径中，用长期记忆补全 location/contact 缺口。"""
    normalized_payload = dict(payload or {})
    memory_snapshot = dict(user_memory_snapshot or {})
    applied = {
        "location": False,
        "contact": False,
    }

    remembered_location = _clean_value(memory_snapshot.get("default_location"))
    remembered_contact = _clean_value(memory_snapshot.get("default_contact"))

    if _clean_value(normalized_payload.get("location")) in (None, "", []) and remembered_location not in (None, "", []):
        normalized_payload["location"] = remembered_location
        applied["location"] = True

    if _clean_value(normalized_payload.get("contact")) in (None, "", []) and remembered_contact not in (None, "", []):
        normalized_payload["contact"] = remembered_contact
        applied["contact"] = True

    return normalized_payload, applied


def _build_memory_applied_payload(payload: dict, applied_flags: dict | None) -> dict | None:
    """把已应用的长期记忆字段整理成可直接返回给前端的结构。"""
    normalized_flags = dict(applied_flags or {})
    visible: dict[str, str] = {}
    if normalized_flags.get("location"):
        visible["location"] = str(payload.get("location") or "")
    if normalized_flags.get("contact"):
        visible["contact"] = str(payload.get("contact") or "")
    if not visible:
        return None
    visible["source"] = "user_memory"
    return visible


def _append_memory_applied_notice(message: str, memory_applied: dict | None) -> str:
    """在用户可见消息里明确提示哪些字段是沿用长期记忆补上的。"""
    if not memory_applied:
        return message

    labels: list[str] = []
    if memory_applied.get("location"):
        labels.append("地点")
    if memory_applied.get("contact"):
        labels.append("联系方式")
    if not labels:
        return message

    suffix = "已沿用上次" + "和".join(labels) + "作为默认信息。"
    base_message = (message or "").strip()
    if not base_message:
        return suffix
    return base_message + suffix


def _build_missing_ticket_reference_response() -> dict:
    """当用户引用“上一单/那张单”但系统无法恢复对象时，返回明确追问。"""
    return {
        "route": "NEED_MORE_INFO",
        "message": "我没找到你提到的“上一单”，请提供工单号（TCK-…）或让我知道你要补充的是哪张单。请发：TCK-2026-XXXXXX + 要补充的内容。",
        "missing_fields": ["ticket_id"],
    }


def _needs_ticket_reference_clarification(
    text: str,
    *,
    resolved_ticket_id: str | None = None,
    effective_draft_id: str | None = None,
) -> bool:
    """判断是否需要在路由前直接追问缺失的 ticket_id。"""
    if resolved_ticket_id or effective_draft_id:
        return False
    return _text_has_any_marker(text, _SHORT_TERM_TICKET_REFERENCE_MARKERS)



def update_ticket_status_workflow(
    db: Session,
    ticket_id: str,
    status: str,
    actor: str | None = None,
    audit_source: str | None = None,
) -> dict:
    """更新工单状态，并记录审计日志。"""
    ticket = crud.get_ticket_by_public_id(db, ticket_id)
    if ticket is None:
        raise LookupError(f"ticket_not_found:{ticket_id}")

    updated = crud.update_ticket_status(db, ticket, status)
    crud.create_audit_log(
        db,
        {
            "actor": actor or "anonymous",
            "action_type": "UPDATE_TICKET",
            "target_type": "TICKET",
            "target_id": updated.public_id,
            "request_id": _new_request_id(),
            "payload_json": _with_audit_source({"status": updated.status}, audit_source),
        },
    )
    return serialize_ticket_detail(db, updated)


def add_ticket_comment_workflow(
    db: Session,
    ticket_id: str,
    comment: str,
    actor: str | None = None,
    audit_source: str | None = None,
) -> dict:
    """向工单追加说明，改为 append-only 写入独立评论表。"""
    ticket = crud.get_ticket_by_public_id(db, ticket_id)
    if ticket is None:
        raise LookupError(f"ticket_not_found:{ticket_id}")

    normalized_actor = actor or "anonymous"
    normalized_comment = (comment or "").strip() or "用户未提供补充内容。"
    audit_request_id = _new_request_id()
    comment_record = models.TicketComment(
        ticket_id=ticket.id,
        actor_user_id=normalized_actor,
        content=normalized_comment,
    )

    ticket.updated_at = _utc_now()
    db.add(comment_record)
    db.add(ticket)
    db.flush()
    _append_audit_log_uncommitted(
        db,
        actor=normalized_actor,
        action_type="ADD_TICKET_COMMENT",
        target_type="TICKET",
        target_id=ticket.public_id,
        request_id=audit_request_id,
        payload_json=_with_audit_source(
            {
                "comment_id": comment_record.comment_id,
                "content": normalized_comment[:_AUDIT_COMMENT_PREVIEW_LIMIT],
            },
            audit_source,
        ),
    )
    db.commit()
    db.refresh(ticket)
    return serialize_ticket_detail(db, ticket)


def escalate_ticket_workflow(
    db: Session,
    ticket_id: str,
    actor: str | None = None,
    reason: str | None = None,
    audit_source: str | None = None,
) -> dict:
    """催办工单：记录审计，并把状态推进到 in_progress。"""
    ticket = crud.get_ticket_by_public_id_for_update(db, ticket_id)
    if ticket is None:
        raise LookupError(f"ticket_not_found:{ticket_id}")

    normalized_actor = actor or "anonymous"
    normalized_reason = (reason or "").strip() or "用户请求催办。"
    audit_request_id = _new_request_id()
    context = dict(ticket.context_json or {})
    escalation_count = int(context.get("escalation_count") or 0) + 1
    context["escalation_count"] = escalation_count
    context["last_escalation_reason"] = normalized_reason
    context["last_escalated_by"] = normalized_actor
    context["last_escalated_at"] = _utc_now().isoformat()
    ticket.context_json = context
    if ticket.status not in ("resolved", "closed", "cancelled"):
        ticket.status = "in_progress"
    db.add(ticket)
    _append_audit_log_uncommitted(
        db,
        actor=normalized_actor,
        action_type="ESCALATE_TICKET",
        target_type="TICKET",
        target_id=ticket.public_id,
        request_id=audit_request_id,
        payload_json=_with_audit_source(
            {
                "reason": normalized_reason,
                "status": ticket.status,
                "escalation_count": escalation_count,
            },
            audit_source,
        ),
    )
    db.commit()
    db.refresh(ticket)
    return serialize_ticket_detail(db, ticket)


def cancel_ticket_workflow(
    db: Session,
    ticket_id: str,
    actor: str | None = None,
    reason: str | None = None,
    audit_source: str | None = None,
) -> dict:
    """取消工单：写原因并把状态设为 cancelled。"""
    ticket = crud.get_ticket_by_public_id_for_update(db, ticket_id)
    if ticket is None:
        raise LookupError(f"ticket_not_found:{ticket_id}")

    normalized_actor = actor or "anonymous"
    normalized_reason = (reason or "").strip() or "用户未提供取消原因。"
    audit_request_id = _new_request_id()
    context = dict(ticket.context_json or {})
    context["cancel_reason"] = normalized_reason
    context["cancelled_by"] = normalized_actor
    context["cancelled_at"] = _utc_now().isoformat()
    ticket.context_json = context
    ticket.status = "cancelled"
    db.add(ticket)
    _append_audit_log_uncommitted(
        db,
        actor=normalized_actor,
        action_type="CANCEL_TICKET",
        target_type="TICKET",
        target_id=ticket.public_id,
        request_id=audit_request_id,
        payload_json=_with_audit_source(
            {"reason": normalized_reason, "status": ticket.status},
            audit_source,
        ),
    )
    db.commit()
    db.refresh(ticket)
    return serialize_ticket_detail(db, ticket)



def _build_need_more_info_response(
    message: str,
    missing_fields: list[str],
    draft=None,
    kb_result: dict | None = None,
    extraction_payload: dict | None = None,
    extractor: str = "draft_merge",
    memory_applied: dict | None = None,
) -> dict:
    """统一构造 `NEED_MORE_INFO` 响应。"""
    response = {
        "route": "NEED_MORE_INFO",
        "message": message,
        "missing_fields": list(missing_fields),
        "memory_applied": memory_applied,
        "draft": serialize_ticket_draft(draft) if draft is not None else None,
        "kb": public_kb_response(kb_result) if isinstance(kb_result, dict) else None,
        "extraction": _build_extraction_view(extraction_payload, missing_fields, extractor)
        if extraction_payload is not None
        else None,
    }
    return response



def _create_draft_for_missing_info(
    db: Session,
    actor: str,
    actor_department: str,
    chain_request_id: str,
    payload: dict,
    missing_fields: list[str],
) -> dict:
    """创建新草稿并写入草稿审计。"""
    draft = crud.create_ticket_draft(
        db,
        {
            "draft_id": _new_draft_public_id(),
            "creator": actor,
            "owner_user_id": actor,
            "department": actor_department,
            "payload_json": payload,
            "missing_fields_json": list(missing_fields),
            "status": "open",
            "expires_at": _draft_expiry(),
            "kb_request_id": chain_request_id,
        },
    )

    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": "DRAFT_CREATED",
            "target_type": "TICKET_DRAFT",
            "target_id": draft.draft_id,
            "request_id": chain_request_id,
            "payload_json": {
                "missing_fields": list(missing_fields),
                "expires_at": draft.expires_at.isoformat(),
                "payload": payload,
            },
        },
    )

    return _build_need_more_info_response(
        message=_build_need_more_info_message(missing_fields),
        missing_fields=missing_fields,
        draft=draft,
        extraction_payload=payload,
        extractor="initial_extract",
    )



def _build_draft_ticket_success_response(
    ticket,
    draft,
    merged_payload: dict,
    message: str,
) -> dict:
    """统一构造草稿续办成功响应，幂等命中和首次建单都复用这一结构。"""
    return {
        "route": "CREATE_TICKET",
        "ticket": {"ticket_id": ticket.public_id, "status": ticket.status},
        "draft": serialize_ticket_draft(draft),
        "extraction": _build_extraction_view(merged_payload, [], "draft_merge"),
        "message": message,
    }


def _resume_ticket_draft_workflow(
    db: Session,
    draft_id: str,
    text: str,
    fields: dict | None,
    actor: str,
    actor_department: str,
) -> dict:
    """续办一个已存在的工单草稿，并保证“建单 + 消费草稿”只发生一次。"""
    lock_stmt = (
        select(models.TicketDraft)
        .where(models.TicketDraft.draft_id == draft_id)
        .with_for_update()
    )
    draft = db.execute(lock_stmt).scalar_one_or_none()
    if draft is None:
        crud.create_audit_log(
            db,
            {
                "actor": actor,
                "action_type": "DRAFT_NOT_FOUND",
                "target_type": "TICKET_DRAFT",
                "target_id": draft_id,
                "request_id": _new_request_id(),
                "payload_json": {"text": text},
            },
        )
        raise LookupError("draft_not_found")

    chain_request_id = str(draft.kb_request_id or _new_request_id())
    if str(draft.owner_user_id or draft.creator or "anonymous") != actor:
        crud.create_audit_log(
            db,
            {
                "actor": actor,
                "action_type": "DRAFT_FORBIDDEN",
                "target_type": "TICKET_DRAFT",
                "target_id": draft.draft_id,
                "request_id": chain_request_id,
                "payload_json": {
                    "owner_user_id": draft.owner_user_id,
                    "draft_status": draft.status,
                },
            },
        )
        raise PermissionError("draft_forbidden")

    existing_ticket = crud.get_ticket_by_source_draft_id(db, draft.draft_id)
    if str(draft.status or "open") in ("consumed", "completed"):
        if existing_ticket is None:
            crud.create_audit_log(
                db,
                {
                    "actor": actor,
                    "action_type": "DRAFT_ALREADY_CONSUMED",
                    "target_type": "TICKET_DRAFT",
                    "target_id": draft.draft_id,
                    "request_id": chain_request_id,
                    "payload_json": {"ticket_lookup": "missing"},
                },
            )
            raise LookupError("draft_not_found")

        crud.create_audit_log(
            db,
            {
                "actor": actor,
                "action_type": "DRAFT_ALREADY_CONSUMED",
                "target_type": "TICKET_DRAFT",
                "target_id": draft.draft_id,
                "request_id": chain_request_id,
                "payload_json": {"ticket_id": existing_ticket.public_id},
            },
        )
        crud.create_audit_log(
            db,
            {
                "actor": actor,
                "action_type": "AGENT_ROUTE",
                "target_type": "TICKET",
                "target_id": existing_ticket.public_id,
                "request_id": chain_request_id,
                "payload_json": {
                    "route": "CREATE_TICKET",
                    "draft_id": draft.draft_id,
                    "continued": True,
                    "idempotent_hit": True,
                },
            },
        )
        return _build_draft_ticket_success_response(
            existing_ticket,
            draft,
            dict(draft.payload_json or {}),
            "该草稿已处理，直接返回既有工单。",
        )

    now = _utc_now()
    draft_expires_at = _normalize_datetime(draft.expires_at) or now
    if str(draft.status or "open") == "expired" or draft_expires_at <= now:
        draft.status = "expired"
        db.add(draft)
        _append_audit_log_uncommitted(
            db,
            actor=actor,
            action_type="DRAFT_EXPIRED",
            target_type="TICKET_DRAFT",
            target_id=draft.draft_id,
            request_id=chain_request_id,
            payload_json={"expired_at": draft_expires_at.isoformat()},
        )
        _append_audit_log_uncommitted(
            db,
            actor=actor,
            action_type="AGENT_ROUTE",
            target_type="TICKET_DRAFT",
            target_id=draft.draft_id,
            request_id=chain_request_id,
            payload_json={"route": "DRAFT_EXPIRED", "draft_id": draft.draft_id},
        )
        db.commit()
        db.refresh(draft)
        return {
            "route": "DRAFT_EXPIRED",
            "message": "该草稿已过期，请重新描述问题。",
            "missing_fields": list(draft.missing_fields_json or []),
            "draft": serialize_ticket_draft(draft),
            "extraction": _build_extraction_view(
                draft.payload_json or {},
                list(draft.missing_fields_json or []),
                "expired_draft",
            ),
        }

    merged_payload = dict(draft.payload_json or {})
    if (text or "").strip():
        extracted = extract_ticket_payload(text, actor, actor_department)
        merged_payload = _merge_draft_payload(merged_payload, extracted)
    if isinstance(fields, dict):
        merged_payload = _merge_draft_payload(merged_payload, fields)

    merged_payload.setdefault("creator", draft.creator or actor)
    merged_payload.setdefault("department", draft.department or actor_department)
    if draft.kb_request_id and not merged_payload.get("source_text"):
        existing_payload = draft.payload_json if isinstance(draft.payload_json, dict) else {}
        merged_payload["source_text"] = str(existing_payload.get("source_text") or "")

    missing_fields = _compute_missing_fields(merged_payload)
    if missing_fields:
        draft.payload_json = merged_payload
        draft.missing_fields_json = list(missing_fields)
        draft.status = "open"
        draft.expires_at = _draft_expiry()
        db.add(draft)
        _append_audit_log_uncommitted(
            db,
            actor=actor,
            action_type="DRAFT_UPDATED",
            target_type="TICKET_DRAFT",
            target_id=draft.draft_id,
            request_id=chain_request_id,
            payload_json={
                "missing_fields": list(missing_fields),
                "payload": merged_payload,
                "has_text_patch": bool((text or "").strip()),
                "has_field_patch": bool(fields),
            },
        )
        _append_audit_log_uncommitted(
            db,
            actor=actor,
            action_type="AGENT_ROUTE",
            target_type="TICKET_DRAFT",
            target_id=draft.draft_id,
            request_id=chain_request_id,
            payload_json={
                "route": "NEED_MORE_INFO",
                "draft_id": draft.draft_id,
                "missing_fields": list(missing_fields),
            },
        )
        db.commit()
        db.refresh(draft)
        return _build_need_more_info_response(
            message=_build_need_more_info_message(missing_fields),
            missing_fields=missing_fields,
            draft=draft,
            extraction_payload=merged_payload,
            extractor="draft_merge",
        )

    try:
        ticket = models.Ticket(
            public_id=_new_ticket_public_id(),
            creator=str(merged_payload.get("creator") or actor),
            department=str(merged_payload.get("department") or actor_department),
            category=str(merged_payload.get("category") or "other"),
            priority=str(merged_payload.get("priority") or "P2"),
            title=str(merged_payload.get("title") or "通用服务请求"),
            description=str(merged_payload.get("description") or text or "用户未提供详细描述。"),
            status="open",
            contact=merged_payload.get("contact"),
            source_draft_id=draft.draft_id,
            context_json={
                "kb_request_id": draft.kb_request_id,
                "kb_attempt_stage": merged_payload.get("kb_attempt_stage"),
                "location": merged_payload.get("location"),
                "source_text": merged_payload.get("source_text"),
                "draft_id": draft.draft_id,
            },
        )
        draft.payload_json = merged_payload
        draft.missing_fields_json = []
        draft.status = "consumed"
        db.add(ticket)
        db.add(draft)
        db.flush()

        _append_audit_log_uncommitted(
            db,
            actor=actor,
            action_type="CREATE_TICKET",
            target_type="TICKET",
            target_id=ticket.public_id,
            request_id=chain_request_id,
            payload_json={
                "department": ticket.department,
                "category": ticket.category,
                "priority": ticket.priority,
                "status": ticket.status,
                "title": ticket.title,
                "context": ticket.context_json,
            },
        )
        _append_audit_log_uncommitted(
            db,
            actor=actor,
            action_type="DRAFT_CONSUMED",
            target_type="TICKET_DRAFT",
            target_id=draft.draft_id,
            request_id=chain_request_id,
            payload_json={
                "ticket_id": ticket.public_id,
                "payload": merged_payload,
            },
        )
        _append_audit_log_uncommitted(
            db,
            actor=actor,
            action_type="AGENT_ROUTE",
            target_type="TICKET",
            target_id=ticket.public_id,
            request_id=chain_request_id,
            payload_json={
                "route": "CREATE_TICKET",
                "draft_id": draft.draft_id,
                "continued": True,
                "idempotent_hit": False,
            },
        )
        db.commit()
        db.refresh(ticket)
        db.refresh(draft)
        _update_user_memory_from_ticket_facts(
            db,
            actor,
            location=merged_payload.get("location"),
            contact=merged_payload.get("contact"),
            source_ticket_id=ticket.public_id,
        )
        return _build_draft_ticket_success_response(
            ticket,
            draft,
            merged_payload,
            "草稿信息已补全，已自动创建工单。",
        )
    except IntegrityError:
        db.rollback()
        recovered_draft = crud.get_ticket_draft_by_draft_id(db, draft_id)
        existing_ticket = crud.get_ticket_by_source_draft_id(db, draft_id)
        if existing_ticket is None or recovered_draft is None:
            raise

        chain_request_id = str(recovered_draft.kb_request_id or _new_request_id())
        crud.create_audit_log(
            db,
            {
                "actor": actor,
                "action_type": "DRAFT_ALREADY_CONSUMED",
                "target_type": "TICKET_DRAFT",
                "target_id": recovered_draft.draft_id,
                "request_id": chain_request_id,
                "payload_json": {"ticket_id": existing_ticket.public_id, "via": "unique_conflict"},
            },
        )
        crud.create_audit_log(
            db,
            {
                "actor": actor,
                "action_type": "AGENT_ROUTE",
                "target_type": "TICKET",
                "target_id": existing_ticket.public_id,
                "request_id": chain_request_id,
                "payload_json": {
                    "route": "CREATE_TICKET",
                    "draft_id": recovered_draft.draft_id,
                    "continued": True,
                    "idempotent_hit": True,
                },
            },
        )
        recovered_payload = dict(recovered_draft.payload_json or {})
        _update_user_memory_from_ticket_facts(
            db,
            actor,
            location=recovered_payload.get("location"),
            contact=recovered_payload.get("contact"),
            source_ticket_id=existing_ticket.public_id,
        )
        return _build_draft_ticket_success_response(
            existing_ticket,
            recovered_draft,
            recovered_payload,
            "该草稿已处理，直接返回既有工单。",
        )


def _handle_kb_intent(
    db: Session,
    text: str,
    actor: str,
    actor_department: str,
) -> dict:
    """按既有 ASK 路径处理纯知识问答。"""
    kb_result = run_ask_workflow(db, text, actor, actor_department)
    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": "AGENT_ROUTE",
            "target_type": "KB_QUERY",
            "target_id": kb_result["query_id"],
            "request_id": kb_result["request_id"],
            "payload_json": {"route": "ASK", "text": text},
        },
    )
    return {"route": "ASK", "kb": public_kb_response(kb_result)}


def _handle_create_ticket_intent(
    db: Session,
    text: str,
    actor: str,
    actor_department: str,
    planner_fields: dict | None = None,
) -> dict:
    """按既有建单链路处理：先问答，再抽取，再建单或进入草稿。"""
    kb_result = run_ask_workflow(db, text, actor, actor_department)
    extracted = extract_ticket_payload(text, actor, actor_department)
    if isinstance(planner_fields, dict):
        extracted = _merge_draft_payload(extracted, planner_fields)
    user_memory_snapshot = _load_user_memory(db, actor)

    payload = _build_ticket_payload(
        extracted,
        actor=actor,
        actor_department=actor_department,
        source_text=text,
        kb_attempt_stage=str(kb_result["meta"].get("attempt_stage") or "unknown"),
    )
    payload, memory_applied = _apply_user_memory_defaults(payload, user_memory_snapshot)
    memory_applied_payload = _build_memory_applied_payload(payload, memory_applied)
    missing_fields = _compute_missing_fields(payload)
    chain_request_id = str(kb_result["request_id"])

    if missing_fields:
        draft_response = _create_draft_for_missing_info(
            db,
            actor=actor,
            actor_department=actor_department,
            chain_request_id=chain_request_id,
            payload=payload,
            missing_fields=missing_fields,
        )
        if memory_applied_payload is not None:
            draft_response["memory_applied"] = memory_applied_payload
            draft_response["message"] = _append_memory_applied_notice(
                str(draft_response.get("message") or ""),
                memory_applied_payload,
            )
        draft_info = draft_response.get("draft") or {}
        crud.create_audit_log(
            db,
            {
                "actor": actor,
                "action_type": "AGENT_ROUTE",
                "target_type": "TICKET_DRAFT",
                "target_id": str(draft_info.get("draft_id") or ""),
                "request_id": chain_request_id,
                "payload_json": {
                    "route": "NEED_MORE_INFO",
                    "text": text,
                    "missing_fields": list(missing_fields),
                    "draft_id": draft_info.get("draft_id"),
                    "user_memory_applied": {key: value for key, value in memory_applied.items() if value},
                },
            },
        )
        draft_response["kb"] = public_kb_response(kb_result)
        return draft_response

    ticket = create_ticket_workflow(
        db,
        creator=actor,
        department=str(payload.get("department") or actor_department),
        category=str(payload.get("category") or "other"),
        priority=str(payload.get("priority") or "P2"),
        title=str(payload.get("title") or "通用服务请求"),
        description=str(payload.get("description") or text or "用户未提供详细描述。"),
        contact=payload.get("contact"),
        context={
            "kb_request_id": kb_result["request_id"],
            "kb_attempt_stage": kb_result["meta"].get("attempt_stage"),
            "location": payload.get("location"),
            "source_text": payload.get("source_text"),
        },
        request_id=chain_request_id,
        audit_action="CREATE_TICKET",
    )

    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": "AGENT_ROUTE",
            "target_type": "TICKET",
            "target_id": ticket["ticket_id"],
            "request_id": chain_request_id,
            "payload_json": {
                "route": "CREATE_TICKET",
                "text": text,
                "extraction": extracted,
                "kb_request_id": kb_result["request_id"],
                "planner_fields_applied": bool(isinstance(planner_fields, dict) and planner_fields),
                "user_memory_applied": {key: value for key, value in memory_applied.items() if value},
            },
        },
    )

    return {
        "route": "CREATE_TICKET",
        "message": _append_memory_applied_notice("", memory_applied_payload) if memory_applied_payload else None,
        "memory_applied": memory_applied_payload,
        "ticket": {"ticket_id": ticket["ticket_id"], "status": ticket["status"]},
        "kb": public_kb_response(kb_result),
        "extraction": _build_extraction_view(payload, [], "initial_extract"),
    }


def _run_agent_workflow_rules(
    db: Session,
    text: str,
    actor: str,
    actor_department: str,
    draft_id: str | None = None,
    resolved_ticket_id: str | None = None,
    fields: dict | None = None,
    confirm_token: str | None = None,
) -> dict:
    """保留原始规则路由，作为 `rules` 模式与 `hybrid` 回退底线。"""
    if draft_id:
        return _resume_ticket_draft_workflow(
            db,
            draft_id=str(draft_id),
            text=text,
            fields=fields,
            actor=actor,
            actor_department=actor_department,
        )

    if confirm_token:
        return _handle_confirmed_pending_action(
            db,
            confirm_token=str(confirm_token),
            actor=actor,
            text=text,
        )

    ticket_id = str(resolved_ticket_id or "") or _extract_ticket_public_id(text)
    if ticket_id:
        action = _route_ticket_tool_action_by_rules(text)
        return _handle_ticket_tool_route_rules(db, action, ticket_id, text, actor)

    if not _should_route_to_ticket(text):
        return _handle_kb_intent(db, text, actor, actor_department)

    return _handle_create_ticket_intent(db, text, actor, actor_department, planner_fields=None)


def _global_plan_target(
    plan: ToolPlan,
    request_id: str,
    provided_ticket_id: str,
    provided_draft_id: str,
) -> tuple[str, str]:
    """根据全局 plan 选择一个稳定的审计目标。"""
    if plan.tool == "continue_ticket_draft":
        return "TICKET_DRAFT", provided_draft_id or request_id
    if plan.tool == "ticket_tool_planner":
        return "TICKET", provided_ticket_id or request_id
    if plan.tool == "kb_answer":
        return "KB_QUERY", request_id
    return "AGENT", request_id


def _validate_global_plan(
    db: Session,
    plan: ToolPlan,
    actor: str,
    request_id: str,
    normalized_text: str,
    provided_ticket_id: str,
    provided_draft_id: str,
) -> dict:
    """对白名单、schema 与系统提供 ID 做最小硬校验。"""
    args_model = _global_plan_args_model(plan.tool)
    target_type, target_id = _global_plan_target(plan, request_id, provided_ticket_id, provided_draft_id)

    if args_model is None:
        return {
            "status": "fallback",
            "reason": "tool_not_supported_by_global_planner",
            "target_type": target_type,
            "target_id": target_id,
        }

    if plan.missing_fields:
        _audit_global_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            target_type=target_type,
            target_id=target_id,
            request_id=request_id,
            payload_json={
                "tool": plan.tool,
                "reason": "missing_fields_from_plan",
                "missing_fields": list(plan.missing_fields),
            },
        )
        return {
            "status": "response",
            "response": _build_tool_plan_missing_fields_response(plan.tool, list(plan.missing_fields)),
        }

    try:
        normalized_args = _dump_pydantic_model(_validate_pydantic_model(args_model, plan.args or {}))
    except ValidationError as exc:
        missing_fields = _extract_missing_fields_from_validation_error(exc)
        if missing_fields:
            _audit_global_plan_event(
                db,
                actor=actor,
                action_type="PLAN_REJECTED",
                target_type=target_type,
                target_id=target_id,
                request_id=request_id,
                payload_json={
                    "tool": plan.tool,
                    "reason": "schema_missing_fields",
                    "missing_fields": list(missing_fields),
                },
            )
            return {
                "status": "response",
                "response": _build_tool_plan_missing_fields_response(plan.tool, missing_fields),
            }
        return {
            "status": "fallback",
            "reason": "schema_invalid",
            "target_type": target_type,
            "target_id": target_id,
        }

    if plan.tool == "continue_ticket_draft":
        if not provided_draft_id:
            return {
                "status": "fallback",
                "reason": "draft_id_missing",
                "target_type": target_type,
                "target_id": target_id,
            }
        normalized_args["draft_id"] = provided_draft_id
    elif plan.tool == "ticket_tool_planner":
        if not provided_ticket_id:
            return {
                "status": "fallback",
                "reason": "ticket_id_missing",
                "target_type": target_type,
                "target_id": target_id,
            }
        normalized_args["ticket_id"] = provided_ticket_id
        normalized_args["raw_text"] = normalized_text or str(normalized_args.get("raw_text") or "")
    elif plan.tool == "kb_answer":
        normalized_args["query"] = normalized_text or str(normalized_args.get("query") or "")
    elif plan.tool == "create_ticket":
        normalized_args["text"] = normalized_text or str(normalized_args.get("text") or "")

    return {
        "status": "validated",
        "args": normalized_args,
        "target_type": target_type,
        "target_id": target_id,
    }


def _handle_global_planner_route(
    db: Session,
    text: str,
    actor: str,
    actor_department: str,
    mode: str,
    draft_id: str | None = None,
    resolved_ticket_id: str | None = None,
    memory_snapshot: dict | None = None,
    fields: dict | None = None,
    confirm_token: str | None = None,
) -> dict:
    """执行 Global Planner；仅做分支选择，ticket 细粒度再交给子规划器。"""
    if confirm_token:
        return _handle_confirmed_pending_action(
            db,
            confirm_token=str(confirm_token),
            actor=actor,
            text=text,
        )

    explicit_ticket_in_text = bool(_extract_ticket_public_id(text))
    explicit_draft_provided = bool(str(draft_id or "").strip())
    provided_ticket_id = str(resolved_ticket_id or "") or _extract_ticket_public_id(text) or ""
    provided_draft_id = str(draft_id or "").strip()
    request_id = _new_request_id()
    planner_context = {
        "actor_user_id": actor,
        "provided_ticket_id": provided_ticket_id,
        "provided_draft_id": provided_draft_id,
        "has_ticket_id": bool(provided_ticket_id),
        "has_draft_id": bool(provided_draft_id),
        "confirm_token_present": False,
        "ticket_tool_mode": bool(provided_ticket_id),
        "draft_mode": bool(provided_draft_id),
        "short_term_memory": dict(memory_snapshot or {}),
        "memory_ticket_applied": bool(
            not explicit_ticket_in_text and (memory_snapshot or {}).get("last_ticket_id") and provided_ticket_id
        ),
        "memory_draft_applied": bool(
            not explicit_draft_provided and (memory_snapshot or {}).get("last_draft_id") and provided_draft_id
        ),
    }
    tools_json = list_global_planner_skill_contracts()

    try:
        plan = planner.run_global_planner(
            user_text=text,
            tools_json=tools_json,
            context=planner_context,
        )
    except PlannerError as exc:
        _audit_global_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            target_type="AGENT",
            target_id=request_id,
            request_id=request_id,
            payload_json={
                "reason": exc.code,
                "planner_mode": mode,
                "planner_scope": "global",
            },
        )
        if mode == "hybrid" and exc.fallback_eligible:
            return _run_agent_workflow_rules(
                db,
                text=text,
                actor=actor,
                actor_department=actor_department,
                draft_id=draft_id,
                resolved_ticket_id=provided_ticket_id,
                fields=fields,
                confirm_token=None,
            )
        return _build_plan_rejected_response("智能规划失败，当前未执行任何操作。")

    target_type, target_id = _global_plan_target(plan, request_id, provided_ticket_id, provided_draft_id)
    _audit_global_plan_event(
        db,
        actor=actor,
        action_type="PLAN_PROPOSED",
        target_type=target_type,
        target_id=target_id,
        request_id=request_id,
        payload_json={
            "tool": plan.tool,
            "args": _plan_args_summary(plan.args),
            "need_confirmation": bool(plan.need_confirmation),
            "missing_fields": list(plan.missing_fields or []),
            "planner_mode": mode,
            "planner_scope": "global",
        },
    )

    validation_result = _validate_global_plan(
        db,
        plan=plan,
        actor=actor,
        request_id=request_id,
        normalized_text=text,
        provided_ticket_id=provided_ticket_id,
        provided_draft_id=provided_draft_id,
    )
    status = str(validation_result.get("status") or "")
    if status == "response":
        return validation_result["response"]
    if status == "fallback":
        _audit_global_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            target_type=str(validation_result.get("target_type") or target_type),
            target_id=str(validation_result.get("target_id") or target_id),
            request_id=request_id,
            payload_json={
                "tool": plan.tool,
                "reason": str(validation_result.get("reason") or "validation_failed"),
                "planner_scope": "global",
            },
        )
        if mode == "hybrid":
            return _run_agent_workflow_rules(
                db,
                text=text,
                actor=actor,
                actor_department=actor_department,
                draft_id=draft_id,
                resolved_ticket_id=provided_ticket_id,
                fields=fields,
                confirm_token=None,
            )
        return _build_plan_rejected_response("规划结果未通过校验，当前未执行任何操作。")

    validated_args = validation_result["args"]
    _audit_global_plan_event(
        db,
        actor=actor,
        action_type="PLAN_EXECUTED",
        target_type=str(validation_result.get("target_type") or target_type),
        target_id=str(validation_result.get("target_id") or target_id),
        request_id=request_id,
        payload_json={
            "tool": plan.tool,
            "args": _plan_args_summary(validated_args),
            "planner_scope": "global",
        },
    )

    if plan.tool == "continue_ticket_draft":
        return _resume_ticket_draft_workflow(
            db,
            draft_id=str(validated_args.get("draft_id") or ""),
            text=text,
            fields=validated_args.get("fields"),
            actor=actor,
            actor_department=actor_department,
        )
    if plan.tool == "ticket_tool_planner":
        return _handle_ticket_tool_route_with_planner(
            db,
            ticket_id=str(validated_args.get("ticket_id") or ""),
            text=str(validated_args.get("raw_text") or text),
            actor=actor,
            mode=mode,
        )
    if plan.tool == "kb_answer":
        return _handle_kb_intent(
            db,
            text=str(validated_args.get("query") or text),
            actor=actor,
            actor_department=actor_department,
        )
    return _handle_create_ticket_intent(
        db,
        text=str(validated_args.get("text") or text),
        actor=actor,
        actor_department=actor_department,
        planner_fields=validated_args.get("fields"),
    )



def _build_ticket_tool_agent_response(route: str, ticket_detail: dict, message: str) -> dict:
    """统一构造工单工具调用的 Agent 响应。"""
    return {
        "route": route,
        "message": message,
        "ticket": {
            "ticket_id": str(ticket_detail.get("ticket_id") or ""),
            "status": str(ticket_detail.get("status") or ""),
        },
        "ticket_detail": ticket_detail,
    }


def _build_need_confirmation_response(ticket_id: str, confirm_token: str, message: str) -> dict:
    """构造高风险动作的确认态响应。"""
    return {
        "route": "NEED_CONFIRMATION",
        "message": message,
        "confirm_token": confirm_token,
        "ticket": {
            "ticket_id": ticket_id,
            "status": "pending_confirmation",
        },
    }


def _build_plan_rejected_response(message: str) -> dict:
    """统一构造 Planner/Validator 的拒绝响应。"""
    return {
        "route": "PLAN_REJECTED",
        "message": message,
    }


def _build_tool_plan_missing_fields_response(tool_name: str, missing_fields: list[str]) -> dict:
    """当计划缺少必要字段时，返回可继续追问的响应。"""
    return {
        "route": "NEED_MORE_INFO",
        "message": f"{tool_name} 还缺少必要参数：" + "、".join(missing_fields) + "。",
        "missing_fields": list(missing_fields),
    }


def _dump_pydantic_model(model_instance) -> dict:
    """兼容 Pydantic v1/v2 的字典导出。"""
    if hasattr(model_instance, "model_dump"):
        return model_instance.model_dump(exclude_none=True)
    return model_instance.dict(exclude_none=True)


def _actor_satisfies_auth_rule(actor: str, auth_rule: str, ticket) -> bool:
    """根据技能声明的 auth_rule 做最小权限校验。"""
    normalized_actor = (actor or "").strip()
    if auth_rule == "owner_or_admin":
        if normalized_actor in _ADMIN_ACTORS:
            return True
        return normalized_actor == str(ticket.creator or "")
    return bool(normalized_actor)


def _build_ticket_tool_dispatch_args(skill, ticket_id: str, stripped_text: str) -> dict:
    """根据技能定义生成当前规则路由下的最小参数。"""
    args = {"ticket_id": ticket_id}
    if skill.default_text_arg:
        args[skill.default_text_arg] = stripped_text or skill.default_text_value or ""
    return args


def _execute_ticket_tool_skill(
    db: Session,
    skill,
    dispatch_args: dict,
    actor: str,
    text: str,
    request_id: str,
    planner_mode: str,
    planned: ToolPlan | None = None,
) -> dict:
    """统一执行一个已校验通过的 ticket 工具，并写执行审计。"""
    registry = get_ticket_tool_registry()
    ticket_id = str(dispatch_args.get("ticket_id") or "")
    ticket_detail = registry.dispatch(
        skill.name,
        db=db,
        args=dispatch_args,
        actor=actor,
        raw_text=text,
    )
    if planned is not None:
        _audit_plan_event(
            db,
            actor=actor,
            action_type="PLAN_EXECUTED",
            ticket_id=ticket_id,
            request_id=request_id,
            payload_json={
                "tool": planned.tool,
                "args": _plan_args_summary(dispatch_args),
                "need_confirmation": bool(planned.need_confirmation),
                "missing_fields": list(planned.missing_fields or []),
            },
        )
    crud.create_audit_log(
        db,
        {
            "actor": actor,
            "action_type": "AGENT_ROUTE",
            "target_type": "TICKET",
            "target_id": ticket_id,
            "request_id": request_id,
            "payload_json": {
                "route": skill.route_name,
                "tool": skill.name,
                "risk_level": skill.risk_level,
                "planner_mode": planner_mode,
                "text": text,
            },
        },
    )
    return _build_ticket_tool_agent_response(skill.route_name, ticket_detail, skill.success_message)


def _route_ticket_tool_action_by_rules(text: str) -> str:
    """沿用旧规则：关键词识别；未命中则默认查单。"""
    return _detect_ticket_tool_action(text) or "LOOKUP_TICKET"


def _handle_ticket_tool_route_rules(
    db: Session,
    action: str,
    ticket_id: str,
    text: str,
    actor: str,
) -> dict:
    """执行既有的规则路由路径；作为 `rules` 与 `hybrid` 的回退底线。"""
    registry = get_ticket_tool_registry()
    skill = registry.get_by_route(action)
    if skill is None:
        raise ValueError(f"unsupported_ticket_tool_action:{action}")

    stripped_text = _strip_ticket_reference(text, ticket_id)
    dispatch_args = _build_ticket_tool_dispatch_args(skill, ticket_id, stripped_text)
    return _execute_ticket_tool_skill(
        db=db,
        skill=skill,
        dispatch_args=dispatch_args,
        actor=actor,
        text=text,
        request_id=_new_request_id(),
        planner_mode="rules",
        planned=None,
    )


def _issue_pending_confirmation(
    db: Session,
    actor: str,
    tool_name: str,
    ticket_id: str,
    request_id: str,
    args: dict,
) -> dict:
    """为高风险动作创建待确认记录，并返回 `NEED_CONFIRMATION`。"""
    pending_action = models.PendingAction(
        user_id=actor,
        tool_name=tool_name,
        args_json=dict(args or {}),
        status="pending",
        expires_at=_pending_action_expiry(),
    )
    db.add(pending_action)
    db.flush()
    _append_audit_log_uncommitted(
        db,
        actor=actor,
        action_type="NEED_CONFIRMATION",
        target_type="TICKET",
        target_id=ticket_id,
        request_id=request_id,
        payload_json={
            "tool": tool_name,
            "confirm_token_prefix": str(pending_action.confirm_id)[:8],
            "expires_at": pending_action.expires_at.isoformat(),
        },
    )
    db.commit()
    db.refresh(pending_action)
    return _build_need_confirmation_response(
        ticket_id=ticket_id,
        confirm_token=str(pending_action.confirm_id),
        message="这是高风险操作。请携带 confirm_token 再次提交以确认执行。",
    )


def _validate_ticket_tool_plan(
    db: Session,
    plan: ToolPlan,
    actor: str,
    request_id: str,
    *,
    confirmation_verified: bool = False,
) -> dict:
    """对白名单、schema、对象与权限做强校验。"""
    registry = get_ticket_tool_registry()
    skill = registry.get(plan.tool)
    ticket_id = str((plan.args or {}).get("ticket_id") or "")

    if skill is None:
        return {
            "status": "fallback",
            "reason": "tool_not_registered",
            "ticket_id": ticket_id,
        }

    if plan.missing_fields:
        _audit_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            ticket_id=ticket_id,
            request_id=request_id,
            payload_json={
                "tool": plan.tool,
                "reason": "missing_fields_from_plan",
                "missing_fields": list(plan.missing_fields),
            },
        )
        return {
            "status": "response",
            "response": _build_tool_plan_missing_fields_response(plan.tool, list(plan.missing_fields)),
        }

    args_model = _ticket_tool_plan_args_model(plan.tool)
    if args_model is None:
        return {
            "status": "fallback",
            "reason": "args_model_not_registered",
            "ticket_id": ticket_id,
        }

    try:
        normalized_args = _dump_pydantic_model(_validate_pydantic_model(args_model, plan.args or {}))
    except ValidationError as exc:
        missing_fields = _extract_missing_fields_from_validation_error(exc)
        if missing_fields:
            _audit_plan_event(
                db,
                actor=actor,
                action_type="PLAN_REJECTED",
                ticket_id=ticket_id,
                request_id=request_id,
                payload_json={
                    "tool": plan.tool,
                    "reason": "schema_missing_fields",
                    "missing_fields": list(missing_fields),
                },
            )
            return {
                "status": "response",
                "response": _build_tool_plan_missing_fields_response(plan.tool, missing_fields),
            }
        return {
            "status": "fallback",
            "reason": "schema_invalid",
            "ticket_id": ticket_id,
        }

    ticket_id = str(normalized_args.get("ticket_id") or "")
    ticket = crud.get_ticket_by_public_id(db, ticket_id)
    if ticket is None:
        _audit_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            ticket_id=ticket_id,
            request_id=request_id,
            payload_json={
                "tool": plan.tool,
                "reason": "ticket_not_found",
            },
        )
        return {
            "status": "response",
            "response": _build_plan_rejected_response("目标工单不存在，当前未执行操作。"),
        }

    if not _actor_satisfies_auth_rule(actor, skill.auth_rule, ticket):
        _audit_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            ticket_id=ticket_id,
            request_id=request_id,
            payload_json={
                "tool": plan.tool,
                "reason": "auth_rejected",
                "auth_rule": skill.auth_rule,
            },
        )
        return {
            "status": "response",
            "response": _build_plan_rejected_response("当前用户无权执行该操作。"),
        }

    if skill.risk_level == "HIGH" and not confirmation_verified:
        response = _issue_pending_confirmation(
            db,
            actor=actor,
            tool_name=skill.name,
            ticket_id=ticket_id,
            request_id=request_id,
            args=normalized_args,
        )
        return {
            "status": "response",
            "response": response,
        }

    return {
        "status": "validated",
        "skill": skill,
        "args": normalized_args,
    }


def _handle_confirmed_pending_action(
    db: Session,
    confirm_token: str,
    actor: str,
    text: str,
) -> dict:
    """消费一个确认 token，并执行对应的高风险动作。"""
    normalized_token = str(confirm_token or "").strip()
    if not normalized_token:
        return _build_plan_rejected_response("confirm_token 不能为空。")

    pending_action = crud.get_pending_action_by_confirm_id(db, normalized_token)
    if pending_action is None:
        return _build_plan_rejected_response("确认令牌无效或已失效。")

    request_id = _new_request_id()
    ticket_id = str((pending_action.args_json or {}).get("ticket_id") or "")
    if str(pending_action.user_id or "") != actor:
        _audit_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            ticket_id=ticket_id,
            request_id=request_id,
            payload_json={
                "tool": pending_action.tool_name,
                "reason": "confirm_token_actor_mismatch",
                "confirm_token_prefix": normalized_token[:8],
            },
        )
        return _build_plan_rejected_response("该确认令牌不属于当前用户。")

    expires_at = _normalize_datetime(pending_action.expires_at)
    if str(pending_action.status or "") != "pending" or expires_at is None or expires_at <= _utc_now():
        if str(pending_action.status or "") == "pending":
            pending_action.status = "expired"
            db.add(pending_action)
            db.commit()
        _audit_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            ticket_id=ticket_id,
            request_id=request_id,
            payload_json={
                "tool": pending_action.tool_name,
                "reason": "confirm_token_expired_or_consumed",
                "confirm_token_prefix": normalized_token[:8],
            },
        )
        return _build_plan_rejected_response("确认令牌已过期或已使用。")

    plan = _validate_pydantic_model(
        ToolPlan,
        {
            "tool": str(pending_action.tool_name),
            "args": dict(pending_action.args_json or {}),
            "need_confirmation": True,
            "missing_fields": [],
        },
    )
    validation_result = _validate_ticket_tool_plan(
        db,
        plan=plan,
        actor=actor,
        request_id=request_id,
        confirmation_verified=True,
    )
    if validation_result.get("status") != "validated":
        return validation_result.get("response") or _build_plan_rejected_response("确认失败，当前未执行操作。")

    response = _execute_ticket_tool_skill(
        db=db,
        skill=validation_result["skill"],
        dispatch_args=validation_result["args"],
        actor=actor,
        text=text,
        request_id=request_id,
        planner_mode="llm_confirmed",
        planned=plan,
    )
    pending_action.status = "consumed"
    db.add(pending_action)
    db.commit()
    return response


def _handle_ticket_tool_route_with_planner(
    db: Session,
    ticket_id: str,
    text: str,
    actor: str,
    mode: str,
) -> dict:
    """走 LLM Planner -> Validator -> Dispatch；必要时可回退旧规则。"""
    request_id = _new_request_id()
    tools_json = list_ticket_tool_skill_contracts()

    try:
        plan = planner.run_ticket_tool_planner(
            user_text=text,
            provided_ticket_id=ticket_id,
            tools_json=tools_json,
        )
    except PlannerError as exc:
        _audit_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            ticket_id=ticket_id,
            request_id=request_id,
            payload_json={
                "reason": exc.code,
                "planner_mode": mode,
            },
        )
        if mode == "hybrid" and exc.fallback_eligible:
            fallback_action = _route_ticket_tool_action_by_rules(text)
            return _handle_ticket_tool_route_rules(db, fallback_action, ticket_id, text, actor)
        return _build_plan_rejected_response("智能规划失败，当前未执行任何工单操作。")

    _audit_plan_event(
        db,
        actor=actor,
        action_type="PLAN_PROPOSED",
        ticket_id=ticket_id,
        request_id=request_id,
        payload_json={
            "tool": plan.tool,
            "args": _plan_args_summary(plan.args),
            "need_confirmation": bool(plan.need_confirmation),
            "missing_fields": list(plan.missing_fields or []),
            "planner_mode": mode,
        },
    )

    validation_result = _validate_ticket_tool_plan(
        db,
        plan=plan,
        actor=actor,
        request_id=request_id,
        confirmation_verified=False,
    )
    status = str(validation_result.get("status") or "")
    if status == "validated":
        return _execute_ticket_tool_skill(
            db=db,
            skill=validation_result["skill"],
            dispatch_args=validation_result["args"],
            actor=actor,
            text=text,
            request_id=request_id,
            planner_mode=mode,
            planned=plan,
        )
    if status == "response":
        return validation_result["response"]
    if status == "fallback":
        _audit_plan_event(
            db,
            actor=actor,
            action_type="PLAN_REJECTED",
            ticket_id=ticket_id,
            request_id=request_id,
            payload_json={
                "tool": plan.tool,
                "reason": str(validation_result.get("reason") or "validation_failed"),
            },
        )
    if status == "fallback" and mode == "hybrid":
        fallback_action = _route_ticket_tool_action_by_rules(text)
        return _handle_ticket_tool_route_rules(db, fallback_action, ticket_id, text, actor)
    return _build_plan_rejected_response("规划结果未通过校验，当前未执行任何工单操作。")


def _handle_ticket_tool_route(
    db: Session,
    ticket_id: str,
    text: str,
    actor: str,
) -> dict:
    """按 Feature Flag 选择规则路由或 LLM Planner 路径。"""
    mode = _agent_planner_mode()
    if mode == "rules":
        action = _route_ticket_tool_action_by_rules(text)
        return _handle_ticket_tool_route_rules(db, action, ticket_id, text, actor)
    return _handle_ticket_tool_route_with_planner(db, ticket_id, text, actor, mode)


def run_agent_workflow(
    db: Session,
    text: str,
    user: str | None = None,
    department: str | None = None,
    draft_id: str | None = None,
    fields: dict | None = None,
    confirm_token: str | None = None,
) -> dict:
    """执行 Agent 主工作流：规则模式或 Global Planner 模式。"""
    actor = user or "anonymous"
    actor_department = department or "general"
    normalized_text = (text or "").strip()
    memory_snapshot = _load_short_term_memory(db, actor)
    explicit_ticket_id = _extract_ticket_public_id(normalized_text)
    effective_draft_id = str(draft_id or "").strip() or _infer_draft_id_from_memory(
        normalized_text,
        memory_snapshot,
        explicit_ticket_id=explicit_ticket_id,
    )
    resolved_ticket_id = str(explicit_ticket_id or "") or str(
        _infer_ticket_id_from_memory(
            normalized_text,
            memory_snapshot,
            explicit_draft_id=effective_draft_id,
        )
        or ""
    )
    if _needs_ticket_reference_clarification(
        normalized_text,
        resolved_ticket_id=resolved_ticket_id or None,
        effective_draft_id=effective_draft_id or None,
    ):
        return _build_missing_ticket_reference_response()
    mode = _agent_planner_mode()
    if mode == "rules":
        response = _run_agent_workflow_rules(
            db,
            text=normalized_text,
            actor=actor,
            actor_department=actor_department,
            draft_id=effective_draft_id or None,
            resolved_ticket_id=resolved_ticket_id or None,
            fields=fields,
            confirm_token=confirm_token,
        )
        _update_short_term_memory_from_response(
            db,
            actor=actor,
            text=normalized_text,
            response=response,
        )
        return response
    response = _handle_global_planner_route(
        db,
        text=normalized_text,
        actor=actor,
        actor_department=actor_department,
        mode=mode,
        draft_id=effective_draft_id or None,
        resolved_ticket_id=resolved_ticket_id or None,
        memory_snapshot=memory_snapshot,
        fields=fields,
        confirm_token=confirm_token,
    )
    _update_short_term_memory_from_response(
        db,
        actor=actor,
        text=normalized_text,
        response=response,
    )
    return response
