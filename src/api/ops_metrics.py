"""
ASK 运行指标聚合模块。

这个模块只做一件事：基于已经落库的 `kb_queries` 记录，统计最近一段时间内
知识库问答链路的运行质量。不调用模型、不访问向量库它、不修改数据库，因此属于
低风险的只读观测能力。

当前统计范围：
1. 请求量：窗口内 ASK 总数、成功数、失败数。
2. 稳定性：JSON 有效率、引用输出率、repair/fallback 比例。
3. 延迟：检索、回答、总耗时的平均值，以及总耗时 p50/p95。
4. 诊断：失败原因分布、attempt_stage 分布、模型分布。
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api import models


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。

    项目里的 `created_at` 字段按 UTC 时间写入更容易跨环境比较；这里统一使用
    timezone-aware datetime，避免本地时区、容器时区和数据库时区不一致时产生
    统计窗口偏移。
    """
    return datetime.now(timezone.utc)


def _safe_hours(hours: int) -> int:
    """把用户传入的统计窗口小时数限制在安全范围内。

    约束：
    - 最小 1 小时，避免传入 0 或负数导致窗口无意义。
    - 最大 7 天，避免一次 API 请求扫太多历史数据，影响演示环境或小型部署环境。

    FastAPI 路由层也做了 query 参数限制；这里再做一次保护，让函数被其它代码直接
    调用时也保持稳定。
    """
    return max(1, min(int(hours), 24 * 7))


def _rate(numerator: int, denominator: int) -> float:
    """计算比例并保留 4 位小数。

    例如：
    - `valid_json_rate = valid_json_count / total`
    - `citation_rate = citation_count / total`

    当分母为 0 时返回 0.0，避免空数据窗口下出现除零错误。
    """
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _avg(values: list[int]) -> int:
    """计算整数平均值，单位通常是毫秒。

    延迟字段在数据库中以整数毫秒保存，因此这里返回整数，方便 API 输出直接展示。
    空列表返回 0，表示当前窗口内没有可统计请求。
    """
    if not values:
        return 0
    return int(round(sum(values) / len(values)))


def _percentile(values: list[int], percentile: float) -> int:
    """计算近似百分位值，用于观察尾部延迟。

    这里使用 nearest-rank 方法：
    1. 将延迟值从小到大排序。
    2. 根据 `ceil(percentile / 100 * n)` 找到对应排名。
    3. 返回该排名位置的值。

    这个算法实现简单、可解释，适合当前项目的轻量观测场景。p50 代表中位数体验，
    p95 更能暴露少数慢请求对用户体验的影响。
    """
    if not values:
        return 0
    ordered = sorted(values)
    rank = math.ceil((float(percentile) / 100.0) * len(ordered))
    index = max(0, min(rank - 1, len(ordered) - 1))
    return int(ordered[index])


def _is_non_empty_list(value: Any) -> bool:
    """判断 JSON 字段是否是非空列表。

    `citations_json` 在 ORM 层是 JSON 类型，正常情况下应为 list；但为了兼容历史数据、
    测试数据或异常写入，这里先做类型判断，再判断长度。
    """
    return isinstance(value, list) and len(value) > 0


def _failure_reason(record: models.KBQuery) -> str:
    """从一条 KBQuery 记录中提取标准化失败原因。

    数据库中 `failure_reason` 可能是 None、空字符串或带空格的字符串。统计分布前统一
    转成去空白字符串，空值表示本次请求没有记录失败原因。
    """
    return str(record.failure_reason or "").strip()


def _attempt_stage(record: models.KBQuery) -> str:
    """从一条 KBQuery 记录中提取标准化执行阶段。

    `attempt_stage` 用来描述回答阶段最终停在哪个路径，例如：
    - `primary`：主模型第一次成功。
    - `primary_repair`：主模型输出经 JSON 修复后成功。
    - `fallback`：走了备用模型。
    - `fallback_refusal`：最终兜底拒答。

    若历史记录缺失该字段，则统一归为 `unknown`，方便分布统计。
    """
    return str(record.attempt_stage or "unknown").strip() or "unknown"


def get_ask_metrics(db: Session, *, hours: int = 24) -> dict[str, Any]:
    """聚合最近一段时间内的 ASK 运行指标。

    输入：
    - `db`：SQLAlchemy Session，由 FastAPI 依赖注入或测试代码传入。
    - `hours`：向前统计多少小时，默认 24 小时。

    输出：
    - 一个可直接作为 API 响应返回的 dict，包含窗口信息和 ASK 指标摘要。

    设计边界：
    - 只读取 `kb_queries`，不访问 Redis、不调用外部服务。
    - token/cost 使用落库字段聚合，不在这里重新估算模型用量。
    - `repair_rate` 优先使用 `repair_used` 字段，`fallback_rate` 仍通过
      `attempt_stage` 字符串推断。
    """
    # 统一收敛统计窗口，避免路由之外的调用传入过大范围。
    window_hours = _safe_hours(hours)
    since = _utc_now() - timedelta(hours=window_hours)

    # 只取窗口内记录，并按时间倒序，便于后续需要扩展最近样例时复用。
    stmt = (
        select(models.KBQuery)
        .where(models.KBQuery.created_at >= since)
        .order_by(models.KBQuery.created_at.desc())
    )
    records = list(db.execute(stmt).scalars().all())

    # 基础请求量。
    total = len(records)

    # 三类延迟：
    # - retrieve：向量/关键词检索耗时
    # - answer：模型生成与 JSON 修复等回答阶段耗时
    # - total：用户体感上更接近的 ASK 主链路总耗时
    retrieve_latencies = [int(record.latency_retrieve_ms or 0) for record in records]
    answer_latencies = [int(record.latency_answer_ms or 0) for record in records]
    total_latencies = [
        int(record.latency_retrieve_ms or 0) + int(record.latency_answer_ms or 0)
        for record in records
    ]

    # 稳定性与可用性指标。
    # citation_count 代表返回中至少包含一条引用，不代表引用一定完全正确。
    valid_json_count = sum(1 for record in records if bool(record.valid_json))
    citation_count = sum(1 for record in records if _is_non_empty_list(record.citations_json))

    # 分布类指标用于排障：
    # - failure_reasons：看失败主要来自无证据、模型调用失败、解析失败等哪一类。
    # - attempt_stages：看主模型成功、repair、fallback、拒答各占多少。
    # - models_seen：看窗口内实际使用了哪些模型。
    failure_reasons = Counter(_failure_reason(record) for record in records if _failure_reason(record))
    attempt_stages = Counter(_attempt_stage(record) for record in records)
    models_seen = Counter(str(record.model or "unknown").strip() or "unknown" for record in records)

    # repair 已经有独立字段；fallback 暂时仍通过阶段名推断。
    repair_count = sum(1 for record in records if bool(getattr(record, "repair_used", False)))
    fallback_count = sum(1 for stage in map(_attempt_stage, records) if "fallback" in stage)

    total_tokens = sum(int(getattr(record, "total_tokens", 0) or 0) for record in records)
    token_usage_estimated_count = sum(
        1 for record in records if bool(getattr(record, "token_usage_estimated", False))
    )
    estimated_cost_usd = round(
        sum(float(getattr(record, "estimated_cost_usd", 0.0) or 0.0) for record in records),
        8,
    )

    # 这里的 failure 定义为“落库了 failure_reason 的请求”。
    # 这比简单用 valid_json=false 更适合当前项目，因为有些保守拒答也可能是系统设计内行为。
    failure_count = sum(failure_reasons.values())
    success_count = total - failure_count

    return {
        "window_hours": window_hours,
        "since": since.isoformat(),
        "ask": {
            "total": total,
            "success": success_count,
            "failure": failure_count,
            "valid_json_rate": _rate(valid_json_count, total),
            "citation_rate": _rate(citation_count, total),
            "repair_rate": _rate(repair_count, total),
            "fallback_rate": _rate(fallback_count, total),
            "avg_retrieve_ms": _avg(retrieve_latencies),
            "avg_answer_ms": _avg(answer_latencies),
            "avg_total_ms": _avg(total_latencies),
            "p50_total_ms": _percentile(total_latencies, 50),
            "p95_total_ms": _percentile(total_latencies, 95),
            "total_tokens": total_tokens,
            "avg_tokens_per_request": _avg([int(getattr(record, "total_tokens", 0) or 0) for record in records]),
            "token_usage_estimated_count": token_usage_estimated_count,
            "estimated_cost_usd": estimated_cost_usd,
            "avg_estimated_cost_usd": round(estimated_cost_usd / total, 8) if total else 0.0,
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "attempt_stages": dict(sorted(attempt_stages.items())),
            "models": dict(sorted(models_seen.items())),
        },
    }


def _status_counts(records: list[Any]) -> dict[str, int]:
    """统计 ORM 记录中的 status 分布。"""
    counter = Counter(str(getattr(record, "status", None) or "unknown").strip() or "unknown" for record in records)
    return dict(sorted(counter.items()))


def _route_from_audit(record: models.AuditLog) -> str:
    """从审计 payload 中提取 route；缺失时回退到 action_type。"""
    payload = dict(record.payload_json or {})
    route = str(payload.get("route") or "").strip()
    if route:
        return route
    return str(record.action_type or "unknown").strip() or "unknown"


def _rejection_reason(record: models.AuditLog) -> str:
    """从失败/拒绝类审计中提取可读原因。"""
    payload = dict(record.payload_json or {})
    for key in ("error_code", "rejection_reason", "failure_reason", "detail"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return str(record.action_type or "rejected").strip() or "rejected"


def _is_failure_or_rejected(record: models.AuditLog) -> bool:
    """判断审计事件是否属于失败或拒绝类事件。"""
    action_type = str(record.action_type or "").upper()
    route = _route_from_audit(record).upper()
    payload = dict(record.payload_json or {})
    if "REJECT" in action_type or "FAILED" in action_type or "ERROR" in action_type:
        return True
    if "REJECT" in route or "FAILED" in route or "ERROR" in route:
        return True
    return any(payload.get(key) for key in ("error_code", "rejection_reason", "failure_reason"))


def get_ticket_metrics(db: Session, *, hours: int = 24) -> dict[str, Any]:
    """聚合工单/Agent 相关的轻量运行指标。"""
    window_hours = _safe_hours(hours)
    since = _utc_now() - timedelta(hours=window_hours)

    audit_stmt = (
        select(models.AuditLog)
        .where(models.AuditLog.created_at >= since)
        .where(models.AuditLog.action_type != "ASK")
        .order_by(models.AuditLog.created_at.desc())
    )
    audit_records = list(db.execute(audit_stmt).scalars().all())
    rejected_records = [record for record in audit_records if _is_failure_or_rejected(record)]

    ticket_records = list(db.execute(select(models.Ticket)).scalars().all())
    draft_records = list(db.execute(select(models.TicketDraft)).scalars().all())
    pending_records = list(db.execute(select(models.PendingAction)).scalars().all())

    action_counts = Counter(str(record.action_type or "unknown").strip() or "unknown" for record in audit_records)
    route_counts = Counter(_route_from_audit(record) for record in audit_records)
    rejection_reasons = Counter(_rejection_reason(record) for record in rejected_records)

    return {
        "total_audit_events": len(audit_records),
        "failure_or_rejected_events": len(rejected_records),
        "action_counts": dict(sorted(action_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "ticket_state": {
            "total": len(ticket_records),
            "status_counts": _status_counts(ticket_records),
        },
        "draft_state": {
            "total": len(draft_records),
            "status_counts": _status_counts(draft_records),
        },
        "confirmation_state": {
            "total": len(pending_records),
            "status_counts": _status_counts(pending_records),
        },
    }


def get_ops_metrics(db: Session, *, hours: int = 24) -> dict[str, Any]:
    """返回 `/ops/metrics` 使用的 ASK + 工单统一观测摘要。"""
    ask_metrics = get_ask_metrics(db, hours=hours)
    return {
        **ask_metrics,
        "tickets": get_ticket_metrics(db, hours=hours),
    }
