"""
Agent Graph 状态定义模块。

作用：
1. 定义图节点间共享的统一状态结构。
2. 约束核心字段（输入、用户、草稿、确认态、记忆、审计上下文）。
3. 作为节点函数输入输出契约，降低字段漂移风险。
"""

from __future__ import annotations
from typing import Any, Literal, TypedDict


# 路由枚举：与现有 `/agent` 响应 route 保持一致或可兼容扩展。
RouteName = Literal[
    "ask",
    "create_ticket",
    "continue_ticket_draft",
    "ticket_tool",
    "confirm",
    "DRAFT_SAVED",
    "DRAFT_EXPIRED",
]


class RequestContext(TypedDict, total=False):
    """请求输入层状态：仅保存本轮调用的原始输入与调用方身份信息。"""

    text: str
    user: str
    department: str
    actor_user_id: str
    actor_role: str
    draft_id: str | None
    confirm_token: str | None
    fields: dict[str, Any]
    request_id: str | None
    explicit_ticket_id: str | None
    resolved_ticket_id: str | None
    effective_draft_id: str | None


class PlannerState(TypedDict, total=False):
    """规划层状态：保存分类、路由、工具选择与参数补全结果。"""

    route: RouteName
    tool: str | None
    args: dict[str, Any]
    need_confirmation: bool
    missing_fields: list[str]
    risk_level: str | None
    reason: str | None
    tool_request: dict[str, Any]


class MemoryState(TypedDict, total=False):
    """记忆层状态：短期记忆、长期记忆快照和本轮记忆应用结果。"""

    short_term: dict[str, Any]
    long_term: dict[str, Any]
    memory_applied: dict[str, Any] | None
    updated: bool


class WorkingMemoryState(TypedDict, total=False):
    """L0 工作记忆：只保存本轮请求内的临时判断、约束与工具结果摘要。"""

    request_id: str | None
    normalized_text: str
    route_source: str | None
    intent: str | None
    selected_tool: str | None
    tool_args_preview: dict[str, Any]
    extracted_entities: dict[str, Any]
    resolved_refs: dict[str, Any]
    missing_fields: list[str]
    auth_context: dict[str, Any]
    permission_context: dict[str, Any]
    risk_context: dict[str, Any]
    memory_context: dict[str, Any]
    tool_result_summary: dict[str, Any]
    error_code: str | None
    error_stage: str | None
    error_reason: str | None


class TicketDraftState(TypedDict, total=False):
    """草稿层状态：草稿对象与缺失字段、续办信息。"""

    draft_id: str
    status: str
    missing_fields: list[str]
    payload: dict[str, Any]
    expires_at: str | None


class ConfirmState(TypedDict, total=False):
    """确认态状态：二次确认 token 与待确认动作。"""

    need_confirmation: bool
    confirm_token: str | None
    action: str | None
    args: dict[str, Any]
    status: Literal["pending", "approved", "rejected"] | None


class ExecutionState(TypedDict, total=False):
    """执行层状态：节点真正执行后的结构化产物。"""

    route: RouteName
    message: str | None
    missing_fields: list[str]
    kb: dict[str, Any] | None
    extraction: dict[str, Any] | None
    ticket: dict[str, Any] | None
    ticket_detail: dict[str, Any] | None
    draft: dict[str, Any] | None
    tool_result: dict[str, Any] | None


class AskState(TypedDict, total=False):
    """ASK 分支中间状态：便于把单节点大逻辑拆成多节点。"""

    request_id: str
    hits: list[dict[str, Any]]
    retrieve_ms: int
    output: dict[str, Any]
    answer_ms: int
    normalized: dict[str, Any]
    query_id: str | None


class ErrorState(TypedDict, total=False):
    """异常状态：统一错误类型，避免节点间依赖字符串硬编码。"""

    code: str
    detail: str
    retryable: bool
    stage: str


class AuditState(TypedDict, total=False):
    """审计状态：节点往这里追加审计事件草稿，由统一层落库。"""

    events: list[dict[str, Any]]


class MetaState(TypedDict, total=False):
    """运行元信息：用于调试、观测、追踪。"""

    next_node: str | None
    finished: bool
    started_at: str | None
    ended_at: str | None
    graph_version: str


class AgentState(TypedDict, total=False):
    """
    Agent Graph 全局状态对象（建议核心字段 9 组）。

    字段分组：
    1. request: 原始请求上下文
    2. planner: 规划结果
    3. memory: 记忆快照与应用结果
    4. working: L0 工作记忆，只在本轮请求内流转
    5. draft: 草稿态信息
    6. confirm: 二次确认信息
    7. execution: 执行结果
    8. error: 错误信息
    9. audit: 审计事件缓存
    10. meta: 调试与追踪信息
    """

    request: RequestContext
    planner: PlannerState
    memory: MemoryState
    working: WorkingMemoryState
    draft: TicketDraftState
    confirm: ConfirmState
    execution: ExecutionState
    ask: AskState
    error: ErrorState
    audit: AuditState
    meta: MetaState
