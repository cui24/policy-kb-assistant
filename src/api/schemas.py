"""
L2/L4 请求与响应模型定义。

一、程序目标
1. 统一 API 输入输出结构。
2. 让 FastAPI 自动完成请求校验与 OpenAPI 文档生成。

二、覆盖的接口
1. `/ask`
2. `/tickets`
3. `/agent`
4. `/kb_queries`
5. `/audit_logs`
6. L4 的草稿续办响应

三、输入输出
1. 输入：HTTP JSON 请求体。
2. 输出：Python 字典在进入响应前的结构约束。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    """单条引用结构。"""

    doc_id: str
    page: int
    snippet: str


class RetrieveHitItem(BaseModel):
    """单条检索命中结构。"""

    doc_id: str | None = None
    page: int | None = None
    score: float = 0.0
    snippet: str = ""


class AskRequest(BaseModel):
    """`/ask` 请求体。"""

    question: str = Field(..., min_length=1)
    user: str | None = None
    department: str | None = None


class AskMeta(BaseModel):
    """`/ask` 响应中的 trace 元信息。"""

    attempt_stage: str
    valid_json: bool
    repair_used: bool
    failure_reason: str | None = None
    retrieve_topk: list[RetrieveHitItem]
    latency_ms: dict[str, int]


class AskResponse(BaseModel):
    """`/ask` 响应体。"""

    request_id: str
    answer: str
    citations: list[CitationItem]
    meta: AskMeta


class TicketCreateRequest(BaseModel):
    """创建工单请求体。"""

    creator: str | None = None
    department: str | None = None
    category: str = "other"
    priority: str = "P2"
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    contact: str | None = None
    context: dict[str, Any] | None = None


class TicketStatusUpdateRequest(BaseModel):
    """更新工单状态请求体。"""

    status: str = Field(..., min_length=1)
    actor: str | None = None


class TicketCommentRequest(BaseModel):
    """追加工单说明请求体。"""

    actor: str | None = None
    comment: str = Field(..., min_length=1)


class TicketEscalateRequest(BaseModel):
    """催办工单请求体。"""

    actor: str | None = None
    reason: str | None = None


class TicketCancelRequest(BaseModel):
    """取消工单请求体。"""

    actor: str | None = None
    reason: str = Field(..., min_length=1)


class TicketResponse(BaseModel):
    """工单摘要响应。"""

    ticket_id: str
    status: str


class TicketCommentItem(BaseModel):
    """工单评论响应项。"""

    comment_id: str | None = None
    actor: str
    content: str
    created_at: str


class TicketDetailResponse(BaseModel):
    """工单详情响应。"""

    ticket_id: str
    creator: str
    assignee: str | None = None
    department: str
    category: str
    priority: str
    title: str
    description: str
    status: str
    contact: str | None = None
    comments: list[TicketCommentItem] = Field(default_factory=list)
    context: dict[str, Any]
    created_at: str
    updated_at: str


class TicketExtractionResponse(BaseModel):
    """Agent 抽取出的结构化工单字段。"""

    creator: str
    department: str
    category: str
    priority: str
    title: str
    description: str
    contact: str | None = None
    location: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    extractor: str


class TicketDraftResponse(BaseModel):
    """工单草稿响应，用于多轮补全。"""

    draft_id: str
    status: str
    missing_fields: list[str] = Field(default_factory=list)
    expires_at: str
    payload: dict[str, Any]
    kb_request_id: str | None = None


ToolName = Literal[
    "ticket_tool_planner",
    "lookup_ticket",
    "add_ticket_comment",
    "escalate_ticket",
    "cancel_ticket",
    "create_ticket",
    "continue_ticket_draft",
    "kb_answer",
]


class ToolPlan(BaseModel):
    """LLM Planner 的结构化执行计划。"""

    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    need_confirmation: bool = False
    missing_fields: list[str] = Field(default_factory=list)


class TicketToolPlannerPlanArgs(BaseModel):
    """全局规划后转交给 ticket 子规划器的入参。"""

    ticket_id: str = Field(..., min_length=1)
    raw_text: str = Field(..., min_length=1)


class LookupTicketPlanArgs(BaseModel):
    """查单工具入参。"""

    ticket_id: str = Field(..., min_length=1)


class AddTicketCommentPlanArgs(BaseModel):
    """评论工具入参。"""

    ticket_id: str = Field(..., min_length=1)
    comment: str = Field(..., min_length=1)


class EscalateTicketPlanArgs(BaseModel):
    """催办工具入参。"""

    ticket_id: str = Field(..., min_length=1)
    reason: str | None = None


class CancelTicketPlanArgs(BaseModel):
    """取消工具入参。"""

    ticket_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    confirm: bool | None = None


class ContinueTicketDraftPlanArgs(BaseModel):
    """草稿续办工具入参。"""

    draft_id: str = Field(..., min_length=1)
    fields: dict[str, Any] = Field(default_factory=dict)


class KBAnswerPlanArgs(BaseModel):
    """知识问答工具入参。"""

    query: str = Field(..., min_length=1)


class CreateTicketPlanArgs(BaseModel):
    """建单工具入参。"""

    text: str = Field(..., min_length=1)
    fields: dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    """`/agent` 请求体。"""

    text: str = ""
    user: str | None = None
    department: str | None = None
    draft_id: str | None = None
    fields: dict[str, Any] | None = None
    confirm_token: str | None = None


class AgentResponse(BaseModel):
    """`/agent` 响应体。"""

    route: str
    message: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    memory_applied: dict[str, Any] | None = None
    confirm_token: str | None = None
    draft: TicketDraftResponse | None = None
    ticket: TicketResponse | None = None
    ticket_detail: TicketDetailResponse | None = None
    kb: AskResponse | None = None
    extraction: TicketExtractionResponse | None = None


class KBQueryDetailResponse(BaseModel):
    """单条问答记录详情，用于追溯与排障。"""

    request_id: str
    user: str
    department: str
    question: str
    answer: str
    citations: list[CitationItem]
    retrieve_topk: list[RetrieveHitItem]
    attempt_stage: str
    latency_ms: dict[str, int]
    model: str
    valid_json: bool
    failure_reason: str | None = None
    created_at: str


class AuditLogResponse(BaseModel):
    """单条审计日志响应。"""

    id: str
    created_at: str
    actor: str
    action_type: str
    target_type: str
    target_id: str
    request_id: str
    payload: dict[str, Any]
