"""
L5 Agent 技能注册表：统一声明 `/agent` 可调用的全部工具契约。

一、程序目标
1. 为 Agent 的全局分支工具与既有工单工具提供唯一的“能力契约”来源。
2. 把工具的名字、用途、参数 schema、风险等级与执行语义集中管理。
3. 让 Planner / Validator / LangChain / MCP 都复用同一份工具描述。

二、当前覆盖范围
1. Global Planner 分支工具：
   - `continue_ticket_draft`
   - `ticket_tool_planner`
   - `kb_answer`
   - `create_ticket`
2. 既有工单工具：
   - `lookup_ticket`
   - `add_ticket_comment`
   - `escalate_ticket`
   - `cancel_ticket`

三、当前阶段取舍
1. Global Planner 分支工具当前主要作为“契约 + 分支入口”存在。
2. 只有既有工单工具直接走 registry.dispatch；全局分支工具仍由 `services.py` 编排执行。
3. `auth_rule` 与 `risk_level` 继续作为显式元信息，由服务层校验器强制执行。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from sqlalchemy.orm import Session


RiskLevel = Literal["LOW", "HIGH"]
PlannerScope = Literal["global", "ticket"]
SkillHandler = Callable[[Session, dict[str, Any], str, str], dict]


@dataclass(frozen=True)
class AgentSkill:
    """单个 Agent 工具的统一定义。"""

    name: str
    route_name: str
    planner_scope: PlannerScope
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    risk_level: RiskLevel
    auth_rule: str
    handler_semantics: str
    audit_event_type: str
    success_message: str
    default_text_arg: str | None
    default_text_value: str | None
    handler: SkillHandler | None


class SkillRegistry:
    """以工具名为主键的最小注册表。"""

    def __init__(self) -> None:
        self._skills_by_name: dict[str, AgentSkill] = {}
        self._skills_by_route: dict[str, AgentSkill] = {}

    def register(self, skill: AgentSkill) -> None:
        """注册单个工具；同名或同 route 不允许重复。"""
        if skill.name in self._skills_by_name:
            raise ValueError(f"duplicate_skill_name:{skill.name}")
        if skill.route_name in self._skills_by_route:
            raise ValueError(f"duplicate_skill_route:{skill.route_name}")
        self._skills_by_name[skill.name] = skill
        self._skills_by_route[skill.route_name] = skill

    def get(self, name: str) -> AgentSkill | None:
        """按工具名读取定义。"""
        return self._skills_by_name.get(name)

    def get_by_route(self, route_name: str) -> AgentSkill | None:
        """按现有 `/agent` route 名读取定义。"""
        return self._skills_by_route.get(route_name)

    def list(self) -> list[AgentSkill]:
        """按注册顺序返回全部工具定义。"""
        return list(self._skills_by_name.values())

    def dispatch(
        self,
        name: str,
        db: Session,
        args: dict[str, Any],
        actor: str,
        raw_text: str = "",
    ) -> dict:
        """按工具名分发到已注册 handler。"""
        skill = self.get(name)
        if skill is None:
            raise KeyError(f"unknown_skill:{name}")
        if skill.handler is None:
            raise NotImplementedError(f"orchestrated_skill_only:{name}")
        return skill.handler(db=db, args=args, actor=actor, raw_text=raw_text)


def _ticket_detail_output_schema() -> dict[str, Any]:
    """统一描述四个既有工单工具的最小返回结构。"""
    return {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "status": {"type": "string"},
            "title": {"type": "string"},
            "comments": {"type": "array"},
            "context": {"type": "object"},
        },
    }


def _agent_branch_output_schema() -> dict[str, Any]:
    """统一描述全局分支工具的最小输出语义。"""
    return {
        "type": "object",
        "properties": {
            "route": {"type": "string"},
            "message": {"type": "string"},
        },
    }


def _lookup_ticket_handler(
    db: Session,
    args: dict[str, Any],
    actor: str,
    raw_text: str = "",
) -> dict:
    """查单 handler：复用既有序列化逻辑。"""
    from src.api import crud, services

    ticket_id = str(args.get("ticket_id") or "")
    ticket = crud.get_ticket_by_public_id(db, ticket_id)
    if ticket is None:
        raise LookupError(f"ticket_not_found:{ticket_id}")
    return services.serialize_ticket_detail(db, ticket)


def _add_ticket_comment_handler(
    db: Session,
    args: dict[str, Any],
    actor: str,
    raw_text: str = "",
) -> dict:
    """补充说明 handler：复用 append-only 评论工作流。"""
    from src.api import services

    return services.add_ticket_comment_workflow(
        db,
        ticket_id=str(args.get("ticket_id") or ""),
        comment=str(args.get("comment") or ""),
        actor=actor,
        audit_source=str(args.get("_audit_source") or "") or None,
    )


def _escalate_ticket_handler(
    db: Session,
    args: dict[str, Any],
    actor: str,
    raw_text: str = "",
) -> dict:
    """催办 handler：复用既有状态推进工作流。"""
    from src.api import services

    return services.escalate_ticket_workflow(
        db,
        ticket_id=str(args.get("ticket_id") or ""),
        actor=actor,
        reason=args.get("reason"),
        audit_source=str(args.get("_audit_source") or "") or None,
    )


def _cancel_ticket_handler(
    db: Session,
    args: dict[str, Any],
    actor: str,
    raw_text: str = "",
) -> dict:
    """取消 handler：当前仅声明高风险，执行仍复用既有工作流。"""
    from src.api import services

    return services.cancel_ticket_workflow(
        db,
        ticket_id=str(args.get("ticket_id") or ""),
        actor=actor,
        reason=args.get("reason"),
        audit_source=str(args.get("_audit_source") or "") or None,
    )


def _build_ticket_tool_skill_definitions() -> list[AgentSkill]:
    """集中定义当前既有工单工具的契约。"""
    return [
        AgentSkill(
            name="lookup_ticket",
            route_name="LOOKUP_TICKET",
            planner_scope="ticket",
            description="按工单号查询工单详情与当前处理状态。",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "要查询的工单号。"},
                },
                "required": ["ticket_id"],
            },
            output_schema=_ticket_detail_output_schema(),
            risk_level="LOW",
            auth_rule="login",
            handler_semantics="read_only",
            audit_event_type="LOOKUP_TICKET",
            success_message="已查询工单进度。",
            default_text_arg=None,
            default_text_value=None,
            handler=_lookup_ticket_handler,
        ),
        AgentSkill(
            name="add_ticket_comment",
            route_name="ADD_TICKET_COMMENT",
            planner_scope="ticket",
            description="向已有工单追加补充说明，评论按 append-only 方式保存。",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "要追加说明的工单号。"},
                    "comment": {"type": "string", "description": "要追加的评论内容。"},
                },
                "required": ["ticket_id", "comment"],
            },
            output_schema=_ticket_detail_output_schema(),
            risk_level="LOW",
            auth_rule="login",
            handler_semantics="append_only",
            audit_event_type="ADD_TICKET_COMMENT",
            success_message="已追加工单说明。",
            default_text_arg="comment",
            default_text_value="用户补充说明。",
            handler=_add_ticket_comment_handler,
        ),
        AgentSkill(
            name="escalate_ticket",
            route_name="ESCALATE_TICKET",
            planner_scope="ticket",
            description="记录催办请求，并在允许时把工单状态推进到 in_progress。",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "要催办的工单号。"},
                    "reason": {"type": "string", "description": "催办原因，可选。"},
                },
                "required": ["ticket_id"],
            },
            output_schema=_ticket_detail_output_schema(),
            risk_level="LOW",
            auth_rule="login",
            handler_semantics="state_transition",
            audit_event_type="ESCALATE_TICKET",
            success_message="已记录催办并更新工单状态。",
            default_text_arg="reason",
            default_text_value="用户请求催办。",
            handler=_escalate_ticket_handler,
        ),
        AgentSkill(
            name="cancel_ticket",
            route_name="CANCEL_TICKET",
            planner_scope="ticket",
            description="取消已有工单并记录取消原因，属于高风险动作。",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "要取消的工单号。"},
                    "reason": {"type": "string", "description": "取消原因。"},
                    "confirm": {"type": "boolean", "description": "后续确认态预留字段。"},
                },
                "required": ["ticket_id", "reason"],
            },
            output_schema=_ticket_detail_output_schema(),
            risk_level="HIGH",
            auth_rule="owner_or_admin",
            handler_semantics="state_transition_high_risk",
            audit_event_type="CANCEL_TICKET",
            success_message="已取消工单。",
            default_text_arg="reason",
            default_text_value="用户请求取消工单。",
            handler=_cancel_ticket_handler,
        ),
    ]


@lru_cache(maxsize=1)
def get_agent_skill_registry() -> SkillRegistry:
    """返回当前阶段的统一 Agent 技能注册表。"""
    registry = SkillRegistry()
    for skill in _build_global_planner_skill_definitions():
        registry.register(skill)
    for skill in _build_ticket_tool_skill_definitions():
        registry.register(skill)
    return registry


def _build_global_planner_skill_definitions() -> list[AgentSkill]:
    """集中定义当前 Global Planner 的四个分支工具契约。"""
    return [
        AgentSkill(
            name="continue_ticket_draft",
            route_name="CONTINUE_TICKET_DRAFT",
            planner_scope="global",
            description="继续补全一个已存在的工单草稿，并沿用现有草稿续办链路。",
            input_schema={
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "要继续处理的草稿号。"},
                    "fields": {"type": "object", "description": "本轮补充的字段集合。"},
                },
                "required": ["draft_id"],
            },
            output_schema=_agent_branch_output_schema(),
            risk_level="LOW",
            auth_rule="login",
            handler_semantics="orchestrated_branch_draft",
            audit_event_type="CONTINUE_TICKET_DRAFT",
            success_message="已进入草稿续办流程。",
            default_text_arg=None,
            default_text_value=None,
            handler=None,
        ),
        AgentSkill(
            name="ticket_tool_planner",
            route_name="TICKET_TOOL_PLANNER",
            planner_scope="global",
            description="把请求转交给既有工单 ticket 子规划器，再细分为查单、评论、催办或取消。",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "目标工单号。"},
                    "raw_text": {"type": "string", "description": "用户原始请求文本。"},
                },
                "required": ["ticket_id", "raw_text"],
            },
            output_schema=_agent_branch_output_schema(),
            risk_level="LOW",
            auth_rule="login",
            handler_semantics="orchestrated_branch_ticket",
            audit_event_type="TICKET_TOOL_PLANNER",
            success_message="已转交既有工单子规划器。",
            default_text_arg=None,
            default_text_value=None,
            handler=None,
        ),
        AgentSkill(
            name="kb_answer",
            route_name="KB_ANSWER",
            planner_scope="global",
            description="按知识问答路径检索制度、政策或流程，并返回带引用的回答。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要查询的知识问答文本。"},
                },
                "required": ["query"],
            },
            output_schema=_agent_branch_output_schema(),
            risk_level="LOW",
            auth_rule="login",
            handler_semantics="orchestrated_branch_kb",
            audit_event_type="KB_ANSWER",
            success_message="已进入知识问答流程。",
            default_text_arg=None,
            default_text_value=None,
            handler=None,
        ),
        AgentSkill(
            name="create_ticket",
            route_name="CREATE_TICKET_FLOW",
            planner_scope="global",
            description="按建单路径继续处理，必要时进入草稿追问并最终创建工单。",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "原始建单描述。"},
                    "fields": {"type": "object", "description": "可保守补充的结构化字段。"},
                },
                "required": ["text"],
            },
            output_schema=_agent_branch_output_schema(),
            risk_level="LOW",
            auth_rule="login",
            handler_semantics="orchestrated_branch_create",
            audit_event_type="CREATE_TICKET_FLOW",
            success_message="已进入建单流程。",
            default_text_arg=None,
            default_text_value=None,
            handler=None,
        ),
    ]


def get_ticket_tool_registry() -> SkillRegistry:
    """兼容旧入口：当前返回统一 registry，ticket 相关调用仍可正常分发。"""
    return get_agent_skill_registry()


def serialize_skill(skill: AgentSkill) -> dict[str, Any]:
    """把技能定义转成可展示、可喂给模型的公共结构。"""
    return {
        "name": skill.name,
        "route_name": skill.route_name,
        "planner_scope": skill.planner_scope,
        "description": skill.description,
        "input_schema": skill.input_schema,
        "output_schema": skill.output_schema,
        "risk_level": skill.risk_level,
        "auth_rule": skill.auth_rule,
        "handler_semantics": skill.handler_semantics,
        "audit_event_type": skill.audit_event_type,
    }


def list_ticket_tool_skills() -> list[dict[str, Any]]:
    """返回既有工单工具的公共技能清单。"""
    return [
        serialize_skill(skill)
        for skill in get_agent_skill_registry().list()
        if skill.planner_scope == "ticket"
    ]


def list_global_planner_skills() -> list[dict[str, Any]]:
    """返回 Global Planner 分支工具的公共技能清单。"""
    return [
        serialize_skill(skill)
        for skill in get_agent_skill_registry().list()
        if skill.planner_scope == "global"
    ]
