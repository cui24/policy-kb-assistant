"""
Agent Graph 装配模块。

作用：
1. 创建并组装 LangGraph 工作流。
2. 注册节点、边与条件分支。
3. 提供可编译图对象给 executor 调用。

约束：
- 本文件只做流程装配，不写具体业务实现细节。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.agent_graph import adapters
from src.agent_graph.memory import hydrate_memory_state
from src.agent_graph.nodes_ask import run_ask_node
from src.agent_graph.nodes_confirm import run_confirm_node
from src.agent_graph.nodes_draft import run_draft_node
from src.agent_graph.nodes_execute_mcp import run_execute_mcp_tool_node
from src.agent_graph.nodes_finalize import run_finalize_node
from src.agent_graph.nodes_global import (
    run_global_plan_node,
    run_global_validate_node,
)
from src.agent_graph.nodes_prepare import run_prepare_node
from src.agent_graph.nodes_routing import (
    run_reference_clarify_node,
    run_resolve_references_node,
    run_rules_router_node,
)
from src.agent_graph.nodes_ticket_create import run_ticket_create_node
from src.agent_graph.nodes_ticket_tool import run_ticket_tool_node
from src.agent_graph.state import AgentState

NODE_PREPARE = "prepare"
NODE_LOAD_MEMORY = "load_memory"
NODE_RESOLVE_REFERENCES = "resolve_references"
NODE_REFERENCE_CLARIFY = "reference_clarify"
NODE_RULES_ROUTER = "rules_router"
NODE_ASK = "ask"
NODE_TICKET_CREATE = "ticket_create"
NODE_DRAFT = "draft"
NODE_CONFIRM = "confirm"
NODE_TICKET_TOOL = "ticket_tool"
NODE_GLOBAL_PLAN = "global_plan"
NODE_GLOBAL_VALIDATE = "global_validate"
NODE_EXECUTE_MCP_TOOL = "execute_mcp_tool"
NODE_FINALIZE = "finalize"


def _run_load_memory_node(db: Session, state: AgentState) -> AgentState:
    return hydrate_memory_state(db, state)


def _route_after_resolve(state: AgentState) -> str:
    """引用恢复后：先处理引用追问，再按 planner 模式分流。"""
    request = dict(state.get("request") or {})
    if str(request.get("confirm_token") or "").strip():
        return NODE_RULES_ROUTER

    if adapters.needs_ticket_reference_clarification(
        str(request.get("text") or ""),
        resolved_ticket_id=request.get("resolved_ticket_id"),
        effective_draft_id=request.get("effective_draft_id"),
    ):
        return NODE_REFERENCE_CLARIFY

    mode = adapters.planner_mode()
    if mode == "rules":
        return NODE_RULES_ROUTER
    return NODE_GLOBAL_PLAN


def _route_after_rules_router(state: AgentState) -> str:
    """规则路由节点后，进入对应业务节点。"""
    planner_state = dict(state.get("planner") or {})
    route = str(planner_state.get("route") or "")
    if route == "ask":
        return NODE_ASK
    if route == "ticket_create":
        return NODE_TICKET_CREATE
    if route == "draft":
        return NODE_DRAFT
    if route == "confirm":
        return NODE_CONFIRM
    if route == "ticket_tool":
        return NODE_TICKET_TOOL
    return NODE_FINALIZE


def _route_after_global_plan(state: AgentState) -> str:
    """全局规划后：去校验、回退 rules 或直接结束。"""
    if dict(state.get("execution") or {}):
        return NODE_FINALIZE
    planner_state = dict(state.get("planner") or {})
    next_step = str(planner_state.get("global_next") or "")
    if next_step == "fallback_rules":
        return NODE_RULES_ROUTER
    if next_step == "validate":
        return NODE_GLOBAL_VALIDATE
    return NODE_FINALIZE


def _route_after_global_validate(state: AgentState) -> str:
    """全局校验后：回退 rules、结束，或按工具分支执行。"""
    planner_state = dict(state.get("planner") or {})
    next_step = str(planner_state.get("global_next") or "")
    if next_step == "fallback_rules":
        return NODE_RULES_ROUTER
    if dict(state.get("execution") or {}):
        return NODE_FINALIZE
    if next_step != "execute":
        return NODE_FINALIZE
    return NODE_EXECUTE_MCP_TOOL


def _fallback_run(db: Session, state: AgentState) -> AgentState:
    """LangGraph 不可用时的回退执行器。"""
    state = run_prepare_node(state)
    state = _run_load_memory_node(db, state)
    state = run_resolve_references_node(state)
    next_node = _route_after_resolve(state)

    if next_node == NODE_REFERENCE_CLARIFY:
        state = run_reference_clarify_node(state)
        return run_finalize_node(db, state)

    if next_node == NODE_GLOBAL_PLAN:
        state = run_global_plan_node(db, state)
        next_after_plan = _route_after_global_plan(state)
        if next_after_plan == NODE_RULES_ROUTER:
            state = run_rules_router_node(state)
            next_after_rules = _route_after_rules_router(state)
            if next_after_rules == NODE_ASK:
                state = run_ask_node(db, state)
            elif next_after_rules == NODE_TICKET_CREATE:
                state = run_ticket_create_node(db, state)
            elif next_after_rules == NODE_DRAFT:
                state = run_draft_node(db, state)
            elif next_after_rules == NODE_CONFIRM:
                state = run_confirm_node(db, state)
            elif next_after_rules == NODE_TICKET_TOOL:
                state = run_ticket_tool_node(db, state)
            return run_finalize_node(db, state)
        if next_after_plan == NODE_GLOBAL_VALIDATE:
            state = run_global_validate_node(db, state)
            next_after_validate = _route_after_global_validate(state)
            if next_after_validate == NODE_RULES_ROUTER:
                state = run_rules_router_node(state)
                next_after_rules = _route_after_rules_router(state)
                if next_after_rules == NODE_ASK:
                    state = run_ask_node(db, state)
                elif next_after_rules == NODE_TICKET_CREATE:
                    state = run_ticket_create_node(db, state)
                elif next_after_rules == NODE_DRAFT:
                    state = run_draft_node(db, state)
                elif next_after_rules == NODE_CONFIRM:
                    state = run_confirm_node(db, state)
                elif next_after_rules == NODE_TICKET_TOOL:
                    state = run_ticket_tool_node(db, state)
                return run_finalize_node(db, state)
            if next_after_validate == NODE_EXECUTE_MCP_TOOL:
                state = run_execute_mcp_tool_node(db, state)
        return run_finalize_node(db, state)

    state = run_rules_router_node(state)
    next_after_rules = _route_after_rules_router(state)
    if next_after_rules == NODE_ASK:
        state = run_ask_node(db, state)
    elif next_after_rules == NODE_TICKET_CREATE:
        state = run_ticket_create_node(db, state)
    elif next_after_rules == NODE_DRAFT:
        state = run_draft_node(db, state)
    elif next_after_rules == NODE_CONFIRM:
        state = run_confirm_node(db, state)
    elif next_after_rules == NODE_TICKET_TOOL:
        state = run_ticket_tool_node(db, state)
    return run_finalize_node(db, state)


def run_graph(db: Session, initial_state: AgentState) -> AgentState:
    """
    执行完整 Agent Graph。

    若 `langgraph` 可用：走 StateGraph 编排。
    若不可用：回退到本地等价调度器。
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return _fallback_run(db, initial_state)

    builder = StateGraph(AgentState)

    builder.add_node(NODE_PREPARE, run_prepare_node)
    builder.add_node(NODE_LOAD_MEMORY, lambda state: _run_load_memory_node(db, state))
    builder.add_node(NODE_RESOLVE_REFERENCES, run_resolve_references_node)
    builder.add_node(NODE_REFERENCE_CLARIFY, run_reference_clarify_node)
    builder.add_node(NODE_RULES_ROUTER, run_rules_router_node)
    builder.add_node(NODE_ASK, lambda state: run_ask_node(db, state))
    builder.add_node(NODE_TICKET_CREATE, lambda state: run_ticket_create_node(db, state))
    builder.add_node(NODE_DRAFT, lambda state: run_draft_node(db, state))
    builder.add_node(NODE_CONFIRM, lambda state: run_confirm_node(db, state))
    builder.add_node(NODE_TICKET_TOOL, lambda state: run_ticket_tool_node(db, state))
    builder.add_node(NODE_GLOBAL_PLAN, lambda state: run_global_plan_node(db, state))
    builder.add_node(NODE_GLOBAL_VALIDATE, lambda state: run_global_validate_node(db, state))
    builder.add_node(NODE_EXECUTE_MCP_TOOL, lambda state: run_execute_mcp_tool_node(db, state))
    builder.add_node(NODE_FINALIZE, lambda state: run_finalize_node(db, state))

    builder.add_edge(START, NODE_PREPARE)
    builder.add_edge(NODE_PREPARE, NODE_LOAD_MEMORY)
    builder.add_edge(NODE_LOAD_MEMORY, NODE_RESOLVE_REFERENCES)

    builder.add_conditional_edges(
        NODE_RESOLVE_REFERENCES,
        _route_after_resolve,
        {
            NODE_REFERENCE_CLARIFY: NODE_REFERENCE_CLARIFY,
            NODE_RULES_ROUTER: NODE_RULES_ROUTER,
            NODE_GLOBAL_PLAN: NODE_GLOBAL_PLAN,
        },
    )

    builder.add_conditional_edges(
        NODE_GLOBAL_PLAN,
        _route_after_global_plan,
        {
            NODE_RULES_ROUTER: NODE_RULES_ROUTER,
            NODE_GLOBAL_VALIDATE: NODE_GLOBAL_VALIDATE,
            NODE_FINALIZE: NODE_FINALIZE,
        },
    )

    builder.add_conditional_edges(
        NODE_GLOBAL_VALIDATE,
        _route_after_global_validate,
        {
            NODE_RULES_ROUTER: NODE_RULES_ROUTER,
            NODE_EXECUTE_MCP_TOOL: NODE_EXECUTE_MCP_TOOL,
            NODE_FINALIZE: NODE_FINALIZE,
        },
    )

    builder.add_conditional_edges(
        NODE_RULES_ROUTER,
        _route_after_rules_router,
        {
            NODE_ASK: NODE_ASK,
            NODE_TICKET_CREATE: NODE_TICKET_CREATE,
            NODE_DRAFT: NODE_DRAFT,
            NODE_CONFIRM: NODE_CONFIRM,
            NODE_TICKET_TOOL: NODE_TICKET_TOOL,
            NODE_FINALIZE: NODE_FINALIZE,
        },
    )

    builder.add_edge(NODE_REFERENCE_CLARIFY, NODE_FINALIZE)
    builder.add_edge(NODE_ASK, NODE_FINALIZE)
    builder.add_edge(NODE_TICKET_CREATE, NODE_FINALIZE)
    builder.add_edge(NODE_DRAFT, NODE_FINALIZE)
    builder.add_edge(NODE_CONFIRM, NODE_FINALIZE)
    builder.add_edge(NODE_TICKET_TOOL, NODE_FINALIZE)
    builder.add_edge(NODE_EXECUTE_MCP_TOOL, NODE_FINALIZE)

    builder.add_edge(NODE_FINALIZE, END)

    compiled = builder.compile()
    return compiled.invoke(initial_state)
