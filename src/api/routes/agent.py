"""
`/agent` 路由：提供 L2/L4 的自然语言动作入口。

一、程序目标
1. 判断用户输入是“问答”还是“建单”。
2. 若是建单，则抽取工单字段。
3. 字段缺失时返回 `NEED_MORE_INFO`，并在 L4 中附带草稿信息。
4. 字段完整时直接创建工单。
5. 若用户带 `draft_id` 续办，则读取草稿并继续处理。
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from redis import Redis
from sqlalchemy.orm import Session

from src.agent_graph import run_agent_graph
from src.api.deps import get_db, get_redis_dep
from src.api.deps_auth import AuthenticatedUser, get_current_active_user
from src.api.rate_limit import (
    RateLimitStoreError,
    agent_limit,
    agent_window_seconds,
    consume_rate_limit,
    extract_client_ip,
    rate_limit_headers,
)
from src.api.schemas import AgentRequest, AgentResponse
from src.api.services import run_agent_workflow


router = APIRouter(tags=["agent"])


def _role_value(role: object) -> str:
    if hasattr(role, "value"):
        return str(getattr(role, "value"))
    return str(role or "")


def _agent_engine() -> str:
    """读取 `/agent` 执行引擎。默认沿用 legacy，便于灰度切换。"""
    return str(os.getenv("AGENT_ENGINE", "legacy")).strip().lower()


@router.post("/agent", response_model=AgentResponse)
def agent(
    payload: AgentRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
    redis: Redis = Depends(get_redis_dep),
) -> AgentResponse:
    """执行最小 Agent 路由。"""
    try:
        decision = consume_rate_limit(
            redis,
            scope="agent:route",
            subject=f"user:{current_user.id}|ip:{extract_client_ip(request)}",
            limit=agent_limit(),
            window_seconds=agent_window_seconds(),
        )
    except RateLimitStoreError as exc:
        raise HTTPException(status_code=503, detail="rate_limit_store_unavailable") from exc

    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="too_many_agent_requests",
            headers=rate_limit_headers(decision),
        )
    for key, value in rate_limit_headers(decision).items():
        response.headers[key] = value

    try:
        runner = run_agent_graph if _agent_engine() == "graph" else run_agent_workflow
        return runner(
            db,
            text=payload.text,
            user=str(current_user.username),
            department=payload.department,
            draft_id=payload.draft_id,
            fields=payload.fields,
            confirm_token=payload.confirm_token,
            actor_user_id=str(current_user.id),
            actor_role=_role_value(current_user.role),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="draft_not_found") from exc
    except LookupError as exc:
        detail = str(exc)
        if detail.startswith("ticket_not_found:"):
            raise HTTPException(status_code=404, detail="ticket_not_found") from exc
        raise HTTPException(status_code=404, detail="draft_not_found") from exc
