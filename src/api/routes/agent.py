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

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import get_db, require_api_key
from src.api.schemas import AgentRequest, AgentResponse
from src.api.services import run_agent_workflow


router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
def agent(payload: AgentRequest, db: Session = Depends(get_db)) -> AgentResponse:
    """执行最小 Agent 路由。"""
    try:
        return run_agent_workflow(
            db,
            text=payload.text,
            user=payload.user,
            department=payload.department,
            draft_id=payload.draft_id,
            fields=payload.fields,
            confirm_token=payload.confirm_token,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="draft_not_found") from exc
    except LookupError as exc:
        detail = str(exc)
        if detail.startswith("ticket_not_found:"):
            raise HTTPException(status_code=404, detail="ticket_not_found") from exc
        raise HTTPException(status_code=404, detail="draft_not_found") from exc
