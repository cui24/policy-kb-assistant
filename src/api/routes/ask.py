"""
`/ask` 路由：把现有 L0/L1 问答链路包装成 API。

一、程序目标
1. 接收 HTTP 请求中的 `question`。
2. 调用业务层执行完整问答。
3. 返回 `answer + citations + trace meta`。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_db, require_api_key
from src.api.schemas import AskRequest, AskResponse
from src.api.services import public_kb_response, run_ask_workflow


router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(payload: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    """执行问答接口。"""
    result = run_ask_workflow(
        db,
        question=payload.question,
        user=payload.user,
        department=payload.department,
    )
    return public_kb_response(result)
