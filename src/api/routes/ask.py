"""
`/ask` 路由：把现有 L0/L1 问答链路包装成 API。

一、程序目标
1. 接收 HTTP 请求中的 `question`。
2. 调用业务层执行完整问答。
3. 返回 `answer + citations + trace meta`。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.api import models
from src.api.deps import get_db
from src.api.deps_auth import get_current_active_user
from src.api.schemas import AskRequest, AskResponse
from src.api.services import public_kb_response, run_ask_workflow_async, run_ask_workflow_stream_async


router = APIRouter(tags=["ask"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
) -> AskResponse:
    """执行问答接口。"""
    result = await run_ask_workflow_async(
        db,
        question=payload.question,
        user=str(current_user.username),
        department=payload.department,
        actor_user_id=str(current_user.id),
    )
    return public_kb_response(result)

@router.post("/ask/stream")
async def ask_stream(
    payload: AskRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
) -> StreamingResponse:
    """SSE 流式问答接口。"""
    async def event_generator():
        try:
            async for event in run_ask_workflow_stream_async(
                db,
                question=payload.question,
                user=str(current_user.username),
                department=payload.department,
                actor_user_id=str(current_user.id),
            ):
                yield _sse(str(event.get("event") or "message"), dict(event.get("data") or {}))
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})
            yield _sse("done", {"ok": False})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
