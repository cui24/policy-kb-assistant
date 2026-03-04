"""
L3-3 追溯路由：把 `kb_queries` 与 `audit_logs` 暴露为可查询 API。

一、程序目标
1. 提供问答记录列表与详情查询。
2. 提供审计日志过滤查询。
3. 让 UI 与排障工具可以按 `request_id` / `ticket_id` 回放一次完整链路。

二、输入输出
1. 输入：HTTP Query 参数，例如 `request_id`、`ticket_id`、`actor`。
2. 输出：结构化 JSON 列表或详情对象。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api import crud
from src.api.deps import get_db
from src.api.schemas import AuditLogResponse, KBQueryDetailResponse
from src.api.services import serialize_audit_log, serialize_kb_query


router = APIRouter(tags=["history"])


@router.get("/kb_queries", response_model=list[KBQueryDetailResponse])
def list_kb_query_items(
    user: str | None = Query(default=None),
    department: str | None = Query(default=None),
    request_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[KBQueryDetailResponse]:
    """列出问答记录，支持按用户、部门与 request_id 过滤。"""
    records = crud.list_kb_queries(
        db,
        user_name=user,
        department=department,
        request_id=request_id,
        limit=limit,
    )
    return [serialize_kb_query(record) for record in records]


@router.get("/kb_queries/{request_id}", response_model=KBQueryDetailResponse)
def get_kb_query_detail(request_id: str, db: Session = Depends(get_db)) -> KBQueryDetailResponse:
    """按 request_id 查询单条问答记录详情。"""
    record = crud.get_kb_query_by_request_id(db, request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="kb_query_not_found")
    return serialize_kb_query(record)


@router.get("/audit_logs", response_model=list[AuditLogResponse])
def list_audit_log_items(
    request_id: str | None = Query(default=None),
    ticket_id: str | None = Query(default=None),
    action_type: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=300),
    db: Session = Depends(get_db),
) -> list[AuditLogResponse]:
    """列出审计日志，支持按 request_id、ticket_id、动作与操作者过滤。"""
    records = crud.list_audit_logs(
        db,
        request_id=request_id,
        ticket_id=ticket_id,
        action_type=action_type,
        actor=actor,
        limit=limit,
    )
    return [serialize_audit_log(record) for record in records]
