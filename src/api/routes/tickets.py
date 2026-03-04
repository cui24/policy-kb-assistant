"""
`/tickets` 路由：提供 L2 最小工单 CRUD。

一、程序目标
1. 创建工单。
2. 查询单个工单。
3. 列出工单。
4. 更新工单状态。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api import crud
from src.api.deps import get_db, require_api_key
from src.api.schemas import (
    TicketCancelRequest,
    TicketCommentRequest,
    TicketCreateRequest,
    TicketDetailResponse,
    TicketEscalateRequest,
    TicketResponse,
    TicketStatusUpdateRequest,
)
from src.api.services import (
    add_ticket_comment_workflow,
    cancel_ticket_workflow,
    create_ticket_workflow,
    escalate_ticket_workflow,
    serialize_ticket_detail,
    serialize_ticket,
    update_ticket_status_workflow,
)


router = APIRouter(tags=["tickets"])


@router.post("/tickets", response_model=TicketResponse, dependencies=[Depends(require_api_key)])
def create_ticket(payload: TicketCreateRequest, db: Session = Depends(get_db)) -> TicketResponse:
    """创建工单，并返回工单号。"""
    return create_ticket_workflow(
        db,
        creator=payload.creator,
        department=payload.department,
        category=payload.category,
        priority=payload.priority,
        title=payload.title,
        description=payload.description,
        contact=payload.contact,
        context=payload.context,
    )


@router.get("/tickets/{ticket_id}", response_model=TicketDetailResponse)
def get_ticket(ticket_id: str, db: Session = Depends(get_db)) -> TicketDetailResponse:
    """按工单号查询工单。"""
    ticket = crud.get_ticket_by_public_id(db, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket_not_found")
    return serialize_ticket_detail(db, ticket)


@router.get("/tickets", response_model=list[TicketDetailResponse])
def list_ticket_items(status: str | None = Query(default=None), db: Session = Depends(get_db)) -> list[TicketDetailResponse]:
    """列出工单，支持按状态过滤。"""
    tickets = crud.list_tickets(db, status=status)
    return [serialize_ticket(ticket) for ticket in tickets]


@router.patch("/tickets/{ticket_id}", response_model=TicketDetailResponse, dependencies=[Depends(require_api_key)])
def update_ticket_status(ticket_id: str, payload: TicketStatusUpdateRequest, db: Session = Depends(get_db)) -> TicketDetailResponse:
    """更新工单状态。"""
    try:
        return update_ticket_status_workflow(db, ticket_id, payload.status, payload.actor)
    except LookupError:
        raise HTTPException(status_code=404, detail="ticket_not_found") from None


@router.post("/tickets/{ticket_id}/comments", response_model=TicketDetailResponse, dependencies=[Depends(require_api_key)])
def add_ticket_comment(ticket_id: str, payload: TicketCommentRequest, db: Session = Depends(get_db)) -> TicketDetailResponse:
    """向工单追加说明。"""
    try:
        return add_ticket_comment_workflow(db, ticket_id, payload.comment, payload.actor)
    except LookupError:
        raise HTTPException(status_code=404, detail="ticket_not_found") from None


@router.post("/tickets/{ticket_id}/escalate", response_model=TicketDetailResponse, dependencies=[Depends(require_api_key)])
def escalate_ticket(ticket_id: str, payload: TicketEscalateRequest, db: Session = Depends(get_db)) -> TicketDetailResponse:
    """催办工单。"""
    try:
        return escalate_ticket_workflow(db, ticket_id, payload.actor, payload.reason)
    except LookupError:
        raise HTTPException(status_code=404, detail="ticket_not_found") from None


@router.post("/tickets/{ticket_id}/cancel", response_model=TicketDetailResponse, dependencies=[Depends(require_api_key)])
def cancel_ticket(ticket_id: str, payload: TicketCancelRequest, db: Session = Depends(get_db)) -> TicketDetailResponse:
    """取消工单。"""
    try:
        return cancel_ticket_workflow(db, ticket_id, payload.actor, payload.reason)
    except LookupError:
        raise HTTPException(status_code=404, detail="ticket_not_found") from None
