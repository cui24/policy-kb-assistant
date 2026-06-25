"""
`/tickets` 路由：提供 L2 最小工单 CRUD。

一、程序目标
1. 创建工单。
2. 查询单个工单。
3. 列出工单。
4. 更新工单状态。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from redis import Redis
from sqlalchemy.orm import Session

from src.api import crud
from src.api.deps import get_db, get_redis_dep
from src.api.deps_auth import AuthenticatedUser, get_current_active_user
from src.api.idempotency import (
    IdempotencyStoreError,
    abort_idempotent_request,
    begin_idempotent_request,
    finish_idempotent_success,
)
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

"""
    创建一个 APIRouter 对象，用来收集这一组和 tickets 相关的接口
    可以把它理解成一个“小路由表”或者“子路由容器”
    在 FastAPI 里，通常不会把所有接口都写在一个 app.py 里，而是拆模块：
        --tickets.py 管工单接口
        --users.py 管用户接口
        --auth.py 管登录接口
    每个文件里各自定义一个router = APIRouter(...)
    然后主程序里再统一挂载到总应用上
    例如通常会有类似
    --app = FastAPI()
    --app.include_router(ask_router)
    --app.include_router(tickets_router)
    --app.include_router(agent_router)
    --app.include_router(history_router)
"""
router = APIRouter(tags=["tickets"])


def _role_value(role: object) -> str:
    if hasattr(role, "value"):
        return str(getattr(role, "value") or "").strip().lower()
    return str(role or "").strip().lower()


def _ensure_ticket_owner_or_admin(db: Session, ticket_id: str, current_user: AuthenticatedUser) -> None:
    """仅允许工单 owner 或 admin/support 执行写操作。"""
    ticket = crud.get_ticket_by_public_id(db, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket_not_found")
    current_role = _role_value(current_user.role)
    if current_role in {"admin", "support"}:
        return
    owner_user_id = str(ticket.creator_user_id or "").strip()
    if owner_user_id and owner_user_id == str(current_user.id):
        return
    if str(ticket.creator or "").strip() == str(current_user.username):
        return
    raise HTTPException(status_code=403, detail="forbidden_ticket_operation")

"""
    tags=["tickets"]主要是给接口文档分类用的，不是决定路由匹配的核心逻辑。
    打开 Swagger 文档 /docs 时，所有这个 router 下面的接口会被归到 tickets 这一组里
    它更像：
    --文档分组标签
    --OpenAPI 元数据
    --方便前端/后端看接口
    它不等于 URL 路径，也不等于“这个路由叫 tickets”
"""

"""
    真正定义路由的不是这一句，而是下面这些 
    @router.get(...) / 
    @router.post(...) / 
    @router.patch(...)
    APIRouter 本身只是一个容器，真正往里面注册“路径 + 方法 + 处理函数”的，是这些装饰器
    @router.post(...)方法含义 --路径：/tickets, --方法：POST, --处理函数：create_ticket
"""
@router.post("/tickets", response_model=TicketResponse)
def create_ticket(
    payload: TicketCreateRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
    redis: Redis = Depends(get_redis_dep),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> TicketResponse:
    """
    创建工单，并返回工单号。

    幂等策略：
    1. 客户端必须传 `Idempotency-Key`。
    2. 同用户 + 同 key + 同 payload：
       - 首次执行创建工单
       - 重试直接返回第一次成功结果（replay）
    3. 同 key + 不同 payload：返回 409，避免“同键多义”。
    4. 同 key 正在处理中：返回 409，提示稍后重试。
    """
    normalized_idempotency_key = str(idempotency_key or "").strip()
    if not normalized_idempotency_key:
        raise HTTPException(status_code=400, detail="idempotency_key_required")

    # 使用请求体生成 payload 指纹，作为“同 key 是否同请求”的判定依据。
    request_payload = payload.model_dump(mode="json")
    try:
        decision = begin_idempotent_request(
            redis,
            scope="tickets:create",
            user_id=str(current_user.id),
            idempotency_key=normalized_idempotency_key,
            payload=request_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IdempotencyStoreError as exc:
        # Redis 不可用时，不做“降级继续写库”，直接失败，避免失去幂等保护后产生重复数据。
        raise HTTPException(status_code=503, detail="idempotency_store_unavailable") from exc

    # 命中历史成功请求，直接重放历史响应，不再执行 create_ticket_workflow。
    if decision.state == "replay":
        cached_response = decision.cached_response
        if not isinstance(cached_response, dict):
            raise HTTPException(status_code=409, detail="idempotency_cache_invalid")
        return TicketResponse.model_validate(cached_response)
    # 同一个 key 对应了不同请求体，属于调用方错误。
    if decision.state == "conflict":
        raise HTTPException(status_code=409, detail="idempotency_key_conflict")
    # 已有同键请求在处理中，避免并发双写。
    if decision.state == "in_progress":
        raise HTTPException(status_code=409, detail="idempotency_request_in_progress")

    try:
        # 只有拿到 `new` 执行权后才会进入真实建单流程。
        response_payload = create_ticket_workflow(
            db,
            creator=str(current_user.username),
            department=payload.department,
            category=payload.category,
            priority=payload.priority,
            title=payload.title,
            description=payload.description,
            contact=payload.contact,
            context=payload.context,
            actor_user_id=str(current_user.id),
        )
    except Exception:
        # 业务失败时撤销 processing 锁，避免 key 长时间卡住无法重试。
        try:
            abort_idempotent_request(redis, decision)
        except IdempotencyStoreError:
            pass
        raise

    try:
        # 业务成功后落 success 记录，后续相同请求可直接 replay。
        finish_idempotent_success(redis, decision, response_payload)
    except IdempotencyStoreError:
        # 工单已创建成功，不因为缓存写失败而中断主流程。
        pass
    return response_payload

"""
    模块加载时就执行了，而不是在请求时执行
    FastAPI 会把这个函数注册到 router.routes
    FastAPI 不是自动扫描“所有函数都变接口”。
    它只会把被装饰器标记过的函数当成接口。普通函数无法变成路由
    请求来了以后:
    1.看请求方法是否匹配
    2.看路径是否匹配
    3.在已注册路由表里找匹配项
    4.如果找到匹配项，则执行处理函数
    5.路径不对 → 404
    6.路径对但方法不对 → 405
    因此GET /tickets/TCK-2026-000123会被解析成：
    --ticket_id=TCK-2026-000123
    --GET方法通过路由找到get_ticket函数运行
"""
@router.get("/tickets/{ticket_id}", response_model=TicketDetailResponse)
def get_ticket(ticket_id: str, db: Session = Depends(get_db)) -> TicketDetailResponse:
    """按工单号查询工单。"""
    ticket = crud.get_ticket_by_public_id(db, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket_not_found")
    return serialize_ticket_detail(db, ticket)

"""
    @router.post(...)Python 装饰器（decorator）
    装饰器的本质就是：
    --拿到下面那个函数对象，对它做额外处理
    在 FastAPI 里，这个“额外处理”主要就是：
    --把函数注册成路由（一个GET /tickets接口）
    --记录请求方法
    --记录路径

    --记录参数校验规则
    response_model=TicketResponse表示这个接口返回的数据，
    应该符合 TicketResponse 这个 Pydantic 模型
    作用通常有三个：
        1. 响应结构校验
        如果你返回的数据不符合 TicketResponse，FastAPI 会报错或过滤。
        2. 自动序列化
        比如对象转 dict / JSON。
        3. 生成接口文档
        Swagger 会知道这个接口返回什么字段。

    --记录依赖项
    current_user: AuthenticatedUser = Depends(get_current_active_user)
    表示：
        在执行这个接口前，先解析并校验 Bearer Token
    这里的意思大概率是：
        这个接口需要登录态
        未登录或 token 无效会直接拒绝
    --记录响应模型
    生成 OpenAPI 文档
"""
@router.get("/tickets", response_model=list[TicketDetailResponse])
def list_ticket_items(status: str | None = Query(default=None), db: Session = Depends(get_db)) -> list[TicketDetailResponse]:
    """列出工单，支持按状态过滤。"""
    tickets = crud.list_tickets(db, status=status)
    return [serialize_ticket(ticket) for ticket in tickets]


@router.patch("/tickets/{ticket_id}", response_model=TicketDetailResponse)
def update_ticket_status(
    ticket_id: str,
    payload: TicketStatusUpdateRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> TicketDetailResponse:
    """更新工单状态。"""
    _ensure_ticket_owner_or_admin(db, ticket_id, current_user)
    try:
        return update_ticket_status_workflow(
            db,
            ticket_id,
            payload.status,
            actor=str(current_user.username),
            actor_user_id=str(current_user.id),
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="ticket_not_found") from None


@router.post("/tickets/{ticket_id}/comments", response_model=TicketDetailResponse)
def add_ticket_comment(
    ticket_id: str,
    payload: TicketCommentRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> TicketDetailResponse:
    """向工单追加说明。"""
    _ensure_ticket_owner_or_admin(db, ticket_id, current_user)
    try:
        return add_ticket_comment_workflow(
            db,
            ticket_id,
            payload.comment,
            actor=str(current_user.username),
            actor_user_id=str(current_user.id),
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="ticket_not_found") from None


@router.post("/tickets/{ticket_id}/escalate", response_model=TicketDetailResponse)
def escalate_ticket(
    ticket_id: str,
    payload: TicketEscalateRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> TicketDetailResponse:
    """催办工单。"""
    _ensure_ticket_owner_or_admin(db, ticket_id, current_user)
    try:
        return escalate_ticket_workflow(
            db,
            ticket_id,
            actor=str(current_user.username),
            actor_user_id=str(current_user.id),
            reason=payload.reason,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="ticket_not_found") from None


@router.post("/tickets/{ticket_id}/cancel", response_model=TicketDetailResponse)
def cancel_ticket(
    ticket_id: str,
    payload: TicketCancelRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_active_user),
) -> TicketDetailResponse:
    """取消工单。"""
    _ensure_ticket_owner_or_admin(db, ticket_id, current_user)
    try:
        return cancel_ticket_workflow(
            db,
            ticket_id,
            actor=str(current_user.username),
            actor_user_id=str(current_user.id),
            reason=payload.reason,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="ticket_not_found") from None
