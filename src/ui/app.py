"""
L3/L4 Web 演示程序：提供“UI 通过 HTTP 调用后端 API”的最小可用闭环。

一、程序目标
1. 让 Streamlit 页面不再直接调用 Python 函数，而是真正走 L2 HTTP API。
2. 默认入口走 `/agent`，展示“一句话 -> 路由 -> 问答或建单”的完整链路。
3. 保留高级模式：可以单独调用 `/ask` 与 `/tickets`，方便调试和演示。
4. 提供工单列表、详情查询和状态更新，让 UI 真正接上 L2 后端。
5. 提供追溯区：可按 `request_id` 或 `ticket_id` 回放问答、审计和关联工单。
6. 在 L4 中支持草稿续办：`NEED_MORE_INFO` 后可直接补地点/联系方式继续提交。

二、程序入口与运行顺序
1. 命令入口：`streamlit run src/ui/app.py`
2. `main()` 内部顺序如下：
   2.1 读取 `.env`
   2.2 配置页面与样式
   2.3 初始化 `session_state`
   2.4 在侧边栏读取 API 地址、默认用户、默认部门
   2.5 创建 `PolicyAPIClient`
   2.6 渲染主操作区：
       - `/agent` 一句话入口
       - `/ask` 仅问答入口
       - `/tickets` 手动建单入口
   2.7 将最近一次 API 响应写入 `session_state`
   2.8 渲染问答结果、工单结果、抽取结果、Trace 信息
   2.9 渲染工单管理区：列表、详情、状态更新
   2.10 渲染追溯区：查询 `kb_queries`、`audit_logs` 与关联工单
   2.11 若存在活跃草稿，则渲染“继续完成工单”表单

三、输入输出数据格式
1. UI 输入：
   - `text`: 一句话自然语言输入，走 `/agent`
   - `question`: 问答输入，走 `/ask`
   - `ticket form`: 手动建单字段，走 `/tickets`
2. API 输出：
   - `/ask`: `AskResponse`
   - `/tickets`: `TicketResponse` / `TicketDetailResponse`
   - `/agent`: `AgentResponse`
3. 页面状态：
   - 把最近一次 API 调用结果放到 `session_state`，便于刷新后继续查看

四、程序可以理解成的伪代码
1. 让用户先配置 API 地址和默认身份
2. 创建 API 客户端
3. 用户点“一句话入口”就调 `/agent`
4. 用户点“仅问答”就调 `/ask`
5. 用户提交手动工单表单就调 `/tickets`
6. 页面根据返回 JSON 决定展示：
   - answer + citations
   - ticket_id
   - route / extraction / trace
7. 页面右侧再调用 `/tickets` 查询列表与详情，并支持更新状态
8. 页面右侧支持按 `request_id` 或 `ticket_id` 回放一次完整链路
9. 当 `/agent` 返回草稿时，页面保留 `draft_id`，并允许用户直接补字段续办
"""

from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from src.ui.api_client import APIClientError, PolicyAPIClient


EXAMPLE_CASES = [
    {
        "label": "一句话报修",
        "text": "我宿舍网络连不上，帮我提交报修工单。地点金明校区，手机号13812345678。",
    },
    {
        "label": "仅问答",
        "text": "统一身份认证的登录地址是什么？",
    },
    {
        "label": "需补信息",
        "text": "帮我提交网络报修工单，我宿舍上不了网。",
    },
]

TICKET_STATUS_OPTIONS = ["open", "in_progress", "resolved", "closed", "cancelled"]
TICKET_CATEGORY_OPTIONS = ["network", "account", "dorm", "other"]
TICKET_PRIORITY_OPTIONS = ["P0", "P1", "P2", "P3"]



def _inject_styles() -> None:
    """注入页面样式，保持现有 demo 观感。"""
    st.markdown(
        """
        <style>
        :root {
          --ink: #14213d;
          --accent: #0b8f8c;
          --accent-soft: #d7f5ef;
          --paper: #fffaf2;
          --line: #d7d3c7;
          --warm: #f3b73f;
          --danger-soft: #fde7e7;
          --danger-ink: #8f2929;
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(243,183,63,0.16), transparent 34%),
            radial-gradient(circle at top right, rgba(11,143,140,0.14), transparent 30%),
            linear-gradient(180deg, #fffdf8 0%, var(--paper) 100%);
          color: var(--ink);
        }
        .block-container {
          padding-top: 2rem;
          padding-bottom: 3rem;
          max-width: 1180px;
        }
        .hero-card,
        .panel-card,
        .answer-card,
        .error-card,
        .route-card,
        .ticket-card {
          border: 1px solid var(--line);
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.92);
          box-shadow: 0 12px 32px rgba(20, 33, 61, 0.07);
        }
        .hero-card {
          padding: 1.2rem 1.25rem 1rem 1.25rem;
          margin-bottom: 1rem;
        }
        .hero-title {
          font-size: 1.8rem;
          font-weight: 700;
          color: var(--ink);
          margin-bottom: 0.35rem;
        }
        .hero-subtitle {
          color: rgba(20, 33, 61, 0.74);
          line-height: 1.55;
          font-size: 0.98rem;
        }
        .answer-card,
        .route-card,
        .ticket-card,
        .error-card {
          padding: 1rem 1.1rem;
          margin-bottom: 0.8rem;
        }
        .answer-card {
          border-left: 6px solid var(--accent);
        }
        .error-card {
          border-left: 6px solid #c94c4c;
          background: rgba(253, 241, 241, 0.96);
        }
        .ticket-card {
          border-left: 6px solid var(--warm);
        }
        .answer-label {
          font-size: 0.78rem;
          letter-spacing: 0.08em;
          font-weight: 700;
          color: var(--accent);
          margin-bottom: 0.45rem;
        }
        .answer-text {
          line-height: 1.75;
          color: var(--ink);
          font-size: 1rem;
        }
        .stage-chip {
          display: inline-block;
          padding: 0.28rem 0.55rem;
          border-radius: 999px;
          background: var(--accent-soft);
          color: var(--accent);
          font-size: 0.76rem;
          font-weight: 700;
          margin-right: 0.45rem;
          margin-bottom: 0.35rem;
        }
        .metric-box {
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 0.75rem 0.9rem;
          background: rgba(255,255,255,0.88);
          margin-bottom: 0.7rem;
        }
        .metric-title {
          color: rgba(20, 33, 61, 0.65);
          font-size: 0.78rem;
          margin-bottom: 0.15rem;
        }
        .metric-value {
          color: var(--ink);
          font-size: 1rem;
          font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def _render_header() -> None:
    """渲染页面标题和说明。"""
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">政策问答、工单与审计演示</div>
          <div class="hero-subtitle">
            当前页面已切换为通过 HTTP 调用 L2 API。默认入口走 <code>/agent</code>，
            能展示“一句话问答 / 建单 / 补信息”的真实后端闭环；同时保留 <code>/ask</code>
            与 <code>/tickets</code> 的单独入口，方便调试与教学演示。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _ensure_state() -> None:
    """初始化页面交互需要的 session_state 键。"""
    defaults = {
        "api_base_url": "http://localhost:8080",
        "ui_api_key": "",
        "ui_user": "alice",
        "ui_department": "IT",
        "agent_input": "",
        "ask_input": "",
        "last_error": None,
        "last_error_context": None,
        "last_ask": None,
        "last_agent": None,
        "last_manual_ticket": None,
        "last_ticket_list": [],
        "selected_ticket_id": "",
        "selected_ticket_detail": None,
        "last_api_health": None,
        "trace_request_id": "",
        "trace_ticket_id": "",
        "trace_kb_detail": None,
        "trace_audit_logs": [],
        "trace_ticket_detail": None,
        "active_draft_id": "",
        "active_draft_missing_fields": [],
        "draft_followup_location": "",
        "draft_followup_contact": "",
        "draft_followup_note": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def _render_sidebar() -> tuple[str, str, str, PolicyAPIClient]:
    """渲染侧边栏配置，并创建 API 客户端。"""
    with st.sidebar:
        st.header("API 设置")
        base_url = st.text_input(
            "L2 API Base URL",
            value=st.session_state.get("api_base_url", "http://localhost:8080"),
            key="api_base_url",
        ).strip()
        user_name = st.text_input(
            "默认用户",
            value=st.session_state.get("ui_user", "alice"),
            key="ui_user",
        ).strip()
        api_key = st.text_input(
            "API Key（写接口必填）",
            value=st.session_state.get("ui_api_key", ""),
            key="ui_api_key",
            type="password",
        ).strip()
        department = st.text_input(
            "默认部门",
            value=st.session_state.get("ui_department", "IT"),
            key="ui_department",
        ).strip()

        client = PolicyAPIClient(base_url=base_url, api_key=api_key or None)

        if st.button("检测 API", use_container_width=True, key="check_api_health"):
            try:
                st.session_state["last_api_health"] = client.health()
                st.session_state["last_error"] = None
                st.session_state["last_error_context"] = None
            except APIClientError as exc:
                st.session_state["last_api_health"] = None
                _set_error("API 健康检查", exc)

        health_info = st.session_state.get("last_api_health")
        if isinstance(health_info, dict):
            st.success(
                f"API 可用：status={health_info.get('status')} · stage={health_info.get('stage')}"
            )
        else:
            st.caption("建议先点击“检测 API”，确认 `make api` 已启动。")

    return base_url, user_name or "anonymous", department or "general", client



def _set_error(context: str, exc: APIClientError) -> None:
    """把 API 错误写入页面状态，供统一展示。"""
    st.session_state["last_error"] = {
        "message": exc.message,
        "status_code": exc.status_code,
        "detail": exc.detail,
    }
    st.session_state["last_error_context"] = context



def _clear_error() -> None:
    """清空上一次错误。"""
    st.session_state["last_error"] = None
    st.session_state["last_error_context"] = None



def _render_error_card() -> None:
    """统一展示最近一次 API 错误。"""
    error_info = st.session_state.get("last_error")
    if not isinstance(error_info, dict):
        return

    context = html.escape(str(st.session_state.get("last_error_context") or "API 调用"))
    message = html.escape(str(error_info.get("message") or "未知错误"))
    status_code = error_info.get("status_code")
    detail = error_info.get("detail")
    detail_text = html.escape(json.dumps(detail, ensure_ascii=False, indent=2))

    st.markdown(
        f"""
        <div class="error-card">
          <div class="answer-label" style="color:#8f2929;">{context}</div>
          <div class="answer-text" style="color:#8f2929;">{message}</div>
          <div style="margin-top:0.5rem; color:#8f2929; font-size:0.88rem;">
            status_code: {status_code if status_code is not None else 'N/A'}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("查看后端错误明细"):
        st.code(detail_text, language="json")



def _render_example_runner() -> str | None:
    """渲染示例按钮，并返回本轮选中的示例文本。"""
    st.caption("一键示例：点击后会把文本填入“一句话入口”输入框。")
    columns = st.columns(len(EXAMPLE_CASES), gap="small")
    selected_text: str | None = None

    for index, (column, item) in enumerate(zip(columns, EXAMPLE_CASES), start=1):
        with column:
            if st.button(item["label"], key=f"example_run_{index}", use_container_width=True):
                selected_text = str(item["text"])

    return selected_text



def _render_answer_block(answer_text: str) -> None:
    """渲染答案主体。"""
    safe_text = html.escape(answer_text.strip())
    st.markdown(
        f"""
        <div class="answer-card">
          <div class="answer-label">ANSWER</div>
          <div class="answer-text">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _render_citations(citations: list[dict[str, Any]]) -> None:
    """渲染 citations 列表。"""
    st.subheader("引用")
    if not citations:
        st.info("当前没有可展示的引用。若答案为拒答，这是正常情况。")
        return

    for index, item in enumerate(citations, start=1):
        doc_id = html.escape(str(item.get("doc_id") or ""))
        page = item.get("page")
        snippet = html.escape(str(item.get("snippet", "") or ""))
        st.markdown(
            f"""
            <div class="panel-card" style="padding:0.85rem 0.95rem; margin-bottom:0.7rem;">
              <div style="font-size:0.82rem; color:rgba(20,33,61,0.64); margin-bottom:0.35rem;">
                引用 {index} · <strong>{doc_id}</strong> · 第 {page} 页
              </div>
              <div style="line-height:1.7; color:#14213d;">{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def _render_hits(hits: list[dict[str, Any]]) -> None:
    """渲染 top-k 命中证据。"""
    st.subheader("命中证据")
    if not hits:
        st.info("当前响应里没有返回检索证据。")
        return

    for index, hit in enumerate(hits, start=1):
        score = float(hit.get("score", 0.0) or 0.0)
        doc_id = html.escape(str(hit.get("doc_id") or ""))
        page = hit.get("page")
        snippet = html.escape(str(hit.get("snippet", "") or ""))
        st.markdown(
            f"""
            <div class="panel-card" style="padding:0.85rem 0.95rem; margin-bottom:0.7rem;">
              <div style="font-size:0.82rem; color:rgba(20,33,61,0.64); margin-bottom:0.35rem;">
                Top {index} · score={score:.3f} · <strong>{doc_id}</strong> · 第 {page} 页
              </div>
              <div style="line-height:1.65; color:#14213d;">{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def _render_trace_block(request_id: str, meta: dict[str, Any]) -> None:
    """用折叠区展示 request_id 和 trace 信息。"""
    with st.expander("查看 Trace / Debug 信息"):
        st.code(
            json.dumps(
                {
                    "request_id": request_id,
                    "meta": meta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )



def _render_kb_response(kb_response: dict[str, Any]) -> None:
    """渲染 `/ask` 或 `/agent.kb` 返回的问答结果。"""
    if not kb_response:
        return

    answer_text = str(kb_response.get("answer") or "")
    citations = kb_response.get("citations", []) or []
    meta = kb_response.get("meta", {}) or {}
    request_id = str(kb_response.get("request_id") or "")

    _render_answer_block(answer_text)
    _render_citations(citations)

    if request_id:
        _render_trace_block(request_id, meta)

    retrieve_hits = meta.get("retrieve_topk", []) or []
    _render_hits(retrieve_hits)



def _render_agent_response(agent_response: dict[str, Any]) -> None:
    """根据 `/agent` 的 route 分支渲染不同结果。"""
    if not agent_response:
        return

    route = str(agent_response.get("route") or "UNKNOWN")
    message = str(agent_response.get("message") or "")
    missing_fields = agent_response.get("missing_fields", []) or []
    ticket = agent_response.get("ticket") or None
    ticket_detail = agent_response.get("ticket_detail") or None
    draft = agent_response.get("draft") or None
    extraction = agent_response.get("extraction") or None
    kb_response = agent_response.get("kb") or None

    st.markdown(
        f"""
        <div class="route-card">
          <div class="answer-label">AGENT ROUTE</div>
          <div class="answer-text">{html.escape(route)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if route == "CREATE_TICKET" and isinstance(ticket, dict):
        st.markdown(
            f"""
            <div class="ticket-card">
              <div class="answer-label" style="color:#9a6a00;">TICKET CREATED</div>
              <div class="answer-text">工单号：{html.escape(str(ticket.get('ticket_id') or ''))}</div>
              <div style="margin-top:0.45rem; color:#7a5d14;">状态：{html.escape(str(ticket.get('status') or ''))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif route == "NEED_MORE_INFO":
        st.warning(message or "当前信息不足，无法直接建单。")
        if missing_fields:
            st.info("缺失字段：" + "、".join(str(item) for item in missing_fields))
        if isinstance(draft, dict):
            st.info("草稿号：" + str(draft.get("draft_id") or ""))
            if draft.get("expires_at"):
                st.caption("草稿有效期至：" + str(draft.get("expires_at")))
    elif route in ("DRAFT_EXPIRED", "DRAFT_NOT_FOUND"):
        st.warning(message or "当前草稿不可继续使用，请重新发起。")
        if isinstance(draft, dict):
            st.info("草稿号：" + str(draft.get("draft_id") or ""))
    elif route == "ASK":
        st.success("当前输入被判定为纯问答请求。")
    elif route == "LOOKUP_TICKET":
        st.success(message or "已查询工单进度。")
    elif route == "ADD_TICKET_COMMENT":
        st.success(message or "已追加工单说明。")
    elif route == "ESCALATE_TICKET":
        st.success(message or "已记录催办。")
    elif route == "CANCEL_TICKET":
        st.success(message or "已取消工单。")

    if extraction:
        with st.expander("查看 Agent 抽取结果"):
            st.code(json.dumps(extraction, ensure_ascii=False, indent=2), language="json")

    if isinstance(ticket_detail, dict):
        _render_ticket_detail_card(ticket_detail)

    if isinstance(kb_response, dict):
        _render_kb_response(kb_response)



def _handle_agent_submit(client: PolicyAPIClient, text: str, user_name: str, department: str) -> None:
    """执行 `/agent` 调用并保存结果。"""
    _clear_error()
    try:
        result = client.agent(text=text, user=user_name, department=department)
    except APIClientError as exc:
        _set_error("调用 /agent", exc)
        return

    st.session_state["last_agent"] = result
    st.session_state["last_ask"] = None

    kb_response = result.get("kb") or {}
    if isinstance(kb_response, dict) and kb_response.get("request_id"):
        st.session_state["trace_request_id"] = str(kb_response.get("request_id"))

    draft = result.get("draft") or {}
    if isinstance(draft, dict) and draft.get("draft_id"):
        st.session_state["active_draft_id"] = str(draft.get("draft_id"))
        st.session_state["active_draft_missing_fields"] = list(draft.get("missing_fields") or [])
        st.session_state["draft_followup_location"] = ""
        st.session_state["draft_followup_contact"] = ""
        st.session_state["draft_followup_note"] = ""
        if not st.session_state.get("trace_request_id") and draft.get("kb_request_id"):
            st.session_state["trace_request_id"] = str(draft.get("kb_request_id"))
    else:
        st.session_state["active_draft_id"] = ""
        st.session_state["active_draft_missing_fields"] = []
        st.session_state["draft_followup_location"] = ""
        st.session_state["draft_followup_contact"] = ""
        st.session_state["draft_followup_note"] = ""

    ticket = result.get("ticket") or {}
    if isinstance(ticket, dict) and ticket.get("ticket_id"):
        st.session_state["selected_ticket_id"] = str(ticket.get("ticket_id"))
        st.session_state["trace_ticket_id"] = str(ticket.get("ticket_id"))
        st.session_state["active_draft_id"] = ""
        st.session_state["active_draft_missing_fields"] = []
        st.session_state["draft_followup_location"] = ""
        st.session_state["draft_followup_contact"] = ""
        st.session_state["draft_followup_note"] = ""



def _handle_ask_submit(client: PolicyAPIClient, question: str, user_name: str, department: str) -> None:
    """执行 `/ask` 调用并保存结果。"""
    _clear_error()
    try:
        result = client.ask(question=question, user=user_name, department=department)
    except APIClientError as exc:
        _set_error("调用 /ask", exc)
        return

    st.session_state["last_ask"] = result
    st.session_state["last_agent"] = None
    if result.get("request_id"):
        st.session_state["trace_request_id"] = str(result.get("request_id"))



def _handle_draft_continue_submit(
    client: PolicyAPIClient,
    draft_id: str,
    user_name: str,
    department: str,
    location: str,
    contact: str,
    note: str,
) -> None:
    """继续提交草稿：补充字段后再次调用 `/agent`。"""
    payload_fields: dict[str, Any] = {}
    if location.strip():
        payload_fields["location"] = location.strip()
    if contact.strip():
        payload_fields["contact"] = contact.strip()

    _clear_error()
    try:
        result = client.agent(
            text=note.strip(),
            user=user_name,
            department=department,
            draft_id=draft_id,
            fields=payload_fields,
        )
    except APIClientError as exc:
        _set_error("调用 /agent（续办草稿）", exc)
        return

    st.session_state["last_agent"] = result
    st.session_state["last_ask"] = None

    updated_draft = result.get("draft") or {}
    route = str(result.get("route") or "")
    if isinstance(updated_draft, dict) and updated_draft.get("draft_id"):
        st.session_state["active_draft_id"] = str(updated_draft.get("draft_id"))
        st.session_state["active_draft_missing_fields"] = list(updated_draft.get("missing_fields") or [])
        if updated_draft.get("kb_request_id"):
            st.session_state["trace_request_id"] = str(updated_draft.get("kb_request_id"))
    else:
        st.session_state["active_draft_id"] = ""
        st.session_state["active_draft_missing_fields"] = []

    ticket = result.get("ticket") or {}
    if isinstance(ticket, dict) and ticket.get("ticket_id"):
        st.session_state["selected_ticket_id"] = str(ticket.get("ticket_id"))
        st.session_state["trace_ticket_id"] = str(ticket.get("ticket_id"))
        st.session_state["active_draft_id"] = ""
        st.session_state["active_draft_missing_fields"] = []
        st.session_state["draft_followup_location"] = ""
        st.session_state["draft_followup_contact"] = ""
        st.session_state["draft_followup_note"] = ""
    elif route in ("DRAFT_EXPIRED", "DRAFT_NOT_FOUND"):
        st.session_state["active_draft_id"] = ""
        st.session_state["active_draft_missing_fields"] = []



def _handle_manual_ticket_submit(
    client: PolicyAPIClient,
    user_name: str,
    department: str,
    title: str,
    description: str,
    contact: str,
    location: str,
    category: str,
    priority: str,
) -> None:
    """执行手动建单，并把 location 放进 context。"""
    _clear_error()
    try:
        result = client.create_ticket(
            title=title,
            description=description,
            creator=user_name,
            department=department,
            category=category,
            priority=priority,
            contact=contact or None,
            context={
                "location": location or None,
                "source": "streamlit_manual_form",
            },
        )
    except APIClientError as exc:
        _set_error("调用 /tickets（创建工单）", exc)
        return

    st.session_state["last_manual_ticket"] = result
    ticket_id = result.get("ticket_id")
    if ticket_id:
        st.session_state["selected_ticket_id"] = str(ticket_id)
        st.session_state["trace_ticket_id"] = str(ticket_id)



def _refresh_ticket_list(client: PolicyAPIClient, status_filter: str | None) -> None:
    """刷新工单列表缓存。"""
    _clear_error()
    try:
        tickets = client.list_tickets(status=status_filter)
    except APIClientError as exc:
        _set_error("调用 /tickets（列表）", exc)
        return

    st.session_state["last_ticket_list"] = tickets



def _load_ticket_detail(client: PolicyAPIClient, ticket_id: str) -> None:
    """按工单号加载详情。"""
    _clear_error()
    try:
        ticket = client.get_ticket(ticket_id)
    except APIClientError as exc:
        _set_error("调用 /tickets/{ticket_id}（详情）", exc)
        return

    st.session_state["selected_ticket_detail"] = ticket



def _update_ticket_status(
    client: PolicyAPIClient,
    ticket_id: str,
    status: str,
    actor: str,
) -> None:
    """更新工单状态并刷新详情。"""
    _clear_error()
    try:
        ticket = client.update_ticket(ticket_id=ticket_id, status=status, actor=actor)
    except APIClientError as exc:
        _set_error("调用 /tickets/{ticket_id}（更新状态）", exc)
        return

    st.session_state["selected_ticket_detail"] = ticket
    refreshed = st.session_state.get("last_ticket_list") or []
    if isinstance(refreshed, list):
        for item in refreshed:
            if isinstance(item, dict) and item.get("ticket_id") == ticket_id:
                item.update(ticket)



def _sort_audit_logs_for_timeline(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把审计日志按时间升序排列，方便从上到下回放。"""
    return sorted(
        [item for item in logs if isinstance(item, dict)],
        key=lambda item: str(item.get("created_at") or ""),
    )



def _load_trace_bundle(
    client: PolicyAPIClient,
    request_id: str,
    ticket_id: str,
) -> None:
    """按 request_id 或 ticket_id 拉取追溯所需的问答、审计与工单详情。"""
    normalized_request_id = request_id.strip()
    normalized_ticket_id = ticket_id.strip()

    if not normalized_request_id and not normalized_ticket_id:
        st.warning("请至少输入 request_id 或 ticket_id。")
        return

    _clear_error()

    ticket_detail: dict[str, Any] | None = None
    resolved_request_id = normalized_request_id
    audit_logs: list[dict[str, Any]] = []

    try:
        if normalized_ticket_id:
            ticket_detail = client.get_ticket(normalized_ticket_id)
            context = ticket_detail.get("context") or {}
            if not resolved_request_id and isinstance(context, dict):
                kb_request_id = context.get("kb_request_id")
                if kb_request_id:
                    resolved_request_id = str(kb_request_id)

            audit_logs.extend(client.list_audit_logs(ticket_id=normalized_ticket_id, limit=100))

        kb_detail: dict[str, Any] | None = None
        if resolved_request_id:
            kb_detail = client.get_kb_query(resolved_request_id)
            audit_logs.extend(client.list_audit_logs(request_id=resolved_request_id, limit=100))
        else:
            kb_detail = None
    except APIClientError as exc:
        _set_error("调用追溯接口（/kb_queries 或 /audit_logs）", exc)
        return

    unique_logs: dict[str, dict[str, Any]] = {}
    for item in audit_logs:
        if isinstance(item, dict):
            log_id = str(item.get("id") or "")
            if log_id:
                unique_logs[log_id] = item

    st.session_state["trace_request_id"] = resolved_request_id
    st.session_state["trace_ticket_id"] = normalized_ticket_id
    st.session_state["trace_kb_detail"] = kb_detail
    st.session_state["trace_audit_logs"] = _sort_audit_logs_for_timeline(list(unique_logs.values()))
    st.session_state["trace_ticket_detail"] = ticket_detail



def _render_draft_continue_form(client: PolicyAPIClient, user_name: str, department: str) -> None:
    """渲染草稿补全表单，让 NEED_MORE_INFO 可以直接续办。"""
    active_draft_id = str(st.session_state.get("active_draft_id") or "")
    if not active_draft_id:
        return

    missing_fields = st.session_state.get("active_draft_missing_fields") or []
    st.subheader("继续完成工单")
    st.caption(
        "当前存在待补全草稿："
        f"{active_draft_id}；仍缺字段："
        + ("、".join(str(item) for item in missing_fields) if missing_fields else "无")
    )

    with st.form("draft_continue_form"):
        location = st.text_input("补充地点", key="draft_followup_location")
        contact = st.text_input("补充联系方式", key="draft_followup_contact")
        note = st.text_area(
            "补充说明（可选，不用重复描述故障）",
            key="draft_followup_note",
            height=80,
        )
        submitted = st.form_submit_button("继续提交草稿", use_container_width=True)

    if submitted:
        if not location.strip() and not contact.strip() and not note.strip():
            st.warning("请至少补充一个字段或一段补充说明。")
        else:
            _handle_draft_continue_submit(
                client=client,
                draft_id=active_draft_id,
                user_name=user_name,
                department=department,
                location=location,
                contact=contact,
                note=note,
            )



def _render_manual_ticket_form(client: PolicyAPIClient, user_name: str, department: str) -> None:
    """渲染手动建单表单。"""
    st.subheader("手动建单（直接调用 /tickets）")
    with st.form("manual_ticket_form"):
        title = st.text_input("标题", value="宿舍区无法上网")
        description = st.text_area(
            "描述",
            value="用户描述：宿舍区断网，需要排查。",
            height=110,
        )
        form_left, form_right = st.columns(2)
        with form_left:
            contact = st.text_input("联系方式", value="13812345678")
            category = st.selectbox("类别", options=TICKET_CATEGORY_OPTIONS, index=0)
        with form_right:
            location = st.text_input("地点（会写入 context.location）", value="金明校区")
            priority = st.selectbox("优先级", options=TICKET_PRIORITY_OPTIONS, index=1)

        submitted = st.form_submit_button("创建工单", use_container_width=True)

    if submitted:
        if not title.strip() or not description.strip():
            st.warning("标题和描述不能为空。")
        else:
            _handle_manual_ticket_submit(
                client=client,
                user_name=user_name,
                department=department,
                title=title.strip(),
                description=description.strip(),
                contact=contact.strip(),
                location=location.strip(),
                category=category,
                priority=priority,
            )

    created = st.session_state.get("last_manual_ticket")
    if isinstance(created, dict):
        st.success(
            "手动建单成功："
            f"ticket_id={created.get('ticket_id')} · status={created.get('status')}"
        )



def _render_ticket_detail_card(ticket: dict[str, Any]) -> None:
    """渲染选中工单的详情。"""
    if not ticket:
        st.caption("选择或加载某个工单后，这里会显示详情。")
        return

    st.markdown(
        f"""
        <div class="ticket-card">
          <div class="answer-label" style="color:#9a6a00;">TICKET DETAIL</div>
          <div class="answer-text">{html.escape(str(ticket.get('ticket_id') or ''))}</div>
          <div style="margin-top:0.4rem; color:#7a5d14; line-height:1.7;">
            状态：{html.escape(str(ticket.get('status') or ''))}<br>
            标题：{html.escape(str(ticket.get('title') or ''))}<br>
            创建人：{html.escape(str(ticket.get('creator') or ''))}<br>
            处理人：{html.escape(str(ticket.get('assignee') or '未分配'))}<br>
            部门：{html.escape(str(ticket.get('department') or ''))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("查看完整工单 JSON"):
        st.code(json.dumps(ticket, ensure_ascii=False, indent=2), language="json")



def _render_ticket_manager(client: PolicyAPIClient, user_name: str) -> None:
    """渲染工单列表、详情和状态更新。"""
    st.subheader("工单管理")

    filter_options = ["全部", *TICKET_STATUS_OPTIONS]
    selected_filter = st.selectbox("状态过滤", options=filter_options, index=0)
    status_filter = None if selected_filter == "全部" else selected_filter

    refresh_list = st.button("刷新工单列表", use_container_width=True, key="refresh_ticket_list")
    if refresh_list:
        _refresh_ticket_list(client, status_filter)

    ticket_list = st.session_state.get("last_ticket_list") or []
    if ticket_list:
        labels = [
            f"{item.get('ticket_id')} · {item.get('status')} · {item.get('title')}"
            for item in ticket_list
            if isinstance(item, dict)
        ]
        mapping = {
            label: str(item.get("ticket_id"))
            for label, item in zip(labels, ticket_list)
            if isinstance(item, dict)
        }
        selected_label = st.selectbox("选择工单", options=["", *labels], index=0)
        if selected_label:
            selected_ticket_id = mapping.get(selected_label, "")
            if selected_ticket_id:
                st.session_state["selected_ticket_id"] = selected_ticket_id
    else:
        st.caption("点击“刷新工单列表”后，这里会出现可选择的工单。")

    manual_ticket_id = st.text_input(
        "或直接输入工单号",
        value=st.session_state.get("selected_ticket_id", ""),
        key="selected_ticket_id",
    ).strip()

    detail_left, detail_right = st.columns([1.25, 1.0])
    with detail_left:
        if st.button("查询工单详情", use_container_width=True, key="load_ticket_detail"):
            if not manual_ticket_id:
                st.warning("请先选择或输入工单号。")
            else:
                _load_ticket_detail(client, manual_ticket_id)

        ticket_detail = st.session_state.get("selected_ticket_detail")
        if isinstance(ticket_detail, dict):
            _render_ticket_detail_card(ticket_detail)
        else:
            st.caption("当前还没有选中的工单详情。")

    with detail_right:
        new_status = st.selectbox("更新状态为", options=TICKET_STATUS_OPTIONS, index=0)
        if st.button("提交状态更新", use_container_width=True, key="update_ticket_status"):
            if not manual_ticket_id:
                st.warning("请先选择或输入工单号。")
            else:
                _update_ticket_status(client, manual_ticket_id, new_status, user_name)



def _render_audit_timeline(logs: list[dict[str, Any]]) -> None:
    """渲染审计日志时间线。"""
    st.subheader("审计时间线")
    if not logs:
        st.caption("当前还没有可展示的审计日志。")
        return

    for item in logs:
        created_at = html.escape(str(item.get("created_at") or ""))
        actor = html.escape(str(item.get("actor") or ""))
        action_type = html.escape(str(item.get("action_type") or ""))
        target_type = html.escape(str(item.get("target_type") or ""))
        target_id = html.escape(str(item.get("target_id") or ""))
        st.markdown(
            f"""
            <div class="panel-card" style="padding:0.8rem 0.9rem; margin-bottom:0.65rem;">
              <div style="font-size:0.8rem; color:rgba(20,33,61,0.64); margin-bottom:0.3rem;">
                {created_at}
              </div>
              <div style="font-weight:700; color:#14213d; margin-bottom:0.25rem;">
                {action_type} · {actor}
              </div>
              <div style="font-size:0.88rem; color:#14213d;">
                {target_type} · {target_id}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"查看日志载荷：{item.get('id')}"):
            st.code(json.dumps(item.get("payload") or {}, ensure_ascii=False, indent=2), language="json")



def _render_trace_explorer(client: PolicyAPIClient) -> None:
    """渲染追溯区，支持按 request_id 或 ticket_id 回放链路。"""
    st.subheader("追溯 / 审计回放")
    st.caption("输入 request_id 或 ticket_id 后，可回放问答记录、审计动作序列和关联工单。")

    trace_left, trace_right = st.columns(2)
    with trace_left:
        request_id = st.text_input("request_id", key="trace_request_id").strip()
    with trace_right:
        ticket_id = st.text_input("ticket_id", key="trace_ticket_id").strip()

    if st.button("查询追溯链路", use_container_width=True, key="load_trace_bundle"):
        _load_trace_bundle(client, request_id, ticket_id)

    trace_ticket = st.session_state.get("trace_ticket_detail")
    if isinstance(trace_ticket, dict):
        st.markdown("#### 关联工单")
        _render_ticket_detail_card(trace_ticket)
        ticket_context = trace_ticket.get("context") or {}
        if isinstance(ticket_context, dict) and ticket_context.get("kb_request_id"):
            st.caption(f"该工单关联的 kb_request_id：{ticket_context.get('kb_request_id')}")

    trace_kb = st.session_state.get("trace_kb_detail")
    if isinstance(trace_kb, dict):
        st.markdown("#### 问答回放")
        st.markdown(
            f"**问题**：{html.escape(str(trace_kb.get('question') or ''))}",
            unsafe_allow_html=True,
        )
        _render_answer_block(str(trace_kb.get("answer") or ""))
        _render_citations(trace_kb.get("citations") or [])
        _render_hits(trace_kb.get("retrieve_topk") or [])
        with st.expander("查看问答记录详情"):
            st.code(json.dumps(trace_kb, ensure_ascii=False, indent=2), language="json")
    else:
        st.caption("查询到 request_id 后，这里会显示当时的问答记录。")

    _render_audit_timeline(st.session_state.get("trace_audit_logs") or [])



def main() -> None:
    """渲染网页入口并驱动 UI -> API 的最小可用闭环。"""
    load_dotenv()

    st.set_page_config(
        page_title="政策问答、工单与审计演示",
        layout="wide",
    )
    _ensure_state()
    _inject_styles()
    _render_header()

    base_url, user_name, department, client = _render_sidebar()

    left_col, right_col = st.columns([1.55, 1.05], gap="large")

    with left_col:
        _render_error_card()

        example_text = _render_example_runner()
        if example_text:
            st.session_state["agent_input"] = example_text
            st.session_state["ask_input"] = example_text

        st.subheader("一句话入口（默认走 /agent）")
        agent_text = st.text_area(
            "输入一句话描述",
            height=130,
            placeholder="例如：我宿舍网络连不上，帮我提交报修工单。地点金明校区，手机号138xxxx。",
            key="agent_input",
        )

        agent_left, agent_right = st.columns(2)
        with agent_left:
            run_agent = st.button("调用 /agent", type="primary", use_container_width=True)
        with agent_right:
            run_ask = st.button("仅调用 /ask", use_container_width=True)

        if run_agent:
            text = agent_text.strip()
            if not text:
                st.warning("请先输入一句话描述。")
            else:
                with st.spinner("正在调用 /agent ..."):
                    t0 = time.time()
                    _handle_agent_submit(client, text, user_name, department)
                    t1 = time.time()
                st.caption(f"本次前端请求耗时：{t1 - t0:.2f}s · base_url={base_url}")

        if run_ask:
            question = str(st.session_state.get("ask_input") or agent_text or "").strip()
            if not question:
                st.warning("请先输入问题。")
            else:
                with st.spinner("正在调用 /ask ..."):
                    t0 = time.time()
                    _handle_ask_submit(client, question, user_name, department)
                    t1 = time.time()
                st.caption(f"本次前端请求耗时：{t1 - t0:.2f}s · base_url={base_url}")

        ask_response = st.session_state.get("last_ask")
        agent_response = st.session_state.get("last_agent")

        if isinstance(ask_response, dict):
            st.subheader("问答结果（/ask）")
            _render_kb_response(ask_response)
        elif isinstance(agent_response, dict):
            st.subheader("一句话结果（/agent）")
            _render_agent_response(agent_response)
        else:
            st.info("点击“调用 /agent”或“仅调用 /ask”后，这里会展示真实后端返回的结果。")

        _render_draft_continue_form(client, user_name, department)

        st.divider()
        _render_manual_ticket_form(client, user_name, department)

    with right_col:
        _render_ticket_manager(client, user_name)
        st.divider()
        _render_trace_explorer(client)


if __name__ == "__main__":
    main()
