"""
L3/L4 Web 演示程序：提供“UI 通过 HTTP 调用后端 API”的最小可用闭环。

一、程序目标
1. 让 Streamlit 页面不再直接调用 Python 函数，而是真正走 L2 HTTP API。
2. 默认入口走 `/agent`，展示“一句话 -> 路由 -> 问答或建单”的完整链路。
3. 页面默认自然语言入口统一走 `/agent`；只保留手动建单表单直调 `/tickets`。
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
       - 经 `/agent` 路由的“仅问答”入口
       - `/tickets` 手动建单入口
   2.7 将最近一次 API 响应写入 `session_state`
   2.8 渲染问答结果、工单结果、抽取结果、Trace 信息
   2.9 渲染工单管理区：列表、详情、状态更新
   2.10 渲染追溯区：查询 `kb_queries`、`audit_logs` 与关联工单
   2.11 若存在活跃草稿，则渲染“继续完成工单”表单

三、输入输出数据格式
1. UI 输入：
   - `text`: 一句话自然语言输入，走 `/agent`
   - `question`: 问答输入，仍走 `/agent`
   - `ticket form`: 手动建单字段，走 `/tickets`
2. API 输出：
   - `/agent`: `AgentResponse`
   - `/tickets`: `TicketResponse` / `TicketDetailResponse`
3. 页面状态：
   - 把最近一次 API 调用结果放到 `session_state`，便于刷新后继续查看

四、程序可以理解成的伪代码
1. 让用户先配置 API 地址和默认身份
2. 创建 API 客户端
3. 用户点“一句话入口”就调 `/agent`
4. 用户点“仅问答”仍走 `/agent`，由后端路由成问答
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
import os
import sys
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
        "label": "一句话建单",
        "text": "我的办公电脑无法连接公司内网，帮我提交 IT 工单。地点北京总部 12 层工位 A1208，联系方式 13812345678。",
    },
    {
        "label": "仅问答",
        "text": "差旅报销流程是什么？",
    },
    {
        "label": "需补信息",
        "text": "帮我提交网络故障工单，我的办公电脑连不上公司内网。",
    },
]

TICKET_STATUS_OPTIONS = ["open", "in_progress", "resolved", "closed", "cancelled"]
TICKET_CATEGORY_OPTIONS = ["network", "account", "hardware", "permission", "other"]
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
        /* 状态/路由/优先级徽章：用底色 + 文字色编码语义，提升可扫描性。 */
        .badge {
          display: inline-block;
          padding: 0.2rem 0.6rem;
          border-radius: 999px;
          font-size: 0.74rem;
          font-weight: 700;
          letter-spacing: 0.02em;
          line-height: 1.5;
          white-space: nowrap;
        }
        .badge-green  { background: #d9f2e3; color: #1d7a48; }
        .badge-blue   { background: #dbeafe; color: #1d4ed8; }
        .badge-amber  { background: #fdecc8; color: #9a6a00; }
        .badge-red    { background: #fde2e2; color: #b42318; }
        .badge-gray   { background: #e7e5df; color: #57534e; }
        .badge-teal   { background: var(--accent-soft); color: var(--accent); }
        .hero-subtitle code,
        .hero-subtitle code * {
          background: var(--accent-soft);
          color: var(--accent);
          padding: 0.05rem 0.35rem;
          border-radius: 6px;
          font-size: 0.86rem;
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



TICKET_STATUS_BADGE = {
    "open": ("blue", "open"),
    "in_progress": ("amber", "in progress"),
    "resolved": ("green", "resolved"),
    "closed": ("gray", "closed"),
    "cancelled": ("gray", "cancelled"),
}

ROUTE_BADGE = {
    "CREATE_TICKET": "green",
    "ASK": "teal",
    "NEED_MORE_INFO": "amber",
    "DRAFT_EXPIRED": "red",
    "DRAFT_NOT_FOUND": "red",
    "LOOKUP_TICKET": "blue",
    "ADD_TICKET_COMMENT": "blue",
    "ESCALATE_TICKET": "amber",
    "CANCEL_TICKET": "gray",
}

PRIORITY_BADGE = {"P0": "red", "P1": "amber", "P2": "blue", "P3": "gray"}


def _badge_html(text: Any, variant: str = "teal") -> str:
    """返回一个彩色徽章的 HTML 片段（不直接渲染，便于内联拼接）。"""
    safe = html.escape(str(text if text not in (None, "") else "—"))
    return f'<span class="badge badge-{variant}">{safe}</span>'


def _status_badge_html(status: Any) -> str:
    """工单状态徽章：按语义着色。"""
    variant, label = TICKET_STATUS_BADGE.get(str(status or ""), ("gray", str(status or "—")))
    return _badge_html(label, variant)


def _route_badge_html(route: Any) -> str:
    """Agent route 徽章。"""
    return _badge_html(route, ROUTE_BADGE.get(str(route or ""), "gray"))


def _priority_badge_html(priority: Any) -> str:
    """工单优先级徽章。"""
    return _badge_html(priority, PRIORITY_BADGE.get(str(priority or ""), "gray"))


def _render_panel_card(meta_html: str, body_html: str, *, body_line_height: str = "1.7") -> None:
    """渲染统一的列表条目卡片（citations / hits / audit 共用），避免重复内联 HTML。"""
    st.markdown(
        f"""
        <div class="panel-card" style="padding:0.85rem 0.95rem; margin-bottom:0.7rem;">
          <div style="font-size:0.82rem; color:rgba(20,33,61,0.64); margin-bottom:0.35rem;">
            {meta_html}
          </div>
          <div style="line-height:{body_line_height}; color:#14213d;">{body_html}</div>
        </div>
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
            当前页面已切换为通过 HTTP 调用 L2 API。默认自然语言入口统一走 <code>/agent</code>，
            能展示“一句话问答 / 建单 / 补信息 / 工单操作”的真实后端闭环；仅保留手动建单表单
            直调 <code>/tickets</code>，便于对比受控编排与原始接口。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def _ensure_state() -> None:
    """初始化页面交互需要的 session_state 键。"""
    env_base_url = str(os.getenv("POLICY_API_BASE_URL") or "").strip() or "http://localhost:8080"
    defaults = {
        "api_base_url": env_base_url,
        "ui_api_key": "",
        "ui_department": "IT",
        "active_page": "Agent 主页",
        "auth_access_token": "",
        "auth_user_profile": None,
        "auth_login_identifier": "",
        "auth_login_password": "",
        "auth_register_username": "",
        "auth_register_password": "",
        "auth_register_email": "",
        "auth_register_phone": "",
        "clear_auth_login_password_next": False,
        "clear_auth_register_password_next": False,
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
        "trace_ticket_id_input": "",
        "trace_kb_detail": None,
        "trace_audit_logs": [],
        "trace_ticket_detail": None,
        "ops_metrics_hours": 24,
        "last_ops_metrics": None,
        "active_draft_id": "",
        "active_draft_missing_fields": [],
        "draft_followup_location": "",
        "draft_followup_contact": "",
        "draft_followup_note": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 避免在 widget 已创建后直接改同 key：改为下一轮渲染前统一清空密码。
    if st.session_state.get("clear_auth_login_password_next"):
        st.session_state["auth_login_password"] = ""
        st.session_state["clear_auth_login_password_next"] = False
    if st.session_state.get("clear_auth_register_password_next"):
        st.session_state["auth_register_password"] = ""
        st.session_state["clear_auth_register_password_next"] = False

    # 若历史会话仍是 localhost 且环境变量已给出容器内地址，则自动矫正一次。
    current_base_url = str(st.session_state.get("api_base_url") or "").strip()
    if current_base_url == "http://localhost:8080" and env_base_url != "http://localhost:8080":
        st.session_state["api_base_url"] = env_base_url



def _get_auth_user_profile() -> dict[str, Any] | None:
    """返回当前登录用户；若 token 或 profile 缺失则视为未登录。"""
    token = str(st.session_state.get("auth_access_token") or "").strip()
    profile = st.session_state.get("auth_user_profile")
    if token and isinstance(profile, dict):
        return profile
    return None


def _build_client() -> PolicyAPIClient:
    """按当前会话状态创建 API 客户端。"""
    base_url = str(st.session_state.get("api_base_url") or "http://localhost:8080").strip()
    api_key = str(st.session_state.get("ui_api_key") or "").strip()
    access_token = str(st.session_state.get("auth_access_token") or "").strip()
    return PolicyAPIClient(
        base_url=base_url,
        api_key=api_key or None,
        access_token=access_token or None,
    )


def _logout_user() -> None:
    """清理登录态并回到登录页。"""
    st.session_state["auth_access_token"] = ""
    st.session_state["auth_user_profile"] = None
    st.session_state["auth_login_password"] = ""
    st.session_state["auth_register_password"] = ""
    st.session_state["active_draft_id"] = ""
    st.session_state["active_draft_missing_fields"] = []
    st.session_state["active_page"] = "Agent 主页"
    _clear_error()
    st.rerun()


def _render_auth_page(client: PolicyAPIClient) -> None:
    """渲染登录/注册页；登录成功后进入 Agent 主页。"""
    st.subheader("登录 / 注册")
    st.caption("请先登录后再进入 Agent、手动问答、工单管理和审计页面。")
    _render_error_card()

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form"):
            identifier = st.text_input("用户名 / 邮箱 / 手机号", key="auth_login_identifier").strip()
            password = st.text_input("密码", type="password", key="auth_login_password")
            submitted = st.form_submit_button("登录", use_container_width=True)

        if submitted:
            if not identifier or not password:
                st.warning("请输入登录标识和密码。")
            else:
                try:
                    auth_result = client.login(identifier=identifier, password=password)
                except APIClientError as exc:
                    _set_error("调用 /auth/login", exc)
                else:
                    token = str(auth_result.get("access_token") or "").strip()
                    profile = auth_result.get("user")
                    if not token or not isinstance(profile, dict):
                        st.error("登录响应缺少 access_token 或 user 字段。")
                    else:
                        st.session_state["auth_access_token"] = token
                        st.session_state["auth_user_profile"] = profile
                        st.session_state["clear_auth_login_password_next"] = True
                        st.session_state["active_page"] = "Agent 主页"
                        client.set_access_token(token)
                        _clear_error()
                        st.toast("登录成功，正在跳转主页。", icon="✅")
                        st.rerun()

    with register_tab:
        with st.form("register_form"):
            register_username = st.text_input("用户名", key="auth_register_username").strip()
            register_password = st.text_input(
                "密码（至少 8 位）",
                type="password",
                key="auth_register_password",
            )
            register_email = st.text_input("邮箱（可选）", key="auth_register_email").strip()
            register_phone = st.text_input("手机号（可选）", key="auth_register_phone").strip()
            submitted = st.form_submit_button("注册并登录", use_container_width=True)

        if submitted:
            if not register_username or not register_password:
                st.warning("请至少填写用户名和密码。")
            else:
                try:
                    auth_result = client.register(
                        username=register_username,
                        password=register_password,
                        email=register_email or None,
                        phone=register_phone or None,
                    )
                except APIClientError as exc:
                    _set_error("调用 /auth/register", exc)
                else:
                    token = str(auth_result.get("access_token") or "").strip()
                    profile = auth_result.get("user")
                    if not token or not isinstance(profile, dict):
                        st.error("注册响应缺少 access_token 或 user 字段。")
                    else:
                        st.session_state["auth_access_token"] = token
                        st.session_state["auth_user_profile"] = profile
                        st.session_state["clear_auth_register_password_next"] = True
                        st.session_state["active_page"] = "Agent 主页"
                        client.set_access_token(token)
                        _clear_error()
                        st.toast("注册成功，正在跳转主页。", icon="✅")
                        st.rerun()


def _render_navigation_sidebar(client: PolicyAPIClient, auth_profile: dict[str, Any]) -> str:
    """渲染登录后侧栏导航。"""
    with st.sidebar:
        st.header("导航")
        nav_icons = {
            "Agent 主页": "🤖 Agent 主页",
            "手动问答": "💬 手动问答",
            "手动建单": "📝 手动建单",
            "工单管理": "📋 工单管理",
            "审计追溯": "🔍 审计追溯",
            "运行监控": "📈 运行监控",
        }
        selected_page = st.radio(
            "选择页面",
            options=["Agent 主页", "手动问答", "手动建单", "工单管理", "审计追溯", "运行监控"],
            format_func=lambda value: nav_icons.get(value, value),
            key="active_page",
        )

        st.divider()
        role = str(auth_profile.get("role") or "user")
        username = str(auth_profile.get("username") or "")
        st.caption(f"当前用户：{username} ({role})")

        if st.button("刷新身份", use_container_width=True, key="refresh_auth_me"):
            try:
                profile = client.me()
            except APIClientError as exc:
                _set_error("调用 /auth/me", exc)
            else:
                st.session_state["auth_user_profile"] = profile
                _clear_error()
                st.toast("身份信息已刷新。", icon="🔄")

        if st.button("退出登录", use_container_width=True, key="logout_user"):
            _logout_user()

    return selected_page



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



def _render_citations(citations: list[dict[str, Any]], *, show_header: bool = True) -> None:
    """渲染 citations 列表。"""
    if show_header:
        st.subheader("引用")
    if not citations:
        st.info("当前没有可展示的引用。若答案为拒答，这是正常情况。")
        return

    for index, item in enumerate(citations, start=1):
        doc_id = html.escape(str(item.get("doc_id") or ""))
        page = item.get("page")
        snippet = html.escape(str(item.get("snippet", "") or ""))
        _render_panel_card(
            f"引用 {index} · <strong>{doc_id}</strong> · 第 {page} 页",
            snippet,
        )



def _render_hits(hits: list[dict[str, Any]], *, show_header: bool = True) -> None:
    """渲染 top-k 命中证据。"""
    if show_header:
        st.subheader("命中证据")
    if not hits:
        st.info("当前响应里没有返回检索证据。")
        return

    for index, hit in enumerate(hits, start=1):
        score = float(hit.get("score", 0.0) or 0.0)
        doc_id = html.escape(str(hit.get("doc_id") or ""))
        page = hit.get("page")
        snippet = html.escape(str(hit.get("snippet", "") or ""))
        _render_panel_card(
            f"Top {index} · {_badge_html(f'score {score:.3f}', 'teal')} "
            f"· <strong>{doc_id}</strong> · 第 {page} 页",
            snippet,
            body_line_height="1.65",
        )



def _render_kb_response(kb_response: dict[str, Any]) -> None:
    """渲染 `/ask` 或 `/agent.kb` 返回的问答结果。"""
    if not kb_response:
        return

    answer_text = str(kb_response.get("answer") or "")
    citations = kb_response.get("citations", []) or []
    meta = kb_response.get("meta", {}) or {}
    request_id = str(kb_response.get("request_id") or "")
    retrieve_hits = meta.get("retrieve_topk", []) or []

    _render_answer_block(answer_text)

    metric_left, metric_right = st.columns(2)
    metric_left.metric("引用条数", len(citations))
    metric_right.metric("命中证据", len(retrieve_hits))

    citations_tab, hits_tab, trace_tab = st.tabs(["引用", "命中证据", "Trace / Debug"])
    with citations_tab:
        _render_citations(citations, show_header=False)
    with hits_tab:
        _render_hits(retrieve_hits, show_header=False)
    with trace_tab:
        if request_id:
            st.code(
                json.dumps(
                    {"request_id": request_id, "meta": meta},
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
            )
        else:
            st.caption("本次响应未携带 request_id，无 Trace 信息。")



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
          <div style="margin-top:0.15rem;">{_route_badge_html(route)}</div>
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
              <div style="margin-top:0.45rem;">状态：{_status_badge_html(ticket.get('status'))}</div>
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
    """执行手动问答入口，直接调用 `/ask`。"""
    _clear_error()
    try:
        result = client.ask(question=question, user=user_name, department=department)
    except APIClientError as exc:
        _set_error("调用 /ask（手动问答）", exc)
        return

    st.session_state["last_agent"] = None
    st.session_state["last_ask"] = result
    if isinstance(result, dict) and result.get("request_id"):
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



def _render_draft_continue_form(
    client: PolicyAPIClient,
    user_name: str,
    department: str,
    is_authenticated: bool,
) -> None:
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
        submitted = st.form_submit_button(
            "继续提交草稿",
            use_container_width=True,
            disabled=not is_authenticated,
        )

    if submitted:
        if not is_authenticated:
            st.warning("请先登录后再继续提交草稿。")
            return
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



def _render_manual_ticket_form(
    client: PolicyAPIClient,
    user_name: str,
    department: str,
    is_authenticated: bool,
) -> None:
    """渲染手动建单表单。"""
    st.subheader("手动建单（直接调用 /tickets）")
    with st.form("manual_ticket_form"):
        title = st.text_input("标题", value="办公电脑无法连接公司内网")
        description = st.text_area(
            "描述",
            value="用户描述：办公电脑无法连接公司内网，需要 IT 服务台排查网络或终端配置。",
            height=110,
        )
        form_left, form_right = st.columns(2)
        with form_left:
            contact = st.text_input("联系方式", value="13812345678")
            category = st.selectbox("类别", options=TICKET_CATEGORY_OPTIONS, index=0)
        with form_right:
            location = st.text_input("地点（会写入 context.location）", value="北京总部 12 层工位 A1208")
            priority = st.selectbox("优先级", options=TICKET_PRIORITY_OPTIONS, index=1)

        submitted = st.form_submit_button(
            "创建工单",
            use_container_width=True,
            disabled=not is_authenticated,
        )

    if submitted:
        if not is_authenticated:
            st.warning("请先登录后再创建工单。")
            return
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

    badges = [f"状态 {_status_badge_html(ticket.get('status'))}"]
    if ticket.get("priority"):
        badges.append(f"优先级 {_priority_badge_html(ticket.get('priority'))}")
    if ticket.get("category"):
        badges.append(f"类别 {_badge_html(ticket.get('category'), 'gray')}")
    badges_html = "&nbsp;&nbsp;".join(badges)

    st.markdown(
        f"""
        <div class="ticket-card">
          <div class="answer-label" style="color:#9a6a00;">TICKET DETAIL</div>
          <div class="answer-text">{html.escape(str(ticket.get('ticket_id') or ''))}</div>
          <div style="margin-top:0.5rem;">{badges_html}</div>
          <div style="margin-top:0.4rem; color:#7a5d14; line-height:1.7;">
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



def _render_ticket_manager(client: PolicyAPIClient, user_name: str, is_authenticated: bool) -> None:
    """渲染工单列表、详情和状态更新。"""
    st.subheader("工单管理")

    filter_options = ["全部", *TICKET_STATUS_OPTIONS]
    selected_filter = st.selectbox("状态过滤", options=filter_options, index=0)
    status_filter = None if selected_filter == "全部" else selected_filter

    refresh_list = st.button("刷新工单列表", use_container_width=True, key="refresh_ticket_list")
    if refresh_list:
        with st.spinner("正在刷新工单列表 ..."):
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
                with st.spinner("正在查询工单详情 ..."):
                    _load_ticket_detail(client, manual_ticket_id)

        ticket_detail = st.session_state.get("selected_ticket_detail")
        if isinstance(ticket_detail, dict):
            _render_ticket_detail_card(ticket_detail)
        else:
            st.caption("当前还没有选中的工单详情。")

    with detail_right:
        new_status = st.selectbox("更新状态为", options=TICKET_STATUS_OPTIONS, index=0)
        if st.button(
            "提交状态更新",
            use_container_width=True,
            key="update_ticket_status",
            disabled=not is_authenticated,
        ):
            if not is_authenticated:
                st.warning("请先登录后再更新工单状态。")
                return
            if not manual_ticket_id:
                st.warning("请先选择或输入工单号。")
            else:
                with st.spinner("正在更新工单状态 ..."):
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
        _render_panel_card(
            created_at,
            f'<div style="font-weight:700; margin-bottom:0.25rem;">{action_type} · {actor}</div>'
            f'<div style="font-size:0.88rem;">{target_type} · {target_id}</div>',
        )

        with st.expander(f"查看日志载荷：{item.get('id')}"):
            st.code(json.dumps(item.get("payload") or {}, ensure_ascii=False, indent=2), language="json")



def _render_trace_explorer(client: PolicyAPIClient) -> None:
    """渲染追溯区：用户侧只输入工单号，内部自动关联 request_id。"""
    st.subheader("追溯 / 审计回放")
    st.caption("输入工单号即可回放关联工单、问答记录和审计动作序列。")

    ticket_id = st.text_input(
        "工单号",
        value=str(st.session_state.get("trace_ticket_id") or ""),
        key="trace_ticket_id_input",
        placeholder="例如：TCK-2026-AB12",
    ).strip()

    if st.button("查询追溯链路", use_container_width=True, key="load_trace_bundle"):
        if not ticket_id:
            st.warning("请先输入工单号。")
        else:
            with st.spinner("正在拉取追溯链路 ..."):
                _load_trace_bundle(client, "", ticket_id)

    resolved_ticket_id = str(st.session_state.get("trace_ticket_id") or "")
    if resolved_ticket_id:
        st.caption(f"当前回放工单：{resolved_ticket_id}")

    trace_ticket = st.session_state.get("trace_ticket_detail")
    if isinstance(trace_ticket, dict):
        st.markdown("#### 关联工单")
        _render_ticket_detail_card(trace_ticket)
        ticket_context = trace_ticket.get("context") or {}
        if isinstance(ticket_context, dict) and ticket_context.get("kb_request_id"):
            with st.expander("查看内部链路 ID"):
                st.code(str(ticket_context.get("kb_request_id")), language="text")

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


def _render_agent_home_page(
    client: PolicyAPIClient,
    user_name: str,
    department: str,
    is_authenticated: bool,
) -> None:
    """渲染 Agent 主页。"""
    _render_error_card()
    example_text = _render_example_runner()
    if example_text:
        st.session_state["agent_input"] = example_text

    st.subheader("Agent 主页")
    agent_text = st.text_area(
        "输入一句话描述",
        height=130,
        placeholder="例如：我的办公电脑无法连接公司内网，帮我提交 IT 工单。地点北京总部 12 层工位 A1208，联系方式 138xxxx。",
        key="agent_input",
    )

    run_agent = st.button(
        "调用 /agent",
        type="primary",
        use_container_width=True,
        disabled=not is_authenticated,
        key="run_agent_home",
    )
    if run_agent:
        text = agent_text.strip()
        if not text:
            st.warning("请先输入一句话描述。")
        else:
            with st.spinner("正在调用 /agent ..."):
                _handle_agent_submit(client, text, user_name, department)

    agent_response = st.session_state.get("last_agent")
    if isinstance(agent_response, dict):
        if str(agent_response.get("route") or "") == "ASK":
            st.subheader("问答结果（经 /agent 路由）")
        else:
            st.subheader("一句话结果（/agent）")
        _render_agent_response(agent_response)
    else:
        st.info("提交一句话后，这里会展示 /agent 返回结果。")

    _render_draft_continue_form(client, user_name, department, is_authenticated=is_authenticated)


def _render_manual_qa_page(
    client: PolicyAPIClient,
    user_name: str,
    department: str,
    is_authenticated: bool,
) -> None:
    """渲染手动问答页。"""
    _render_error_card()
    st.subheader("手动问答（直接调用 /ask）")
    question = st.text_area(
        "输入问题",
        key="ask_input",
        height=130,
        placeholder="例如：统一身份认证的登录地址是什么？",
    )

    submitted = st.button(
        "调用 /ask",
        use_container_width=True,
        type="primary",
        disabled=not is_authenticated,
        key="run_manual_qa",
    )
    if submitted:
        normalized_question = question.strip()
        if not normalized_question:
            st.warning("请先输入问题。")
        else:
            with st.spinner("正在调用 /ask ..."):
                _handle_ask_submit(client, normalized_question, user_name, department)

    ask_result = st.session_state.get("last_ask")
    if isinstance(ask_result, dict):
        _render_kb_response(ask_result)
    else:
        st.info("提交问题后，这里会显示手动问答结果。")


def _render_manual_ticket_page(
    client: PolicyAPIClient,
    user_name: str,
    department: str,
    is_authenticated: bool,
) -> None:
    """渲染手动建单页。"""
    _render_error_card()
    _render_manual_ticket_form(client, user_name, department, is_authenticated=is_authenticated)


def _render_ticket_management_page(
    client: PolicyAPIClient,
    user_name: str,
    is_authenticated: bool,
) -> None:
    """渲染工单管理页。"""
    _render_error_card()
    _render_ticket_manager(client, user_name, is_authenticated=is_authenticated)


def _render_audit_page(client: PolicyAPIClient) -> None:
    """渲染审计追溯页。"""
    _render_error_card()
    _render_trace_explorer(client)


def _format_rate(value: Any) -> str:
    """把 0-1 的比例格式化成百分比；异常值按 0% 展示。"""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _format_ms(value: Any) -> str:
    """把毫秒数字格式化成 UI 中易扫读的文本。"""
    try:
        return f"{int(value)} ms"
    except (TypeError, ValueError):
        return "0 ms"


def _format_cost_usd(value: Any) -> str:
    """把美元成本格式化成较短文本；小额成本保留更多小数。"""
    try:
        return f"${float(value):.6f}"
    except (TypeError, ValueError):
        return "$0.000000"


def _dict_to_rows(data: Any, key_label: str, value_label: str) -> list[dict[str, Any]]:
    """把后端返回的分布字典转为 `st.dataframe` 可展示的行列表。"""
    if not isinstance(data, dict):
        return []
    return [
        {key_label: str(key), value_label: value}
        for key, value in sorted(data.items(), key=lambda item: str(item[0]))
    ]


def _render_distribution_table(title: str, data: Any, key_label: str = "名称") -> None:
    """渲染一张轻量分布表；空数据时给出占位提示。"""
    st.markdown(f"**{title}**")
    rows = _dict_to_rows(data, key_label, "数量")
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("当前统计窗口内暂无数据。")


def _render_ops_page(client: PolicyAPIClient) -> None:
    """渲染运行监控页，展示现有 `/ops/metrics` 聚合结果。"""
    _render_error_card()
    st.subheader("运行监控")
    st.caption("演示环境直接展示监控数据；真实上线时可按用户角色只对管理员开放。")

    left, right = st.columns([2, 1])
    with left:
        hours = st.slider(
            "统计窗口（小时）",
            min_value=1,
            max_value=24 * 7,
            value=int(st.session_state.get("ops_metrics_hours") or 24),
            step=1,
            key="ops_metrics_hours",
        )
    with right:
        refresh = st.button("刷新监控", use_container_width=True, type="primary")

    if refresh or not isinstance(st.session_state.get("last_ops_metrics"), dict):
        try:
            st.session_state["last_ops_metrics"] = client.ops_metrics(hours=hours)
        except APIClientError as exc:
            _set_error("调用 /ops/metrics", exc)
            _render_error_card()
            return
        else:
            _clear_error()
            st.toast("运行监控已刷新。", icon="📈")

    metrics = st.session_state.get("last_ops_metrics")
    if not isinstance(metrics, dict):
        st.info("点击刷新后，这里会展示最近一段时间的运行指标。")
        return

    ask = metrics.get("ask")
    tickets = metrics.get("tickets")
    ask_metrics = ask if isinstance(ask, dict) else {}
    ticket_metrics = tickets if isinstance(tickets, dict) else {}

    st.markdown("**ASK 问答链路**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("请求总数", int(ask_metrics.get("total") or 0))
    col2.metric("成功 / 失败", f"{int(ask_metrics.get('success') or 0)} / {int(ask_metrics.get('failure') or 0)}")
    col3.metric("JSON 有效率", _format_rate(ask_metrics.get("valid_json_rate")))
    col4.metric("引用输出率", _format_rate(ask_metrics.get("citation_rate")))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("平均总延迟", _format_ms(ask_metrics.get("avg_total_ms")))
    col6.metric("P95 总延迟", _format_ms(ask_metrics.get("p95_total_ms")))
    col7.metric("Token 总量", int(ask_metrics.get("total_tokens") or 0))
    col8.metric("估算成本", _format_cost_usd(ask_metrics.get("estimated_cost_usd")))

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("平均检索延迟", _format_ms(ask_metrics.get("avg_retrieve_ms")))
    col10.metric("平均回答延迟", _format_ms(ask_metrics.get("avg_answer_ms")))
    col11.metric("JSON 修复率", _format_rate(ask_metrics.get("repair_rate")))
    col12.metric("Fallback 比例", _format_rate(ask_metrics.get("fallback_rate")))

    dist_left, dist_mid, dist_right = st.columns(3)
    with dist_left:
        _render_distribution_table("执行阶段分布", ask_metrics.get("attempt_stages"), "阶段")
    with dist_mid:
        _render_distribution_table("失败原因分布", ask_metrics.get("failure_reasons"), "原因")
    with dist_right:
        _render_distribution_table("模型分布", ask_metrics.get("models"), "模型")

    st.divider()
    st.markdown("**Agent / 工单链路**")
    ticket_state = ticket_metrics.get("ticket_state") if isinstance(ticket_metrics.get("ticket_state"), dict) else {}
    draft_state = ticket_metrics.get("draft_state") if isinstance(ticket_metrics.get("draft_state"), dict) else {}
    confirmation_state = (
        ticket_metrics.get("confirmation_state")
        if isinstance(ticket_metrics.get("confirmation_state"), dict)
        else {}
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("审计事件", int(ticket_metrics.get("total_audit_events") or 0))
    col2.metric("失败/拒绝事件", int(ticket_metrics.get("failure_or_rejected_events") or 0))
    col3.metric("工单总数", int(ticket_state.get("total") or 0))
    col4.metric("草稿 / 确认", f"{int(draft_state.get('total') or 0)} / {int(confirmation_state.get('total') or 0)}")

    ticket_left, ticket_mid, ticket_right = st.columns(3)
    with ticket_left:
        _render_distribution_table("Action 分布", ticket_metrics.get("action_counts"), "Action")
        _render_distribution_table("工单状态", ticket_state.get("status_counts"), "状态")
    with ticket_mid:
        _render_distribution_table("Route 分布", ticket_metrics.get("route_counts"), "Route")
        _render_distribution_table("草稿状态", draft_state.get("status_counts"), "状态")
    with ticket_right:
        _render_distribution_table("拒绝原因", ticket_metrics.get("rejection_reasons"), "原因")
        _render_distribution_table("确认状态", confirmation_state.get("status_counts"), "状态")

    with st.expander("查看原始监控 JSON"):
        st.code(json.dumps(metrics, ensure_ascii=False, indent=2), language="json")


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

    client = _build_client()
    auth_user = _get_auth_user_profile()
    is_authenticated = auth_user is not None

    if not is_authenticated:
        _render_auth_page(client)
        return

    assert isinstance(auth_user, dict)
    selected_page = _render_navigation_sidebar(client, auth_user)

    user_name = "anonymous"
    if isinstance(auth_user, dict):
        user_name = str(auth_user.get("username") or "anonymous")
    department = str(auth_user.get("department") or st.session_state.get("ui_department") or "IT")

    if selected_page == "Agent 主页":
        _render_agent_home_page(client, user_name, department, is_authenticated=is_authenticated)
    elif selected_page == "手动问答":
        _render_manual_qa_page(client, user_name, department, is_authenticated=is_authenticated)
    elif selected_page == "手动建单":
        _render_manual_ticket_page(client, user_name, department, is_authenticated=is_authenticated)
    elif selected_page == "工单管理":
        _render_ticket_management_page(client, user_name, is_authenticated=is_authenticated)
    elif selected_page == "审计追溯":
        _render_audit_page(client)
    else:
        _render_ops_page(client)


if __name__ == "__main__":
    main()
