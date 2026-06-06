"""
Streamlit UI smoke tests：验证页面可渲染、可做最小交互，不依赖真实后端。

一、测试目标
1. 确认 `src/ui/app.py` 能被 Streamlit `AppTest` 正常加载。
2. 确认关键控件仍存在，避免页面结构回归时无人察觉。
3. 确认聊天页默认输入、模式分流等关键交互不会回归。
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.ui import app as ui_app


_APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
_DEFAULT_CHAT_INPUT_BY_MODE = {
    "自动": "新员工申请开通 ERP 系统账户，从审批通过到技术支持开通权限一般需要多久？",
    "问答": "新员工申请开通 ERP 系统账户，从审批通过到技术支持开通权限一般需要多久？",
    "工单": "我的办公电脑无法连接公司内网，帮我提交 IT 工单。地点北京总部 12 层工位 A1208，联系方式 13812345678。",
}


def _render_app(
    authenticated: bool = True,
    active_page: str | None = None,
    chat_mode: str | None = None,
    chat_input_text: str | None = None,
) -> AppTest:
    """加载页面并完成首轮渲染；默认注入登录态以测试主功能页。"""
    app = AppTest.from_file(str(_APP_PATH))
    if authenticated:
        app.session_state["auth_access_token"] = "ui-smoke-token"
        app.session_state["auth_user_profile"] = {
            "username": "ui-smoke-user",
            "role": "user",
            "department": "IT",
        }
    if active_page:
        app.session_state["active_page"] = active_page
    if chat_mode:
        app.session_state["chat_mode"] = chat_mode
    if chat_input_text is not None:
        app.session_state["chat_input_text"] = chat_input_text
    app.run()
    return app


def test_streamlit_app_renders_expected_controls() -> None:
    """页面首屏应可渲染，且包含当前演示依赖的核心控件。"""
    app = _render_app()

    assert len(app.exception) == 0

    button_labels = [button.label for button in app.button]
    assert "清空对话" in button_labels

    subheaders = [subheader.value for subheader in app.subheader]
    assert "聊天" in subheaders

    text_area_values = {text_area.label: text_area.value for text_area in app.text_area}
    assert text_area_values["输入问题或工单需求"] == _DEFAULT_CHAT_INPUT_BY_MODE["自动"]

    mode_radio = next(radio for radio in app.radio if radio.label == "模式")
    assert mode_radio.value == "自动"

    radio_options = [
        option
        for radio in app.radio
        for option in getattr(radio, "options", [])
    ]
    assert any("聊天" in option for option in radio_options)
    assert any("运行监控" in option for option in radio_options)
    assert not any("Agent主页" in option for option in radio_options)


def test_chat_page_uses_mode_specific_default_input() -> None:
    """聊天页应按模式展示不同默认输入，并清理旧默认残留。"""
    for mode, expected_input in _DEFAULT_CHAT_INPUT_BY_MODE.items():
        app = _render_app(chat_mode=mode)

        assert len(app.exception) == 0
        text_area_values = {text_area.label: text_area.value for text_area in app.text_area}
        assert text_area_values["输入问题或工单需求"] == expected_input

    app = _render_app(
        chat_mode="工单",
        chat_input_text="员工可以通过哪些渠道进行投诉举报？",
    )
    text_area_values = {text_area.label: text_area.value for text_area in app.text_area}
    assert text_area_values["输入问题或工单需求"] == _DEFAULT_CHAT_INPUT_BY_MODE["工单"]


def test_chat_mode_execution_path_separates_streaming_ask_and_direct_agent() -> None:
    """问答模式走 ASK stream，工单模式走 Agent；自动模式按文本分流。"""
    qa_text = "新员工申请开通 ERP 系统账户，从审批通过到技术支持开通权限一般需要多久？"
    ticket_text = "我的办公电脑无法连接公司内网，帮我提交 IT 工单。地点北京总部 12 层工位 A1208，联系方式 13812345678。"

    assert ui_app._chat_execution_path("问答", ticket_text) == "ask_stream"
    assert ui_app._chat_execution_path("工单", qa_text) == "agent"
    assert ui_app._chat_execution_path("自动", qa_text) == "ask_stream"
    assert ui_app._chat_execution_path("自动", ticket_text) == "agent"
