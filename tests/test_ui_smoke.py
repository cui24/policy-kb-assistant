"""
Streamlit UI smoke tests：验证页面可渲染、可做最小交互，不依赖真实后端。

一、测试目标
1. 确认 `src/ui/app.py` 能被 Streamlit `AppTest` 正常加载。
2. 确认关键控件仍存在，避免页面结构回归时无人察觉。
3. 确认在空输入下点击 `/agent` 不会触发异常，而是展示前端告警。
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


_APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"


def _render_app() -> AppTest:
    """加载页面并完成首轮渲染。"""
    app = AppTest.from_file(str(_APP_PATH))
    app.run()
    return app


def test_streamlit_app_renders_expected_controls() -> None:
    """页面首屏应可渲染，且包含当前演示依赖的核心控件。"""
    app = _render_app()

    assert len(app.exception) == 0

    button_labels = [button.label for button in app.button]
    assert "调用 /agent" in button_labels
    assert "仅问答（走 /agent）" in button_labels
    assert "创建工单" in button_labels
    assert "检测 API" in button_labels

    text_area_labels = [text_area.label for text_area in app.text_area]
    assert "输入一句话描述" in text_area_labels
    assert "描述" in text_area_labels


def test_streamlit_app_empty_agent_submit_shows_warning() -> None:
    """空输入点击 `/agent` 时，应由前端直接拦截并提示。"""
    app = _render_app()

    next(button for button in app.button if button.label == "调用 /agent").click().run()

    assert len(app.exception) == 0
    warning_messages = [warning.value for warning in app.warning]
    assert "请先输入一句话描述。" in warning_messages
