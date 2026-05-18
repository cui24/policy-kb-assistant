from __future__ import annotations

from src.agent_graph import mcp_client
from src.mcp_wrapper.contracts import ToolCallResult


def test_remote_mcp_circuit_opens_after_failures(monkeypatch) -> None:
    mcp_client.reset_remote_mcp_circuit_breaker()

    async def _always_fail(*, server_url: str, tool: str, args: dict, raw_text: str, timeout_seconds: int) -> ToolCallResult:
        return ToolCallResult(
            ok=False,
            route="PLAN_REJECTED",
            payload={"route": "PLAN_REJECTED", "message": "远程 MCP 调用失败。"},
            error_code="remote_mcp_error",
            message="remote_mcp_error:Exception",
            retryable=True,
        )

    monkeypatch.setattr(mcp_client, "_call_remote_tool_async", _always_fail)

    first = mcp_client.call_remote_tool(
        server_url="http://127.0.0.1:9000/mcp",
        tool="ask_policy",
        args={"query": "x"},
        raw_text="x",
        timeout_seconds=1,
        circuit_breaker_enabled=True,
        circuit_fail_threshold=2,
        circuit_open_seconds=30,
    )
    second = mcp_client.call_remote_tool(
        server_url="http://127.0.0.1:9000/mcp",
        tool="ask_policy",
        args={"query": "x"},
        raw_text="x",
        timeout_seconds=1,
        circuit_breaker_enabled=True,
        circuit_fail_threshold=2,
        circuit_open_seconds=30,
    )
    third = mcp_client.call_remote_tool(
        server_url="http://127.0.0.1:9000/mcp",
        tool="ask_policy",
        args={"query": "x"},
        raw_text="x",
        timeout_seconds=1,
        circuit_breaker_enabled=True,
        circuit_fail_threshold=2,
        circuit_open_seconds=30,
    )

    assert first.error_code == "remote_mcp_error"
    assert second.error_code == "remote_mcp_error"
    assert third.error_code == "remote_mcp_circuit_open"
    assert third.retryable is True


def test_remote_mcp_circuit_resets_after_success(monkeypatch) -> None:
    mcp_client.reset_remote_mcp_circuit_breaker()

    # 先制造一次失败，累计失败计数。
    async def _fail(**kwargs) -> ToolCallResult:
        return ToolCallResult(
            ok=False,
            route="PLAN_REJECTED",
            payload={"route": "PLAN_REJECTED", "message": "远程 MCP 调用失败。"},
            error_code="remote_mcp_error",
            message="remote_mcp_error:Exception",
            retryable=True,
        )

    monkeypatch.setattr(
        mcp_client,
        "_call_remote_tool_async",
        _fail,
    )
    _ = mcp_client.call_remote_tool(
        server_url="http://127.0.0.1:9000/mcp",
        tool="ask_policy",
        args={"query": "x"},
        raw_text="x",
        timeout_seconds=1,
        circuit_breaker_enabled=True,
        circuit_fail_threshold=3,
        circuit_open_seconds=30,
    )

    # 再改为成功，确认熔断计数会被清零。
    async def _success(**kwargs) -> ToolCallResult:
        return ToolCallResult(
            ok=True,
            route="ASK",
            payload={"route": "ASK", "success": True, "data": {"route": "ASK"}},
            error_code=None,
            message=None,
            retryable=False,
        )

    monkeypatch.setattr(
        mcp_client,
        "_call_remote_tool_async",
        _success,
    )
    success = mcp_client.call_remote_tool(
        server_url="http://127.0.0.1:9000/mcp",
        tool="ask_policy",
        args={"query": "x"},
        raw_text="x",
        timeout_seconds=1,
        circuit_breaker_enabled=True,
        circuit_fail_threshold=3,
        circuit_open_seconds=30,
    )

    assert success.ok is True
    opened, _ = mcp_client._is_circuit_open()
    assert opened is False
