from __future__ import annotations

from typing import Any

from src.mcp_wrapper.contracts import ToolCallRequest
import src.mcp_wrapper.executor as executor


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def set(self, key: str, value: str, ex: int | None = None, nx: bool = False, xx: bool = False) -> bool:
        if nx and key in self._store:
            return False
        if xx and key not in self._store:
            return False
        self._store[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0


def _request(*, text: str, idempotency_key: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool="create_ticket",
        args={"text": text, "idempotency_key": idempotency_key},
        actor="alice",
        actor_user_id="alice",
        actor_role="user",
        department="IT",
        raw_text=text,
    )


def test_invoke_tool_idempotent_replay_success(monkeypatch) -> None:
    fake_redis = _FakeRedis()
    calls = {"count": 0}

    def _fake_dispatch_tool(_db: Any, _request: ToolCallRequest) -> tuple[dict[str, Any], str]:
        calls["count"] += 1
        return (
            {
                "route": "CREATE_TICKET",
                "message": "ok",
                "ticket": {"ticket_id": "TCK-1"},
            },
            "create_ticket",
        )

    monkeypatch.setattr(executor, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(executor, "dispatch_tool", _fake_dispatch_tool)

    first = executor.invoke_tool(None, _request(text="network down", idempotency_key="idem-1"))
    second = executor.invoke_tool(None, _request(text="network down", idempotency_key="idem-1"))

    assert first.ok is True
    assert second.ok is True
    assert first.payload == second.payload
    assert calls["count"] == 1


def test_invoke_tool_idempotent_conflict(monkeypatch) -> None:
    fake_redis = _FakeRedis()
    calls = {"count": 0}

    def _fake_dispatch_tool(_db: Any, _request: ToolCallRequest) -> tuple[dict[str, Any], str]:
        calls["count"] += 1
        return (
            {
                "route": "CREATE_TICKET",
                "message": "ok",
                "ticket": {"ticket_id": "TCK-1"},
            },
            "create_ticket",
        )

    monkeypatch.setattr(executor, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(executor, "dispatch_tool", _fake_dispatch_tool)

    first = executor.invoke_tool(None, _request(text="network down", idempotency_key="idem-1"))
    second = executor.invoke_tool(None, _request(text="vpn down", idempotency_key="idem-1"))

    assert first.ok is True
    assert second.ok is False
    assert second.error_code == "idempotency_conflict"
    assert calls["count"] == 1


def test_invoke_tool_idempotent_replay_plan_rejected(monkeypatch) -> None:
    fake_redis = _FakeRedis()
    calls = {"count": 0}

    def _fake_dispatch_tool(_db: Any, _request: ToolCallRequest) -> tuple[dict[str, Any], str]:
        calls["count"] += 1
        return (
            {
                "route": "PLAN_REJECTED",
                "message": "confirm_token 不能为空。",
            },
            "confirm_action",
        )

    monkeypatch.setattr(executor, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(executor, "dispatch_tool", _fake_dispatch_tool)

    req = ToolCallRequest(
        tool="confirm_action",
        args={"confirm_token": "", "text": "确认", "idempotency_key": "idem-2"},
        actor="alice",
        actor_user_id="alice",
        actor_role="user",
        department="IT",
        raw_text="确认",
    )
    first = executor.invoke_tool(None, req)
    second = executor.invoke_tool(None, req)

    assert first.ok is False
    assert first.error_code == "plan_rejected"
    assert second.ok is False
    assert second.error_code == "plan_rejected"
    assert first.payload == second.payload
    assert calls["count"] == 1
