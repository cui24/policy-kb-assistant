#!/usr/bin/env python3
"""MCP client smoke test：独立验证 ask_policy / create_ticket / get_ticket_detail。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


REQUIRED_TOOLS = {"ask_policy", "create_ticket", "get_ticket_detail"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test MCP tools via stdio client.")
    parser.add_argument("--actor", default="mcp-smoke-user", help="MCP_ACTOR_USER_ID")
    parser.add_argument("--department", default="IT", help="Default department")
    parser.add_argument("--question", default="宿舍断网怎么报修？", help="Question for ask_policy")
    parser.add_argument(
        "--create-text",
        default="我宿舍网络连不上，帮我提交报修工单。地点金明校区，手机号13812345678。",
        help="Input text for create_ticket",
    )
    parser.add_argument(
        "--ticket-id",
        default="",
        help="Fallback ticket id for get_ticket_detail when create_ticket doesn't return one.",
    )
    parser.add_argument(
        "--python-cmd",
        default=sys.executable,
        help="Python executable used to launch MCP stdio server.",
    )
    parser.add_argument(
        "--embed-model",
        default="",
        help="Optional EMBED_MODEL passed to MCP subprocess.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode for HF/Transformers in MCP subprocess.",
    )
    return parser.parse_args()


def _payload_from_result(result) -> dict[str, Any]:
    if result.structuredContent is not None:
        payload = result.structuredContent
        return payload if isinstance(payload, dict) else {"value": payload}

    text_payload = "".join(
        block.text
        for block in result.content
        if getattr(block, "type", "") == "text"
    )
    if not text_payload:
        return {}
    try:
        parsed = json.loads(text_payload)
    except json.JSONDecodeError:
        return {"raw_text": text_payload}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _extract_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        return dict(data)
    return dict(payload or {})


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _run(args: argparse.Namespace) -> int:
    embed_model = str(args.embed_model or os.getenv("EMBED_MODEL") or "").strip()
    force_offline = bool(args.offline)
    hf_hub_offline = "1" if force_offline else str(os.getenv("HF_HUB_OFFLINE") or "").strip()
    transformers_offline = "1" if force_offline else str(os.getenv("TRANSFORMERS_OFFLINE") or "").strip()

    env = {
        "PYTHONPATH": os.getcwd(),
        "MCP_ACTOR_USER_ID": str(args.actor or "").strip() or "mcp-smoke-user",
        "MCP_DEPARTMENT": str(args.department or "").strip() or "IT",
    }
    if embed_model:
        env["EMBED_MODEL"] = embed_model
    if hf_hub_offline:
        env["HF_HUB_OFFLINE"] = hf_hub_offline
    if transformers_offline:
        env["TRANSFORMERS_OFFLINE"] = transformers_offline

    server_params = StdioServerParameters(
        command=str(args.python_cmd),
        args=["-m", "src.mcp_stdio_server"],
        env=env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tool_names = {tool.name for tool in tools_result.tools}
            print("[TOOLS]", ", ".join(sorted(tool_names)))
            missing = sorted(REQUIRED_TOOLS - tool_names)
            _assert(not missing, f"missing tools: {missing}")

            ask_payload = _payload_from_result(
                await session.call_tool(
                    "ask_policy",
                    {
                        "question": args.question,
                    },
                )
            )
            print("[ASK_POLICY]", json.dumps(ask_payload, ensure_ascii=False))
            _assert(str(ask_payload.get("contract_version") or "") == "v1", "ask_policy contract_version must be v1")
            _assert(bool(ask_payload.get("success")) is True, "ask_policy success must be true")
            ask_data = _extract_data(ask_payload)
            _assert(str(ask_data.get("route") or "").upper() == "ASK", "ask_policy route must be ASK")
            kb = ask_data.get("kb") if isinstance(ask_data.get("kb"), dict) else {}
            _assert(bool(str(kb.get("answer") or "").strip()), "ask_policy answer is empty")

            create_payload = _payload_from_result(
                await session.call_tool(
                    "create_ticket",
                    {
                        "text": args.create_text,
                        "fields": {"department": args.department},
                    },
                )
            )
            print("[CREATE_TICKET]", json.dumps(create_payload, ensure_ascii=False))
            _assert(str(create_payload.get("contract_version") or "") == "v1", "create_ticket contract_version must be v1")
            _assert(bool(create_payload.get("success")) is True, "create_ticket success must be true")
            create_data = _extract_data(create_payload)
            create_route = str(create_data.get("route") or "").upper()
            _assert(
                create_route in {"CREATE_TICKET", "NEED_MORE_INFO"},
                f"unexpected create_ticket route: {create_route}",
            )

            ticket_id = ""
            if isinstance(create_data.get("ticket"), dict):
                ticket_id = str(create_data.get("ticket", {}).get("ticket_id") or "").strip()
            if not ticket_id:
                ticket_id = str(args.ticket_id or "").strip()
            _assert(ticket_id, "no ticket_id from create_ticket; pass --ticket-id to continue get_ticket_detail")

            detail_payload = _payload_from_result(
                await session.call_tool("get_ticket_detail", {"ticket_id": ticket_id})
            )
            print("[GET_TICKET_DETAIL]", json.dumps(detail_payload, ensure_ascii=False))
            _assert(str(detail_payload.get("contract_version") or "") == "v1", "get_ticket_detail contract_version must be v1")
            _assert(bool(detail_payload.get("success")) is True, "get_ticket_detail success must be true")
            detail_data = _extract_data(detail_payload)
            returned_id = str(detail_data.get("ticket_id") or detail_data.get("ticket_detail", {}).get("ticket_id") or "")
            _assert(returned_id == ticket_id, f"ticket id mismatch: expected {ticket_id}, got {returned_id}")

            print("[PASS] MCP client smoke test completed successfully.")
            return 0


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
