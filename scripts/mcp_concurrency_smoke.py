#!/usr/bin/env python3
"""MCP 工具层并发测试脚本。

当前目标：
1. 通过标准 MCP Streamable HTTP client 调用 MCP server。
2. 测试 MCP list_tools、get_ticket_detail、create_ticket 三类场景。
3. 和 `scripts/concurrency_smoke.py` 的 FastAPI 直连压测结果做对照。

注意：
- 这个脚本需要安装 `mcp` Python 包。
- 当前宿主机 conda 环境可能没有安装 `mcp`，Docker 的 api/mcp 镜像里通常有。
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_OUTPUT_PATH = "outputs/mcp_concurrency_setup.json"


def _env(name: str, default: str) -> str:
    """读取环境变量，空值时返回默认值。

    实现流程：
    1. 从系统环境变量读取指定名称。
    2. 去掉首尾空白。
    3. 如果变量不存在或为空字符串，返回默认值。
    """
    value = str(os.getenv(name) or "").strip()
    return value or default


def _parse_args() -> argparse.Namespace:
    """解析 MCP 并发测试命令行参数。

    实现流程：
    1. 读取 MCP server URL，默认 `http://localhost:9000/mcp`。
    2. 读取 API URL，用于准备 MCP lookup 所需的种子工单。
    3. 读取 MCP actor 对应的测试账号，默认和 compose 里的 `demo_user` 对齐。
    4. 读取测试 case、请求总数、并发数、超时时间等压测参数。
    """
    parser = argparse.ArgumentParser(description="MCP 工具层并发测试脚本。")
    parser.add_argument("--mcp-url", default=_env("LOADTEST_MCP_URL", "http://localhost:9000/mcp"))
    parser.add_argument("--api-base-url", default=_env("LOADTEST_API_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--actor", default=_env("LOADTEST_MCP_ACTOR", "demo_user"))
    parser.add_argument("--password", default=_env("LOADTEST_MCP_PASSWORD", "loadtest_password_123"))
    parser.add_argument("--email", default=_env("LOADTEST_MCP_EMAIL", "demo_user@example.com"))
    parser.add_argument("--department", default=_env("LOADTEST_DEPARTMENT", "IT"))
    parser.add_argument("--timeout", type=float, default=float(_env("LOADTEST_MCP_TIMEOUT_SECONDS", "15")))
    parser.add_argument("--output", default=_env("LOADTEST_MCP_SETUP_OUTPUT", DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--case",
        choices=["health", "lookup", "create"],
        default="health",
        help="MCP 测试场景：health=list_tools，lookup=get_ticket_detail，create=create_ticket。",
    )
    parser.add_argument("--requests", type=int, default=int(_env("LOADTEST_REQUESTS", "50")))
    parser.add_argument("--concurrency", type=int, default=int(_env("LOADTEST_CONCURRENCY", "5")))
    parser.add_argument("--ticket-id", default=_env("LOADTEST_MCP_TICKET_ID", ""))
    return parser.parse_args()


def _normalize_url(value: str, default: str) -> str:
    """标准化 URL 字符串。

    实现流程：
    1. 空值时使用 default。
    2. 去掉首尾空白。
    3. 去掉末尾 `/`，让路径拼接更稳定。
    """
    return str(value or default).strip().rstrip("/")


def _load_mcp_sdk() -> tuple[Any, Any, Any]:
    """延迟加载 MCP client 依赖。

    输出：
    - `anyio`
    - `ClientSession`
    - `streamablehttp_client`

    实现流程：
    1. 在函数内部 import，而不是模块加载时 import。
    2. 如果当前 Python 环境缺少 `mcp` 包，抛出带说明的 RuntimeError。

    用途：
    - 让 `python -m py_compile` 在没有安装 MCP SDK 的环境里也能通过。
    - 运行真实 MCP 测试时再要求依赖存在。
    """
    try:
        import anyio
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "当前 Python 环境缺少 MCP SDK。请在安装了 mcp 包的环境运行，"
            "或在 Docker api/mcp 镜像中运行该脚本。"
        ) from exc
    return anyio, ClientSession, streamablehttp_client


def _request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    """发送同步 HTTP 请求，并尽量把响应解析成 JSON。

    用途：
    - MCP lookup 需要一张归属于 MCP actor 的工单。
    - 这里复用 API 的注册、登录、建单接口准备这张种子工单。
    """
    response = client.request(method, path, json=json_body, headers=headers)
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    return response.status_code, payload


def _raise_for_unexpected(status_code: int, payload: Any, *, action: str, allowed: set[int]) -> None:
    """检查 HTTP 状态码是否符合预期，不符合时终止准备流程。"""
    if status_code in allowed:
        return
    raise RuntimeError(f"{action}_failed status={status_code} payload={payload!r}")


def _register_or_ignore(client: httpx.Client, *, username: str, password: str, email: str) -> None:
    """注册 MCP actor 对应的 API 测试用户，用户已存在时忽略。

    实现流程：
    1. 调用 `POST /auth/register`。
    2. 如果返回 409，说明用户已存在，继续后续流程。
    3. 如果返回 200，说明注册成功。
    4. 其他状态码视为准备失败。
    """
    status_code, payload = _request_json(
        client,
        "POST",
        "/auth/register",
        json_body={"username": username, "password": password, "email": email},
    )
    if status_code == 409:
        print(f"[AUTH] MCP actor 用户已存在：{username}")
        return
    _raise_for_unexpected(status_code, payload, action="register", allowed={200})
    print(f"[AUTH] 已注册 MCP actor 用户：{username}")


def _login(client: httpx.Client, *, username: str, password: str) -> str:
    """登录 MCP actor 对应的 API 用户，并返回 Bearer Token。

    用途：
    - 只用于准备 lookup 种子工单。
    - 真正 MCP 调用不使用 Bearer Token，MCP server 使用固定 actor。
    """
    status_code, payload = _request_json(
        client,
        "POST",
        "/auth/login",
        json_body={"identifier": username, "password": password},
    )
    _raise_for_unexpected(status_code, payload, action="login", allowed={200})
    if not isinstance(payload, dict) or not str(payload.get("access_token") or "").strip():
        raise RuntimeError(f"login_missing_access_token payload={payload!r}")
    token = str(payload["access_token"])
    print(f"[AUTH] MCP actor 登录成功：{username}，token_prefix={token[:12]}...")
    return token


def _seed_lookup_ticket(
    client: httpx.Client,
    *,
    token: str,
    actor: str,
    department: str,
) -> dict[str, Any]:
    """创建或复用 MCP lookup 测试用工单。

    实现流程：
    1. 用 MCP actor 对应的 API 用户创建工单。
    2. 使用稳定 Idempotency-Key，重复运行脚本时复用同一张工单。
    3. 返回包含 `ticket_id` 的工单摘要。

    注意：
    - MCP server 的固定 actor 默认是 `demo_user`。
    - lookup 权限校验会检查工单 owner，因此种子工单也要由同名 actor 创建。
    """
    status_code, payload = _request_json(
        client,
        "POST",
        "/tickets",
        json_body={
            "department": department,
            "category": "network",
            "priority": "P2",
            "title": "MCP 并发查单种子工单",
            "description": "用于 MCP get_ticket_detail 并发测试。",
            "contact": f"{actor}@example.com",
            "context": {"source": "mcp_concurrency_setup"},
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "mcp-concurrency-lookup-stable-v1",
        },
    )
    _raise_for_unexpected(status_code, payload, action="seed_lookup_ticket", allowed={200})
    if not isinstance(payload, dict) or not str(payload.get("ticket_id") or "").strip():
        raise RuntimeError(f"seed_lookup_ticket_missing_ticket_id payload={payload!r}")
    print(f"[SEED] mcp_lookup: {payload['ticket_id']} 状态={payload.get('status')}")
    return payload


def _write_output(path: str, payload: dict[str, Any]) -> None:
    """把 MCP 测试准备结果写入本地 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OUTPUT] 已写入 {output_path}")


def _payload_from_call_result(call_result: Any) -> dict[str, Any]:
    """从 MCP call_tool 返回值中提取结构化 payload。

    实现流程：
    1. 优先读取 `structuredContent`。
    2. 如果没有结构化内容，则拼接 text content。
    3. text content 是 JSON 时解析为字典。
    4. 解析失败时保留为 `raw_text`。
    """
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    text_payload = "".join(
        str(getattr(block, "text", "") or "")
        for block in getattr(call_result, "content", []) or []
        if str(getattr(block, "type", "") or "") == "text"
    )
    if not text_payload:
        return {}
    try:
        parsed = json.loads(text_payload)
    except json.JSONDecodeError:
        return {"raw_text": text_payload}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _extract_ticket_id(payload: Any) -> str:
    """从多种可能的 MCP payload 结构中提取 ticket_id。

    实现流程：
    1. 检查 payload 顶层。
    2. 检查 wrapper contract 的 `data` 字段。
    3. 检查常见嵌套字段 `ticket` / `ticket_detail`。
    4. 找不到时返回空字符串。
    """
    if not isinstance(payload, dict):
        return ""
    for candidate in (
        payload,
        payload.get("data") if isinstance(payload.get("data"), dict) else None,
        payload.get("ticket") if isinstance(payload.get("ticket"), dict) else None,
        payload.get("ticket_detail") if isinstance(payload.get("ticket_detail"), dict) else None,
    ):
        if isinstance(candidate, dict) and str(candidate.get("ticket_id") or "").strip():
            return str(candidate["ticket_id"])

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("ticket", "ticket_detail"):
            nested = data.get(key)
            if isinstance(nested, dict) and str(nested.get("ticket_id") or "").strip():
                return str(nested["ticket_id"])
    return ""


def _mcp_payload_success(payload: dict[str, Any]) -> bool:
    """判断 MCP wrapper payload 是否表示业务成功。

    实现流程：
    1. 如果 payload 明确有 `success` 字段，按该字段判断。
    2. 如果存在 `error` 字段，认为失败。
    3. 否则认为 MCP 调用成功。
    """
    if "success" in payload:
        return bool(payload.get("success"))
    if isinstance(payload.get("error"), dict):
        return False
    return True


def _mcp_status_from_payload(payload: dict[str, Any]) -> str:
    """把 MCP payload 转成压测统计里的状态标签。"""
    if _mcp_payload_success(payload):
        return "ok"
    error = payload.get("error")
    if isinstance(error, dict) and str(error.get("error_code") or "").strip():
        return str(error["error_code"])
    return "tool_error"


async def _call_mcp_tool_once(*, server_url: str, tool: str, arguments: dict[str, Any], timeout: float) -> dict[str, Any]:
    """打开一个 MCP session 并调用一次工具。

    实现流程：
    1. 使用 Streamable HTTP transport 连接 MCP server。
    2. 初始化 ClientSession。
    3. 调用指定 tool。
    4. 解析 tool payload，并转换成统一的 ok/status/payload 结构。

    注意：
    - 每个请求独立建立 session，更接近“多个 MCP 客户端并发”的压力模型。
    - 这会比复用一个 session 更重，但也更贴近面试题里的多客户端场景。
    """
    _anyio, ClientSession, streamablehttp_client = _load_mcp_sdk()
    async with streamablehttp_client(
        str(server_url),
        timeout=float(timeout),
        sse_read_timeout=float(max(timeout, 30)),
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            call_result = await session.call_tool(tool, arguments=dict(arguments or {}))
            payload = _payload_from_call_result(call_result)
            return {
                "ok": _mcp_payload_success(payload),
                "status": _mcp_status_from_payload(payload),
                "payload": payload,
            }


async def _list_mcp_tools_once(*, server_url: str, timeout: float) -> dict[str, Any]:
    """打开一个 MCP session 并执行 list_tools。

    用途：
    - 作为 MCP 层的 health case。
    - 测试 MCP server 建连、initialize、列工具这条轻量链路。
    """
    _anyio, ClientSession, streamablehttp_client = _load_mcp_sdk()
    async with streamablehttp_client(
        str(server_url),
        timeout=float(timeout),
        sse_read_timeout=float(max(timeout, 30)),
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            tool_names = [str(tool.name) for tool in getattr(result, "tools", []) or []]
            return {"ok": True, "status": "ok", "payload": {"tools": tool_names}}


def _percentile(values: list[float], ratio: float) -> float:
    """计算延迟列表的近似百分位值。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def _print_benchmark_summary(name: str, results: list[dict[str, Any]], elapsed_seconds: float) -> None:
    """打印 MCP 并发测试统计摘要。"""
    latencies = [float(item["latency_ms"]) for item in results]
    statuses = Counter(str(item.get("status") or "unknown") for item in results)
    ok_count = sum(1 for item in results if bool(item.get("ok")))
    total = len(results)
    error_count = total - ok_count
    rps = (total / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    print(f"[MCP_BENCH] case={name}")
    print(f"[MCP_BENCH] total={total} ok={ok_count} error={error_count} rps={rps:.2f}")
    print(
        "[MCP_BENCH] latency_ms "
        f"avg={(sum(latencies) / len(latencies)) if latencies else 0.0:.2f} "
        f"p50={_percentile(latencies, 0.50):.2f} "
        f"p95={_percentile(latencies, 0.95):.2f} "
        f"p99={_percentile(latencies, 0.99):.2f} "
        f"max={(max(latencies) if latencies else 0.0):.2f}"
    )
    print(f"[MCP_BENCH] statuses={dict(statuses)}")


async def _run_benchmark(*, name: str, total_requests: int, concurrency: int, request_once) -> list[dict[str, Any]]:
    """执行通用 MCP 异步并发测试。

    实现流程：
    1. 使用 Semaphore 控制最大并发。
    2. 每个请求记录耗时。
    3. 单个请求异常不会中断整批测试。
    4. 统一打印总量、成功数、错误数、RPS 和延迟分布。
    """
    safe_total = max(1, int(total_requests))
    safe_concurrency = max(1, int(concurrency))
    semaphore = asyncio.Semaphore(safe_concurrency)
    started_at = time.perf_counter()

    async def _one(index: int) -> dict[str, Any]:
        async with semaphore:
            request_started_at = time.perf_counter()
            try:
                result = await request_once(index)
                latency_ms = (time.perf_counter() - request_started_at) * 1000
                return {
                    "ok": bool(result.get("ok")),
                    "status": str(result.get("status") or "unknown"),
                    "latency_ms": latency_ms,
                    "payload": result.get("payload") if isinstance(result, dict) else None,
                }
            except Exception as exc:
                latency_ms = (time.perf_counter() - request_started_at) * 1000
                return {
                    "ok": False,
                    "status": f"exception:{exc.__class__.__name__}",
                    "latency_ms": latency_ms,
                    "payload": None,
                }

    results = await asyncio.gather(*(_one(index) for index in range(safe_total)))
    elapsed_seconds = time.perf_counter() - started_at
    _print_benchmark_summary(name, results, elapsed_seconds)
    return results


async def _run_health_case(*, server_url: str, total_requests: int, concurrency: int, timeout: float) -> None:
    """执行 MCP list_tools 并发测试。"""
    last_tools: list[str] = []

    async def _request_once(_index: int) -> dict[str, Any]:
        nonlocal last_tools
        result = await _list_mcp_tools_once(server_url=server_url, timeout=timeout)
        payload = result.get("payload")
        if isinstance(payload, dict):
            last_tools = list(payload.get("tools") or [])
        return result

    await _run_benchmark(
        name="mcp_health",
        total_requests=total_requests,
        concurrency=concurrency,
        request_once=_request_once,
    )
    print(f"[MCP_HEALTH] tools={last_tools}")


async def _run_lookup_case(
    *,
    server_url: str,
    ticket_id: str,
    total_requests: int,
    concurrency: int,
    timeout: float,
) -> None:
    """执行 MCP get_ticket_detail 并发测试。"""
    async def _request_once(_index: int) -> dict[str, Any]:
        return await _call_mcp_tool_once(
            server_url=server_url,
            tool="get_ticket_detail",
            arguments={"ticket_id": ticket_id},
            timeout=timeout,
        )

    await _run_benchmark(
        name="mcp_lookup",
        total_requests=total_requests,
        concurrency=concurrency,
        request_once=_request_once,
    )


async def _run_create_case(
    *,
    server_url: str,
    department: str,
    total_requests: int,
    concurrency: int,
    timeout: float,
) -> None:
    """执行 MCP create_ticket 并发测试。"""
    run_id = int(time.time())
    created_ticket_ids: list[str] = []

    async def _request_once(index: int) -> dict[str, Any]:
        result = await _call_mcp_tool_once(
            server_url=server_url,
            tool="create_ticket",
            arguments={
                "text": f"请创建一张网络故障工单，标题是 MCP 并发建单 {run_id}-{index}，地点在测试区，联系方式 mcp@example.com。",
                "fields": {
                    "department": department,
                    "category": "network",
                    "priority": "P2",
                    "title": f"MCP 并发建单 {run_id}-{index}",
                    "description": f"用于 MCP create_ticket 并发测试的第 {index} 个请求。",
                    "contact": "mcp@example.com",
                    "location": "mcp-load-test-lab",
                },
                "idempotency_key": f"mcp-create-{run_id}-{index}",
            },
            timeout=timeout,
        )
        if bool(result.get("ok")):
            ticket_id = _extract_ticket_id(result.get("payload"))
            if ticket_id:
                created_ticket_ids.append(ticket_id)
        return result

    results = await _run_benchmark(
        name="mcp_create",
        total_requests=total_requests,
        concurrency=concurrency,
        request_once=_request_once,
    )
    ok_count = sum(1 for item in results if bool(item.get("ok")))
    print(
        "[MCP_VERIFY] created_tickets "
        f"ok_requests={ok_count} returned_ticket_ids={len(created_ticket_ids)} "
        f"unique_ticket_ids={len(set(created_ticket_ids))}"
    )


def main() -> int:
    """脚本主入口。

    实现流程：
    1. 解析参数。
    2. 对 lookup case 准备归属于 MCP actor 的种子工单。
    3. 根据 case 调用对应 MCP 并发测试。
    4. 输出准备信息和压测统计。
    """
    args = _parse_args()
    mcp_url = _normalize_url(str(args.mcp_url), "http://localhost:9000/mcp")
    api_base_url = _normalize_url(str(args.api_base_url), "http://localhost:8080")
    actor = str(args.actor or "demo_user").strip() or "demo_user"
    print(f"[CONFIG] mcp_url={mcp_url}")
    print(f"[CONFIG] api_base_url={api_base_url}")
    print(f"[CONFIG] actor={actor} department={args.department}")
    _load_mcp_sdk()

    ticket_id = str(args.ticket_id or "").strip()
    if str(args.case) == "lookup" and not ticket_id:
        with httpx.Client(base_url=api_base_url, timeout=float(args.timeout)) as client:
            health_status, health_payload = _request_json(client, "GET", "/health")
            _raise_for_unexpected(health_status, health_payload, action="api_health", allowed={200})
            _register_or_ignore(
                client,
                username=actor,
                password=str(args.password),
                email=str(args.email),
            )
            token = _login(client, username=actor, password=str(args.password))
            seed = _seed_lookup_ticket(
                client,
                token=token,
                actor=actor,
                department=str(args.department),
            )
            ticket_id = str(seed["ticket_id"])
            _write_output(
                str(args.output),
                {
                    "mcp_url": mcp_url,
                    "api_base_url": api_base_url,
                    "actor": actor,
                    "lookup_ticket": seed,
                    "generated_at": int(time.time()),
                },
            )

    if str(args.case) == "health":
        asyncio.run(
            _run_health_case(
                server_url=mcp_url,
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )
    elif str(args.case) == "lookup":
        if not ticket_id:
            raise RuntimeError("mcp_lookup_ticket_id_missing")
        asyncio.run(
            _run_lookup_case(
                server_url=mcp_url,
                ticket_id=ticket_id,
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )
    elif str(args.case) == "create":
        asyncio.run(
            _run_create_case(
                server_url=mcp_url,
                department=str(args.department),
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )

    print("[DONE] MCP 并发测试完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
