#!/usr/bin/env python3
"""并发测试准备脚本。

当前范围：
1. 读取 API 与压测配置。
2. 注册或登录测试用户，并获取 Bearer Token。
3. 创建后续并发测试使用的确定性种子工单。

当前已经包含多类轻量场景：
1. setup：只准备账号、Token 和种子工单。
2. health：并发访问 `/health`，测试入口层基线。
3. lookup：并发查询同一张工单，测试鉴权 + DB 读路径。
4. comment：并发给同一张工单追加评论，测试低风险写路径。
5. escalate：并发催办同一张工单，测试同一行更新的一致性。
6. escalate_many：并发催办多张工单，测试分散写路径。
7. create：并发创建工单，测试 Redis 幂等 + DB 插入写路径。
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


DEFAULT_OUTPUT_PATH = "outputs/concurrency_setup.json"


def _env(name: str, default: str) -> str:
    """读取环境变量，空值时返回默认值。

    实现流程：
    1. 从 `os.getenv` 读取变量。
    2. 转成字符串并去掉首尾空白。
    3. 如果读取结果为空，使用调用方传入的默认值。
    """
    value = str(os.getenv(name) or "").strip()
    return value or default


def _parse_args() -> argparse.Namespace:
    """解析命令行参数，并把环境变量作为默认配置来源。

    实现流程：
    1. 定义 API 地址、测试账号、部门、超时等基础参数。
    2. 定义 `--case`，控制只做 setup，还是额外执行某个压测场景。
    3. 定义 `--requests` 和 `--concurrency`，控制请求总数和并发数。
    4. 定义 `--distributed-tickets`，控制分散写场景要准备多少张工单。
    5. 定义 `--fresh`，用于选择复用稳定幂等键还是每次创建新工单。
    """
    parser = argparse.ArgumentParser(description="准备并发测试需要的登录态和种子工单。")
    parser.add_argument("--api-base-url", default=_env("LOADTEST_API_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--username", default=_env("LOADTEST_USERNAME", "loadtest_user"))
    parser.add_argument("--password", default=_env("LOADTEST_PASSWORD", "loadtest_password_123"))
    parser.add_argument("--email", default=_env("LOADTEST_EMAIL", "loadtest_user@example.com"))
    parser.add_argument("--department", default=_env("LOADTEST_DEPARTMENT", "IT"))
    parser.add_argument("--timeout", type=float, default=float(_env("LOADTEST_TIMEOUT_SECONDS", "15")))
    parser.add_argument("--output", default=_env("LOADTEST_SETUP_OUTPUT", DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--case",
        choices=["setup", "health", "lookup", "comment", "escalate", "escalate_many", "create"],
        default="setup",
        help=(
            "要执行的测试场景：setup 只准备数据，health 测入口，lookup 测查单，"
            "comment 测并发评论，escalate 测单工单并发催办，"
            "escalate_many 测多工单分散催办，create 测并发建单。"
        ),
    )
    parser.add_argument("--requests", type=int, default=int(_env("LOADTEST_REQUESTS", "100")))
    parser.add_argument("--concurrency", type=int, default=int(_env("LOADTEST_CONCURRENCY", "10")))
    parser.add_argument(
        "--distributed-tickets",
        type=int,
        default=int(_env("LOADTEST_DISTRIBUTED_TICKETS", "20")),
        help="escalate_many 场景中要准备的工单数量。",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="使用带时间戳的幂等键创建全新工单，而不是复用上一次准备结果。",
    )
    return parser.parse_args()


def _normalize_base_url(value: str) -> str:
    """标准化 API Base URL。

    实现流程：
    1. 空值时回退到 `http://localhost:8080`。
    2. 去掉首尾空白。
    3. 去掉末尾 `/`，避免后续拼接路径时出现双斜杠。
    """
    return str(value or "http://localhost:8080").strip().rstrip("/")


def _request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    """发送同步 HTTP 请求，并尽量把响应解析成 JSON。

    输入：
    - `client`：已配置 base_url / timeout 的 `httpx.Client`。
    - `method`：HTTP 方法，例如 GET / POST。
    - `path`：接口路径，例如 `/auth/login`。
    - `json_body`：可选 JSON 请求体。
    - `headers`：可选请求头。

    输出：
    - `(status_code, payload)`。

    实现流程：
    1. 使用传入的 client 发起请求。
    2. 优先调用 `response.json()` 解析响应。
    3. 如果响应不是 JSON，则回退为 `response.text`。
    """
    response = client.request(method, path, json=json_body, headers=headers)
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    return response.status_code, payload


def _raise_for_unexpected(status_code: int, payload: Any, *, action: str, allowed: set[int]) -> None:
    """统一检查 HTTP 状态码是否符合预期。

    实现流程：
    1. 如果状态码在 allowed 集合里，直接返回。
    2. 否则抛出 RuntimeError，并带上动作名、状态码和响应体。

    作用：
    - 让注册、登录、建单等步骤失败时尽早停止，而不是继续跑无效压测。
    """
    if status_code in allowed:
        return
    raise RuntimeError(f"{action}_failed status={status_code} payload={payload!r}")


def _register_or_ignore(client: httpx.Client, *, username: str, password: str, email: str) -> dict[str, Any] | None:
    """注册测试用户；如果用户已存在则忽略。

    输入：
    - `username` / `password` / `email`：测试用户凭证。

    输出：
    - 注册成功时返回接口响应。
    - 用户已存在时返回 None。

    实现流程：
    1. 调用 `POST /auth/register`。
    2. 如果返回 409，说明用户或邮箱已存在，打印提示后继续。
    3. 如果返回 200，说明注册成功。
    4. 其他状态码视为准备失败。
    """
    status_code, payload = _request_json(
        client,
        "POST",
        "/auth/register",
        json_body={
            "username": username,
            "password": password,
            "email": email,
        },
    )
    if status_code == 409:
        print(f"[AUTH] 用户已存在：{username}")
        return None
    _raise_for_unexpected(status_code, payload, action="register", allowed={200})
    print(f"[AUTH] 已注册用户：{username}")
    return payload if isinstance(payload, dict) else {"raw": payload}


def _login(client: httpx.Client, *, username: str, password: str) -> dict[str, Any]:
    """登录测试用户，并获取 Bearer Token。

    输入：
    - `username`：登录 identifier。
    - `password`：登录密码。

    输出：
    - `/auth/login` 返回的完整 JSON，其中必须包含 `access_token`。

    实现流程：
    1. 调用 `POST /auth/login`。
    2. 校验 HTTP 状态码必须为 200。
    3. 校验响应里必须有非空 `access_token`。
    4. 打印 token 前缀，避免在终端暴露完整 token。
    """
    status_code, payload = _request_json(
        client,
        "POST",
        "/auth/login",
        json_body={
            "identifier": username,
            "password": password,
        },
    )
    _raise_for_unexpected(status_code, payload, action="login", allowed={200})
    if not isinstance(payload, dict) or not str(payload.get("access_token") or "").strip():
        raise RuntimeError(f"login_missing_access_token payload={payload!r}")
    token = str(payload.get("access_token") or "")
    print(f"[AUTH] 登录成功：{username}，token_prefix={token[:12]}...")
    return payload


def _ticket_payload(*, department: str, case: str, title: str, description: str) -> dict[str, Any]:
    """构造创建测试工单所需的请求体。

    输入：
    - `department`：工单部门。
    - `case`：测试场景名称，例如 lookup / comment / escalate。
    - `title` / `description`：工单标题和描述。

    输出：
    - 可直接传给 `POST /tickets` 的 JSON payload。

    实现流程：
    1. 为工单设置统一的 category / priority / contact。
    2. 在 context 中写入 `source=concurrency_setup` 和 case，便于后续追溯。
    3. 返回稳定 payload，配合稳定幂等键可以重复运行 setup。
    """
    return {
        "department": department,
        "category": "network" if case in {"lookup", "comment", "escalate", "escalate_many", "cancel"} else "other",
        "priority": "P2",
        "title": title,
        "description": description,
        "contact": "loadtest@example.com",
        "context": {
            "source": "concurrency_setup",
            "case": case,
            "location": "load-test-lab",
        },
    }


def _create_ticket(
    client: httpx.Client,
    *,
    token: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """调用 `POST /tickets` 创建一张测试工单。

    输入：
    - `token`：登录后获得的 Bearer Token。
    - `idempotency_key`：建单幂等键。
    - `payload`：建单请求体。

    输出：
    - 创建成功后的工单摘要，至少包含 `ticket_id`。

    实现流程：
    1. 在请求头中附加 `Authorization` 和 `Idempotency-Key`。
    2. 调用 `POST /tickets`。
    3. 校验状态码必须为 200。
    4. 校验响应中必须包含 `ticket_id`。
    """
    status_code, response_payload = _request_json(
        client,
        "POST",
        "/tickets",
        json_body=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": idempotency_key,
        },
    )
    _raise_for_unexpected(status_code, response_payload, action="create_ticket", allowed={200})
    if not isinstance(response_payload, dict) or not str(response_payload.get("ticket_id") or "").strip():
        raise RuntimeError(f"create_ticket_missing_ticket_id payload={response_payload!r}")
    return response_payload


def _seed_tickets(
    client: httpx.Client,
    *,
    token: str,
    department: str,
    fresh: bool,
) -> dict[str, dict[str, Any]]:
    """创建后续压测所需的种子工单。

    输入：
    - `token`：Bearer Token。
    - `department`：测试部门。
    - `fresh`：是否使用时间戳生成新幂等键。

    输出：
    - 以 case 为键的工单信息字典，例如 `lookup -> ticket_id`。

    实现流程：
    1. 根据 fresh 决定幂等键后缀：
       - false：使用 `stable-v1`，重复运行会 replay 之前结果。
       - true：使用当前时间戳，每次创建新工单。
    2. 构造 lookup/comment/escalate/cancel 四类测试工单 payload。
    3. 逐个调用 `_create_ticket`。
    4. 记录每个 case 的 ticket_id、status 和 idempotency_key。
    """
    suffix = str(int(time.time())) if fresh else "stable-v1"
    cases = {
        "lookup": _ticket_payload(
            department=department,
            case="lookup",
            title="并发查单基线工单",
            description="用于并发测试中的只读查单场景。",
        ),
        "comment": _ticket_payload(
            department=department,
            case="comment",
            title="并发评论测试工单",
            description="用于多个客户端同时追加评论的测试场景。",
        ),
        "escalate": _ticket_payload(
            department=department,
            case="escalate",
            title="并发催办测试工单",
            description="用于多个客户端同时催办同一张工单的测试场景。",
        ),
        "cancel": _ticket_payload(
            department=department,
            case="cancel",
            title="并发取消确认测试工单",
            description="用于取消、确认态和状态冲突测试场景。",
        ),
    }

    seeded: dict[str, dict[str, Any]] = {}
    for case, payload in cases.items():
        idempotency_key = f"concurrency-setup-{case}-{suffix}"
        ticket = _create_ticket(
            client,
            token=token,
            idempotency_key=idempotency_key,
            payload=payload,
        )
        seeded[case] = {
            "ticket_id": ticket["ticket_id"],
            "status": ticket.get("status"),
            "idempotency_key": idempotency_key,
        }
        print(f"[SEED] {case}: {ticket['ticket_id']} 状态={ticket.get('status')}")
    return seeded


def _seed_distributed_escalate_tickets(
    client: httpx.Client,
    *,
    token: str,
    department: str,
    ticket_count: int,
    fresh: bool,
) -> list[dict[str, Any]]:
    """创建多工单分散催办场景所需的测试工单。

    输入：
    - `client`：同步 HTTP client。
    - `token`：Bearer Token。
    - `department`：测试部门。
    - `ticket_count`：需要准备的工单数量。
    - `fresh`：是否使用时间戳创建全新幂等键。

    输出：
    - 工单信息列表，每个元素包含 `ticket_id`、`status` 和 `idempotency_key`。

    实现流程：
    1. 对 ticket_count 做下限保护，至少准备 1 张工单。
    2. 根据 fresh 选择稳定后缀或时间戳后缀。
    3. 循环构造多张工单的 payload。
    4. 使用不同的 Idempotency-Key 创建或复用这些工单。
    5. 返回后续 `escalate_many` 场景可以轮询使用的工单列表。

    用途：
    - 单工单 escalate 测的是“热点行更新”。
    - 多工单 escalate_many 把写请求分散到不同工单，用来判断瓶颈是否主要来自同一行锁竞争。
    """
    safe_count = max(1, int(ticket_count))
    suffix = str(int(time.time())) if fresh else "stable-v1"
    tickets: list[dict[str, Any]] = []

    for index in range(safe_count):
        idempotency_key = f"concurrency-distributed-escalate-{index}-{suffix}"
        payload = _ticket_payload(
            department=department,
            case="escalate_many",
            title=f"分散催办测试工单 {index + 1}",
            description=f"用于多工单分散写压测的第 {index + 1} 张工单。",
        )
        ticket = _create_ticket(
            client,
            token=token,
            idempotency_key=idempotency_key,
            payload=payload,
        )
        tickets.append(
            {
                "ticket_id": ticket["ticket_id"],
                "status": ticket.get("status"),
                "idempotency_key": idempotency_key,
            }
        )

    print(f"[SEED] distributed_escalate: 已准备 {len(tickets)} 张工单")
    return tickets


def _write_output(path: str, payload: dict[str, Any]) -> None:
    """把 setup 结果写入本地 JSON 文件。

    输入：
    - `path`：输出文件路径。
    - `payload`：要写入的 JSON 内容。

    实现流程：
    1. 确保父目录存在。
    2. 用 UTF-8 写入格式化 JSON。
    3. 打印输出文件位置。

    注意：
    - 输出内容包含本地测试用 access_token，不要提交或分享。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OUTPUT] 已写入 {output_path}")


def _percentile(values: list[float], ratio: float) -> float:
    """计算延迟列表的近似百分位值。

    输入：
    - `values`：延迟列表，单位通常是毫秒。
    - `ratio`：百分位比例，例如 0.95 表示 p95。

    输出：
    - 对应百分位上的值。

    实现流程：
    1. 空列表直接返回 0。
    2. 对数值排序。
    3. 根据 ratio 计算索引，并做边界保护。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def _extract_escalation_count(payload: Any) -> int:
    """从工单详情响应中提取催办次数。

    输入：
    - `payload`：`GET /tickets/{ticket_id}` 或催办接口返回的响应体。

    输出：
    - `context.escalation_count` 的整数值。
    - 如果响应体格式异常、context 不存在或字段不可转整数，则返回 0。

    实现流程：
    1. 先确认 payload 是字典。
    2. 从 payload 中取出 `context` 字段。
    3. 从 context 中读取 `escalation_count`。
    4. 尝试转换成 int，失败时按 0 处理。

    用途：
    - 并发催办不能只看 HTTP 200。
    - 如果 100 次催办都返回 200，但计数只增加了 70，就说明存在并发丢更新。
    """
    if not isinstance(payload, dict):
        return 0
    context = payload.get("context") or {}
    if not isinstance(context, dict):
        return 0
    try:
        return int(context.get("escalation_count") or 0)
    except (TypeError, ValueError):
        return 0


def _safe_response_json(response: httpx.Response) -> Any:
    """安全解析 HTTP 响应 JSON。

    输入：
    - `response`：`httpx` 返回的响应对象。

    输出：
    - JSON 解析成功时返回解析后的对象。
    - JSON 解析失败时返回空字典。

    实现流程：
    1. 调用 `response.json()`。
    2. 如果响应不是 JSON 或响应体异常，捕获异常并返回 `{}`。

    用途：
    - 压测结束后的校验请求不能因为响应格式异常直接让脚本崩溃。
    - 真正的异常应该体现在 benchmark 的状态码和错误统计里。
    """
    try:
        return response.json()
    except Exception:
        return {}


def _print_benchmark_summary(name: str, results: list[dict[str, Any]], elapsed_seconds: float) -> None:
    """打印一次并发测试的统计摘要。

    输入：
    - `name`：测试场景名称。
    - `results`：每个请求的结果，包含 ok/status/latency_ms。
    - `elapsed_seconds`：整批请求总耗时。

    输出：
    - 直接打印 total、成功数、错误数、RPS、延迟和状态码分布。

    实现流程：
    1. 提取所有请求延迟。
    2. 统计 HTTP 状态码或异常类型分布。
    3. 统计成功/失败数量。
    4. 计算吞吐量和延迟百分位。
    """
    latencies = [float(item["latency_ms"]) for item in results]
    statuses = Counter(str(item.get("status") or "unknown") for item in results)
    ok_count = sum(1 for item in results if bool(item.get("ok")))
    total = len(results)
    error_count = total - ok_count
    rps = (total / elapsed_seconds) if elapsed_seconds > 0 else 0.0

    print(f"[BENCH] case={name}")
    print(f"[BENCH] total={total} ok={ok_count} error={error_count} rps={rps:.2f}")
    print(
        "[BENCH] latency_ms "
        f"avg={(sum(latencies) / len(latencies)) if latencies else 0.0:.2f} "
        f"p50={_percentile(latencies, 0.50):.2f} "
        f"p95={_percentile(latencies, 0.95):.2f} "
        f"p99={_percentile(latencies, 0.99):.2f} "
        f"max={(max(latencies) if latencies else 0.0):.2f}"
    )
    print(f"[BENCH] statuses={dict(statuses)}")


async def _run_benchmark(
    *,
    name: str,
    total_requests: int,
    concurrency: int,
    request_once,
) -> list[dict[str, Any]]:
    """执行一个通用异步并发测试。

    输入：
    - `name`：测试场景名称。
    - `total_requests`：请求总数。
    - `concurrency`：最大并发数。
    - `request_once`：单次请求协程函数，输入 index，输出 HTTP 状态码。

    输出：
    - 每个请求的结果列表。

    实现流程：
    1. 对请求总数和并发数做下限保护。
    2. 用 `asyncio.Semaphore` 控制同时在飞的请求数。
    3. 为每个请求记录开始时间、状态码和耗时。
    4. 单个请求异常不会中断整批测试，而是记录为 `exception:<类型>`。
    5. 全部请求完成后打印统计摘要。
    """
    safe_total = max(1, int(total_requests))
    safe_concurrency = max(1, int(concurrency))
    semaphore = asyncio.Semaphore(safe_concurrency)
    started_at = time.perf_counter()

    async def _one(index: int) -> dict[str, Any]:
        async with semaphore:
            request_started_at = time.perf_counter()
            try:
                status = await request_once(index)
                latency_ms = (time.perf_counter() - request_started_at) * 1000
                return {
                    "ok": 200 <= int(status) < 400,
                    "status": int(status),
                    "latency_ms": latency_ms,
                }
            except Exception as exc:
                latency_ms = (time.perf_counter() - request_started_at) * 1000
                return {
                    "ok": False,
                    "status": f"exception:{exc.__class__.__name__}",
                    "latency_ms": latency_ms,
                }

    results = await asyncio.gather(*(_one(index) for index in range(safe_total)))
    elapsed_seconds = time.perf_counter() - started_at
    _print_benchmark_summary(name, results, elapsed_seconds)
    return results


async def _run_health_case(*, base_url: str, total_requests: int, concurrency: int, timeout: float) -> None:
    """执行 `/health` 入口层基线并发测试。

    输入：
    - `base_url`：API 地址。
    - `total_requests`：请求总数。
    - `concurrency`：最大并发数。
    - `timeout`：单请求超时时间。

    实现流程：
    1. 创建一个异步 HTTP client。
    2. 定义单次请求：`GET /health`。
    3. 交给 `_run_benchmark` 统一调度和统计。

    用途：
    - 判断入口层和容器基础状态是否稳定。
    - 这条链路不依赖鉴权和数据库业务查询。
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=float(timeout)) as client:
        async def _request_once(_index: int) -> int:
            response = await client.get("/health")
            return response.status_code

        await _run_benchmark(
            name="health",
            total_requests=total_requests,
            concurrency=concurrency,
            request_once=_request_once,
        )


async def _run_lookup_case(
    *,
    base_url: str,
    token: str,
    ticket_id: str,
    total_requests: int,
    concurrency: int,
    timeout: float,
) -> None:
    """执行带鉴权的工单查询并发测试。

    输入：
    - `base_url`：API 地址。
    - `token`：Bearer Token。
    - `ticket_id`：用于查询的种子工单号。
    - `total_requests`：请求总数。
    - `concurrency`：最大并发数。
    - `timeout`：单请求超时时间。

    实现流程：
    1. 创建带 `Authorization` 请求头的异步 HTTP client。
    2. 定义单次请求：`GET /tickets/{ticket_id}`。
    3. 交给 `_run_benchmark` 统一调度和统计。

    用途：
    - 测试鉴权 + DB 读 + 工单序列化这一条链路。
    - 可用于和 `/health` 对比，观察业务读路径的额外开销。
    """
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=base_url, timeout=float(timeout), headers=headers) as client:
        async def _request_once(_index: int) -> int:
            response = await client.get(f"/tickets/{ticket_id}")
            return response.status_code

        await _run_benchmark(
            name="lookup",
            total_requests=total_requests,
            concurrency=concurrency,
            request_once=_request_once,
        )


async def _run_comment_case(
    *,
    base_url: str,
    token: str,
    ticket_id: str,
    total_requests: int,
    concurrency: int,
    timeout: float,
) -> None:
    """执行带鉴权的工单评论并发测试。

    输入：
    - `base_url`：API 地址。
    - `token`：Bearer Token。
    - `ticket_id`：用于追加评论的种子工单号。
    - `total_requests`：请求总数。
    - `concurrency`：最大并发数。
    - `timeout`：单请求超时时间。

    实现流程：
    1. 创建带 `Authorization` 请求头的异步 HTTP client。
    2. 为每个请求生成唯一评论内容，避免后续排查时分不清是哪次请求写入。
    3. 调用 `POST /tickets/{ticket_id}/comments`。
    4. 交给 `_run_benchmark` 统一控制并发、捕获异常并打印统计摘要。

    用途：
    - 这是第一个写路径压测 case。
    - 它能观察同一工单被并发追加评论时，DB session、事务提交、
      评论表 append-only 写入和响应序列化是否稳定。
    - 相比并发取消/并发状态修改，评论是低风险写操作，更适合作为第一步。
    """
    headers = {"Authorization": f"Bearer {token}"}
    run_id = int(time.time())
    async with httpx.AsyncClient(base_url=base_url, timeout=float(timeout), headers=headers) as client:
        async def _request_once(index: int) -> int:
            payload = {
                "comment": f"并发评论测试 run={run_id} index={index}",
            }
            response = await client.post(f"/tickets/{ticket_id}/comments", json=payload)
            return response.status_code

        await _run_benchmark(
            name="comment",
            total_requests=total_requests,
            concurrency=concurrency,
            request_once=_request_once,
        )


async def _run_escalate_case(
    *,
    base_url: str,
    token: str,
    ticket_id: str,
    total_requests: int,
    concurrency: int,
    timeout: float,
) -> None:
    """执行带鉴权的工单催办并发测试。

    输入：
    - `base_url`：API 地址。
    - `token`：Bearer Token。
    - `ticket_id`：用于并发催办的种子工单号。
    - `total_requests`：请求总数。
    - `concurrency`：最大并发数。
    - `timeout`：单请求超时时间。

    实现流程：
    1. 先查询一次工单详情，记录压测前的 `escalation_count`。
    2. 并发调用 `POST /tickets/{ticket_id}/escalate`，每次请求带唯一 reason。
    3. 使用 `_run_benchmark` 输出吞吐、错误和延迟分布。
    4. 压测结束后再次查询工单详情，记录压测后的 `escalation_count`。
    5. 对比 “成功请求数” 和 “计数实际增加量”：
       - 如果二者一致，说明这个 case 下没有观察到催办计数丢更新。
       - 如果实际增加量小于成功请求数，说明并发更新可能发生覆盖。

    用途：
    - 这是比 comment 更敏感的写路径压测。
    - comment 是 append-only 插入；escalate 会更新同一张工单的 context 和状态。
    - 因此它更适合发现同一行并发更新、事务锁等待和数据库连接池压力。
    """
    headers = {"Authorization": f"Bearer {token}"}
    run_id = int(time.time())
    async with httpx.AsyncClient(base_url=base_url, timeout=float(timeout), headers=headers) as client:
        before_response = await client.get(f"/tickets/{ticket_id}")
        before_count = _extract_escalation_count(_safe_response_json(before_response) if before_response.status_code == 200 else {})

        async def _request_once(index: int) -> int:
            payload = {
                "reason": f"并发催办测试 run={run_id} index={index}",
            }
            response = await client.post(f"/tickets/{ticket_id}/escalate", json=payload)
            return response.status_code

        results = await _run_benchmark(
            name="escalate",
            total_requests=total_requests,
            concurrency=concurrency,
            request_once=_request_once,
        )

        ok_count = sum(1 for item in results if bool(item.get("ok")))
        try:
            after_response = await client.get(f"/tickets/{ticket_id}")
            after_count = _extract_escalation_count(_safe_response_json(after_response) if after_response.status_code == 200 else {})
            actual_delta = after_count - before_count
            print(
                "[VERIFY] escalation_count "
                f"before={before_count} after={after_count} "
                f"ok_requests={ok_count} actual_delta={actual_delta}"
            )
            if actual_delta != ok_count:
                print("[VERIFY] WARNING: 催办成功数和计数增量不一致，可能存在并发覆盖或请求后置失败。")
        except Exception as exc:
            print(f"[VERIFY] WARNING: 压测后查询工单详情失败，无法校验计数：{exc.__class__.__name__}")


async def _run_escalate_many_case(
    *,
    base_url: str,
    token: str,
    ticket_ids: list[str],
    total_requests: int,
    concurrency: int,
    timeout: float,
) -> None:
    """执行多工单分散催办并发测试。

    输入：
    - `base_url`：API 地址。
    - `token`：Bearer Token。
    - `ticket_ids`：用于分散写入的多张工单号。
    - `total_requests`：请求总数。
    - `concurrency`：最大并发数。
    - `timeout`：单请求超时时间。

    实现流程：
    1. 对 ticket_ids 去空并去重，确保至少有一张可用工单。
    2. 压测前查询每张工单的 `escalation_count`。
    3. 请求按 `index % len(ticket_ids)` 轮询分配到不同工单。
    4. 每个请求成功后，在本地记录对应工单的成功次数。
    5. 压测后再次查询每张工单的 `escalation_count`。
    6. 对比每张工单的成功请求数和实际计数增量。

    用途：
    - 和单工单 escalate 对照。
    - 如果单工单 60 并发失败，但多工单 60 并发成功，说明主要问题是热点行锁竞争。
    - 如果多工单也失败，说明整体 DB 连接池、同步线程池或审计写入路径也存在压力。
    """
    unique_ticket_ids = list(dict.fromkeys(str(ticket_id).strip() for ticket_id in ticket_ids if str(ticket_id).strip()))
    if not unique_ticket_ids:
        raise RuntimeError("distributed_ticket_ids_missing")

    headers = {"Authorization": f"Bearer {token}"}
    run_id = int(time.time())
    ok_by_ticket: Counter[str] = Counter()

    async with httpx.AsyncClient(base_url=base_url, timeout=float(timeout), headers=headers) as client:
        before_counts: dict[str, int] = {}
        for ticket_id in unique_ticket_ids:
            response = await client.get(f"/tickets/{ticket_id}")
            before_counts[ticket_id] = _extract_escalation_count(_safe_response_json(response) if response.status_code == 200 else {})

        async def _request_once(index: int) -> int:
            ticket_id = unique_ticket_ids[index % len(unique_ticket_ids)]
            payload = {
                "reason": f"多工单分散催办测试 run={run_id} index={index}",
            }
            response = await client.post(f"/tickets/{ticket_id}/escalate", json=payload)
            if 200 <= response.status_code < 400:
                ok_by_ticket[ticket_id] += 1
            return response.status_code

        results = await _run_benchmark(
            name="escalate_many",
            total_requests=total_requests,
            concurrency=concurrency,
            request_once=_request_once,
        )

        try:
            mismatches = 0
            checked = 0
            total_actual_delta = 0
            total_expected_delta = 0
            for ticket_id in unique_ticket_ids:
                response = await client.get(f"/tickets/{ticket_id}")
                after_count = _extract_escalation_count(_safe_response_json(response) if response.status_code == 200 else {})
                expected_delta = int(ok_by_ticket.get(ticket_id) or 0)
                actual_delta = after_count - int(before_counts.get(ticket_id) or 0)
                checked += 1
                total_expected_delta += expected_delta
                total_actual_delta += actual_delta
                if actual_delta != expected_delta:
                    mismatches += 1

            ok_count = sum(1 for item in results if bool(item.get("ok")))
            print(
                "[VERIFY] distributed_escalation_count "
                f"tickets={checked} ok_requests={ok_count} "
                f"expected_delta={total_expected_delta} actual_delta={total_actual_delta} "
                f"mismatched_tickets={mismatches}"
            )
            if mismatches:
                print("[VERIFY] WARNING: 存在工单计数增量和成功请求数不一致，可能有并发覆盖或校验期间仍有请求未完成。")
        except Exception as exc:
            print(f"[VERIFY] WARNING: 压测后查询多工单详情失败，无法校验计数：{exc.__class__.__name__}")


async def _run_create_case(
    *,
    base_url: str,
    token: str,
    department: str,
    total_requests: int,
    concurrency: int,
    timeout: float,
) -> None:
    """执行并发建单测试。

    输入：
    - `base_url`：API 地址。
    - `token`：Bearer Token。
    - `department`：建单部门。
    - `total_requests`：请求总数。
    - `concurrency`：最大并发数。
    - `timeout`：单请求超时时间。

    实现流程：
    1. 创建带 `Authorization` 请求头的异步 HTTP client。
    2. 为每个请求生成唯一 Idempotency-Key，避免被幂等机制当成重复建单。
    3. 为每个请求生成唯一标题和描述，便于后续排查。
    4. 调用 `POST /tickets` 创建工单。
    5. 响应成功时记录返回的 `ticket_id`。
    6. 压测结束后打印成功建单数和唯一工单号数量。

    用途：
    - 测试创建工单这条写路径，而不是更新已有工单。
    - 覆盖 Redis 幂等检查、DB 插入 tickets、审计日志写入和事务提交。
    - 它比查询更接近真实 MCP 客户端批量建单的场景。
    """
    headers = {"Authorization": f"Bearer {token}"}
    run_id = int(time.time())
    created_ticket_ids: list[str] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=float(timeout), headers=headers) as client:
        async def _request_once(index: int) -> int:
            payload = _ticket_payload(
                department=department,
                case="create",
                title=f"并发建单测试 run={run_id} index={index}",
                description=f"用于并发建单压测的第 {index} 个请求。",
            )
            response = await client.post(
                "/tickets",
                json=payload,
                headers={"Idempotency-Key": f"concurrency-create-{run_id}-{index}"},
            )
            if 200 <= response.status_code < 400:
                response_payload = _safe_response_json(response)
                if isinstance(response_payload, dict) and str(response_payload.get("ticket_id") or "").strip():
                    created_ticket_ids.append(str(response_payload["ticket_id"]))
            return response.status_code

        results = await _run_benchmark(
            name="create",
            total_requests=total_requests,
            concurrency=concurrency,
            request_once=_request_once,
        )

    ok_count = sum(1 for item in results if bool(item.get("ok")))
    unique_created = len(set(created_ticket_ids))
    print(
        "[VERIFY] created_tickets "
        f"ok_requests={ok_count} returned_ticket_ids={len(created_ticket_ids)} "
        f"unique_ticket_ids={unique_created}"
    )
    if unique_created != ok_count:
        print("[VERIFY] WARNING: 成功请求数和唯一工单号数量不一致，可能存在响应缺字段或幂等键冲突。")


def main() -> int:
    """脚本主入口。

    实现流程：
    1. 解析命令行参数和环境变量配置。
    2. 标准化 API 地址。
    3. 同步执行准备阶段：
       - 检查 `/health`。
       - 注册测试用户，已存在则跳过。
       - 登录获取 Bearer Token。
       - 创建四张种子工单。
       - 如果执行 `escalate_many`，额外创建多张分散写测试工单。
       - 写入 `outputs/concurrency_setup.json`。
    4. 根据 `--case` 决定是否追加执行并发测试：
       - `setup`：只做准备。
       - `health`：测入口层基线。
       - `lookup`：测带鉴权的查单读路径。
       - `comment`：测同一工单并发追加评论的写路径。
       - `escalate`：测同一工单并发催办的更新路径，并检查计数是否丢失。
       - `escalate_many`：测多工单分散催办，并检查总计数是否丢失。
       - `create`：测并发创建工单，检查成功请求是否都返回唯一工单号。
    """
    args = _parse_args()
    base_url = _normalize_base_url(args.api_base_url)
    print(f"[CONFIG] api_base_url={base_url}")
    print(f"[CONFIG] username={args.username} department={args.department}")

    distributed_escalate_tickets: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, timeout=float(args.timeout)) as client:
        # 第一步：确认 API 已启动，避免后续注册/建单报连接错误。
        health_status, health_payload = _request_json(client, "GET", "/health")
        _raise_for_unexpected(health_status, health_payload, action="health", allowed={200})
        print(f"[HEALTH] {health_payload}")

        # 第二步：准备登录态。注册是幂等准备动作，用户已存在时继续登录。
        _register_or_ignore(
            client,
            username=str(args.username),
            password=str(args.password),
            email=str(args.email),
        )
        login_payload = _login(client, username=str(args.username), password=str(args.password))
        token = str(login_payload.get("access_token") or "")

        # 第三步：创建或复用后续压测需要的种子工单。
        seeded_tickets = _seed_tickets(
            client,
            token=token,
            department=str(args.department),
            fresh=bool(args.fresh),
        )

        # 第三点五步：只有多工单分散写场景需要额外准备一组工单。
        if str(args.case) == "escalate_many":
            distributed_escalate_tickets = _seed_distributed_escalate_tickets(
                client,
                token=token,
                department=str(args.department),
                ticket_count=int(args.distributed_tickets),
                fresh=bool(args.fresh),
            )

    output_payload = {
        "api_base_url": base_url,
        "username": str(args.username),
        "department": str(args.department),
        "access_token": token,
        "token_prefix": token[:12] + "...",
        "seeded_tickets": seeded_tickets,
        "distributed_escalate_tickets": distributed_escalate_tickets,
        "generated_at": int(time.time()),
    }
    _write_output(str(args.output), output_payload)

    # 第四步：如果指定了压测 case，则在准备完成后追加执行。
    if str(args.case) == "health":
        asyncio.run(
            _run_health_case(
                base_url=base_url,
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )
    elif str(args.case) == "lookup":
        lookup_ticket_id = str(seeded_tickets.get("lookup", {}).get("ticket_id") or "").strip()
        if not lookup_ticket_id:
            raise RuntimeError("lookup_ticket_id_missing")
        asyncio.run(
            _run_lookup_case(
                base_url=base_url,
                token=token,
                ticket_id=lookup_ticket_id,
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )
    elif str(args.case) == "comment":
        comment_ticket_id = str(seeded_tickets.get("comment", {}).get("ticket_id") or "").strip()
        if not comment_ticket_id:
            raise RuntimeError("comment_ticket_id_missing")
        asyncio.run(
            _run_comment_case(
                base_url=base_url,
                token=token,
                ticket_id=comment_ticket_id,
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )
    elif str(args.case) == "escalate":
        escalate_ticket_id = str(seeded_tickets.get("escalate", {}).get("ticket_id") or "").strip()
        if not escalate_ticket_id:
            raise RuntimeError("escalate_ticket_id_missing")
        asyncio.run(
            _run_escalate_case(
                base_url=base_url,
                token=token,
                ticket_id=escalate_ticket_id,
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )
    elif str(args.case) == "escalate_many":
        distributed_ticket_ids = [
            str(item.get("ticket_id") or "").strip()
            for item in distributed_escalate_tickets
            if isinstance(item, dict)
        ]
        if not distributed_ticket_ids:
            raise RuntimeError("distributed_escalate_ticket_ids_missing")
        asyncio.run(
            _run_escalate_many_case(
                base_url=base_url,
                token=token,
                ticket_ids=distributed_ticket_ids,
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )
    elif str(args.case) == "create":
        asyncio.run(
            _run_create_case(
                base_url=base_url,
                token=token,
                department=str(args.department),
                total_requests=int(args.requests),
                concurrency=int(args.concurrency),
                timeout=float(args.timeout),
            )
        )

    print("[DONE] 并发测试准备完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
