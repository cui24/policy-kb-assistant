"""
L3/L4 UI API 客户端：作为 Streamlit 页面访问后端 HTTP API 的唯一入口。

一、程序目标
1. 把 UI 中所有与后端交互的逻辑收敛到一个文件里。
2. 统一处理 `base_url`、请求头、超时和错误提示。
3. 让 `src/ui/app.py` 只关心“拿到什么 JSON、如何展示”。

二、核心调用顺序
1. 页面创建 `PolicyAPIClient(base_url=...)`
2. UI 调用：
   - `ask(...)`
   - `agent(...)`
   - `create_ticket(...)`
   - `list_tickets(...)`
   - `get_ticket(...)`
   - `update_ticket(...)`
   - `add_ticket_comment(...)`
   - `escalate_ticket(...)`
   - `cancel_ticket(...)`
   - `list_kb_queries(...)`
   - `get_kb_query(...)`
   - `list_audit_logs(...)`
3. 所有方法最终都会走 `_request(...)`
4. `_request(...)` 负责：
   - 拼接 URL
   - 发 HTTP 请求
   - 解析 JSON
   - 将 4xx/5xx 转成统一异常
5. L4 中 `agent(...)` 还支持：
   - `draft_id`
   - `fields`
   让 UI 可以继续提交草稿，而不必让用户重说一遍

三、输入输出
1. 输入：
   - `base_url: str`
   - 各接口的请求 JSON 字段
2. 输出：
   - 成功：API 返回的 JSON（`dict` 或 `list[dict]`）
   - 失败：抛出 `APIClientError`

四、错误处理策略
1. 连接失败：提示 API 未启动或地址错误
2. 401：提示鉴权失败
3. 422：提示请求体字段不符合 schema
4. 500：提示后端内部错误
5. 其他非 2xx：统一转成可展示的错误消息
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv


class APIClientError(RuntimeError):
    """统一表示 UI 到 API 的调用失败。"""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        detail: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail



def _normalize_base_url(raw_base_url: str) -> str:
    """去掉末尾斜杠，保证后续拼接路径时不会重复。"""
    return (raw_base_url or "http://localhost:8080").strip().rstrip("/")



def _parse_response_body(response: httpx.Response) -> Any:
    """尽量把响应体解析成 JSON；若不是 JSON，则退回纯文本。"""
    content_type = str(response.headers.get("content-type") or "")
    if "application/json" in content_type:
        try:
            return response.json()
        except Exception:
            return response.text
    return response.text



def _default_error_message(status_code: int) -> str:
    """根据状态码生成更适合 UI 展示的错误说明。"""
    if status_code == 401:
        return "API 鉴权失败（401）。请检查 API Key。"
    if status_code == 403:
        return "当前请求被拒绝（403）。你没有权限执行该操作。"
    if status_code == 404:
        return "请求的 API 路径不存在（404）。请确认后端已升级到当前版本。"
    if status_code == 422:
        return "请求参数不符合接口 schema（422）。请检查字段名和必填项。"
    if status_code >= 500:
        return "后端内部错误（5xx）。请查看 API 服务日志。"
    return f"API 请求失败（{status_code}）。"


class PolicyAPIClient:
    """封装 L2 API 的同步调用。"""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float = 45.0,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        env_base_url = os.getenv("POLICY_API_BASE_URL", "http://localhost:8080")
        env_api_key = str(os.getenv("POLICY_API_KEY") or "").strip()
        self.base_url = _normalize_base_url(base_url or env_base_url)
        self.timeout_seconds = float(timeout_seconds)
        normalized_api_key = None if api_key is None else api_key.strip()
        self.api_key = normalized_api_key if normalized_api_key else env_api_key

    def _build_headers(self) -> dict[str, str]:
        """构造请求头；若配置了 API Key，则自动附加。"""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """统一发请求、解析结果并转换错误。"""
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.request(
                    method=method.upper(),
                    url=url,
                    headers=self._build_headers(),
                    json=json_body,
                    params=params,
                )
        except httpx.RequestError as exc:
            raise APIClientError(
                message="无法连接到 L2 API。请确认 `make api` 正在运行，且 base_url 正确。",
                detail=str(exc),
            ) from exc

        parsed_body = _parse_response_body(response)
        if response.status_code >= 400:
            raise APIClientError(
                message=_default_error_message(response.status_code),
                status_code=response.status_code,
                detail=parsed_body,
            )

        return parsed_body

    def health(self) -> dict[str, Any]:
        """调用 `/health`。"""
        data = self._request("GET", "/health")
        return data if isinstance(data, dict) else {"raw": data}

    def ask(
        self,
        question: str,
        user: str | None = None,
        department: str | None = None,
    ) -> dict[str, Any]:
        """调用 `/ask`。"""
        data = self._request(
            "POST",
            "/ask",
            json_body={
                "question": question,
                "user": user,
                "department": department,
            },
        )
        return data if isinstance(data, dict) else {"raw": data}

    def agent(
        self,
        text: str,
        user: str | None = None,
        department: str | None = None,
        draft_id: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """调用 `/agent`。"""
        data = self._request(
            "POST",
            "/agent",
            json_body={
                "text": text,
                "user": user,
                "department": department,
                "draft_id": draft_id,
                "fields": fields,
            },
        )
        return data if isinstance(data, dict) else {"raw": data}

    def create_ticket(
        self,
        title: str,
        description: str,
        creator: str | None = None,
        department: str | None = None,
        category: str = "other",
        priority: str = "P2",
        contact: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """调用 `POST /tickets`。"""
        data = self._request(
            "POST",
            "/tickets",
            json_body={
                "creator": creator,
                "department": department,
                "category": category,
                "priority": priority,
                "title": title,
                "description": description,
                "contact": contact,
                "context": context or {},
            },
        )
        return data if isinstance(data, dict) else {"raw": data}

    def list_tickets(self, status: str | None = None) -> list[dict[str, Any]]:
        """调用 `GET /tickets`。"""
        params = None if not status else {"status": status}
        data = self._request("GET", "/tickets", params=params)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """调用 `GET /tickets/{ticket_id}`。"""
        data = self._request("GET", f"/tickets/{ticket_id}")
        return data if isinstance(data, dict) else {"raw": data}

    def update_ticket(
        self,
        ticket_id: str,
        status: str,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """调用 `PATCH /tickets/{ticket_id}`。"""
        data = self._request(
            "PATCH",
            f"/tickets/{ticket_id}",
            json_body={
                "status": status,
                "actor": actor,
            },
        )
        return data if isinstance(data, dict) else {"raw": data}

    def add_ticket_comment(
        self,
        ticket_id: str,
        comment: str,
        actor: str | None = None,
    ) -> dict[str, Any]:
        """调用 `POST /tickets/{ticket_id}/comments`。"""
        data = self._request(
            "POST",
            f"/tickets/{ticket_id}/comments",
            json_body={
                "actor": actor,
                "comment": comment,
            },
        )
        return data if isinstance(data, dict) else {"raw": data}

    def escalate_ticket(
        self,
        ticket_id: str,
        actor: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """调用 `POST /tickets/{ticket_id}/escalate`。"""
        data = self._request(
            "POST",
            f"/tickets/{ticket_id}/escalate",
            json_body={
                "actor": actor,
                "reason": reason,
            },
        )
        return data if isinstance(data, dict) else {"raw": data}

    def cancel_ticket(
        self,
        ticket_id: str,
        actor: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """调用 `POST /tickets/{ticket_id}/cancel`。"""
        data = self._request(
            "POST",
            f"/tickets/{ticket_id}/cancel",
            json_body={
                "actor": actor,
                "reason": reason,
            },
        )
        return data if isinstance(data, dict) else {"raw": data}

    def list_kb_queries(
        self,
        user: str | None = None,
        department: str | None = None,
        request_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """调用 `GET /kb_queries`。"""
        params = {
            "user": user,
            "department": department,
            "request_id": request_id,
            "limit": limit,
        }
        normalized_params = {key: value for key, value in params.items() if value not in (None, "", [])}
        data = self._request("GET", "/kb_queries", params=normalized_params)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def get_kb_query(self, request_id: str) -> dict[str, Any]:
        """调用 `GET /kb_queries/{request_id}`。"""
        data = self._request("GET", f"/kb_queries/{request_id}")
        return data if isinstance(data, dict) else {"raw": data}

    def list_audit_logs(
        self,
        request_id: str | None = None,
        ticket_id: str | None = None,
        action_type: str | None = None,
        actor: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """调用 `GET /audit_logs`。"""
        params = {
            "request_id": request_id,
            "ticket_id": ticket_id,
            "action_type": action_type,
            "actor": actor,
            "limit": limit,
        }
        normalized_params = {key: value for key, value in params.items() if value not in (None, "", [])}
        data = self._request("GET", "/audit_logs", params=normalized_params)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []
