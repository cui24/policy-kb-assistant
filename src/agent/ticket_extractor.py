"""
L2 工单字段抽取器：把自然语言报修描述转换成结构化工单字段。

一、程序目标
1. 接收用户自然语言描述。
2. 尝试用 LLM 抽取 `category/priority/title/contact/location` 等字段。
3. 若 LLM 不可用或输出不稳定，则回退到启发式抽取。
4. 返回结构化字段与 `missing_fields`，供 `/agent` 决定是否创建工单。

二、调用顺序
1. `extract_ticket_payload(...)`
2. 先走 `_extract_with_llm(...)`
3. 失败则走 `_heuristic_extract(...)`
4. 最后 `_normalize_payload(...)` 统一字段

三、输入输出
1. 输入：用户原始文本、默认 user/department。
2. 输出：
   {
     "department": str,
     "category": str,
     "priority": str,
     "title": str,
     "description": str,
     "contact": str | None,
     "location": str | None,
     "missing_fields": list[str],
     "extractor": "llm" | "heuristic"
   }
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


_ACTION_CATEGORY_HINTS = {
    "network": ("网络", "上不了网", "连不上", "校园网", "wifi", "无线网", "认证"),
    "account": ("账号", "密码", "权限", "登录", "认证", "开通"),
    "dorm": ("宿舍", "公寓", "床位", "寝室"),
}



def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中容错提取最外层 JSON 对象。"""
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                return data if isinstance(data, dict) else None
            except Exception:
                return None
    return None



def _pick_category(text: str) -> str:
    """根据关键词估计工单类别。"""
    lowered = (text or "").lower()
    for category, terms in _ACTION_CATEGORY_HINTS.items():
        if any(term.lower() in lowered for term in terms):
            return category
    return "other"



def _pick_department(category: str, default_department: str | None) -> str:
    """根据类别推断处理部门。"""
    if default_department:
        return default_department
    if category in {"network", "account"}:
        return "IT"
    if category == "dorm":
        return "Admin"
    return "General"



def _pick_priority(text: str) -> str:
    """根据故障严重程度给出保守优先级。"""
    if any(token in text for token in ("紧急", "立刻", "完全无法", "中断")):
        return "P0"
    if any(token in text for token in ("无法", "连不上", "故障", "异常", "报修")):
        return "P1"
    return "P2"



def _extract_contact(text: str) -> str | None:
    """提取手机号或邮箱。"""
    phone_match = re.search(r"1\d{10}", text)
    if phone_match:
        return phone_match.group(0)
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if email_match:
        return email_match.group(0)
    return None



def _extract_location(text: str) -> str | None:
    """提取地点信息。"""
    for pattern in (
        r"地点[：: ]*([^，。,.；;]+)",
        r"位置[：: ]*([^，。,.；;]+)",
        r"在([^，。,.；;]*校区[^，。,.；;]*)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    for campus in ("金明校区", "明伦校区", "郑州校区"):
        if campus in text:
            return campus
    return None



def _build_title(category: str, location: str | None) -> str:
    """生成适合作为工单标题的短句。"""
    title_prefix = {
        "network": "网络故障报修",
        "account": "账号权限支持",
        "dorm": "宿舍事务咨询",
        "other": "通用服务请求",
    }.get(category, "通用服务请求")
    if location:
        return f"{location}{title_prefix}"
    return title_prefix



def _heuristic_extract(text: str, default_department: str | None) -> dict[str, Any]:
    """当 LLM 不可用时，用可解释的启发式规则兜底。"""
    category = _pick_category(text)
    location = _extract_location(text)
    contact = _extract_contact(text)
    payload = {
        "department": _pick_department(category, default_department),
        "category": category,
        "priority": _pick_priority(text),
        "title": _build_title(category, location),
        "description": text.strip(),
        "contact": contact,
        "location": location,
        "missing_fields": [],
        "extractor": "heuristic",
    }
    return payload



def _extract_with_llm(text: str, default_department: str | None) -> dict[str, Any] | None:
    """用 LLM 做结构化抽取；失败则返回 `None`。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
    )
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    department_hint = default_department or "未指定"

    system_prompt = (
        "你是企业服务台工单抽取器。"
        "请从用户报修/求助描述中提取结构化字段。"
        "你必须只输出严格 JSON，不要 Markdown，不要额外说明。"
        "输出格式："
        '{"department":"IT","category":"network","priority":"P1","title":"...","description":"...","contact":"...","location":"...","missing_fields":["contact"]}'
        "若某字段不存在，填 null；缺失的必填字段放进 missing_fields。"
        "必填字段只有：contact、location。"
    )
    user_prompt = (
        f"默认部门：{department_hint}\n"
        f"用户原话：{text}\n"
        "请抽取工单字段并输出 JSON。"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
    except Exception:
        return None

    content = (response.choices[0].message.content or "").strip()
    data = _extract_json_object(content)
    if not isinstance(data, dict):
        return None
    data["extractor"] = "llm"
    return data



def _normalize_payload(raw: dict[str, Any], text: str, default_department: str | None) -> dict[str, Any]:
    """把 LLM 或启发式结果统一收敛成稳定字段。"""
    category = str(raw.get("category") or _pick_category(text)).strip() or "other"
    location = raw.get("location")
    location = str(location).strip() if isinstance(location, str) and location.strip() else _extract_location(text)
    contact = raw.get("contact")
    contact = str(contact).strip() if isinstance(contact, str) and str(contact).strip() else _extract_contact(text)
    priority = str(raw.get("priority") or _pick_priority(text)).strip() or "P2"
    department = str(raw.get("department") or _pick_department(category, default_department)).strip() or _pick_department(category, default_department)
    title = str(raw.get("title") or _build_title(category, location)).strip() or _build_title(category, location)
    description = str(raw.get("description") or text).strip() or text.strip()

    missing_fields: list[str] = []
    for field_name in ("location", "contact"):
        field_value = location if field_name == "location" else contact
        if not field_value:
            missing_fields.append(field_name)

    llm_missing = raw.get("missing_fields")
    if isinstance(llm_missing, list):
        for item in llm_missing:
            if isinstance(item, str) and item not in missing_fields:
                missing_fields.append(item)

    return {
        "department": department,
        "category": category,
        "priority": priority,
        "title": title,
        "description": description,
        "contact": contact,
        "location": location,
        "missing_fields": missing_fields,
        "extractor": str(raw.get("extractor") or "heuristic"),
    }



def extract_ticket_payload(
    text: str,
    default_user: str | None = None,
    default_department: str | None = None,
) -> dict[str, Any]:
    """对外统一入口：优先 LLM，失败则启发式，并补齐标准字段。"""
    llm_payload = _extract_with_llm(text, default_department)
    raw_payload = llm_payload or _heuristic_extract(text, default_department)
    normalized = _normalize_payload(raw_payload, text, default_department)
    normalized["creator"] = default_user or "anonymous"
    return normalized
