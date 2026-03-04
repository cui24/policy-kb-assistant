"""
L5 Agent Planner：负责把自然语言请求转换成结构化 `ToolPlan`。

一、程序目标
1. 在 `/agent` 中提供“全局分支规划 + ticket 子规划”的最小双层 Planner。
2. 通过固定 prompt + JSON 解析 + 一次修复重试，尽量减少模型跑偏。
3. 为 `/agent` 提供可切换的 `rules / llm / hybrid` 路由模式。

二、当前范围
1. Global Planner 只做分支选择：`kb_answer / create_ticket / ticket_tool_planner / continue_ticket_draft`。
2. ticket 工具的细粒度落参仍交给 ticket 子规划器处理。
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.api.schemas import ToolPlan


class PlannerError(RuntimeError):
    """Planner 失败；`fallback_eligible` 表示 hybrid 模式可否安全回退。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "planner_failed",
        fallback_eligible: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.fallback_eligible = fallback_eligible


def agent_planner_mode() -> str:
    """读取当前 Agent Planner 模式。"""
    raw_value = str(os.getenv("AGENT_PLANNER_MODE", "rules")).strip().lower()
    if raw_value in {"rules", "llm", "hybrid"}:
        return raw_value
    return "rules"


def planner_model_name() -> str:
    """读取 Planner 使用的模型名。"""
    return str(os.getenv("AGENT_PLANNER_MODEL", os.getenv("OPENAI_MODEL", "deepseek-chat")))


def planner_backend() -> str:
    """读取 Planner 后端实现。"""
    raw_value = str(os.getenv("AGENT_PLANNER_BACKEND", "raw")).strip().lower()
    if raw_value in {"raw", "langchain_structured", "langchain_tools"}:
        return raw_value
    return "raw"


def planner_timeout_seconds() -> float:
    """读取 Planner 请求超时。"""
    raw_value = os.getenv("AGENT_PLANNER_TIMEOUT_SECONDS", os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
    try:
        return max(5.0, min(float(raw_value), 60.0))
    except Exception:
        return 20.0


def _extract_json_object_text(text: str) -> str | None:
    """从模型输出中尽量提取最外层 JSON 对象文本。"""
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return None


def _validate_tool_plan_from_json(raw_text: str) -> ToolPlan:
    """把原始 JSON 文本校验成 `ToolPlan`。"""
    json_text = _extract_json_object_text(raw_text)
    if not json_text:
        raise ValueError("no_json_object")
    if hasattr(ToolPlan, "model_validate_json"):
        return ToolPlan.model_validate_json(json_text)
    return ToolPlan.parse_raw(json_text)


def _call_planner_llm(system_prompt: str, user_prompt: str) -> str:
    """调用 LLM 并返回原始文本。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise PlannerError("missing OPENAI_API_KEY", code="missing_api_key", fallback_eligible=True)

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=planner_timeout_seconds(),
    )
    try:
        response = client.chat.completions.create(
            model=planner_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=260,
        )
    except Exception as exc:
        raise PlannerError(str(exc), code="llm_call_failed", fallback_eligible=True) from exc

    return (response.choices[0].message.content or "").strip()


def _build_openai_client_kwargs(api_key: str) -> dict[str, Any]:
    """统一整理 OpenAI 兼容客户端配置。"""
    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        "timeout": planner_timeout_seconds(),
    }


def _call_planner_langchain_structured(system_prompt: str, user_prompt: str) -> ToolPlan:
    """使用 LangChain structured output 直接产出 `ToolPlan`。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise PlannerError("missing OPENAI_API_KEY", code="missing_api_key", fallback_eligible=True)

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise PlannerError(
            "langchain_openai_unavailable",
            code="langchain_unavailable",
            fallback_eligible=True,
        ) from exc

    llm = _build_langchain_chat_model(ChatOpenAI, api_key)
    try:
        structured_llm = llm.with_structured_output(ToolPlan)
        result = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:
        raise PlannerError(
            str(exc),
            code="langchain_structured_failed",
            fallback_eligible=True,
        ) from exc

    return _validate_tool_plan_from_payload(
        result,
        invalid_code="langchain_structured_invalid",
    )


def _build_langchain_chat_model(chat_openai_cls, api_key: str):
    """构造一个兼容新版/旧版参数名的 LangChain ChatOpenAI 实例。"""
    llm_kwargs = _build_openai_client_kwargs(api_key)
    llm_kwargs.update(
        {
            "model": planner_model_name(),
            "temperature": 0.0,
            "max_tokens": 260,
        }
    )
    try:
        return chat_openai_cls(**llm_kwargs)
    except TypeError:
        legacy_kwargs = {
            "model": planner_model_name(),
            "temperature": 0.0,
            "max_tokens": 260,
            "openai_api_key": api_key,
            "openai_api_base": os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
            "request_timeout": planner_timeout_seconds(),
        }
        return chat_openai_cls(**legacy_kwargs)


def _validate_tool_plan_from_payload(payload: Any, *, invalid_code: str) -> ToolPlan:
    """把 LangChain 返回的对象统一整理成合法 `ToolPlan`。"""
    if isinstance(payload, ToolPlan):
        return payload
    if isinstance(payload, dict):
        try:
            if hasattr(ToolPlan, "model_validate"):
                return ToolPlan.model_validate(payload)
            return ToolPlan.parse_obj(payload)
        except Exception as exc:
            raise PlannerError(
                invalid_code,
                code=invalid_code,
                fallback_eligible=True,
            ) from exc

    raise PlannerError(
        f"{invalid_code}_type",
        code=invalid_code,
        fallback_eligible=True,
    )


def _tool_specs_from_contracts(tools_json: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把统一工具契约转成 LangChain/OpenAI tool calling 可接受的函数定义。"""
    tool_specs: list[dict[str, Any]] = []
    for item in tools_json:
        tool_specs.append(
            {
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "description": str(item.get("description") or ""),
                    "parameters": dict(item.get("input_schema") or {"type": "object", "properties": {}}),
                },
            }
        )
    return tool_specs


def _extract_langchain_tool_calls(message: Any) -> list[dict[str, Any]]:
    """从 LangChain AIMessage 中提取标准化后的 tool_calls。"""
    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list):
        return [item for item in tool_calls if isinstance(item, dict)]

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if not isinstance(additional_kwargs, dict):
        return []
    raw_tool_calls = additional_kwargs.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []

    normalized_calls: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        function_payload = item.get("function")
        if not isinstance(function_payload, dict):
            continue
        args_payload = function_payload.get("arguments")
        try:
            args = json.loads(args_payload) if isinstance(args_payload, str) else dict(args_payload or {})
        except Exception:
            args = {}
        normalized_calls.append(
            {
                "name": function_payload.get("name"),
                "args": args if isinstance(args, dict) else {},
            }
        )
    return normalized_calls


def _tool_plan_from_langchain_tool_calls(tool_calls: list[dict[str, Any]], tools_json: list[dict[str, Any]]) -> ToolPlan:
    """把 LangChain tool_calls 转成当前系统统一的 `ToolPlan`。"""
    if not tool_calls:
        raise PlannerError(
            "langchain_tools_missing_tool_call",
            code="langchain_tools_missing_tool_call",
            fallback_eligible=True,
        )

    first_call = tool_calls[0]
    tool_name = str(first_call.get("name") or "").strip()
    args = first_call.get("args")
    if not tool_name:
        raise PlannerError(
            "langchain_tools_missing_name",
            code="langchain_tools_invalid",
            fallback_eligible=True,
        )
    if not isinstance(args, dict):
        args = {}

    matching_tool = next((item for item in tools_json if str(item.get("name") or "") == tool_name), None)
    if matching_tool is None:
        raise PlannerError(
            "langchain_tools_unknown_tool",
            code="langchain_tools_invalid",
            fallback_eligible=True,
        )

    return ToolPlan(
        tool=tool_name,
        args=args,
        need_confirmation=str(matching_tool.get("risk_level") or "").upper() == "HIGH",
        missing_fields=[],
    )


def _call_planner_langchain_tools(
    system_prompt: str,
    user_prompt: str,
    tools_json: list[dict[str, Any]],
) -> ToolPlan:
    """使用 LangChain tool calling 产出 `ToolPlan`，但不自动执行工具。"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise PlannerError("missing OPENAI_API_KEY", code="missing_api_key", fallback_eligible=True)

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise PlannerError(
            "langchain_openai_unavailable",
            code="langchain_unavailable",
            fallback_eligible=True,
        ) from exc

    llm = _build_langchain_chat_model(ChatOpenAI, api_key)
    tool_specs = _tool_specs_from_contracts(tools_json)
    tool_system_prompt = (
        system_prompt
        + "当前是 tool-calling 模式：你必须且只能调用一个工具，不要输出普通文本；不要自己执行工具。"
    )
    try:
        try:
            bound_llm = llm.bind_tools(tool_specs, tool_choice="required")
        except TypeError:
            bound_llm = llm.bind_tools(tool_specs)
        message = bound_llm.invoke(
            [
                {"role": "system", "content": tool_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:
        raise PlannerError(
            str(exc),
            code="langchain_tools_failed",
            fallback_eligible=True,
        ) from exc

    tool_calls = _extract_langchain_tool_calls(message)
    if not tool_calls:
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            try:
                return _validate_tool_plan_from_json(content)
            except Exception:
                pass
    return _tool_plan_from_langchain_tool_calls(tool_calls, tools_json)


def _tools_json_text(tools_json: list[dict[str, Any]]) -> str:
    """把工具清单稳定序列化为 prompt 文本。"""
    return json.dumps(tools_json, ensure_ascii=False, sort_keys=True)


def _agent_data_path(filename: str) -> Path:
    """返回 Agent 本地数据文件路径。"""
    return Path(__file__).resolve().parents[2] / "data" / "agent" / filename


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """读取一个本地 JSONL 文件；缺失时返回空列表。"""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


@lru_cache(maxsize=1)
def load_global_planner_tool_docs() -> list[dict[str, Any]]:
    """读取 Global Planner 的 tool-doc 文档库。"""
    return _load_jsonl_records(_agent_data_path("tool_docs.jsonl"))


@lru_cache(maxsize=1)
def load_global_planner_regression_cases() -> list[dict[str, Any]]:
    """读取 Global Planner 的模糊表达回归集。"""
    return _load_jsonl_records(_agent_data_path("global_planner_regression_cases.jsonl"))


def _score_tool_doc(doc: dict[str, Any], user_text: str, context: dict[str, Any]) -> int:
    """用轻量关键词规则给 tool-doc 打分。"""
    normalized_text = str(user_text or "")
    tool_name = str(doc.get("tool_name") or "")
    keywords = doc.get("keywords") or []
    score = 0

    for keyword in keywords:
        normalized_keyword = str(keyword or "").strip()
        if normalized_keyword and normalized_keyword in normalized_text:
            score += 4 + min(len(normalized_keyword), 4)

    if tool_name and tool_name in normalized_text:
        score += 4

    has_ticket_id = bool(context.get("has_ticket_id"))
    has_draft_id = bool(context.get("has_draft_id"))
    draft_mode = bool(context.get("draft_mode"))
    ticket_tool_mode = bool(context.get("ticket_tool_mode"))

    if tool_name == "continue_ticket_draft" and (has_draft_id or draft_mode):
        score += 10
    if tool_name == "ticket_tool_planner" and (has_ticket_id or ticket_tool_mode):
        score += 10
    if tool_name == "kb_answer" and any(
        marker in normalized_text for marker in ("流程", "规定", "政策", "制度", "怎么", "SLA")
    ):
        score += 5
    if tool_name == "create_ticket" and any(
        marker in normalized_text for marker in ("报修", "工单", "提交", "处理", "维修")
    ):
        score += 3

    return score


def retrieve_global_planner_tool_docs(
    user_text: str,
    context: dict[str, Any] | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """对本地 tool-doc 做最小 top-k 检索，供 Global Planner prompt 使用。"""
    ctx = dict(context or {})
    corpus = load_global_planner_tool_docs()
    if not corpus:
        return []

    scored_items: list[tuple[int, int, dict[str, Any]]] = []
    for index, doc in enumerate(corpus):
        scored_items.append((_score_tool_doc(doc, user_text, ctx), index, doc))

    scored_items.sort(key=lambda item: (-item[0], item[1]))
    selected_docs = [doc for score, _, doc in scored_items if score > 0][: max(1, top_k)]

    if not selected_docs:
        fallback_tool_names: list[str]
        if bool(ctx.get("has_draft_id")) or bool(ctx.get("draft_mode")):
            fallback_tool_names = ["continue_ticket_draft", "create_ticket"]
        elif bool(ctx.get("has_ticket_id")) or bool(ctx.get("ticket_tool_mode")):
            fallback_tool_names = ["ticket_tool_planner", "lookup_ticket"]
        else:
            fallback_tool_names = ["kb_answer", "create_ticket"]

        for tool_name in fallback_tool_names:
            fallback_doc = next((doc for doc in corpus if str(doc.get("tool_name") or "") == tool_name), None)
            if fallback_doc and fallback_doc not in selected_docs:
                selected_docs.append(fallback_doc)
            if len(selected_docs) >= max(1, top_k):
                break

    return selected_docs[: max(1, top_k)]


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    """判断文本中是否命中任一标记词。"""
    normalized_text = str(text or "")
    return any(marker in normalized_text for marker in markers)


def _filter_tools_by_name(
    tools_json: list[dict[str, Any]],
    allowed_names: set[str],
) -> list[dict[str, Any]]:
    """按工具名筛选工具，保持原顺序；若筛空则回退原列表。"""
    selected_tools = [item for item in tools_json if str(item.get("name") or "") in allowed_names]
    return selected_tools or list(tools_json)


def _filter_tool_docs_for_available_tools(
    docs: list[dict[str, Any]],
    tools_json: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """让 prompt 中的 tool-doc 与当前可用工具集合保持一致。"""
    allowed_names = {str(item.get("name") or "") for item in tools_json}
    if not allowed_names:
        return docs

    filtered_docs = [doc for doc in docs if str(doc.get("tool_name") or "") in allowed_names]
    if filtered_docs:
        return filtered_docs

    fallback_docs: list[dict[str, Any]] = []
    for tool_name in allowed_names:
        matched_doc = next(
            (item for item in load_global_planner_tool_docs() if str(item.get("tool_name") or "") == tool_name),
            None,
        )
        if matched_doc and matched_doc not in fallback_docs:
            fallback_docs.append(matched_doc)
    return fallback_docs or docs


def _looks_like_kb_question(user_text: str) -> bool:
    """判断是否更像制度/流程类问答。"""
    return _contains_any_marker(
        user_text,
        ("流程", "规定", "政策", "制度", "SLA", "怎么", "如何", "多久响应"),
    )


def _looks_like_new_ticket_request(user_text: str) -> bool:
    """判断是否明确是“新建工单”请求。"""
    return _contains_any_marker(
        user_text,
        ("报修", "提交工单", "建单", "创建工单", "帮我报修", "需要上门", "维修", "处理一下"),
    )


def _looks_like_unresolved_ticket_follow_up(user_text: str) -> bool:
    """判断是否像“已有对象的跟进”，但当前未显式给出 ticket_id。"""
    reference_markers = (
        "那个工单",
        "那张单",
        "那单",
        "上一单",
        "上一个工单",
        "上一张单",
        "上次那个",
        "之前那个",
        "刚建的单",
        "刚才那张",
        "还是那张单",
        "上次报的",
    )
    follow_up_markers = (
        "补充",
        "补个",
        "补一句",
        "追加",
        "备注",
        "留言",
        "催办",
        "催一下",
        "加急",
        "升级",
        "取消",
        "撤销",
        "关掉",
        "关闭",
        "撤掉",
        "不用修了",
        "算了",
        "别弄了",
        "更新一下信息",
        "加一句话",
    )
    return _contains_any_marker(user_text, reference_markers) or (
        _contains_any_marker(user_text, ("工单", "那单", "单"))
        and _contains_any_marker(user_text, follow_up_markers)
    )


def _looks_like_unresolved_draft_follow_up(user_text: str) -> bool:
    """判断是否更像草稿续办或字段纠正。"""
    return _contains_any_marker(user_text, ("草稿", "draft")) or (
        _contains_any_marker(user_text, ("刚才", "上次", "继续", "还是"))
        and _contains_any_marker(user_text, ("填错", "改一下", "改地址", "地址", "联系方式", "电话"))
    )


def select_global_langchain_tool_candidates(
    user_text: str,
    tools_json: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """为 `langchain_tools` 收窄 Global Planner 的候选工具，减少无对象 follow-up 被误判成建单。"""
    ctx = dict(context or {})

    if bool(ctx.get("has_draft_id")) or bool(ctx.get("draft_mode")):
        return _filter_tools_by_name(tools_json, {"continue_ticket_draft"})
    if bool(ctx.get("has_ticket_id")) or bool(ctx.get("ticket_tool_mode")):
        return _filter_tools_by_name(tools_json, {"ticket_tool_planner"})

    if _looks_like_unresolved_draft_follow_up(user_text):
        return _filter_tools_by_name(tools_json, {"continue_ticket_draft", "ticket_tool_planner"})
    if _looks_like_unresolved_ticket_follow_up(user_text):
        return _filter_tools_by_name(tools_json, {"ticket_tool_planner"})
    if _looks_like_kb_question(user_text) and not _looks_like_new_ticket_request(user_text):
        return _filter_tools_by_name(tools_json, {"kb_answer"})
    if _looks_like_new_ticket_request(user_text):
        return _filter_tools_by_name(tools_json, {"create_ticket"})

    return list(tools_json)


def _tool_doc_context_text(docs: list[dict[str, Any]]) -> str:
    """把检索结果转成适合直接放入 prompt 的精简上下文。"""
    if not docs:
        return "[]"

    lines: list[str] = []
    for doc in docs:
        lines.append(
            json.dumps(
                {
                    "doc_id": doc.get("doc_id"),
                    "tool_name": doc.get("tool_name"),
                    "content": doc.get("content"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def build_global_planner_prompt(
    user_text: str,
    tools_json: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """构造全局 Planner prompt：只负责分支选择，不做 ticket 细粒度落参。"""
    ctx = dict(context or {})
    rag_docs = _filter_tool_docs_for_available_tools(
        retrieve_global_planner_tool_docs(user_text, ctx, top_k=3),
        tools_json,
    )
    system_prompt = (
        "你是企业内部助手的总规划器（Global Planner）。"
        "你的职责只有一件事：根据用户原话与系统提供的状态字段，在可用工具中选择一个最合适的 tool，并输出严格 JSON 的 ToolPlan。"
        "你不执行工具，不解释，不回答自然语言，不编造系统外事实。"
        "你只能从系统提供的 tools 列表中选择一个 tool，tool 必须与 tools 列表中的 name 完全一致，禁止输出列表外工具。"
        "你必须信任系统提供的状态字段，并遵守以下硬约束："
        "1) 不得编造或猜测 ticket_id / draft_id；若 has_ticket_id=true，必须使用 provided_ticket_id；若 has_draft_id=true，必须使用 provided_draft_id；"
        "2) 若用户请求涉及 cancel_ticket，系统后续会做确认态；当前全局规划阶段只需要在进入 ticket_tool_planner 时把原始文本转交给子规划器；"
        "3) 只输出 JSON，禁止 Markdown 与解释。"
        "system_state 中还可能包含 short_term_memory、memory_ticket_applied、memory_draft_applied；这些字段由系统提供，只能用于辅助理解上下文，不能覆盖系统已给定的 ID。"
        "最小决策策略必须严格按顺序执行："
        "A) 若 draft_mode=true 或 has_draft_id=true，则选择 continue_ticket_draft，"
        "args={\"draft_id\":provided_draft_id,\"fields\":<保守抽取的 fields>}；"
        "B) 否则若 ticket_tool_mode=true 或 has_ticket_id=true，则选择 ticket_tool_planner，"
        "args={\"ticket_id\":provided_ticket_id,\"raw_text\":<用户原话>}；"
        "C) 否则若用户明显是在咨询制度、政策、流程、规定等知识问答，则选择 kb_answer，"
        "args={\"query\":<用户原话>}；"
        "D) 否则默认 create_ticket，args={\"text\":<用户原话>,\"fields\":<保守抽取的 fields>}。"
        "fields 只能是 JSON 对象；不确定的字段不要编造。location/contact/title/description 能明确抽取才填，不能确定就留空。"
        "missing_fields 在当前阶段通常输出空数组。"
        "输出格式固定为："
        '{"tool":"kb_answer","args":{"query":"示例"},"need_confirmation":false,"missing_fields":[]}'
    )
    user_prompt = (
        f"tools:\n{_tools_json_text(tools_json)}\n\n"
        "retrieved_tool_docs:\n"
        f"{_tool_doc_context_text(rag_docs)}\n\n"
        "system_state:\n"
        f"{json.dumps(ctx, ensure_ascii=False, sort_keys=True)}\n\n"
        f"user_text:\n{user_text}\n\n"
        "请根据 system 规则，只输出严格 JSON。"
    )
    return system_prompt, user_prompt


def build_global_repair_prompt(
    user_text: str,
    tools_json: list[dict[str, Any]],
    context: dict[str, Any] | None,
    invalid_output: str,
) -> tuple[str, str]:
    """构造全局 Planner 的修复 prompt：只修复结构，不重新自由发挥。"""
    ctx = dict(context or {})
    rag_docs = _filter_tool_docs_for_available_tools(
        retrieve_global_planner_tool_docs(user_text, ctx, top_k=3),
        tools_json,
    )
    system_prompt = (
        "你是 ToolPlan JSON 修复器。"
        "你的任务不是重新解释业务，而是把给定的错误输出修复成严格 JSON。"
        "你必须只输出一个合法的 ToolPlan JSON 对象，不得输出解释。"
        "tool 只能是提供 tools 列表中的 name；provided_ticket_id 与 provided_draft_id 若存在，必须原样使用。"
    )
    user_prompt = (
        f"tools:\n{_tools_json_text(tools_json)}\n\n"
        "retrieved_tool_docs:\n"
        f"{_tool_doc_context_text(rag_docs)}\n\n"
        "system_state:\n"
        f"{json.dumps(ctx, ensure_ascii=False, sort_keys=True)}\n\n"
        f"user_text:\n{user_text}\n\n"
        f"invalid_output:\n{invalid_output}\n\n"
        "请修复为严格合法的 ToolPlan JSON。"
    )
    return system_prompt, user_prompt


def build_ticket_subplanner_prompt(
    user_text: str,
    provided_ticket_id: str,
    tools_json: list[dict[str, Any]],
) -> tuple[str, str]:
    """构造“既有工单工具”的行为对齐版 Planner prompt。"""
    system_prompt = (
        "你是企业工单系统的既有工单工具规划器。"
        "你的职责是：基于用户原话和给定 ticket_id，输出一个严格 JSON 的 ToolPlan。"
        "你不执行工具，不解释，不输出 Markdown。"
        "你必须只从提供的 tools 列表中选择 tool。"
        "你必须直接使用提供的 ticket_id，禁止改写、猜测或编造新的 ticket_id。"
        "动作选择必须按以下优先级："
        "1) 若命中取消关键词【取消、撤销、关闭工单】=> cancel_ticket；"
        "2) 否则若命中催办关键词【催办、催一下、加急、升级】=> escalate_ticket；"
        "3) 否则若命中补充关键词【补充、补充说明、追加、备注、留言】=> add_ticket_comment；"
        "4) 否则若命中查询关键词【查工单、查一下、查询、进度、状态】=> lookup_ticket；"
        "5) 否则默认 lookup_ticket。"
        "你必须先生成 text_tail：从用户原话中删除 ticket_id，压缩多余空格，并去掉首尾空格与这些标点【：:，,。】。"
        "如果 tool=lookup_ticket，则 args 只能包含 ticket_id。"
        "如果 tool=add_ticket_comment，comment 使用 text_tail；若 text_tail 为空，则 comment='用户补充说明。'。"
        "如果 tool=escalate_ticket，reason 使用 text_tail；若 text_tail 为空，则 reason='用户请求催办。'。"
        "如果 tool=cancel_ticket，reason 使用 text_tail；若 text_tail 为空，则 reason='用户请求取消工单。'。"
        "如果 tool=cancel_ticket，则 need_confirmation 必须为 true；其它工具必须为 false。"
        "在当前行为对齐模式下，missing_fields 必须输出空数组。"
        "输出格式固定为："
        '{"tool":"lookup_ticket","args":{"ticket_id":"TCK-2026-XXXXXX"},"need_confirmation":false,"missing_fields":[]}'
    )
    user_prompt = (
        f"tools:\n{_tools_json_text(tools_json)}\n\n"
        f"ticket_id:\n{provided_ticket_id}\n\n"
        f"user_text:\n{user_text}\n\n"
        "请根据规则，只输出严格 JSON。"
    )
    return system_prompt, user_prompt


def build_repair_prompt(
    user_text: str,
    provided_ticket_id: str,
    tools_json: list[dict[str, Any]],
    invalid_output: str,
) -> tuple[str, str]:
    """构造修复 prompt：只修复结构，不重做自由发挥。"""
    system_prompt = (
        "你是 ToolPlan JSON 修复器。"
        "你的任务不是重新解释业务，而是把给定的错误输出修复成严格 JSON。"
        "你必须只输出一个合法的 ToolPlan JSON 对象，不得输出解释。"
        "tool 只能是提供 tools 列表中的 name，ticket_id 必须使用提供的 ticket_id。"
    )
    user_prompt = (
        f"tools:\n{_tools_json_text(tools_json)}\n\n"
        f"ticket_id:\n{provided_ticket_id}\n\n"
        f"user_text:\n{user_text}\n\n"
        f"invalid_output:\n{invalid_output}\n\n"
        "请修复为严格合法的 ToolPlan JSON。"
    )
    return system_prompt, user_prompt


def run_global_planner(
    user_text: str,
    tools_json: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> ToolPlan:
    """执行一次全局 Planner，必要时自动做一次修复重试。"""
    backend = planner_backend()
    planner_tools_json = (
        select_global_langchain_tool_candidates(user_text, tools_json, context)
        if backend == "langchain_tools"
        else list(tools_json)
    )
    system_prompt, user_prompt = build_global_planner_prompt(
        user_text=user_text,
        tools_json=planner_tools_json,
        context=context,
    )
    if backend == "langchain_structured":
        return _call_planner_langchain_structured(system_prompt, user_prompt)
    if backend == "langchain_tools":
        return _call_planner_langchain_tools(system_prompt, user_prompt, planner_tools_json)
    first_output = _call_planner_llm(system_prompt, user_prompt)
    try:
        return _validate_tool_plan_from_json(first_output)
    except Exception:
        repair_system_prompt, repair_user_prompt = build_global_repair_prompt(
            user_text=user_text,
            tools_json=planner_tools_json,
            context=context,
            invalid_output=first_output,
        )
        second_output = _call_planner_llm(repair_system_prompt, repair_user_prompt)
        try:
            return _validate_tool_plan_from_json(second_output)
        except Exception as exc:
            raise PlannerError(
                "global_planner_output_invalid_after_repair",
                code="global_repair_failed",
                fallback_eligible=True,
            ) from exc


def run_ticket_tool_planner(
    user_text: str,
    provided_ticket_id: str,
    tools_json: list[dict[str, Any]],
) -> ToolPlan:
    """执行一次 ticket 工具 Planner，必要时自动做一次修复重试。"""
    backend = planner_backend()
    system_prompt, user_prompt = build_ticket_subplanner_prompt(
        user_text=user_text,
        provided_ticket_id=provided_ticket_id,
        tools_json=tools_json,
    )
    if backend == "langchain_structured":
        return _call_planner_langchain_structured(system_prompt, user_prompt)
    if backend == "langchain_tools":
        return _call_planner_langchain_tools(system_prompt, user_prompt, tools_json)
    first_output = _call_planner_llm(system_prompt, user_prompt)
    try:
        return _validate_tool_plan_from_json(first_output)
    except Exception:
        repair_system_prompt, repair_user_prompt = build_repair_prompt(
            user_text=user_text,
            provided_ticket_id=provided_ticket_id,
            tools_json=tools_json,
            invalid_output=first_output,
        )
        second_output = _call_planner_llm(repair_system_prompt, repair_user_prompt)
        try:
            return _validate_tool_plan_from_json(second_output)
        except Exception as exc:
            raise PlannerError(
                "planner_output_invalid_after_repair",
                code="repair_failed",
                fallback_eligible=True,
            ) from exc
