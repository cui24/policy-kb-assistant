"""
L0 回答程序：基于检索证据调用 LLM，输出带引用的严格 JSON。

一、程序目标
1. 接收用户问题和检索证据。
2. 强制模型只根据证据回答。
3. 把模型输出约束成统一 JSON 结构。
4. 若模型输出不可靠，则返回“证据不足”而不是胡编。

二、程序在整个项目中的位置
1. 上游输入来自 `retrieve(...)`
2. 下游输出给：
   - `src/cli/demo_cli.py`
   - `src/eval/run_regression.py`
   - 未来的 API 层

三、核心调用顺序
1. 主要调用入口：`answer_with_citations(question, evidence)`
2. 其内部顺序如下：
   2.1 `load_dotenv()`：加载模型配置
   2.2 `load_level_config(level)`：读取引用长度和生成参数
   2.3 校验 `evidence` 和 `OPENAI_API_KEY`
   2.4 把证据列表整理成 prompt 中的证据块文本
   2.5 构造 `system_prompt` 与 `user_prompt`
   2.6 初始化 `OpenAI(...)` 客户端
   2.7 依次执行重试策略中的模型调用：
       - 调 `_call_llm_final_only(...)`
       - 调 `_extract_json(...)`
       - 调 `_normalize_output(...)`
       - 必要时调 `_repair_output_to_json(...)`
       - 再次解析并校验
       - 如果模型拒答但检索结果同时满足分数、margin、词面重合三重门槛，则尝试程序侧抽取式兜底
   2.8 如果任一步得到有效结果则立即返回
   2.9 全部失败则返回拒答结果，并附带失败原因

四、主要函数的输入输出
1. `load_level_config(level: str) -> dict[str, Any]`
   - 输入：level 名称
   - 输出：配置字典

2. `_extract_json(text: str) -> dict[str, Any] | None`
   - 输入：模型原始文本输出
   - 输出：
     - 成功：解析后的字典
     - 失败：`None`
   - 作用：容忍模型在 JSON 前后夹带多余内容

3. `_call_llm_final_only(client, model, messages, max_tokens) -> str`
   - 输入：
     - `client`: `OpenAI`
     - `model`: `str`
     - `messages`: `list[{"role": str, "content": str}]`
     - `max_tokens`: `int`
   - 输出：模型返回的最终文本内容 `str`
   - 若接口超时或报错：抛出 `RuntimeError`

4. `_normalize_output(data, evidence, max_snippet_chars) -> dict[str, Any]`
   - 输入：
     - `data`: 模型解析出的 JSON 字典
     - `evidence`: 检索证据列表
     - `max_snippet_chars`: 引用最大长度
   - 输出：
     {
       "answer": str,
       "citations": list[{
         "doc_id": str,
         "page": int,
         "snippet": str
       }]
     }
   - 作用：把不稳定的模型输出收敛成固定 schema

5. `_repair_output_to_json(client, model, question, evidence_block, raw_output, max_tokens) -> str`
   - 输入：
     - `client`: `OpenAI`
     - `model`: `str`
     - `question`: `str`
     - `evidence_block`: `str`
     - `raw_output`: `str`
     - `max_tokens`: `int`
   - 输出：修复后的模型文本输出 `str`
   - 作用：当第一次输出不是严格 JSON 或结构不合格时，要求模型做一次“格式修复”

6. `_build_extractive_fallback(question, evidence, max_snippet_chars, min_score, min_margin, min_overlap) -> dict[str, Any] | None`
   - 输入：
     - `question`: 用户问题 `str`
     - `evidence`: 检索证据列表
     - `max_snippet_chars`: 引用最大长度
     - `min_score`: 最小检索分数阈值
     - `min_margin`: top1 相对 top2 的最小领先幅度
     - `min_overlap`: query 与 top1 证据的最小词面重合数
   - 输出：
     - 成功：程序侧构造的回答和 citation
     - 失败：`None`
   - 作用：当模型过度保守拒答，且可回答性闸门通过时，直接使用最高分证据做抽取式兜底

7. `answer_with_citations(question, evidence) -> dict[str, Any]`
   - 输入：
     - `question`: 用户问题 `str`
     - `evidence`: `list[dict]`，通常由 `retrieve(...)` 返回
   - `evidence` 中每项格式：
     {
       "score": float,
       "doc_id": str | None,
       "page": int | None,
       "snippet": str,
       "text": str
     }
   - 输出：
     {
       "answer": str,
       "citations": list[{
         "doc_id": str,
         "page": int,
         "snippet": str
       }],
       "meta": {
         "json_ok": bool,
         "repair_used": bool,
         "failure_reason": str | None,
         "attempt_stage": str
       }
     }

五、重试与兜底逻辑
1. 第一轮：主模型，正常 token
2. 第二轮：主模型，更大 token
3. 第三轮：fallback 模型
4. 任一轮如果输出不可解析或缺少必要 citations，会触发一次修复型提示
5. 对远程接口超时或异常，直接记录失败原因并继续按当前策略兜底
6. 如果仍失败，则直接返回拒答结果，并在 `meta.failure_reason` 中记录原因

六、程序可以理解成的伪代码
1. 读模型配置
2. 如果没证据，直接拒答
3. 把证据拼成 prompt
4. 调模型
5. 尝试解析 JSON
6. 如果成功，检查是否缺少 citations
7. 如果解析失败或结构不合格，触发一次修复型提示
8. 如果模型拒答但高分证据同时通过可回答性闸门，走程序侧抽取式兜底
9. 如果仍失败，返回拒答
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI


_QUERY_SPLIT_MARKERS = (
    "请概括要点",
    "请概括主要内容",
    "请列举条款中的要点",
    "请列举要点",
    "请概括",
    "请列举",
    "请按手册给出的方式概括步骤",
    "请按手册给出的方式",
    "请按",
    "主要包括",
    "主要内容",
    "有哪些",
    "是什么",
    "什么是",
    "什么",
    "如何",
    "怎么",
    "哪些",
    "多少",
    "多久",
    "分别",
    "是否",
    "如果",
    "为什么",
    "需要",
    "请",
)

_GENERIC_OVERLAP_TERMS = {
    "规定",
    "要求",
    "处理",
    "方式",
    "流程",
    "内容",
    "主要",
    "概括",
    "说明",
    "相关",
    "进行",
    "支持",
    "是否",
    "可以",
    "如何",
    "什么",
}


def load_level_config(level: str) -> dict[str, Any]:
    """读取回答阶段相关配置，例如引用长度和生成 token 上限。"""
    with open(f"configs/levels/{level}.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _normalize_overlap_text(text: str) -> str:
    """把词面重合计算所需文本归一化成更稳定的格式。"""
    lowered = (text or "").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", lowered)


def _extract_overlap_terms(query: str) -> list[str]:
    """
    从 query 中提取用于 answerability gate 的词面锚点。
    这里不做重型中文分词，只保留一层便宜、可解释、可调参的词面特征。
    """
    normalized = _normalize_overlap_text(query)
    for marker in _QUERY_SPLIT_MARKERS:
        normalized = normalized.replace(marker, " ")

    tokens = re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]+", normalized)
    terms: list[str] = []
    seen: set[str] = set()

    def add_term(term: str) -> None:
        candidate = term.strip()
        if len(candidate) < 2:
            return
        if candidate in _GENERIC_OVERLAP_TERMS:
            return
        if candidate in seen:
            return
        seen.add(candidate)
        terms.append(candidate)

    for token in tokens:
        if re.fullmatch(r"[0-9a-z]+", token):
            add_term(token)
            continue

        if len(token) <= 4:
            add_term(token)
            continue

        """
        中文长串如果只保留整段，往往过于严格；
        这里拆成 2/3 字片段，兼顾召回真实关键词与实现成本。
        """
        for size in (2, 3):
            for index in range(0, len(token) - size + 1):
                add_term(token[index : index + size])

    return terms


def _count_lexical_overlap(query: str, snippet: str) -> int:
    """计算 query 与 top1 证据片段的词面重合数。"""
    evidence_text = _normalize_overlap_text(snippet)
    terms = _extract_overlap_terms(query)
    return sum(1 for term in terms if term in evidence_text)


def _evaluate_extractive_gate(
    question: str,
    evidence: list[dict[str, Any]],
    min_score: float,
    min_margin: float,
    min_overlap: int,
) -> tuple[bool, dict[str, Any]]:
    """
    判断是否允许程序侧抽取式兜底。
    三重门槛全部满足才放行：
    1. top1 分数足够高
    2. top1 相对 top2 有明显领先
    3. query 与 top1 片段至少有一定词面重合
    """
    if not evidence:
        return False, {
            "reason": "no_evidence",
            "top_score": 0.0,
            "second_score": 0.0,
            "score_margin": 0.0,
            "lexical_overlap": 0,
        }

    best = evidence[0]
    top_score = float(best.get("score", 0.0) or 0.0)
    second_score = float(evidence[1].get("score", 0.0) or 0.0) if len(evidence) > 1 else 0.0
    score_margin = top_score - second_score
    snippet = str(best.get("snippet") or best.get("text", "")).strip()
    lexical_overlap = _count_lexical_overlap(question, snippet)

    details = {
        "top_score": top_score,
        "second_score": second_score,
        "score_margin": score_margin,
        "lexical_overlap": lexical_overlap,
    }

    if top_score < min_score:
        details["reason"] = "score_below_threshold"
        return False, details
    if score_margin < min_margin:
        details["reason"] = "margin_below_threshold"
        return False, details
    if lexical_overlap < min_overlap:
        details["reason"] = "lexical_overlap_below_threshold"
        return False, details

    details["reason"] = "passed"
    return True, details


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    尝试把模型输出解析成 JSON。
    有些模型会在 JSON 前后混入解释文本，因此这里额外尝试截取最外层 {...}。
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def _call_llm_final_only(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    """调用 OpenAI-compatible 接口，并只返回最终的文本内容。"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"llm_call_failed:{exc.__class__.__name__}") from exc
    message = response.choices[0].message
    return (message.content or "").strip()


def _is_refusal_answer(answer: str) -> bool:
    """判断当前答案是否属于“证据不足”类拒答。"""
    return answer.startswith("证据不足")


def _normalize_output(
    data: dict[str, Any],
    evidence: list[dict[str, Any]],
    max_snippet_chars: int,
) -> dict[str, Any]:
    """
    把模型返回的 JSON 归一化成稳定结构：
    {"answer": str, "citations": list[{doc_id,page,snippet}]}
    """
    answer = str(data.get("answer", "")).strip()
    citations_raw = data.get("citations", [])
    citations = []

    """
    有些模型会给出 doc_id/page，但漏掉 snippet。
    这里提前建立索引表，后面可以根据 (doc_id, page) 把 snippet 补回来。
    """
    snippet_by_doc_page: dict[tuple[str, int], str] = {}
    for ev in evidence:
        doc_id = str(ev.get("doc_id", ""))
        page = int(ev.get("page", 0) or 0)
        snippet = str(ev.get("snippet") or ev.get("text", ""))[:max_snippet_chars]
        if doc_id and page > 0 and snippet:
            snippet_by_doc_page[(doc_id, page)] = snippet

    if isinstance(citations_raw, list):
        for item in citations_raw:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id", "")).strip()
            page = int(item.get("page", 0) or 0)
            snippet = str(item.get("snippet", "")).strip()

            if not doc_id or page <= 0:
                """doc_id/page 是引用最基本的定位信息，缺了就不保留。"""
                continue
            if not snippet:
                snippet = snippet_by_doc_page.get((doc_id, page), "")
            if snippet:
                snippet = snippet[:max_snippet_chars]

            citations.append({"doc_id": doc_id, "page": page, "snippet": snippet})

    return {"answer": answer, "citations": citations}


def _validate_output(result: dict[str, Any]) -> tuple[bool, str | None]:
    """
    校验归一化后的结果是否满足当前 L0 约束。
    当前约束是：只要不是拒答，就必须至少带 1 条 citation。
    """
    answer = str(result.get("answer", "")).strip()
    citations = result.get("citations", []) or []

    if not answer:
        return False, "missing_answer"
    if _is_refusal_answer(answer):
        return True, None
    if not citations:
        return False, "missing_citations"
    return True, None


def _repair_output_to_json(
    client: OpenAI,
    model: str,
    question: str,
    evidence_block: str,
    raw_output: str,
    max_tokens: int,
) -> str:
    """
    当第一次输出不是严格 JSON，或虽然可解析但结构不合格时，
    再发起一次“修复型提示”，要求模型只做结构化修复。
    """
    repair_system = (
        "你是一个 JSON 修复器。"
        "你的任务是把已有回答转换成严格 JSON。"
        "输出必须以 { 开始，以 } 结束。"
        "禁止输出 Markdown、解释或其它额外文字。"
        "如果答案不是“证据不足”，citations 至少 1 条。"
        "citations 中的 doc_id/page/snippet 必须来自给定证据。"
        "格式："
        '{"answer":"...","citations":[{"doc_id":"...","page":1,"snippet":"..."}]}'
    )
    repair_user = (
        f"问题：{question}\n\n"
        f"证据片段：\n{evidence_block}\n\n"
        f"原始输出：\n{raw_output[:4000]}\n\n"
        "请把原始输出修复成严格 JSON，仅输出 JSON。"
    )
    repair_messages = [
        {"role": "system", "content": repair_system},
        {"role": "user", "content": repair_user},
    ]
    try:
        return _call_llm_final_only(client, model, repair_messages, max_tokens)
    except RuntimeError:
        return ""


def _attach_meta(
    result: dict[str, Any],
    *,
    json_ok: bool,
    repair_used: bool,
    failure_reason: str | None,
    attempt_stage: str,
) -> dict[str, Any]:
    """把调试和评测需要的元信息挂到返回结果中。"""
    enriched = dict(result)
    enriched["meta"] = {
        "json_ok": json_ok,
        "repair_used": repair_used,
        "failure_reason": failure_reason,
        "attempt_stage": attempt_stage,
    }
    return enriched


def _build_extractive_fallback(
    question: str,
    evidence: list[dict[str, Any]],
    max_snippet_chars: int,
    min_score: float,
    min_margin: float,
    min_overlap: int,
) -> dict[str, Any] | None:
    """
    当模型给出“证据不足”，但检索结果通过可回答性闸门时，
    直接使用最高分证据构造一个保守但可用的抽取式回答。
    """
    gate_passed, _ = _evaluate_extractive_gate(
        question,
        evidence,
        min_score,
        min_margin,
        min_overlap,
    )
    if not gate_passed:
        return None

    best = evidence[0]
    doc_id = str(best.get("doc_id") or "").strip()
    page = int(best.get("page", 0) or 0)
    snippet = str(best.get("snippet") or best.get("text", "")).strip()[:max_snippet_chars]

    if not doc_id or page <= 0 or not snippet:
        return None

    return {
        "answer": f"根据检索到的原文证据，相关规定为：{snippet}",
        "citations": [
            {
                "doc_id": doc_id,
                "page": page,
                "snippet": snippet,
            }
        ],
    }


def answer_with_citations(
    question: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    生成严格基于证据的最终答案。
    如果模型输出无法稳定解析，就直接拒答，避免“看起来像对其实是编的”。
    """
    """统一从 .env 读取模型配置，便于后续从云 API 切换到本地 vLLM。"""
    load_dotenv()

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    fallback_model_raw = str(os.getenv("OPENAI_FALLBACK_MODEL", "") or "").strip()
    fallback_model = (
        model
        if not fallback_model_raw or fallback_model_raw.lower() == "none"
        else fallback_model_raw
    )
    level = os.getenv("APP_LEVEL", "l0")
    cfg = load_level_config(level)
    generation_cfg = cfg.get("generation", {})
    max_snippet_chars = int(cfg["citations"]["max_snippet_chars"])
    max_tokens = int(generation_cfg.get("max_tokens", 400))
    retry_max_tokens = int(generation_cfg.get("retry_max_tokens", max(max_tokens, 500)))
    refusal_fallback_score = float(os.getenv("REFUSAL_FALLBACK_SCORE", "0.60"))
    refusal_fallback_margin = float(os.getenv("REFUSAL_FALLBACK_MARGIN", "0.08"))
    refusal_fallback_min_overlap = int(os.getenv("REFUSAL_FALLBACK_MIN_OVERLAP", "3"))
    request_timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))

    if not evidence:
        """没证据直接拒答，这是政策问答场景的基本安全策略。"""
        return _attach_meta(
            {"answer": "证据不足：未检索到与问题相关的文档片段。", "citations": []},
            json_ok=False,
            repair_used=False,
            failure_reason="no_evidence",
            attempt_stage="short_circuit",
        )
    if not api_key:
        return _attach_meta(
            {"answer": "证据不足：缺少 OPENAI_API_KEY。", "citations": []},
            json_ok=False,
            repair_used=False,
            failure_reason="missing_api_key",
            attempt_stage="short_circuit",
        )

    evidence_lines = []
    for idx, ev in enumerate(evidence, start=1):
        doc_id = ev.get("doc_id")
        page = ev.get("page")
        snippet = str(ev.get("snippet") or ev.get("text", ""))[:max_snippet_chars]
        evidence_lines.append(f"[E{idx}] doc_id={doc_id} page={page} snippet={snippet}")
    evidence_block = "\n".join(evidence_lines)

    """
    system prompt 负责强约束输出格式，核心目标有两个：
    1) 只能基于证据回答，不能自由发挥
    2) 必须产出严格 JSON，方便后续程序解析和回归统计
    """
    system_prompt = (
        "你是企业内部政策/手册问答助手。"
        "你只能依据给定证据回答，禁止编造。"
        "若证据不足，回答“证据不足”。"
        "你必须输出严格 JSON（不要 Markdown，不要额外文字）。"
        "输出必须以 { 开始，以 } 结束。"
        "格式："
        '{"answer":"...","citations":[{"doc_id":"...","page":1,"snippet":"..."}]}'
        "如果答案不是“证据不足”，citations 至少 1 条。"
        "其中 citations.snippet 必须来自证据原文，可截断。"
    )
    user_prompt = (
        f"问题：{question}\n\n"
        f"证据片段：\n{evidence_block}\n\n"
        "请按要求输出 JSON。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=request_timeout_seconds,
    )

    """
    重试策略分三步：
    1) 先用当前主模型正常请求
    2) 如果可能因为输出截断或格式不完整，再给更大的 token 重试一次
    3) 仍失败则切到 fallback 模型，优先保证“能稳定出 JSON”
    """
    attempts = [
        (model, max_tokens, "primary"),
        (model, retry_max_tokens, "primary_retry"),
        (fallback_model, max_tokens, "fallback"),
    ]
    repair_used = False
    last_failure_reason = "unknown"

    for attempt_model, attempt_max_tokens, attempt_stage in attempts:
        try:
            content = _call_llm_final_only(client, attempt_model, messages, attempt_max_tokens)
        except RuntimeError as exc:
            last_failure_reason = str(exc)
            content = ""
        data = _extract_json(content)
        if isinstance(data, dict) and "answer" in data:
            normalized = _normalize_output(data, evidence, max_snippet_chars)
            is_valid, failure_reason = _validate_output(normalized)
            if is_valid:
                if _is_refusal_answer(str(normalized.get("answer", "")).strip()):
                    """
                    如果模型直接拒答，但检索结果通过三重门槛，
                    才允许程序侧抽取式回答兜底，避免把“中等相似但不可回答”的负例强行答出来。
                    """
                    extractive = _build_extractive_fallback(
                        question,
                        evidence,
                        max_snippet_chars,
                        refusal_fallback_score,
                        refusal_fallback_margin,
                        refusal_fallback_min_overlap,
                    )
                    if extractive is not None:
                        return _attach_meta(
                            extractive,
                            json_ok=True,
                            repair_used=repair_used,
                            failure_reason=None,
                            attempt_stage=f"{attempt_stage}_extractive",
                        )
                return _attach_meta(
                    normalized,
                    json_ok=True,
                    repair_used=repair_used,
                    failure_reason=None,
                    attempt_stage=attempt_stage,
                )
            last_failure_reason = failure_reason or "invalid_output"
        else:
            last_failure_reason = "parse_failed"

        """
        只做一次修复型调用，避免把失败的单次回答放大成过多额外请求。
        这里既处理“完全解析失败”，也处理“能解析但缺少 citations”等结构性问题。
        """
        if content and not repair_used:
            repaired_content = _repair_output_to_json(
                client,
                fallback_model,
                question,
                evidence_block,
                content,
                min(max_tokens, 1000),
            )
            repair_used = True
            repaired_data = _extract_json(repaired_content)
            if isinstance(repaired_data, dict) and "answer" in repaired_data:
                repaired_normalized = _normalize_output(repaired_data, evidence, max_snippet_chars)
                is_valid, failure_reason = _validate_output(repaired_normalized)
                if is_valid:
                    if _is_refusal_answer(str(repaired_normalized.get("answer", "")).strip()):
                        extractive = _build_extractive_fallback(
                            question,
                            evidence,
                            max_snippet_chars,
                            refusal_fallback_score,
                            refusal_fallback_margin,
                            refusal_fallback_min_overlap,
                        )
                        if extractive is not None:
                            return _attach_meta(
                                extractive,
                                json_ok=True,
                                repair_used=True,
                                failure_reason=None,
                                attempt_stage=f"{attempt_stage}_repair_extractive",
                            )
                    return _attach_meta(
                        repaired_normalized,
                        json_ok=True,
                        repair_used=True,
                        failure_reason=None,
                        attempt_stage=f"{attempt_stage}_repair",
                    )
                last_failure_reason = failure_reason or "repair_invalid_output"
            else:
                last_failure_reason = "repair_parse_failed"

    """兜底拒答比“返回不可信答案”更安全，也更适合你的 demo 场景。"""
    return _attach_meta(
        {"answer": "证据不足：模型返回结果无法解析为有效 JSON。", "citations": []},
        json_ok=False,
        repair_used=repair_used,
        failure_reason=last_failure_reason,
        attempt_stage="fallback_refusal",
    )
