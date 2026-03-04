"""
L1 Gate 网格搜索程序：用混合问题集自动搜索更合适的 extractive Gate 阈值。

一、程序目标
1. 使用 `l0_mix.json` 同时观察正例回答与负例拒答。
2. 自动遍历 `score / margin / lexical overlap` 三个门槛。
3. 找到在当前样本上更平衡的 Gate 参数组合。
4. 避免为每组参数重复调用 LLM，控制搜索成本。

二、为什么这个脚本不能“每组参数都重新跑一遍主链路”
1. 如果 45 组参数都完整执行 `retrieve -> answer_with_citations`：
   - 每组都要重新调用 LLM
   - 总成本和总耗时都会非常高
2. 当前 Gate 只影响“模型已经拒答之后，是否允许 extractive 兜底”
3. 因此可以拆成两步：
   3.1 先用“极严格阈值”跑一轮，得到不含 extractive 的基线结果
   3.2 再离线重放不同 Gate 参数，只模拟“拒答是否会被 extractive 救回”
4. 这样只需要调用一次模型，就能评估多组阈值。

三、程序入口与运行顺序
1. 命令入口：`python -m src.eval.grid_search_gate`
2. 可选参数：
   - `--questions`
   - `--score-grid`
   - `--margin-grid`
   - `--overlap-grid`
   - `--refresh-cache`
   - `--topn`
3. 内部顺序：
   3.1 读取混合问题集
   3.2 采集或读取“禁用 extractive 的基线缓存”
   3.3 遍历所有参数组合
   3.4 对每组参数离线模拟 Gate 是否放行
   3.5 计算 `pos/neg` 指标
   3.6 按目标函数选出最优组合
   3.7 写入结果文件并打印最佳参数

四、输出文件
1. `outputs/grid_search_gate_base_cache.json`
   - 保存禁用 extractive 后的基线结果，避免下次重复调用 LLM
2. `outputs/grid_search_gate_results.json`
   - 保存所有参数组合的评测结果
3. `outputs/grid_search_gate_best.json`
   - 保存最佳参数和对应指标

五、程序可以理解成的伪代码
1. 读混合问题集
2. 若没有缓存，先用超严格阈值跑一轮，拿到“模型原始决策 + 检索结果”
3. 对每组 `score/margin/overlap`：
   3.1 只对“基线中被拒答”的样本尝试 `_build_extractive_fallback(...)`
   3.2 重新汇总 `pos/neg` 指标
4. 按“先保负例拒答，再尽量保正例回答”的规则选最佳组合
5. 写文件并打印结果
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from src.eval.run_regression import _normalize_question_item, _summarize_rows, _top_scores
from src.kb.answer import _build_extractive_fallback, answer_with_citations, load_level_config
from src.kb.retrieve import retrieve


def _default_questions_path() -> str:
    """默认优先搜索混合问题集。"""
    mix_path = Path("data/questions/l0_mix.json")
    if mix_path.exists():
        return str(mix_path)
    return "data/questions/l0.json"


def _parse_float_grid(text: str) -> list[float]:
    """把命令行里逗号分隔的浮点网格解析成列表。"""
    values = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("Empty float grid.")
    return values


def _parse_int_grid(text: str) -> list[int]:
    """把命令行里逗号分隔的整数网格解析成列表。"""
    values = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("Empty int grid.")
    return values


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    """临时覆盖环境变量，离开上下文后恢复原值。"""
    originals: dict[str, str | None] = {}
    for key, value in overrides.items():
        originals[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, original in originals.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _load_questions(path: str) -> list[dict[str, Any]]:
    """读取并标准化问题集。"""
    question_path = Path(path)
    if not question_path.exists():
        raise SystemExit(f"Missing questions file: {question_path}")

    raw_questions = json.loads(question_path.read_text(encoding="utf-8"))
    return [
        _normalize_question_item(item, index)
        for index, item in enumerate(raw_questions, start=1)
    ]


def _build_base_cache(
    questions: list[dict[str, Any]],
    questions_path: str,
    cache_path: Path,
) -> dict[str, Any]:
    """
    采集“禁用 extractive”后的基线结果。
    只在这一步真正调用 LLM；后续所有参数组合都离线重放。
    """
    level = os.getenv("APP_LEVEL", "l0")
    cfg = load_level_config(level)
    max_snippet_chars = int(cfg["citations"]["max_snippet_chars"])
    rows: list[dict[str, Any]] = []

    """
    这里用极端阈值把 extractive 完全关掉。
    这样基线结果就只反映“模型自身会不会回答/拒答”，便于后续离线模拟。
    """
    with _temporary_env(
        {
            "REFUSAL_FALLBACK_SCORE": "9.0",
            "REFUSAL_FALLBACK_MARGIN": "9.0",
            "REFUSAL_FALLBACK_MIN_OVERLAP": "999",
        }
    ):
        for item in questions:
            question = item["q"]
            expected_doc = item["expected_doc"]

            t0 = time.time()
            hits = retrieve(question)
            t1 = time.time()
            output = answer_with_citations(question, hits)
            t2 = time.time()

            citations = output.get("citations", []) or []
            answer_text = (output.get("answer", "") or "").strip()
            is_refusal = answer_text.startswith("证据不足")
            meta = output.get("meta", {}) or {}
            top_score, second_score, score_margin, top_doc = _top_scores(hits)
            doc_hit = any(hit.get("doc_id") == expected_doc for hit in hits) if expected_doc else False

            rows.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "neg_type": item["neg_type"],
                    "q": question,
                    "expected_doc": expected_doc,
                    "top_doc": top_doc,
                    "top_score": top_score,
                    "second_score": second_score,
                    "score_margin": score_margin,
                    "doc_hit": doc_hit,
                    "citations_n": len(citations),
                    "refused": is_refusal,
                    "json_ok": bool(meta.get("json_ok", False)),
                    "repair_used": bool(meta.get("repair_used", False)),
                    "failure_reason": meta.get("failure_reason"),
                    "attempt_stage": str(meta.get("attempt_stage", "unknown")),
                    "retrieve_ms": int((t1 - t0) * 1000),
                    "answer_ms": int((t2 - t1) * 1000),
                    "hits": hits,
                    "max_snippet_chars": max_snippet_chars,
                }
            )

    payload = {
        "questions_path": questions_path,
        "n": len(rows),
        "base_mode": "no_extractive",
        "rows": rows,
    }
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_or_build_base_cache(
    questions: list[dict[str, Any]],
    questions_path: str,
    cache_path: Path,
    refresh_cache: bool,
) -> dict[str, Any]:
    """优先复用已有基线缓存，减少重复调用 LLM。"""
    if cache_path.exists() and not refresh_cache:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("questions_path") == questions_path:
            return payload
    return _build_base_cache(questions, questions_path, cache_path)


def _simulate_rows(
    base_rows: list[dict[str, Any]],
    score_threshold: float,
    margin_threshold: float,
    overlap_threshold: int,
) -> list[dict[str, Any]]:
    """在固定基线上离线模拟某一组 Gate 参数。"""
    simulated_rows: list[dict[str, Any]] = []

    for row in base_rows:
        simulated = {
            key: value
            for key, value in row.items()
            if key not in {"hits", "max_snippet_chars"}
        }

        """
        只有“基线中已经拒答”的样本，才可能被新的 Gate 参数救回。
        非拒答样本的模型输出与当前 Gate 无关，直接沿用基线结果。
        """
        if row["refused"]:
            extractive = _build_extractive_fallback(
                row["q"],
                row["hits"],
                int(row["max_snippet_chars"]),
                score_threshold,
                margin_threshold,
                overlap_threshold,
            )
            if extractive is not None:
                simulated["citations_n"] = len(extractive.get("citations", []) or [])
                simulated["refused"] = False
                simulated["attempt_stage"] = f"{row['attempt_stage']}_grid_extractive"

        simulated_rows.append(simulated)

    return simulated_rows


def _rank_tuple(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    目标函数优先级：
    1. 先把负例拒答做对
    2. 再尽量保住正例引用率
    3. 再尽量降低正例拒答
    4. 最后尽量减少正例对 extractive 的依赖
    """
    neg = summary.get("neg", {})
    pos = summary.get("pos", {})
    return (
        float(neg.get("refusal_correct_rate", 0.0)),
        float(pos.get("citation_rate", 0.0)),
        1.0 - float(pos.get("refusal_rate", 0.0)),
        1.0 - float(pos.get("extractive_rate", 0.0)),
    )


def _search(
    base_payload: dict[str, Any],
    questions_path: str,
    score_grid: list[float],
    margin_grid: list[float],
    overlap_grid: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """遍历网格并找出最佳参数。"""
    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for score_threshold, margin_threshold, overlap_threshold in product(
        score_grid,
        margin_grid,
        overlap_grid,
    ):
        simulated_rows = _simulate_rows(
            base_payload["rows"],
            score_threshold,
            margin_threshold,
            overlap_threshold,
        )
        summary = _summarize_rows(simulated_rows, questions_path)
        result = {
            "params": {
                "score": score_threshold,
                "margin": margin_threshold,
                "overlap": overlap_threshold,
            },
            "summary": summary,
            "rank": _rank_tuple(summary),
        }
        results.append(result)

        if best_result is None or result["rank"] > best_result["rank"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("Grid search produced no result.")

    results.sort(key=lambda item: item["rank"], reverse=True)
    return results, best_result


def main() -> None:
    """执行 Gate 网格搜索，并写出结果文件。"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="为 extractive Gate 做网格搜索")
    parser.add_argument(
        "--questions",
        default=_default_questions_path(),
        help="问题集 JSON 路径，建议使用 l0_mix.json",
    )
    parser.add_argument(
        "--score-grid",
        default="0.60,0.62,0.65,0.68,0.70",
        help="逗号分隔的 score 阈值网格",
    )
    parser.add_argument(
        "--margin-grid",
        default="0.06,0.08,0.10",
        help="逗号分隔的 margin 阈值网格",
    )
    parser.add_argument(
        "--overlap-grid",
        default="1,2,3",
        help="逗号分隔的词面重合阈值网格",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="打印前 N 组结果",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="忽略已有基线缓存，重新采样模型输出",
    )
    args = parser.parse_args()

    questions_path = str(Path(args.questions))
    questions = _load_questions(questions_path)
    cache_path = Path("outputs/grid_search_gate_base_cache.json")
    base_payload = _load_or_build_base_cache(
        questions,
        questions_path,
        cache_path,
        args.refresh_cache,
    )

    score_grid = _parse_float_grid(args.score_grid)
    margin_grid = _parse_float_grid(args.margin_grid)
    overlap_grid = _parse_int_grid(args.overlap_grid)

    results, best_result = _search(
        base_payload,
        questions_path,
        score_grid,
        margin_grid,
        overlap_grid,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results_payload = {
        "questions_path": questions_path,
        "search_space": {
            "score_grid": score_grid,
            "margin_grid": margin_grid,
            "overlap_grid": overlap_grid,
        },
        "base_cache_path": str(cache_path),
        "n_results": len(results),
        "results": results,
    }
    best_payload = {
        "questions_path": questions_path,
        "search_space": results_payload["search_space"],
        "best": best_result,
    }

    (output_dir / "grid_search_gate_results.json").write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "grid_search_gate_best.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("BEST")
    print(json.dumps(best_result, ensure_ascii=False, indent=2))
    print("TOP_RESULTS")
    print(json.dumps(results[: max(args.topn, 1)], ensure_ascii=False, indent=2))
    print(
        "Wrote: outputs/grid_search_gate_base_cache.json, "
        "outputs/grid_search_gate_results.json, outputs/grid_search_gate_best.json"
    )


if __name__ == "__main__":
    main()
