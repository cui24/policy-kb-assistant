"""
L1 回归评测程序：批量跑问题集，得到当前 L0/L1 的基础质量指标。

一、程序目标
1. 读取问题集 JSON。
2. 对每道题依次执行检索和回答。
3. 兼容“纯正例问题集”和“正负混合问题集”。
4. 把总体指标、按 label 分组指标、逐题明细写入 `outputs/`。

二、程序入口与运行顺序
1. 命令入口：`python -m src.eval.run_regression`
2. 可选参数：`--questions <path>`
3. 默认行为：
   - 若 `data/questions/l0_mix.json` 存在，优先使用它
   - 否则退回 `data/questions/l0.json`
4. `main()` 内部顺序如下：
   4.1 `load_dotenv()`：加载环境变量
   4.2 解析命令行参数
   4.3 读取问题集 JSON
   4.4 将每条样本归一化成统一结构
   4.5 遍历每道题：
       - 调用 `retrieve(question)`
       - 调用 `answer_with_citations(question, hits)`
       - 统计单题结果
       - 记录单题耗时
   4.6 计算总体 summary
   4.7 额外按 `label=pos/neg` 输出分项指标
   4.8 写入 `outputs/regress_summary.json`
   4.9 写入 `outputs/regress_rows.json`
   4.10 打印 summary

三、输入输出数据格式
1. 旧版正例问题集输入格式：
   [
     {
       "doc": "期望命中的文档ID",
       "q": "问题文本"
     }
   ]
2. 混合问题集输入格式：
   [
     {
       "id": "pos-001",
       "label": "pos" | "neg",
       "expected_doc": "文档ID或null",
       "neg_type": "可选，负例类型",
       "q": "问题文本"
     }
   ]
3. 单题明细输出格式（`regress_rows.json`）：
   [
     {
       "id": str,
       "label": "pos" | "neg",
       "neg_type": str | None,
       "q": str,
       "expected_doc": str | None,
       "doc_hit": bool,
       "citations_n": int,
       "refused": bool,
       "json_ok": bool,
       "repair_used": bool,
       "failure_reason": str | None,
       "attempt_stage": str,
       "top_score": float,
       "second_score": float,
       "score_margin": float,
       "retrieve_ms": int,
       "answer_ms": int
     }
   ]
4. 汇总输出格式（`regress_summary.json`）：
   {
     "n": int,
     "pos_n": int,
     "neg_n": int,
     "citation_rate": float,
     "refusal_rate": float,
     "valid_json_rate": float,
     "repair_used_rate": float,
     "pos": {...},
     "neg": {...}
   }

四、主要函数的输入输出
1. `main() -> None`
   - 输入：无显式参数，依赖命令行参数和问题集文件
   - 输出：无返回值
   - 副作用：
     - 调用检索和回答模块
     - 写两个 JSON 文件到 `outputs/`
     - 向终端打印 summary

五、程序可以理解成的伪代码
1. 选定问题集路径
2. 打开问题集 JSON
3. 把每条样本标准化成统一结构
4. 对每道题：
   4.1 检索
   4.2 回答
   4.3 统计命中、引用、拒答、耗时
   4.4 记录 top1/top2 分数
5. 汇总总体指标
6. 再按正例和负例分别计算指标
7. 写出 summary 和 rows
8. 打印 summary
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.kb.answer import answer_with_citations
from src.kb.retrieve import retrieve


def _default_questions_path() -> str:
    """优先使用混合集；如果还没创建，再回退到旧版正例集。"""
    mix_path = Path("data/questions/l0_mix.json")
    if mix_path.exists():
        return str(mix_path)
    return "data/questions/l0.json"


def _normalize_question_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    """
    把旧版问题集和新版混合集归一化成统一结构。
    这样评测逻辑就不需要分两套分支。
    """
    expected_doc = item.get("expected_doc", item.get("doc"))
    if expected_doc is not None:
        expected_doc = str(expected_doc).strip() or None

    label = str(item.get("label") or "").strip().lower()
    if label not in {"pos", "neg"}:
        label = "pos" if expected_doc else "neg"

    sample_id = str(item.get("id") or f"{label}-{index:03d}")
    neg_type = item.get("neg_type")
    neg_type = str(neg_type).strip() if neg_type is not None else None

    return {
        "id": sample_id,
        "label": label,
        "expected_doc": expected_doc,
        "neg_type": neg_type,
        "q": str(item["q"]).strip(),
    }


def _top_scores(hits: list[dict[str, Any]]) -> tuple[float, float, float, str | None]:
    """提取 top1、top2 和 margin，方便后续分析检索与 Gate 的关系。"""
    top_score = float(hits[0].get("score", 0.0) or 0.0) if hits else 0.0
    second_score = float(hits[1].get("score", 0.0) or 0.0) if len(hits) > 1 else 0.0
    top_doc = str(hits[0].get("doc_id")).strip() if hits and hits[0].get("doc_id") is not None else None
    return top_score, second_score, top_score - second_score, top_doc


def _is_extractive_stage(stage: str) -> bool:
    """判断当前回答是否走了程序侧抽取式兜底。"""
    return "extractive" in stage


def _is_primary_stage(stage: str) -> bool:
    """判断当前回答是否由主模型链路直接完成，而不是抽取式兜底。"""
    return stage.startswith("primary") and not _is_extractive_stage(stage)


def _safe_rate(numerator: int, denominator: int) -> float:
    """统一处理分母为 0 的比例计算。"""
    return (numerator / denominator) if denominator else 0.0


def _summarize_rows(rows: list[dict[str, Any]], questions_path: str) -> dict[str, Any]:
    """同时输出总体、正例、负例三层指标。"""
    n = len(rows)
    pos_rows = [row for row in rows if row["label"] == "pos"]
    neg_rows = [row for row in rows if row["label"] == "neg"]

    citation_count = sum(1 for row in rows if row["citations_n"] > 0)
    refusal_count = sum(1 for row in rows if row["refused"])
    valid_json_count = sum(1 for row in rows if row["json_ok"])
    repair_used_count = sum(1 for row in rows if row["repair_used"])
    attempt_stage = dict(Counter(row["attempt_stage"] for row in rows))
    avg_retrieve_ms = int(sum(row["retrieve_ms"] for row in rows) / n) if n else 0
    avg_answer_ms = int(sum(row["answer_ms"] for row in rows) / n) if n else 0

    pos_n = len(pos_rows)
    pos_doc_hit = sum(1 for row in pos_rows if row["doc_hit"])
    pos_citation = sum(1 for row in pos_rows if row["citations_n"] > 0)
    pos_refusal = sum(1 for row in pos_rows if row["refused"])
    pos_primary = sum(1 for row in pos_rows if _is_primary_stage(row["attempt_stage"]))
    pos_extractive = sum(1 for row in pos_rows if _is_extractive_stage(row["attempt_stage"]))

    neg_n = len(neg_rows)
    neg_refusal = sum(1 for row in neg_rows if row["refused"])
    neg_extractive = sum(1 for row in neg_rows if _is_extractive_stage(row["attempt_stage"]))

    return {
        "questions_path": questions_path,
        "n": n,
        "pos_n": pos_n,
        "neg_n": neg_n,
        "citation_rate": _safe_rate(citation_count, n),
        "refusal_rate": _safe_rate(refusal_count, n),
        "valid_json_rate": _safe_rate(valid_json_count, n),
        "repair_used_rate": _safe_rate(repair_used_count, n),
        "attempt_stage": attempt_stage,
        "avg_retrieve_ms": avg_retrieve_ms,
        "avg_answer_ms": avg_answer_ms,
        "pos": {
            "n": pos_n,
            "doc_hit_rate": _safe_rate(pos_doc_hit, pos_n),
            "citation_rate": _safe_rate(pos_citation, pos_n),
            "refusal_rate": _safe_rate(pos_refusal, pos_n),
            "primary_rate": _safe_rate(pos_primary, pos_n),
            "extractive_rate": _safe_rate(pos_extractive, pos_n),
        },
        "neg": {
            "n": neg_n,
            "refusal_correct_rate": _safe_rate(neg_refusal, neg_n),
            "false_accept_rate": _safe_rate(neg_n - neg_refusal, neg_n),
            "extractive_rate": _safe_rate(neg_extractive, neg_n),
        },
    }


def main() -> None:
    """回归脚本也走统一配置入口，避免和 demo/生产配置脱节。"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="运行 L0/L1 回归问题集")
    parser.add_argument(
        "--questions",
        default=_default_questions_path(),
        help="问题集 JSON 路径",
    )
    args = parser.parse_args()

    question_path = Path(args.questions)
    if not question_path.exists():
        raise SystemExit(f"Missing questions file: {question_path}")

    raw_questions = json.loads(question_path.read_text(encoding="utf-8"))
    questions = [
        _normalize_question_item(item, index)
        for index, item in enumerate(raw_questions, start=1)
    ]
    rows = []

    for item in questions:
        question = item["q"]
        expected_doc = item["expected_doc"]

        """把检索和回答耗时拆开统计，是为了区分瓶颈在向量检索还是模型生成。"""
        t0 = time.time()
        hits = retrieve(question)
        t1 = time.time()
        output = answer_with_citations(question, hits)
        t2 = time.time()

        """
        正例看 doc_hit，负例固定为 False。
        这样旧指标的语义仍然成立：它只用于观察“检索是否至少命中期望文档”。
        """
        doc_hit = any(hit.get("doc_id") == expected_doc for hit in hits) if expected_doc else False
        citations = output.get("citations", []) or []
        answer_text = (output.get("answer", "") or "").strip()
        is_refusal = answer_text.startswith("证据不足")
        meta = output.get("meta", {}) or {}
        json_ok = bool(meta.get("json_ok", False))
        repair_used = bool(meta.get("repair_used", False))
        failure_reason = meta.get("failure_reason")
        attempt_stage = str(meta.get("attempt_stage", "unknown"))
        top_score, second_score, score_margin, top_doc = _top_scores(hits)

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
                "json_ok": json_ok,
                "repair_used": repair_used,
                "failure_reason": failure_reason,
                "attempt_stage": attempt_stage,
                "retrieve_ms": int((t1 - t0) * 1000),
                "answer_ms": int((t2 - t1) * 1000),
            }
        )

    summary = _summarize_rows(rows, str(question_path))

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    """汇总和明细分开落盘，方便你展示总体指标，也能追查单题失败原因。"""
    (output_dir / "regress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "regress_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote: outputs/regress_summary.json, outputs/regress_rows.json")


if __name__ == "__main__":
    main()
