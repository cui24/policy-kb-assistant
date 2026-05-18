#!/usr/bin/env python3
"""对 policy_eval_100.csv 运行端到端评测并输出检索/回答指标。

评测目标：
1. 同一份评测集，横向对比多个 Qdrant collection（如 baseline/overlap/structured）。
2. 同时统计：
   - 检索层：Recall@3、Recall@5、MRR、GoldDoc Hit@5。
   - 回答层：Answer Point Coverage（答案点覆盖）、引用率、拒答率。
3. 结果写入机器可读 JSON，便于后续画图/回归对比。

输出产物：
- <collection>.rows.json：逐题明细。
- <collection>.summary.json：该库汇总指标。
- compare_summary.json：多库并排汇总。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# 允许直接运行 scripts/*.py 时导入项目包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    """解析评测参数。

    - eval-set：题集 CSV（默认 policy_eval_100.csv）。
    - collections：逗号分隔的集合名列表。
    - top-k：检索深度，至少 5 才能计算 Recall@5。
    - skip-answer：只跑检索层，适合快速 smoke test。
    """
    parser = argparse.ArgumentParser(description="Evaluate policy eval set on one or more collections")
    parser.add_argument("--eval-set", default="data/eval/policy_eval_100.csv")
    parser.add_argument(
        "--collections",
        default="policy_kb_baseline_v1,policy_kb_overlap_v1,policy_kb_structured_v1",
        help="Comma-separated Qdrant collection names",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Retrieval top_k (must be >=5 for Recall@5)")
    parser.add_argument("--out-dir", default="outputs/policy_eval")
    parser.add_argument("--skip-answer", action="store_true", help="Only run retrieval metrics")
    return parser.parse_args()


def _safe_div(n: float, d: float) -> float:
    """安全除法，防止分母为 0。"""
    return (n / d) if d else 0.0


def _normalize_text(text: str) -> str:
    """归一化文本：仅保留中英数字，便于宽松匹配。"""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (text or "").lower())


def _extract_terms(text: str) -> list[str]:
    """抽取词面锚点，用于答案点覆盖率的鲁棒判断。

    说明：
    - 中文长串会被拆成 2/3 字片段，降低严格字符串包含的脆弱性。
    - 这是“轻量启发式”，不是语义等价判定；用于快速自动评估。
    """
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", (text or "").lower())
    tokens = re.findall(r"[0-9a-z]+|[\u4e00-\u9fff]+", normalized)
    terms: list[str] = []
    for tok in tokens:
        if re.fullmatch(r"[0-9a-z]+", tok):
            if len(tok) >= 2:
                terms.append(tok)
            continue
        if len(tok) <= 4:
            if len(tok) >= 2:
                terms.append(tok)
        else:
            # 长中文串拆成 2/3 字片段，增强鲁棒性
            for size in (2, 3):
                for i in range(0, len(tok) - size + 1):
                    terms.append(tok[i : i + size])
    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def _point_hit(answer: str, point: str) -> bool:
    """判断单个 answer_point 是否被回答命中。

    两级策略：
    1) 严格子串：point 归一化后在 answer 中出现，直接命中；
    2) 关键词覆盖：point 关键词覆盖率 >= 60% 视为命中。
    """
    ans = _normalize_text(answer)
    p = _normalize_text(point)
    if not p:
        return False
    if p in ans:
        return True

    # 关键词覆盖兜底：点的关键词至少覆盖 60%
    terms = _extract_terms(point)
    if not terms:
        return False
    covered = sum(1 for t in terms if t in ans)
    return (covered / len(terms)) >= 0.6


def _split_points(text: str) -> list[str]:
    """按中英文分号拆分 answer_points。"""
    points = [p.strip() for p in re.split(r"[；;]", text or "") if p.strip()]
    return points


def _canonical_doc_id(doc_id: str) -> str:
    """把文档标识归一化为可对齐的主键。

    例如：
    - `HR-01` -> `HR-01`
    - `HR-01博雷《员工手册》` -> `HR-01`
    - `fin_02_xxx` -> `FIN-02`（若能识别）
    """
    raw = str(doc_id or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    # 不使用单词边界：中文后缀（如 HR-01博雷《员工手册》）也要能命中
    m = re.search(r"([A-Z]{2,4})[-_](\d{1,3})", upper)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return upper


def _rank_of_gold(hits: list[dict[str, Any]], gold_doc_id: str) -> int | None:
    """计算 gold_doc_id 在检索结果中的 1-based 排名。"""
    gold_norm = _canonical_doc_id(gold_doc_id)
    for idx, hit in enumerate(hits, start=1):
        doc_raw = str(hit.get("doc_id") or "").strip()
        if not doc_raw:
            continue
        if doc_raw == gold_doc_id:
            return idx
        if _canonical_doc_id(doc_raw) == gold_norm:
            return idx
    return None


def _load_eval_rows(path: Path) -> list[dict[str, str]]:
    """读取评测集 CSV 并统一 strip。"""
    with path.open("r", encoding="utf-8", newline="") as rf:
        rows = list(csv.DictReader(rf))
    out: list[dict[str, str]] = []
    for row in rows:
        out.append({k: str(v or "").strip() for k, v in row.items()})
    return out


def _summarize_rows(rows: list[dict[str, Any]], *, skip_answer: bool) -> dict[str, Any]:
    """聚合逐题结果，输出总体和分 category 指标。"""
    n = len(rows)
    r3 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 3)
    r5 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5)
    mrr = sum((1.0 / r["rank"]) for r in rows if r["rank"] is not None)

    retrieve_ms = [r["retrieve_ms"] for r in rows]
    # 检索层核心指标
    summary: dict[str, Any] = {
        "n": n,
        "recall_at_3": _safe_div(r3, n),
        "recall_at_5": _safe_div(r5, n),
        "gold_doc_hit_at_5": _safe_div(r5, n),
        "mrr": _safe_div(mrr, n),
        "retrieve_ms_p50": int(statistics.median(retrieve_ms)) if retrieve_ms else 0,
        "retrieve_ms_p95": int(sorted(retrieve_ms)[max(0, int(0.95 * len(retrieve_ms)) - 1)]) if retrieve_ms else 0,
    }

    if not skip_answer:
        # 回答层指标（仅在未跳过回答时计算）
        answer_ms = [r["answer_ms"] for r in rows]
        refusal_n = sum(1 for r in rows if r["refused"])
        citation_n = sum(1 for r in rows if r["citations_n"] > 0)
        point_cov = [r["answer_point_coverage"] for r in rows]
        summary.update(
            {
                "answer_point_coverage_avg": _safe_div(sum(point_cov), len(point_cov)),
                "citation_rate": _safe_div(citation_n, n),
                "refusal_rate": _safe_div(refusal_n, n),
                "answer_ms_p50": int(statistics.median(answer_ms)) if answer_ms else 0,
                "answer_ms_p95": int(sorted(answer_ms)[max(0, int(0.95 * len(answer_ms)) - 1)]) if answer_ms else 0,
            }
        )

    # 分类别指标，便于定位某类文档是否异常（如 finance 明显低于其他类）
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    by_category: dict[str, Any] = {}
    for cat, cat_rows in sorted(by_cat.items()):
        cn = len(cat_rows)
        c3 = sum(1 for r in cat_rows if r["rank"] is not None and r["rank"] <= 3)
        c5 = sum(1 for r in cat_rows if r["rank"] is not None and r["rank"] <= 5)
        cmrr = sum((1.0 / r["rank"]) for r in cat_rows if r["rank"] is not None)
        item: dict[str, Any] = {
            "n": cn,
            "recall_at_3": _safe_div(c3, cn),
            "recall_at_5": _safe_div(c5, cn),
            "mrr": _safe_div(cmrr, cn),
        }
        if not skip_answer:
            point_cov = [r["answer_point_coverage"] for r in cat_rows]
            item["answer_point_coverage_avg"] = _safe_div(sum(point_cov), len(point_cov))
        by_category[cat] = item
    summary["by_category"] = by_category

    return summary


def _evaluate_one_collection(
    *,
    collection: str,
    eval_rows: list[dict[str, str]],
    top_k: int,
    skip_answer: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """评测单个 collection。

    执行顺序（每题）：
    1) 检索 hits；
    2) 计算 gold 排名与命中；
    3) 可选回答生成与答案点覆盖统计；
    4) 记录耗时与明细。
    """
    try:
        from src.kb.answer import answer_with_citations
        from src.kb.retrieve import retrieve
    except Exception as exc:  # pragma: no cover - env bootstrap guard
        raise RuntimeError(
            "failed to import src.kb modules; please install runtime deps (e.g. `pip install -r requirements.txt`)"
        ) from exc

    # 动态切换检索目标库，复用同一套 retrieve/answer 逻辑。
    os.environ["QDRANT_COLLECTION"] = collection

    rows_out: list[dict[str, Any]] = []
    for row in eval_rows:
        qid = row["query_id"]
        query = row["query"]
        gold_doc_id = row["gold_doc_id"]
        answer_points = _split_points(row.get("answer_points", ""))

        # --- 检索阶段 ---
        t0 = time.time()
        hits = retrieve(query, top_k=top_k)
        t1 = time.time()

        rank = _rank_of_gold(hits, gold_doc_id)
        top_docs = [str(h.get("doc_id") or "") for h in hits[:5]]
        item: dict[str, Any] = {
            "query_id": qid,
            "category": row["category"],
            "difficulty": row["difficulty"],
            "query_type": row["query_type"],
            "query": query,
            "gold_doc_id": gold_doc_id,
            "rank": rank,
            "doc_hit_at_3": bool(rank is not None and rank <= 3),
            "doc_hit_at_5": bool(rank is not None and rank <= 5),
            "top_docs": top_docs,
            "retrieve_ms": int((t1 - t0) * 1000),
        }

        if not skip_answer:
            # --- 回答阶段 ---
            t2 = time.time()
            output = answer_with_citations(query, hits)
            t3 = time.time()

            answer_text = str(output.get("answer") or "").strip()
            citations = output.get("citations") or []
            refused = answer_text.startswith("证据不足")
            # 统计答案点命中数量和覆盖率（Coverage = hit / total）
            hit_n = sum(1 for p in answer_points if _point_hit(answer_text, p))
            coverage = _safe_div(hit_n, len(answer_points))

            item.update(
                {
                    "answer": answer_text,
                    "answer_point_total": len(answer_points),
                    "answer_point_hit": hit_n,
                    "answer_point_coverage": coverage,
                    "citations_n": len(citations),
                    "refused": refused,
                    "answer_ms": int((t3 - t2) * 1000),
                    "meta": output.get("meta") or {},
                }
            )
        else:
            item.update(
                {
                    "answer_point_total": len(answer_points),
                    "answer_point_hit": 0,
                    "answer_point_coverage": 0.0,
                    "citations_n": 0,
                    "refused": False,
                    "answer_ms": 0,
                    "meta": {},
                }
            )

        rows_out.append(item)

    summary = _summarize_rows(rows_out, skip_answer=skip_answer)
    summary["collection"] = collection
    summary["top_k"] = top_k
    summary["skip_answer"] = skip_answer
    return summary, rows_out


def _write_json(path: Path, payload: Any) -> None:
    """写 JSON 文件（UTF-8，带缩进）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    """程序入口：按 collection 逐个评测并生成对比报告。"""
    args = _parse_args()
    load_dotenv()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        raise SystemExit(f"[ERROR] missing eval file: {eval_path}")

    # top_k<5 无法计算 Recall@5，因此直接拒绝。
    if args.top_k < 5:
        raise SystemExit("[ERROR] --top-k must be >= 5")

    eval_rows = _load_eval_rows(eval_path)
    collections = [c.strip() for c in str(args.collections).split(",") if c.strip()]
    if not collections:
        raise SystemExit("[ERROR] empty collections")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    for collection in collections:
        print(f"[INFO] evaluating collection={collection} rows={len(eval_rows)}")
        summary, rows = _evaluate_one_collection(
            collection=collection,
            eval_rows=eval_rows,
            top_k=int(args.top_k),
            skip_answer=bool(args.skip_answer),
        )
        all_summaries.append(summary)

        slug = re.sub(r"[^0-9a-zA-Z._-]+", "_", collection)
        _write_json(out_dir / f"{slug}.rows.json", rows)
        _write_json(out_dir / f"{slug}.summary.json", summary)
        print(
            "[OK]",
            collection,
            f"R@5={summary['recall_at_5']:.3f}",
            f"MRR={summary['mrr']:.3f}",
            (
                f"APC={summary.get('answer_point_coverage_avg', 0.0):.3f}"
                if not args.skip_answer
                else "APC=skip"
            ),
        )

    # 生成跨库对比汇总，后续画图/汇报直接读这份即可。
    compare = {
        "eval_set": str(eval_path),
        "top_k": int(args.top_k),
        "skip_answer": bool(args.skip_answer),
        "collections": all_summaries,
    }
    _write_json(out_dir / "compare_summary.json", compare)

    # 控制台表格
    print("\n=== Compare Summary ===")
    for s in all_summaries:
        print(
            f"{s['collection']}: "
            f"R@3={s['recall_at_3']:.3f} "
            f"R@5={s['recall_at_5']:.3f} "
            f"MRR={s['mrr']:.3f} "
            + (
                f"APC={s.get('answer_point_coverage_avg', 0.0):.3f} "
                f"Refusal={s.get('refusal_rate', 0.0):.3f}"
                if not args.skip_answer
                else "APC=skip Refusal=skip"
            )
        )

    print(f"[OK] outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
