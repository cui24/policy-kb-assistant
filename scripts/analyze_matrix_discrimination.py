#!/usr/bin/env python3
"""矩阵评测结果的“区分度体检”脚本。

为什么需要这个脚本
------------------
在 RAG 入库策略/编码器矩阵实验中，只看 `R@5` 往往会出现“大家都差不多”的错觉：
- `top_k` 较大时，很多组合都能在前 5 命中 gold doc，指标容易饱和；
- 平均值会掩盖分层差异（例如 hard 题、comparison 题其实拉开明显差距）。

本脚本专门做以下几件事：
1) 读取一个 run 目录下所有组合的逐题结果（`*.rows.json`）；
2) 计算整体指标（R@1/R@3/R@5/MRR）并给出 spread/std；
3) 按 `difficulty / query_type / category` 分层出排行榜；
4) 给出“题目级波动”诊断（哪些题在不同组合下最不稳定）；
5) 输出机器可读文件（csv/json）+ 人类可读摘要（markdown）。

输入约定
--------
run 目录结构通常如下（由 `benchmark_matrix_8gb.sh` 生成）：

outputs/matrix_bench/run_YYYYMMDD_HHMMSS/
  ├─ fixed__baai_bge-small-zh-v1.5/
  │   └─ eval/*.rows.json
  ├─ overlap__baai_bge-small-zh-v1.5/
  │   └─ eval/*.rows.json
  └─ ...

每个 rows.json 需要至少包含以下字段（来自 evaluate_policy_eval_set.py）：
- query_id
- category
- difficulty
- query_type
- rank
- doc_hit_at_3
- doc_hit_at_5

输出文件
--------
默认写到 `<run_dir>/analysis/`：
- combo_overall.csv：每个组合的整体指标
- combo_stratified.csv：每个组合在每个分层子集的指标
- query_variability.csv：每道题在所有组合下的波动程度
- discrimination_summary.json：区分度汇总（spread/std/波动统计）
- discrimination_report.md：可直接阅读的结论报告
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ========== 数据结构定义 ==========


@dataclass(frozen=True)
class Metrics:
    """单个集合上的一组核心检索指标。"""

    n: int
    r1: float
    r3: float
    r5: float
    mrr: float


# ========== 基础工具函数 ==========


def _safe_div(n: float, d: float) -> float:
    """安全除法，分母为 0 时返回 0。"""

    return (n / d) if d else 0.0


def _rank_to_rr(rank: int | None) -> float:
    """把 rank 转为 reciprocal rank（RR）。

    - rank = 1 -> 1.0
    - rank = 2 -> 0.5
    - rank = None -> 0.0
    """

    if rank is None:
        return 0.0
    if rank <= 0:
        return 0.0
    return 1.0 / rank


def _coerce_rank(value: Any) -> int | None:
    """把各种 rank 值稳健地转换成 int 或 None。"""

    if value is None:
        return None
    try:
        iv = int(value)
    except Exception:
        return None
    return iv if iv > 0 else None


def _pct(x: float) -> str:
    """把 0~1 浮点转成百分比字符串，保留 1 位小数。"""

    return f"{x * 100:.1f}%"


# ========== 指标计算 ==========


def compute_metrics(rows: list[dict[str, Any]]) -> Metrics:
    """基于逐题 rows 计算整体检索指标。"""

    n = len(rows)
    if n == 0:
        return Metrics(n=0, r1=0.0, r3=0.0, r5=0.0, mrr=0.0)

    r1_hits = 0
    r3_hits = 0
    r5_hits = 0
    rr_sum = 0.0

    for row in rows:
        rank = _coerce_rank(row.get("rank"))

        # 优先信任 rank，其次回退到布尔命中字段。
        if rank is not None:
            if rank <= 1:
                r1_hits += 1
            if rank <= 3:
                r3_hits += 1
            if rank <= 5:
                r5_hits += 1
        else:
            if bool(row.get("doc_hit_at_3")):
                r3_hits += 1
            if bool(row.get("doc_hit_at_5")):
                r5_hits += 1

        rr_sum += _rank_to_rr(rank)

    return Metrics(
        n=n,
        r1=_safe_div(r1_hits, n),
        r3=_safe_div(r3_hits, n),
        r5=_safe_div(r5_hits, n),
        mrr=_safe_div(rr_sum, n),
    )


def _subset_rows(rows: list[dict[str, Any]], field: str, value: str) -> list[dict[str, Any]]:
    """提取分层子集。"""

    return [r for r in rows if str(r.get(field, "")).strip() == value]


# ========== 结果读取 ==========


def discover_rows_files(run_dir: Path) -> list[Path]:
    """自动发现 run 目录下所有 rows.json。

    匹配模式：`<run_dir>/*/eval/*.rows.json`
    """

    return sorted(run_dir.glob("*/eval/*.rows.json"))


def combo_name_from_rows_path(rows_path: Path) -> str:
    """从 rows 路径提取组合名。

    例如：
    - .../overlap__baai_bge-small-zh-v1.5/eval/xxx.rows.json
      -> overlap__baai_bge-small-zh-v1.5
    """

    # rows_path: run/COMBO/eval/file.rows.json
    return rows_path.parent.parent.name


def load_combo_rows(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """加载 run 目录下所有组合的逐题结果。"""

    combo_rows: dict[str, list[dict[str, Any]]] = {}
    rows_files = discover_rows_files(run_dir)
    for fp in rows_files:
        combo = combo_name_from_rows_path(fp)
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"rows file is not list: {fp}")
        combo_rows[combo] = data
    return combo_rows


# ========== 区分度统计 ==========


def summarize_spread(overall: dict[str, Metrics]) -> dict[str, dict[str, float]]:
    """计算每个指标在所有组合间的 spread/std。"""

    out: dict[str, dict[str, float]] = {}
    for key in ("r1", "r3", "r5", "mrr"):
        vals = [getattr(m, key) for m in overall.values()]
        if not vals:
            out[key] = {"min": 0.0, "max": 0.0, "spread": 0.0, "std": 0.0}
            continue
        out[key] = {
            "min": min(vals),
            "max": max(vals),
            "spread": max(vals) - min(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def query_variability(combo_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """统计每道题在不同组合下的波动。

    输出字段：
    - query_id / category / difficulty / query_type
    - hit3_changed / hit5_changed
    - rank_min / rank_max / rank_spread
    - rank_spread_hit_only（仅在命中样本中统计 rank 波动）
    - rr_mean / rr_min / rr_max
    """

    if not combo_rows:
        return []

    combos = sorted(combo_rows)
    base_rows = combo_rows[combos[0]]
    qids = [str(r.get("query_id", "")).strip() for r in base_rows]

    # 建索引，减少后续 O(n^2) 扫描。
    idx: dict[str, dict[str, dict[str, Any]]] = {}
    for combo, rows in combo_rows.items():
        m: dict[str, dict[str, Any]] = {}
        for r in rows:
            qid = str(r.get("query_id", "")).strip()
            if qid:
                m[qid] = r
        idx[combo] = m

    out: list[dict[str, Any]] = []
    for qid in qids:
        qrows: list[dict[str, Any]] = []
        for combo in combos:
            row = idx[combo].get(qid)
            if row is not None:
                qrows.append(row)

        if not qrows:
            continue

        ranks_raw = [_coerce_rank(r.get("rank")) for r in qrows]
        # 用 999 作为未命中占位，便于观察“命中波动是否极端”。
        ranks_for_spread = [rk if rk is not None else 999 for rk in ranks_raw]
        hit_ranks = [rk for rk in ranks_raw if rk is not None]
        rr_vals = [_rank_to_rr(rk) for rk in ranks_raw]

        hit3 = [bool(r.get("doc_hit_at_3")) if r.get("rank") is None else (_coerce_rank(r.get("rank")) or 999) <= 3 for r in qrows]
        hit5 = [bool(r.get("doc_hit_at_5")) if r.get("rank") is None else (_coerce_rank(r.get("rank")) or 999) <= 5 for r in qrows]

        exemplar = qrows[0]
        out.append(
            {
                "query_id": qid,
                "category": str(exemplar.get("category", "")).strip(),
                "difficulty": str(exemplar.get("difficulty", "")).strip(),
                "query_type": str(exemplar.get("query_type", "")).strip(),
                "query": str(exemplar.get("query", "")).strip(),
                "hit3_changed": int(min(hit3) != max(hit3)),
                "hit5_changed": int(min(hit5) != max(hit5)),
                "rank_min": min(ranks_for_spread),
                "rank_max": max(ranks_for_spread),
                "rank_spread": max(ranks_for_spread) - min(ranks_for_spread),
                "rank_spread_hit_only": (max(hit_ranks) - min(hit_ranks)) if len(hit_ranks) >= 2 else 0,
                "rr_mean": _safe_div(sum(rr_vals), len(rr_vals)),
                "rr_min": min(rr_vals),
                "rr_max": max(rr_vals),
            }
        )

    # 波动最大的题优先展示
    out.sort(
        key=lambda x: (x["hit5_changed"], x["rank_spread_hit_only"], x["rank_spread"], -x["rr_mean"]),
        reverse=True,
    )
    return out


# ========== 文件输出 ==========


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写 UTF-8 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    """写格式化 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown_report(
    *,
    run_dir: Path,
    overall_rows: list[dict[str, Any]],
    spread: dict[str, dict[str, float]],
    variability_rows: list[dict[str, Any]],
    stratified_rows: list[dict[str, Any]],
    top_n: int,
) -> str:
    """生成可阅读 markdown 摘要。"""

    # 按 mrr 排序得到 overall leaderboard
    overall_sorted = sorted(overall_rows, key=lambda x: (x["mrr"], x["r5"], x["r3"]), reverse=True)
    best = overall_sorted[0] if overall_sorted else None
    worst = overall_sorted[-1] if overall_sorted else None

    # hard 子集排行榜
    hard_rows = [r for r in stratified_rows if r["subset_kind"] == "difficulty" and r["subset_value"] == "hard"]
    hard_sorted = sorted(hard_rows, key=lambda x: (x["mrr"], x["r5"]), reverse=True)

    hit5_changed = sum(1 for r in variability_rows if r["hit5_changed"] == 1)
    hit3_changed = sum(1 for r in variability_rows if r["hit3_changed"] == 1)

    lines: list[str] = []
    lines.append("# Matrix 区分度分析报告")
    lines.append("")
    lines.append(f"- Run 目录: `{run_dir}`")
    lines.append(f"- 组合数: {len(overall_rows)}")
    lines.append(f"- 题目数: {len(variability_rows)}")
    lines.append("")

    lines.append("## 1) 整体区分度")
    lines.append("")
    for key, label in [("r1", "R@1"), ("r3", "R@3"), ("r5", "R@5"), ("mrr", "MRR")]:
        s = spread.get(key, {})
        lines.append(
            f"- {label}: min={s.get('min', 0):.4f}, max={s.get('max', 0):.4f}, spread={s.get('spread', 0):.4f}, std={s.get('std', 0):.4f}"
        )
    lines.append("")

    if best and worst:
        lines.append(
            f"- 最优组合（按 MRR）: `{best['combo']}` (MRR={best['mrr']:.4f}, R@5={best['r5']:.4f})"
        )
        lines.append(
            f"- 最差组合（按 MRR）: `{worst['combo']}` (MRR={worst['mrr']:.4f}, R@5={worst['r5']:.4f})"
        )
    lines.append("")

    lines.append("## 2) 题目级波动")
    lines.append("")
    lines.append(f"- hit@5 会随组合变化的题目数: {hit5_changed}/{len(variability_rows)}")
    lines.append(f"- hit@3 会随组合变化的题目数: {hit3_changed}/{len(variability_rows)}")
    lines.append("")
    lines.append(f"- 波动最大的前 {top_n} 题（按 hit5_changed + rank_spread 排序）:")
    for row in variability_rows[:top_n]:
        lines.append(
            f"  - {row['query_id']} | {row['difficulty']}/{row['query_type']} | "
            f"hit5_changed={row['hit5_changed']} | "
            f"rank_spread_hit_only={row['rank_spread_hit_only']} | "
            f"rank_spread={row['rank_spread']} | rr_mean={row['rr_mean']:.3f}"
        )
    lines.append("")

    lines.append("## 3) Hard 子集排行榜（按 MRR）")
    lines.append("")
    for row in hard_sorted[: max(5, top_n)]:
        lines.append(
            f"- `{row['combo']}`: MRR={row['mrr']:.4f}, R@1={row['r1']:.4f}, R@5={row['r5']:.4f}, n={row['n']}"
        )
    lines.append("")

    lines.append("## 4) 结论建议")
    lines.append("")
    lines.append("- 若整体 R@5 差距小但 R@1/MRR 差距明显，说明差异主要体现在排序质量而非“是否命中”。")
    lines.append("- 建议后续主指标使用：`R@1 + MRR (+ nDCG@3)`，而非仅看 R@5。")
    lines.append("- 建议固定输出 hard/comparison/multi_hop 子榜单，避免平均值掩盖差异。")

    return "\n".join(lines) + "\n"


# ========== 主流程 ==========


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze discrimination of matrix benchmark run")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/matrix_bench/run_YYYYMMDD_HHMMSS")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory; default is <run-dir>/analysis",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top-N rows to show in markdown report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    combo_rows = load_combo_rows(run_dir)
    if not combo_rows:
        raise SystemExit(f"no rows json found under: {run_dir}")

    # 1) 每个组合的 overall 指标
    overall_metrics: dict[str, Metrics] = {}
    overall_rows: list[dict[str, Any]] = []
    for combo, rows in sorted(combo_rows.items()):
        m = compute_metrics(rows)
        overall_metrics[combo] = m
        overall_rows.append(
            {
                "combo": combo,
                "n": m.n,
                "r1": m.r1,
                "r3": m.r3,
                "r5": m.r5,
                "mrr": m.mrr,
            }
        )

    # 2) 分层指标（category / difficulty / query_type）
    stratified_rows: list[dict[str, Any]] = []
    for combo, rows in sorted(combo_rows.items()):
        for kind in ("category", "difficulty", "query_type"):
            values = sorted({str(r.get(kind, "")).strip() for r in rows if str(r.get(kind, "")).strip()})
            for value in values:
                sub = _subset_rows(rows, kind, value)
                m = compute_metrics(sub)
                stratified_rows.append(
                    {
                        "combo": combo,
                        "subset_kind": kind,
                        "subset_value": value,
                        "n": m.n,
                        "r1": m.r1,
                        "r3": m.r3,
                        "r5": m.r5,
                        "mrr": m.mrr,
                    }
                )

    # 3) 区分度汇总
    spread = summarize_spread(overall_metrics)
    variability_rows = query_variability(combo_rows)

    summary = {
        "run_dir": str(run_dir),
        "combo_count": len(combo_rows),
        "query_count": len(variability_rows),
        "spread": spread,
        "hit5_changed_queries": sum(1 for r in variability_rows if r["hit5_changed"] == 1),
        "hit3_changed_queries": sum(1 for r in variability_rows if r["hit3_changed"] == 1),
    }

    # 4) 落盘
    write_csv(
        out_dir / "combo_overall.csv",
        sorted(overall_rows, key=lambda x: (x["mrr"], x["r5"], x["r3"]), reverse=True),
        ["combo", "n", "r1", "r3", "r5", "mrr"],
    )
    write_csv(
        out_dir / "combo_stratified.csv",
        sorted(stratified_rows, key=lambda x: (x["subset_kind"], x["subset_value"], -x["mrr"])),
        ["combo", "subset_kind", "subset_value", "n", "r1", "r3", "r5", "mrr"],
    )
    write_csv(
        out_dir / "query_variability.csv",
        variability_rows,
        [
            "query_id",
            "category",
            "difficulty",
            "query_type",
            "query",
            "hit3_changed",
            "hit5_changed",
            "rank_min",
            "rank_max",
            "rank_spread",
            "rank_spread_hit_only",
            "rr_mean",
            "rr_min",
            "rr_max",
        ],
    )
    write_json(out_dir / "discrimination_summary.json", summary)

    report_md = build_markdown_report(
        run_dir=run_dir,
        overall_rows=overall_rows,
        spread=spread,
        variability_rows=variability_rows,
        stratified_rows=stratified_rows,
        top_n=max(1, args.top_n),
    )
    (out_dir / "discrimination_report.md").write_text(report_md, encoding="utf-8")

    # 5) 控制台打印关键信息，方便一眼判断
    print(f"[OK] analyzed run: {run_dir}")
    print(f"[OK] combo_count={len(combo_rows)} query_count={len(variability_rows)}")
    print(
        "[OK] spread "
        f"R@1={spread['r1']['spread']:.4f}, "
        f"R@3={spread['r3']['spread']:.4f}, "
        f"R@5={spread['r5']['spread']:.4f}, "
        f"MRR={spread['mrr']['spread']:.4f}"
    )
    print(f"[OK] outputs: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
