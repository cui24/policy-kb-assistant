#!/usr/bin/env python3
"""不同 RAG chunk/window 大小的消融实验脚本。

这个脚本做的是“调度”和“汇总”，不重新实现入库或评测逻辑：

1. 对每个窗口档位创建一个独立的 Qdrant collection。
2. 通过环境变量调整 `src.kb.ingest` 的切块参数并重新入库。
3. 调用 `scripts/evaluate_policy_eval_set.py` 对所有 collection 做横向评测。
4. 汇总入库统计和评测指标，生成 Markdown/JSON 报告。

典型用途：
- 快速只看检索层：加 `--skip-answer`，不调用大模型回答。
- 完整看端到端：不加 `--skip-answer`，会调用回答生成链路，耗时更长。
- 先检查命令：加 `--dry-run`，只打印将要执行的命令。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SET = "data/eval/policy_eval_total_130.csv"
DEFAULT_OUT_DIR = "outputs/rag_window_size_benchmark"


def _env(name: str, default: str) -> str:
    """读取环境变量；变量不存在或为空时返回默认值。"""
    value = str(os.getenv(name) or "").strip()
    return value or default


def _parse_sizes(raw: str) -> list[int]:
    """解析 `400,600,800` 这种档位字符串，并过滤非法值。"""
    sizes: list[int] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            size = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid size: {token!r}") from exc
        if size < 50:
            raise argparse.ArgumentTypeError(f"size must be >= 50: {size}")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one size is required")
    return sizes


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    参数设计思路：
    - sizes 控制实验档位。
    - strategy 默认使用项目当前最有代表性的 structured_hybrid。
    - skip-answer 支持先做低成本检索评测，再决定是否跑完整回答评测。
    """
    parser = argparse.ArgumentParser(description="Benchmark RAG chunk/window sizes across Qdrant collections.")
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=_parse_sizes("400,600,800,1000,1200"),
        help="逗号分隔的 chunk/window 大小档位，例如：400,600,800,1000,1200。",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "overlap", "structured_hybrid"],
        default="structured_hybrid",
        help="入库切块策略，默认测试 structured_hybrid。",
    )
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET, help="评测集 CSV 路径。")
    parser.add_argument("--top-k", type=int, default=8, help="检索 top_k，必须 >= 5。")
    parser.add_argument("--collection-prefix", default="policy_kb_window", help="测试 collection 名称前缀。")
    parser.add_argument("--qdrant-url", default=_env("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--embed-model", default=_env("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="实验输出目录。")
    parser.add_argument("--skip-answer", action="store_true", help="只跑检索层指标，不调用回答生成。")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不实际入库或评测。")
    parser.add_argument("--reuse-existing", action="store_true", help="跳过入库，直接评测已有 collections。")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="不入库、不评测，只根据已有 compare_summary.json 和 ingest report 重建汇总报告。",
    )
    parser.add_argument(
        "--secondary-overlap",
        type=int,
        default=None,
        help="二次切分 overlap 固定值；不传则按 secondary-overlap-ratio 计算。",
    )
    parser.add_argument(
        "--secondary-overlap-ratio",
        type=float,
        default=0.15,
        help="未显式指定 overlap 时，按 size 的这个比例计算 overlap。",
    )
    parser.add_argument(
        "--preserve-soft-max",
        action="store_true",
        help="不覆盖 KB_STRUCTURED_SOFT_MAX_CHARS，保留 ingest.py 默认值或外部环境变量。",
    )
    parser.add_argument("--window-pages", type=int, default=1, help="每个页窗口最多包含的页数。")
    parser.add_argument("--window-overlap-pages", type=int, default=0, help="相邻页窗口重叠页数。")
    parser.add_argument("--window-max-chars", type=int, default=30000, help="单个页窗口最大字符数。")
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    """把相对路径转换成项目根目录下的绝对路径。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _slug(value: str) -> str:
    """生成适合文件名和 collection 名的安全字符串。"""
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", str(value)).strip("_")


def _overlap_for_size(args: argparse.Namespace, size: int) -> int:
    """根据策略和参数计算当前档位使用的 overlap。

    structured_hybrid/overlap 默认使用比例 overlap；fixed 默认不重叠。
    如果用户显式传了 --secondary-overlap，则三个策略都使用这个固定值。
    """
    if args.secondary_overlap is not None:
        return max(0, int(args.secondary_overlap))
    if args.strategy == "fixed":
        return 0
    ratio = max(0.0, float(args.secondary_overlap_ratio))
    return max(0, int(round(size * ratio)))


def _collection_name(prefix: str, strategy: str, size: int, overlap: int) -> str:
    """按参数生成稳定 collection 名，避免不同实验互相覆盖。"""
    return _slug(f"{prefix}_{strategy}_s{size}_o{overlap}")


def _build_env(args: argparse.Namespace, *, size: int, overlap: int, collection: str) -> dict[str, str]:
    """构造某个档位入库和评测时使用的环境变量。"""
    env = dict(os.environ)
    env["QDRANT_URL"] = str(args.qdrant_url)
    env["QDRANT_COLLECTION"] = collection
    env["EMBED_MODEL"] = str(args.embed_model)
    env["KB_CHUNK_STRATEGY"] = str(args.strategy)
    env["KB_WINDOW_PAGES"] = str(max(1, int(args.window_pages)))
    env["KB_WINDOW_OVERLAP_PAGES"] = str(max(0, int(args.window_overlap_pages)))
    env["KB_WINDOW_MAX_CHARS"] = str(max(0, int(args.window_max_chars)))

    if args.strategy == "structured_hybrid":
        env["KB_STRUCTURED_CHUNK_SIZE"] = str(size)
        env["KB_STRUCTURED_SECONDARY_OVERLAP"] = str(overlap)
        if not args.preserve_soft_max:
            # 默认让 soft max 跟随实验档位，否则 ingest.py 默认 520 会削弱大窗口实验差异。
            env["KB_STRUCTURED_SOFT_MAX_CHARS"] = str(size)
    elif args.strategy == "overlap":
        env["KB_OVERLAP_CHUNK_SIZE"] = str(size)
        env["KB_OVERLAP_OVERLAP"] = str(overlap)
    else:
        env["KB_FIXED_CHUNK_SIZE"] = str(size)
        env["KB_FIXED_OVERLAP"] = str(overlap)

    return env


def _run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    """执行子命令；dry-run 时只打印命令。"""
    printable = shlex.join(cmd)
    if dry_run:
        print(f"[DRY_RUN] {printable}")
        return
    print(f"[RUN] {printable}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件；不存在时返回空字典。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """写 JSON 文件，统一使用 UTF-8 和中文友好的缩进格式。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_ingest_report(*, out_dir: Path, collection: str, dry_run: bool) -> dict[str, Any]:
    """保存本轮入库报告副本，避免下一轮入库覆盖 `outputs/ingest_report.json`。"""
    if dry_run:
        return {}
    source = PROJECT_ROOT / "outputs" / "ingest_report.json"
    report = _read_json(source)
    if not report:
        print(f"[WARN] ingest report not found: {source}")
        return {}
    target = out_dir / "ingest_reports" / f"{_slug(collection)}.ingest_report.json"
    _write_json(target, report)
    print(f"[REPORT] copied ingest report: {target.relative_to(PROJECT_ROOT)}")
    return report


def _load_saved_ingest_report(*, out_dir: Path, collection: str) -> dict[str, Any]:
    """读取之前保存过的入库报告，供 reuse-existing/report-only 模式复用。"""
    report_path = out_dir / "ingest_reports" / f"{_slug(collection)}.ingest_report.json"
    report = _read_json(report_path)
    if not report:
        print(f"[WARN] saved ingest report not found: {report_path.relative_to(PROJECT_ROOT)}")
    return report


def _ingest_summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """从入库报告中提取实验对比最需要的字段。"""
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    docs = report.get("documents") if isinstance(report.get("documents"), list) else []
    avg_chars_values = [
        float(item.get("avg_chunk_chars") or 0)
        for item in docs
        if isinstance(item, dict) and float(item.get("avg_chunk_chars") or 0) > 0
    ]
    avg_chunk_chars = sum(avg_chars_values) / len(avg_chars_values) if avg_chars_values else 0.0
    return {
        "total_chunks": int(summary.get("total_chunks") or 0),
        "total_points_in_qdrant": int(summary.get("total_points_in_qdrant") or 0),
        "avg_chunk_chars_by_doc": round(avg_chunk_chars, 2),
        "documents": int(summary.get("documents") or 0),
    }


def _format_float(value: Any, digits: int = 3) -> str:
    """把指标格式化成固定小数；缺失时显示短横线。"""
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _format_int(value: Any) -> str:
    """把整数指标格式化；缺失时显示短横线。"""
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _write_markdown_report(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    experiment_rows: list[dict[str, Any]],
    compare_summary: dict[str, Any],
) -> None:
    """生成便于阅读和放进简历/面试材料的 Markdown 汇总报告。"""
    summaries = {
        str(item.get("collection") or ""): item
        for item in compare_summary.get("collections", [])
        if isinstance(item, dict)
    }

    lines: list[str] = []
    lines.append("# RAG 窗口大小消融实验")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- 切块策略：`{args.strategy}`")
    lines.append(f"- Embedding 模型：`{args.embed_model}`")
    lines.append(f"- 评测集：`{args.eval_set}`")
    lines.append(f"- 检索深度：`top_k={args.top_k}`")
    lines.append(f"- 是否跳过回答生成：`{bool(args.skip_answer)}`")
    lines.append("")
    lines.append("## 对比结果")
    lines.append("")
    header = [
        "size",
        "overlap",
        "chunks",
        "avg_chars",
        "R@3",
        "R@5",
        "MRR",
        "APC",
        "Citation",
        "Refusal",
        "retrieve_p50_ms",
        "retrieve_p95_ms",
        "collection",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in experiment_rows:
        collection = str(row["collection"])
        ingest = row.get("ingest") if isinstance(row.get("ingest"), dict) else {}
        eval_summary = summaries.get(collection, {})
        values = [
            _format_int(row.get("size")),
            _format_int(row.get("overlap")),
            _format_int(ingest.get("total_chunks")),
            _format_float(ingest.get("avg_chunk_chars_by_doc"), 1),
            _format_float(eval_summary.get("recall_at_3")),
            _format_float(eval_summary.get("recall_at_5")),
            _format_float(eval_summary.get("mrr")),
            _format_float(eval_summary.get("answer_point_coverage_avg")),
            _format_float(eval_summary.get("citation_rate")),
            _format_float(eval_summary.get("refusal_rate")),
            _format_int(eval_summary.get("retrieve_ms_p50")),
            _format_int(eval_summary.get("retrieve_ms_p95")),
            f"`{collection}`",
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## 读数建议")
    lines.append("")
    lines.append("- 先看 R@5/MRR：判断不同窗口大小对“能不能检索到正确文档”的影响。")
    lines.append("- 再看 APC/Citation/Refusal：判断检索结果进入回答链路后，答案完整性和拒答是否变好。")
    lines.append("- 结合 chunks/avg_chars：窗口越小通常 chunk 更多、检索更精细；窗口越大上下文更完整，但也可能引入噪声。")
    lines.append("- 这个实验是本地消融结果，面试或简历里建议表述为“在本项目评测集上”，不要泛化成生产指标。")
    lines.append("")

    report_path = out_dir / "window_size_compare.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] wrote markdown: {report_path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    """主流程：按档位入库、统一评测、生成汇总报告。"""
    args = _parse_args()
    if int(args.top_k) < 5:
        raise SystemExit("[ERROR] --top-k must be >= 5")

    eval_set = _project_path(args.eval_set)
    if not eval_set.exists():
        raise SystemExit(f"[ERROR] missing eval set: {eval_set}")

    out_dir = _project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir = out_dir / "policy_eval"

    experiment_rows: list[dict[str, Any]] = []
    collections: list[str] = []

    for size in args.sizes:
        overlap = _overlap_for_size(args, size)
        collection = _collection_name(args.collection_prefix, args.strategy, size, overlap)
        collections.append(collection)
        env = _build_env(args, size=size, overlap=overlap, collection=collection)

        print(
            "[PLAN]",
            f"size={size}",
            f"overlap={overlap}",
            f"collection={collection}",
            f"reuse_existing={bool(args.reuse_existing)}",
            f"report_only={bool(args.report_only)}",
        )
        report: dict[str, Any] = {}
        if args.reuse_existing or args.report_only:
            report = _load_saved_ingest_report(out_dir=out_dir, collection=collection)
        if not args.reuse_existing and not args.report_only:
            _run_command([sys.executable, "-m", "src.kb.ingest"], env=env, dry_run=bool(args.dry_run))
            report = _copy_ingest_report(out_dir=out_dir, collection=collection, dry_run=bool(args.dry_run))

        experiment_rows.append(
            {
                "size": size,
                "overlap": overlap,
                "strategy": args.strategy,
                "collection": collection,
                "ingest": _ingest_summary_from_report(report),
            }
        )

    base_env = dict(os.environ)
    base_env["QDRANT_URL"] = str(args.qdrant_url)
    base_env["EMBED_MODEL"] = str(args.embed_model)

    if args.report_only:
        compare_summary = _read_json(eval_out_dir / "compare_summary.json")
        if not compare_summary:
            raise SystemExit(f"[ERROR] missing compare summary: {eval_out_dir / 'compare_summary.json'}")
    else:
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_policy_eval_set.py",
            "--eval-set",
            str(eval_set),
            "--collections",
            ",".join(collections),
            "--top-k",
            str(args.top_k),
            "--out-dir",
            str(eval_out_dir),
        ]
        if args.skip_answer:
            eval_cmd.append("--skip-answer")
        _run_command(eval_cmd, env=base_env, dry_run=bool(args.dry_run))

        compare_summary = {} if args.dry_run else _read_json(eval_out_dir / "compare_summary.json")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            "sizes": args.sizes,
            "strategy": args.strategy,
            "eval_set": str(eval_set),
            "top_k": int(args.top_k),
            "skip_answer": bool(args.skip_answer),
            "qdrant_url": str(args.qdrant_url),
            "embed_model": str(args.embed_model),
            "reuse_existing": bool(args.reuse_existing),
        },
        "experiments": experiment_rows,
        "evaluation": compare_summary,
    }
    if not args.dry_run:
        _write_json(out_dir / "window_size_compare.json", result)
        _write_markdown_report(
            out_dir=out_dir,
            args=args,
            experiment_rows=experiment_rows,
            compare_summary=compare_summary,
        )
    else:
        print("[DRY_RUN] no files written")

    print("[DONE] RAG window size benchmark finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
