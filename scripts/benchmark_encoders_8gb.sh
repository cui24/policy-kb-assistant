#!/usr/bin/env bash
set -euo pipefail

# 编码器对比实验脚本（面向 8GB 显存机器）
#
# 目标：
# 1) 固定切块策略（默认 APP_LEVEL=l0），只比较 EMBED_MODEL 差异。
# 2) 对每个编码器执行：入库 -> 评测 -> 汇总。
# 3) 输出统一结果到 outputs/encoder_bench，方便横向对比。
#
# 默认评测模式：
# - 先跑检索层（--skip-answer），不消耗 LLM。
# - 需要回答层再加 --with-answer。
#
# 用法：
#   bash scripts/benchmark_encoders_8gb.sh
#   bash scripts/benchmark_encoders_8gb.sh --with-answer
#   bash scripts/benchmark_encoders_8gb.sh --models-file data/eval/encoder_candidates_8gb.txt --app-level l0

MODE="retrieval"
MODELS_FILE="data/eval/encoder_candidates_8gb.txt"
APP_LEVEL="l0"
EVAL_SET="data/eval/policy_eval_100.csv"
TOP_K="8"
OUT_DIR="outputs/encoder_bench"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-answer)
      MODE="full"
      shift
      ;;
    --models-file)
      MODELS_FILE="$2"
      shift 2
      ;;
    --app-level)
      APP_LEVEL="$2"
      shift 2
      ;;
    --eval-set)
      EVAL_SET="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$MODELS_FILE" ]]; then
  echo "[ERROR] models file not found: $MODELS_FILE" >&2
  exit 2
fi

if [[ ! -f "$EVAL_SET" ]]; then
  echo "[ERROR] eval set not found: $EVAL_SET" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_DIR/run_${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "[INFO] run_dir=$RUN_DIR"
echo "[INFO] mode=$MODE app_level=$APP_LEVEL top_k=$TOP_K"
echo "[INFO] models_file=$MODELS_FILE"

slugify_model() {
  # 统一 collection 命名，避免特殊字符
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's#[/ ]#_#g; s#[^a-z0-9._-]#-#g'
}

while IFS= read -r raw_model; do
  model="$(echo "$raw_model" | sed 's/#.*$//' | xargs)"
  if [[ -z "$model" ]]; then
    continue
  fi

  slug="$(slugify_model "$model")"
  collection="policy_kb_enc_${slug}_v1"
  one_dir="$RUN_DIR/$slug"
  mkdir -p "$one_dir"

  echo
  echo "=================================================="
  echo "[INFO] model=$model"
  echo "[INFO] collection=$collection"
  echo "=================================================="

  # 1) 入库（固定切块，仅比较编码器）
  start_ingest="$(date +%s)"
  EMBED_MODEL="$model" \
  QDRANT_COLLECTION="$collection" \
  APP_LEVEL="$APP_LEVEL" \
  python -m src.kb.ingest | tee "$one_dir/ingest.log"
  end_ingest="$(date +%s)"
  ingest_sec="$((end_ingest - start_ingest))"

  # ingest 报告默认写 outputs/ingest_report.json，这里按模型归档一份
  if [[ -f "outputs/ingest_report.json" ]]; then
    cp "outputs/ingest_report.json" "$one_dir/ingest_report.json"
  fi

  # 2) 评测（检索层或全链路）
  start_eval="$(date +%s)"
  if [[ "$MODE" == "retrieval" ]]; then
    python scripts/evaluate_policy_eval_set.py \
      --eval-set "$EVAL_SET" \
      --collections "$collection" \
      --top-k "$TOP_K" \
      --skip-answer \
      --out-dir "$one_dir/eval" | tee "$one_dir/eval.log"
  else
    python scripts/evaluate_policy_eval_set.py \
      --eval-set "$EVAL_SET" \
      --collections "$collection" \
      --top-k "$TOP_K" \
      --out-dir "$one_dir/eval" | tee "$one_dir/eval.log"
  fi
  end_eval="$(date +%s)"
  eval_sec="$((end_eval - start_eval))"

  # 3) 记录单模型运行摘要
  python - <<PY
import json
from pathlib import Path

out = Path("$one_dir") / "run_meta.json"
payload = {
    "model": "$model",
    "collection": "$collection",
    "app_level": "$APP_LEVEL",
    "mode": "$MODE",
    "top_k": int("$TOP_K"),
    "ingest_seconds": int("$ingest_sec"),
    "eval_seconds": int("$eval_sec"),
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] wrote {out}")
PY

done < "$MODELS_FILE"

# 4) 聚合排行榜
python - <<PY
import csv
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
rows = []
for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
    meta_path = model_dir / "run_meta.json"
    summary_candidates = list((model_dir / "eval").glob("*.summary.json"))
    if not meta_path.exists() or not summary_candidates:
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # evaluate_policy_eval_set.py 单 collection 会产出一个 summary
    summary = json.loads(summary_candidates[0].read_text(encoding="utf-8"))
    row = {
        "model": meta["model"],
        "collection": meta["collection"],
        "mode": meta["mode"],
        "ingest_seconds": meta["ingest_seconds"],
        "eval_seconds": meta["eval_seconds"],
        "recall_at_3": summary.get("recall_at_3", 0.0),
        "recall_at_5": summary.get("recall_at_5", 0.0),
        "mrr": summary.get("mrr", 0.0),
        "answer_point_coverage_avg": summary.get("answer_point_coverage_avg", 0.0),
        "citation_rate": summary.get("citation_rate", 0.0),
        "refusal_rate": summary.get("refusal_rate", 0.0),
    }
    rows.append(row)

rows.sort(key=lambda r: (r["recall_at_5"], r["mrr"], r["answer_point_coverage_avg"]), reverse=True)

out_csv = run_dir / "leaderboard.csv"
with out_csv.open("w", encoding="utf-8", newline="") as wf:
    writer = csv.DictWriter(
        wf,
        fieldnames=[
            "model",
            "collection",
            "mode",
            "ingest_seconds",
            "eval_seconds",
            "recall_at_3",
            "recall_at_5",
            "mrr",
            "answer_point_coverage_avg",
            "citation_rate",
            "refusal_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] leaderboard: {out_csv}")
for idx, row in enumerate(rows, start=1):
    print(
        f"{idx:02d}. {row['model']} "
        f"R@5={row['recall_at_5']:.3f} "
        f"MRR={row['mrr']:.3f} "
        f"APC={row['answer_point_coverage_avg']:.3f}"
    )
PY

echo
echo "[DONE] encoder benchmark finished. see: $RUN_DIR"

