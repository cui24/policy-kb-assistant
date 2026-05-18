#!/usr/bin/env bash
set -euo pipefail

# 编码器 × 切分策略矩阵实验脚本（8GB 显存机器）
#
# 默认组合：
# - 编码器：data/eval/encoder_candidates_8gb.txt
# - 策略：fixed,overlap,structured_hybrid
# - 模式：先跑检索层（--skip-answer）
#
# 用法示例：
#   bash scripts/benchmark_matrix_8gb.sh
#   bash scripts/benchmark_matrix_8gb.sh --with-answer
#   bash scripts/benchmark_matrix_8gb.sh --strategies fixed,overlap,structured_hybrid

MODE="retrieval"
MODELS_FILE="data/eval/encoder_candidates_8gb.txt"
STRATEGIES="fixed,overlap,structured_hybrid"
APP_LEVEL="l0"
EVAL_SET="data/eval/policy_eval_100.csv"
TOP_K="8"
OUT_DIR="outputs/matrix_bench"

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
    --strategies)
      STRATEGIES="$2"
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
echo "[INFO] strategies=$STRATEGIES"

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's#[/ ]#_#g; s#[^a-z0-9._-]#-#g'
}

set_strategy_env() {
  local strategy="$1"
  case "$strategy" in
    fixed)
      export KB_CHUNK_STRATEGY="fixed"
      export KB_FIXED_CHUNK_SIZE="800"
      export KB_FIXED_OVERLAP="0"
      ;;
    overlap)
      export KB_CHUNK_STRATEGY="overlap"
      export KB_OVERLAP_CHUNK_SIZE="800"
      export KB_OVERLAP_OVERLAP="150"
      ;;
    structured_hybrid)
      export KB_CHUNK_STRATEGY="structured_hybrid"
      export KB_STRUCTURED_CHUNK_SIZE="800"
      export KB_STRUCTURED_SECONDARY_OVERLAP="100"
      export KB_STRUCTURED_MERGE_SHORT="260"
      ;;
    *)
      echo "[ERROR] unsupported strategy: $strategy" >&2
      exit 2
      ;;
  esac
}

IFS=',' read -r -a STRATEGY_ARR <<< "$STRATEGIES"
for i in "${!STRATEGY_ARR[@]}"; do
  STRATEGY_ARR[$i]="$(echo "${STRATEGY_ARR[$i]}" | xargs)"
done

while IFS= read -r raw_model; do
  model="$(echo "$raw_model" | sed 's/#.*$//' | xargs)"
  if [[ -z "$model" ]]; then
    continue
  fi
  model_slug="$(slugify "$model")"

  for strategy in "${STRATEGY_ARR[@]}"; do
    if [[ -z "$strategy" ]]; then
      continue
    fi
    set_strategy_env "$strategy"

    combo_slug="${strategy}__${model_slug}"
    collection="policy_kb_${combo_slug}_v1"
    one_dir="$RUN_DIR/$combo_slug"
    mkdir -p "$one_dir"

    echo
    echo "=================================================="
    echo "[INFO] model=$model"
    echo "[INFO] strategy=$strategy"
    echo "[INFO] collection=$collection"
    echo "=================================================="

    # 1) 入库
    start_ingest="$(date +%s)"
    EMBED_MODEL="$model" \
    QDRANT_COLLECTION="$collection" \
    APP_LEVEL="$APP_LEVEL" \
    python -m src.kb.ingest | tee "$one_dir/ingest.log"
    end_ingest="$(date +%s)"
    ingest_sec="$((end_ingest - start_ingest))"

    if [[ -f "outputs/ingest_report.json" ]]; then
      cp "outputs/ingest_report.json" "$one_dir/ingest_report.json"
    fi

    # 2) 评测
    start_eval="$(date +%s)"
    if [[ "$MODE" == "retrieval" ]]; then
      EMBED_MODEL="$model" \
      python scripts/evaluate_policy_eval_set.py \
        --eval-set "$EVAL_SET" \
        --collections "$collection" \
        --top-k "$TOP_K" \
        --skip-answer \
        --out-dir "$one_dir/eval" | tee "$one_dir/eval.log"
    else
      EMBED_MODEL="$model" \
      python scripts/evaluate_policy_eval_set.py \
        --eval-set "$EVAL_SET" \
        --collections "$collection" \
        --top-k "$TOP_K" \
        --out-dir "$one_dir/eval" | tee "$one_dir/eval.log"
    fi
    end_eval="$(date +%s)"
    eval_sec="$((end_eval - start_eval))"

    # 3) 单组合摘要
    python - <<PY
import json
from pathlib import Path

out = Path("$one_dir") / "run_meta.json"
payload = {
    "model": "$model",
    "model_slug": "$model_slug",
    "strategy": "$strategy",
    "collection": "$collection",
    "app_level": "$APP_LEVEL",
    "mode": "$MODE",
    "top_k": int("$TOP_K"),
    "ingest_seconds": int("$ingest_sec"),
    "eval_seconds": int("$eval_sec"),
    "chunk_env": {
        "KB_CHUNK_STRATEGY": "$KB_CHUNK_STRATEGY",
        "KB_FIXED_CHUNK_SIZE": "${KB_FIXED_CHUNK_SIZE:-}",
        "KB_FIXED_OVERLAP": "${KB_FIXED_OVERLAP:-}",
        "KB_OVERLAP_CHUNK_SIZE": "${KB_OVERLAP_CHUNK_SIZE:-}",
        "KB_OVERLAP_OVERLAP": "${KB_OVERLAP_OVERLAP:-}",
        "KB_STRUCTURED_CHUNK_SIZE": "${KB_STRUCTURED_CHUNK_SIZE:-}",
        "KB_STRUCTURED_SECONDARY_OVERLAP": "${KB_STRUCTURED_SECONDARY_OVERLAP:-}",
        "KB_STRUCTURED_MERGE_SHORT": "${KB_STRUCTURED_MERGE_SHORT:-}",
    },
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] wrote {out}")
PY
  done
done < "$MODELS_FILE"

# 4) 聚合排行榜
python - <<PY
import csv
import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
rows = []
for combo_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
    meta_path = combo_dir / "run_meta.json"
    summary_candidates = list((combo_dir / "eval").glob("*.summary.json"))
    if not meta_path.exists() or not summary_candidates:
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_candidates[0].read_text(encoding="utf-8"))
    rows.append(
        {
            "model": meta["model"],
            "strategy": meta["strategy"],
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
    )

rows.sort(
    key=lambda r: (r["recall_at_5"], r["mrr"], r["answer_point_coverage_avg"]),
    reverse=True,
)

out_csv = run_dir / "leaderboard.csv"
with out_csv.open("w", encoding="utf-8", newline="") as wf:
    writer = csv.DictWriter(
        wf,
        fieldnames=[
            "model",
            "strategy",
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
        f"{idx:02d}. [{row['strategy']}] {row['model']} "
        f"R@5={row['recall_at_5']:.3f} "
        f"MRR={row['mrr']:.3f} "
        f"APC={row['answer_point_coverage_avg']:.3f} "
        f"Ingest={row['ingest_seconds']}s Eval={row['eval_seconds']}s"
    )
PY

echo
echo "[DONE] matrix benchmark finished. see: $RUN_DIR"
