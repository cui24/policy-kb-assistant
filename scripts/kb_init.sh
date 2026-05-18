#!/usr/bin/env bash
set -euo pipefail

if python - <<'PY'
import json
import os
import sys
import time
import urllib.request

qdrant_url = str(os.getenv("QDRANT_URL", "http://qdrant:6333")).rstrip("/")
collection = str(os.getenv("QDRANT_COLLECTION", "policy_kb_l0")).strip() or "policy_kb_l0"
wait_seconds = int(os.getenv("KB_INIT_WAIT_SECONDS", "120"))

deadline = time.time() + max(1, wait_seconds)
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{qdrant_url}/collections", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        break
    except Exception as exc:
        last_error = exc
        time.sleep(2)
else:
    print(f"[KB-INIT] qdrant is not ready after {wait_seconds}s: {last_error}", file=sys.stderr)
    sys.exit(2)

collections = {
    str(item.get("name") or "")
    for item in (payload.get("result", {}).get("collections") or [])
}
if collection not in collections:
    print(f"[KB-INIT] collection {collection!r} missing; need ingest")
    sys.exit(10)

with urllib.request.urlopen(f"{qdrant_url}/collections/{collection}", timeout=5) as response:
    detail = json.loads(response.read().decode("utf-8"))

points_count = int((detail.get("result") or {}).get("points_count") or 0)
if points_count <= 0:
    print(f"[KB-INIT] collection {collection!r} has no points; need ingest")
    sys.exit(10)

print(
    f"[KB-INIT] collection {collection!r} already ready with points_count={points_count}; skip ingest"
)
sys.exit(0)
PY
then
    echo "[KB-INIT] done"
else
    status="$?"
    if [ "$status" -eq 10 ]; then
        echo "[KB-INIT] running ingest..."
        python -m src.kb.ingest
    else
        exit "$status"
    fi
fi
