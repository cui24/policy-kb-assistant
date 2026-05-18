#!/usr/bin/env bash
set -euo pipefail

# 用途：
# 1) 验证 /auth/login 是否会触发限流（429）
# 2) 验证 /agent 是否会触发限流（429）
#
# 使用示例：
#   bash scripts/rate_limit_smoke.sh
#   BASE_URL=http://localhost:8080 LOGIN_LOOP=20 AGENT_LOOP=40 bash scripts/rate_limit_smoke.sh

BASE_URL="${BASE_URL:-http://localhost:8080}"
PASSWORD="${PASSWORD:-TestPass123!}"
LOGIN_LOOP="${LOGIN_LOOP:-15}"
AGENT_LOOP="${AGENT_LOOP:-35}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-15}"
USERNAME="${USERNAME:-rl_user_$(date +%s)}"
IDENTIFIER="${IDENTIFIER:-$USERNAME}"
BAD_PASSWORD="${BAD_PASSWORD:-wrong-password}"
AGENT_TEXT="${AGENT_TEXT:-我宿舍网络连不上，帮我提交报修工单。地点金明校区，手机号13812345678。}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

body_file="$tmp_dir/body.json"
header_file="$tmp_dir/headers.txt"
agent_payload="$(
  python - <<'PY' "$AGENT_TEXT"
import json
import sys
text = str(sys.argv[1] if len(sys.argv) > 1 else "")
print(json.dumps({"text": text}, ensure_ascii=False))
PY
)"

echo "[1/5] health check: ${BASE_URL}/health"
if ! curl -sS --max-time "$REQUEST_TIMEOUT" "${BASE_URL}/health" >"$body_file"; then
  echo "health 检查失败：API 没启动或地址不对"
  exit 1
fi
echo "health: $(cat "$body_file")"

echo "[2/5] register test user: ${USERNAME}"
register_status="$(
  curl -sS --max-time "$REQUEST_TIMEOUT" \
    -o "$body_file" \
    -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -X POST "${BASE_URL}/auth/register" \
    -d "{\"username\":\"${USERNAME}\",\"password\":\"${PASSWORD}\"}"
)"
echo "register status: ${register_status}"
if [[ "${register_status}" != "200" && "${register_status}" != "409" ]]; then
  echo "register 失败，响应：$(tr -d '\n' < "$body_file" | cut -c1-300)"
  exit 1
fi

echo "[3/5] login once to get token"
login_ok_status="$(
  curl -sS --max-time "$REQUEST_TIMEOUT" \
    -o "$body_file" \
    -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -X POST "${BASE_URL}/auth/login" \
    -d "{\"identifier\":\"${IDENTIFIER}\",\"password\":\"${PASSWORD}\"}"
)"
echo "login status: ${login_ok_status}"
if [[ "${login_ok_status}" != "200" ]]; then
  echo "首次登录失败，响应：$(tr -d '\n' < "$body_file" | cut -c1-300)"
  exit 1
fi

access_token="$(
  python - <<'PY' "$body_file"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(payload.get("access_token") or ""))
PY
)"
if [[ -z "${access_token}" ]]; then
  echo "未拿到 access_token，响应：$(tr -d '\n' < "$body_file" | cut -c1-300)"
  exit 1
fi
echo "token acquired: ${access_token:0:20}..."

echo "[4/5] flood /auth/login with wrong password to trigger 429"
login_limited="false"
for i in $(seq 1 "$LOGIN_LOOP"); do
  status="$(
    curl -sS --max-time "$REQUEST_TIMEOUT" \
      -D "$header_file" \
      -o "$body_file" \
      -w "%{http_code}" \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/auth/login" \
      -d "{\"identifier\":\"${IDENTIFIER}\",\"password\":\"${BAD_PASSWORD}\"}"
  )"
  preview="$(tr -d '\n' < "$body_file" | cut -c1-180)"
  echo "  login attempt #${i}: status=${status}, body=${preview}"
  if [[ "${status}" == "429" ]]; then
    login_limited="true"
    echo "  -> login rate limited (429)"
    echo "  -> headers:"
    grep -iE '^retry-after:|^x-ratelimit-' "$header_file" || true
    break
  fi
done
if [[ "${login_limited}" != "true" ]]; then
  echo "  !! 在 ${LOGIN_LOOP} 次内未触发 /auth/login 429"
fi

echo "[5/5] flood /agent to trigger 429"
agent_limited="false"
for i in $(seq 1 "$AGENT_LOOP"); do
  status="$(
    curl -sS --max-time "$REQUEST_TIMEOUT" \
      -D "$header_file" \
      -o "$body_file" \
      -w "%{http_code}" \
      -H "Authorization: Bearer ${access_token}" \
      -H "Content-Type: application/json" \
      -X POST "${BASE_URL}/agent" \
      -d "${agent_payload}"
  )"
  preview="$(tr -d '\n' < "$body_file" | cut -c1-180)"
  echo "  agent attempt #${i}: status=${status}, body=${preview}"
  if [[ "${status}" == "429" ]]; then
    agent_limited="true"
    echo "  -> agent rate limited (429)"
    echo "  -> headers:"
    grep -iE '^retry-after:|^x-ratelimit-' "$header_file" || true
    break
  fi
done
if [[ "${agent_limited}" != "true" ]]; then
  echo "  !! 在 ${AGENT_LOOP} 次内未触发 /agent 429"
fi

echo "done"
