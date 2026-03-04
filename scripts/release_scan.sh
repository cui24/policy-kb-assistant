#!/usr/bin/env bash
set -euo pipefail

# Release scan helper:
# Lists large files, local-only artifacts, and suspicious hardcoded secret patterns.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

printf '== Release scan in %s ==\n' "$ROOT_DIR"

printf '\n[1/3] Files larger than 10MB\n'
LARGE_FILES="$(find . -type f -size +10M ! -path './.git/*' | sort || true)"
if [[ -n "$LARGE_FILES" ]]; then
  printf '%s\n' "$LARGE_FILES"
else
  printf '(none)\n'
fi

printf '\n[2/3] Common local-only artifacts currently present\n'
LOCAL_ARTIFACTS="$(find . -maxdepth 2 \( -name '.env' -o -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' -o -name '__pycache__' -o -name '.pytest_cache' -o -name 'logs' -o -name 'outputs' -o -name 'tmp' -o -name 'qdrant_storage' -o -name '.specstory' \) | sort || true)"
if [[ -n "$LOCAL_ARTIFACTS" ]]; then
  printf '%s\n' "$LOCAL_ARTIFACTS"
else
  printf '(none)\n'
fi

printf '\n[3/3] Suspicious secret-like patterns\n'
if command -v rg >/dev/null 2>&1; then
  MATCHES="$(rg -n --hidden -g '!*.pyc' -g '!.git/*' -g '!data/**' -g '!outputs/**' -g '!logs/**' -g '!*.pdf' -g '!scripts/release_scan.sh' "(OPENAI_API_KEY|POLICY_API_KEY|sk-[A-Za-z0-9]|-----BEGIN [A-Z ]*PRIVATE KEY-----|api_key\\s*=\\s*['\\\"])" . || true)"
else
  MATCHES=""
fi

if [[ -n "$MATCHES" ]]; then
  printf '%s\n' "$MATCHES"
else
  printf '(none)\n'
fi

printf '\n== Done ==\n'
