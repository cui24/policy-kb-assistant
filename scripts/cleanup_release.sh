#!/usr/bin/env bash
set -euo pipefail

# Release cleanup helper:
# Removes common local-only artifacts that should not be committed before publishing.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

remove_path() {
  local target="$1"
  if [[ ! -e "$target" ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY-RUN] remove %s\n' "$target"
  else
    rm -rf "$target"
    printf '[REMOVED] %s\n' "$target"
  fi
}

printf '== Release cleanup in %s ==\n' "$ROOT_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Mode: dry-run\n'
else
  printf 'Mode: apply\n'
fi

remove_path ".pytest_cache"
remove_path ".specstory"
remove_path "logs"
remove_path "outputs"
remove_path "tmp"
remove_path "qdrant_storage"

while IFS= read -r -d '' dir_path; do
  remove_path "$dir_path"
done < <(find . -type d -name '__pycache__' -print0)

while IFS= read -r -d '' file_path; do
  remove_path "$file_path"
done < <(find . -type f \( -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' \) -print0)

printf '== Done ==\n'
