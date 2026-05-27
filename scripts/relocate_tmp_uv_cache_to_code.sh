#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/use_code_storage_env.sh
source "$repo_root/scripts/use_code_storage_env.sh"

tmp_cache="${1:-/tmp/uv-cache}"
target="$UV_CACHE_DIR"

if [[ "$target" == /tmp/* ]]; then
    echo "Refusing to point uv cache at $target" >&2
    exit 1
fi

mkdir -p "$target"

if [[ -L "$tmp_cache" ]]; then
    linked_target="$(readlink -f "$tmp_cache")"
    echo "$tmp_cache is already a symlink to $linked_target"
    exit 0
fi

if [[ -e "$tmp_cache" ]]; then
    migrated_root="$CODE_CACHE_ROOT/uv-migrated"
    migrated_path="$migrated_root/$(hostname)-$(date +%Y%m%d%H%M%S)"
    mkdir -p "$migrated_root"
    mv "$tmp_cache" "$migrated_path"
    echo "Moved existing $tmp_cache to $migrated_path"
fi

ln -s "$target" "$tmp_cache"
echo "Created $tmp_cache -> $target"
