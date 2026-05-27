#!/usr/bin/env bash
# Source this file before creating Python environments or running uv/pip installs.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file instead of executing it:"
    echo "  source scripts/use_code_storage_env.sh"
    exit 2
fi

if [[ -z "${CODE_ROOT:-}" || "${CODE_ROOT:-}" == /tmp/* ]]; then
    if [[ -e "$HOME/code" ]]; then
        CODE_ROOT="$(readlink -f "$HOME/code")"
    else
        CODE_ROOT="$HOME/code"
    fi
fi
export CODE_ROOT

if [[ -z "${CODE_CACHE_ROOT:-}" || "${CODE_CACHE_ROOT:-}" == /tmp/* ]]; then
    CODE_CACHE_ROOT="$CODE_ROOT/.cache"
fi
export CODE_CACHE_ROOT

_code_env_set_if_tmp_or_empty() {
    local name="$1"
    local value="$2"
    local current="${!name:-}"

    if [[ -z "$current" || "$current" == /tmp/* ]]; then
        export "$name=$value"
    fi
}

_code_env_set_if_tmp_or_empty UV_CACHE_DIR "$CODE_CACHE_ROOT/uv"
_code_env_set_if_tmp_or_empty PIP_CACHE_DIR "$CODE_CACHE_ROOT/pip"
_code_env_set_if_tmp_or_empty HF_HOME "$CODE_CACHE_ROOT/huggingface"
_code_env_set_if_tmp_or_empty TORCH_HOME "$CODE_CACHE_ROOT/torch"
_code_env_set_if_tmp_or_empty TRITON_CACHE_DIR "$CODE_CACHE_ROOT/triton"
_code_env_set_if_tmp_or_empty XDG_CACHE_HOME "$CODE_CACHE_ROOT/xdg"
_code_env_set_if_tmp_or_empty TMPDIR "$CODE_ROOT/.tmp"
_code_env_set_if_tmp_or_empty TMP "$TMPDIR"
_code_env_set_if_tmp_or_empty TEMP "$TMPDIR"

mkdir -p \
    "$UV_CACHE_DIR" \
    "$PIP_CACHE_DIR" \
    "$HF_HOME" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$TMPDIR"

unset -f _code_env_set_if_tmp_or_empty

if [[ "${CODE_STORAGE_ENV_QUIET:-0}" != "1" ]]; then
    echo "CODE_ROOT=$CODE_ROOT"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "TMPDIR=$TMPDIR"
fi
