#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/use_code_storage_env.sh
source "$repo_root/scripts/use_code_storage_env.sh"

env_prefix="${1:-$repo_root/.conda/ltx23}"
python_version="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not on PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "$env_prefix")"

if [[ ! -x "$env_prefix/bin/python" ]]; then
    conda create -y -p "$env_prefix" "python=$python_version"
fi

"$env_prefix/bin/python" -m pip install --upgrade pip uv

hook_dir="$env_prefix/etc/conda/activate.d"
mkdir -p "$hook_dir"
cat > "$hook_dir/code_storage_env.sh" <<EOF_HOOK_SH
export CODE_STORAGE_ENV_QUIET=1
source "$repo_root/scripts/use_code_storage_env.sh"
unset CODE_STORAGE_ENV_QUIET
EOF_HOOK_SH
cat > "$hook_dir/code_storage_env.csh" <<EOF_HOOK_CSH
setenv CODE_STORAGE_ENV_QUIET 1
source "$repo_root/scripts/use_code_storage_env.csh"
unsetenv CODE_STORAGE_ENV_QUIET
EOF_HOOK_CSH

cat <<EOF

Environment is ready at:
  $env_prefix

Conda activation hooks were installed under:
  $hook_dir

Before uv/pip installs or model runs, use:
  source $repo_root/scripts/use_code_storage_env.sh
  conda activate $env_prefix

Example install command:
  uv pip install -e "$repo_root/python[diffusion]" --prerelease=allow
EOF
