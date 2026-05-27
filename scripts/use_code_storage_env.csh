# Source this file before creating Python environments or running uv/pip installs.

if ( ! $?CODE_ROOT ) then
    if ( -e "$HOME/code" ) then
        setenv CODE_ROOT `readlink -f "$HOME/code"`
    else
        setenv CODE_ROOT "$HOME/code"
    endif
else if ( "$CODE_ROOT" =~ /tmp/* ) then
    if ( -e "$HOME/code" ) then
        setenv CODE_ROOT `readlink -f "$HOME/code"`
    else
        setenv CODE_ROOT "$HOME/code"
    endif
endif

if ( ! $?CODE_CACHE_ROOT ) then
    setenv CODE_CACHE_ROOT "$CODE_ROOT/.cache"
else if ( "$CODE_CACHE_ROOT" =~ /tmp/* ) then
    setenv CODE_CACHE_ROOT "$CODE_ROOT/.cache"
endif

if ( ! $?UV_CACHE_DIR ) then
    setenv UV_CACHE_DIR "$CODE_CACHE_ROOT/uv"
else if ( "$UV_CACHE_DIR" =~ /tmp/* ) then
    setenv UV_CACHE_DIR "$CODE_CACHE_ROOT/uv"
endif

if ( ! $?PIP_CACHE_DIR ) then
    setenv PIP_CACHE_DIR "$CODE_CACHE_ROOT/pip"
else if ( "$PIP_CACHE_DIR" =~ /tmp/* ) then
    setenv PIP_CACHE_DIR "$CODE_CACHE_ROOT/pip"
endif

if ( ! $?HF_HOME ) then
    setenv HF_HOME "$CODE_CACHE_ROOT/huggingface"
else if ( "$HF_HOME" =~ /tmp/* ) then
    setenv HF_HOME "$CODE_CACHE_ROOT/huggingface"
endif

if ( ! $?TORCH_HOME ) then
    setenv TORCH_HOME "$CODE_CACHE_ROOT/torch"
else if ( "$TORCH_HOME" =~ /tmp/* ) then
    setenv TORCH_HOME "$CODE_CACHE_ROOT/torch"
endif

if ( ! $?TRITON_CACHE_DIR ) then
    setenv TRITON_CACHE_DIR "$CODE_CACHE_ROOT/triton"
else if ( "$TRITON_CACHE_DIR" =~ /tmp/* ) then
    setenv TRITON_CACHE_DIR "$CODE_CACHE_ROOT/triton"
endif

if ( ! $?XDG_CACHE_HOME ) then
    setenv XDG_CACHE_HOME "$CODE_CACHE_ROOT/xdg"
else if ( "$XDG_CACHE_HOME" =~ /tmp/* ) then
    setenv XDG_CACHE_HOME "$CODE_CACHE_ROOT/xdg"
endif

if ( ! $?TMPDIR ) then
    setenv TMPDIR "$CODE_ROOT/.tmp"
else if ( "$TMPDIR" =~ /tmp/* ) then
    setenv TMPDIR "$CODE_ROOT/.tmp"
endif

if ( ! $?TMP ) then
    setenv TMP "$TMPDIR"
else if ( "$TMP" =~ /tmp/* ) then
    setenv TMP "$TMPDIR"
endif

if ( ! $?TEMP ) then
    setenv TEMP "$TMPDIR"
else if ( "$TEMP" =~ /tmp/* ) then
    setenv TEMP "$TMPDIR"
endif

mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

set _code_storage_env_quiet = 0
if ( $?CODE_STORAGE_ENV_QUIET ) then
    if ( "$CODE_STORAGE_ENV_QUIET" == "1" ) then
        set _code_storage_env_quiet = 1
    endif
endif

if ( "$_code_storage_env_quiet" != "1" ) then
    echo "CODE_ROOT=$CODE_ROOT"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "TMPDIR=$TMPDIR"
endif
unset _code_storage_env_quiet
