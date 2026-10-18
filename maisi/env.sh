# Source this in every new UCloud terminal:  source /work/<YourFolder>/BT2026/maisi/env.sh
# Sets PROJ (the persistent /work folder that contains BT2026), activates venv-maisi
# and points all caches into PROJ so nothing large lands in the ephemeral home dir.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ="${PROJ:-$(dirname "$(dirname "$_here")")}"
export MAISI_DIR="${MAISI_DIR:-$PROJ/NV-Generate-CTMR}"
export MAISI_RUNS="${MAISI_RUNS:-$PROJ/maisi_runs}"

export UV_INSTALL_DIR="$PROJ/tools"
export UV_CACHE_DIR="$PROJ/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJ/tools/python"
export HF_HOME="$PROJ/.cache/huggingface"
export PATH="$PROJ/tools:$PATH"

mkdir -p "$PROJ/logs" "$MAISI_RUNS"

if [ -f "$PROJ/venv-maisi/bin/activate" ]; then
    source "$PROJ/venv-maisi/bin/activate"
fi
unset _here
