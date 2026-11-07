#!/bin/bash
# One-time: venv-nnunet in $PROJ for B200 (sm_100): Python 3.11 (uv), torch 2.15.0 (cu132), nnunetv2 2.6.4 (it excludes torch 2.9).
# Runs on a CPU session too. Replaces the thesis setup.sh (torch 2.3.0+cu121 has no Blackwell kernels).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../maisi/env.sh"
deactivate 2>/dev/null || true
cd "$PROJ"
[ -d venv-nnunet ] || uv venv --python 3.11 venv-nnunet
source venv-nnunet/bin/activate
# nnunetv2 2.6.4 excludes torch 2.9.x; torch 2.15.0 (PyPI, CUDA 13.2 build) has sm_100 kernels, node driver = CUDA 13.4
uv pip install "torch==2.15.0" "nnunetv2==2.6.4" "SimpleITK" "nibabel" "pandas" "scipy" "scikit-image" "matplotlib" "seaborn"
python - <<'PY'
import torch, nnunetv2, nibabel, SimpleITK
print("torch", torch.__version__, "cuda build", torch.version.cuda, "archs", torch.cuda.get_arch_list() if torch.cuda.is_available() else "(no GPU here)")
print("nnunetv2", nnunetv2.__file__)
PY
uv pip freeze > "$PROJ/venv-nnunet-freeze.txt"
echo "setup_nnunet done"
