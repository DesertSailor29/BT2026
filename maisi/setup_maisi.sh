#!/usr/bin/env bash
# One-time setup of the MAISI-v2 (rflow-ct) environment on UCloud. Everything goes into $PROJ (persistent).
# Safe to re-run: finished steps are skipped.
#   export PROJ=/work/<YourFolder>
#   nohup bash "$PROJ/BT2026/maisi/setup_maisi.sh" > "$PROJ/logs/setup_maisi_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MAISI_REPO=https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git
MAISI_COMMIT=dba166e
PYTHON_VERSION=3.11
TORCH_VERSION=2.9.1          # cu128 build: has Blackwell (sm_100) kernels
MONAI_VERSION=1.6.1
VENV="$PROJ/venv-maisi"

echo "== $(date) | PROJ=$PROJ"

echo "== 1/6 GPU"
nvidia-smi

echo "== 2/6 uv (Python/package manager, installed into $PROJ/tools)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PROJ/tools" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

echo "== 3/6 NV-Generate-CTMR @ $MAISI_COMMIT"
if [ ! -d "$MAISI_DIR/.git" ]; then
    git clone "$MAISI_REPO" "$MAISI_DIR"
fi
git -C "$MAISI_DIR" fetch --quiet origin
git -C "$MAISI_DIR" checkout --quiet "$MAISI_COMMIT"
git -C "$MAISI_DIR" log --oneline -1

echo "== 4/6 venv-maisi (Python $PYTHON_VERSION, torch $TORCH_VERSION+cu128, MONAI $MONAI_VERSION)"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi
uv pip install --python "$VENV/bin/python" "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu128
# torch==X also matches the installed X+cu128, so this step keeps the CUDA build.
uv pip install --python "$VENV/bin/python" "torch==$TORCH_VERSION" "monai==$MONAI_VERSION" -r "$MAISI_DIR/requirements.txt"
uv pip freeze --python "$VENV/bin/python" > "$PROJ/venv-maisi-freeze.txt"

echo "== 5/6 GPU check inside venv"
"$VENV/bin/python" - <<'EOF'
import torch, monai
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| MONAI", monai.__version__)
assert torch.cuda.is_available(), "CUDA not available"
print("GPU:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok:", float((x @ x).abs().mean()) > 0)
EOF

echo "== 6/6 rflow-ct weights (Path A needs models only, no mask database)"
cd "$MAISI_DIR"
"$VENV/bin/python" -m scripts.download_model_data --version rflow-ct --root_dir ./ --model_only
ls -lh "$MAISI_DIR/models"

echo "== setup done $(date)"
