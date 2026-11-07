# Source this for nnU-Net work on UCloud:  source /work/<YourFolder>/BT2026/nnunet/env.sh
# Sets PROJ + caches (via maisi/env.sh), then switches to venv-nnunet and sets the nnU-Net paths in $PROJ.
_nn_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$_nn_here/../maisi/env.sh"
deactivate 2>/dev/null || true
[ -f "$PROJ/venv-nnunet/bin/activate" ] && source "$PROJ/venv-nnunet/bin/activate"
export nnUNet_raw="$PROJ/nnunet_data/raw"
export nnUNet_preprocessed="$PROJ/nnunet_data/preprocessed"
export nnUNet_results="$PROJ/nnunet_data/results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
unset _nn_here
