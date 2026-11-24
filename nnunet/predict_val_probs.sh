#!/usr/bin/env bash
# Softmax probabilities of the real validation cases, for post-hoc rules that need more than the saved label maps
# (e.g. growing the predicted gland where the model was almost sure; autopilot Phase A, autopilot/MISSION.md).
# GPU, short (pancreas: 43 cases, ~25 min per run). checkpoint_final, as nnU-Net's own fold_X/validation predictions.
#
#   bash predict_val_probs.sh <dataset id> <fold> [<fold> ...]        e.g.  bash predict_val_probs.sh 102 0 1
# -> $nnUNet_results/<dataset>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_<f>/val_probs/  (<case>.nii.gz + <case>.npz)
# Resumable (--continue_prediction). Inputs: links to the real val images of the dataset (msd_* / lits_* in imagesTr).
set -euo pipefail
DID="${1:?dataset id}"; shift
[ $# -gt 0 ] || { echo "folds?"; exit 2; }
: "${nnUNet_raw:?source nnunet/env.sh first}" "${nnUNet_results:?}"
found=("$nnUNet_raw"/Dataset$(printf %03d "$DID")_*)
[ ${#found[@]} -eq 1 ] && [ -d "${found[0]}" ] || { echo "Dataset$DID not found once in $nnUNet_raw"; exit 1; }
DS="$(basename "${found[0]}")"
IN="$nnUNet_raw/$DS/val_inputs"
mkdir -p "$IN"
for f in "$nnUNet_raw/$DS"/imagesTr/msd_*_0000.nii.gz "$nnUNet_raw/$DS"/imagesTr/lits_*_0000.nii.gz; do
  [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "$IN/$(basename "$f")"
done
n=$(find "$IN" -name '*_0000.nii.gz' | wc -l)
echo "$DS: $n real val inputs in $IN"
for F in "$@"; do
  OUT="$nnUNet_results/$DS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_$F/val_probs"
  echo "=== $(date '+%F %T') fold $F -> $OUT"
  nnUNetv2_predict -i "$IN" -o "$OUT" -d "$DID" -c 3d_fullres -f "$F" -chk checkpoint_final.pth \
    --save_probabilities -npp 3 -nps 3 --continue_prediction
  echo "fold $F: $(find "$OUT" -name '*.npz' | wc -l) / $n probability files"
done
