#!/usr/bin/env bash
# Softmax probabilities of the real *test* cases, for frozen post-hoc rules that need them (autopilot A3,
# autopilot/MISSION.md; e.g. grow0.1 in nnunet/post_rules_test.py). Only for rules frozen in notes/AUTOPILOT.md
# before any test run. GPU (pancreas: 238 cases, ~2.2 h and ~32 GB per run). checkpoint_final, as test_final.
#
#   bash predict_test_probs.sh <dataset id> <fold> [<fold> ...]        e.g.  bash predict_test_probs.sh 102 0 1
# -> $nnUNet_results/<dataset>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_<f>/test_probs/  (<case>.nii.gz + .npz)
# Resumable (--continue_prediction). Inputs: the dataset's imagesTs_real (predict_test.py prepare).
set -euo pipefail
DID="${1:?dataset id}"; shift
[ $# -gt 0 ] || { echo "folds?"; exit 2; }
: "${nnUNet_raw:?source nnunet/env.sh first}" "${nnUNet_results:?}"
found=("$nnUNet_raw"/Dataset$(printf %03d "$DID")_*)
[ ${#found[@]} -eq 1 ] && [ -d "${found[0]}" ] || { echo "Dataset$DID not found once in $nnUNet_raw"; exit 1; }
DS="$(basename "${found[0]}")"
IN="$nnUNet_raw/$DS/imagesTs_real"
n=$(find "$IN" -name '*_0000.nii.gz' | wc -l)
echo "$DS: $n real test inputs in $IN"
for F in "$@"; do
  OUT="$nnUNet_results/$DS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_$F/test_probs"
  echo "=== $(date '+%F %T') fold $F -> $OUT"
  nnUNetv2_predict -i "$IN" -o "$OUT" -d "$DID" -c 3d_fullres -f "$F" -chk checkpoint_final.pth \
    --save_probabilities -npp 3 -nps 3 --continue_prediction
  echo "fold $F: $(find "$OUT" -name '*.npz' | wc -l) / $n probability files"
done
