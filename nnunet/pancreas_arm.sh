#!/usr/bin/env bash
# One pancreas arm end to end (autopilot plan, notes/AUTOPILOT.md): build/preprocess its dataset if needed, register
# its fold, train (1000 epochs, default trainer, as `all`), predict the 238 MSD07 test cases (checkpoint_final), eval
# against the `all` + `all_rep2` baseline. Strictly sequential (nothing shares the GPU). Resumable: every step skips
# what is done, so the same command continues after a session end (training resumes with --c).
#
#   bash pancreas_arm.sh <arm>                                   known arms:
#     thickSlice        Dataset112 (build_thick_dataset.py), every synthetic case          E1
#     thickSlice_rep2   Dataset112, same cases again (run-to-run noise of E1)
#   bash pancreas_arm.sh <arm> <dataset id> <make_splits args...>  any other arm, e.g. a rule arm whose dataset an
#     autopilot session built + preprocessed beforehand (plans copied from D102, as build_thick_dataset.py does):
#       bash pancreas_arm.sh shapeRepair 113 --all-synth
#       bash pancreas_arm.sh shapeReject 102 --ids "$MAISI_RUNS/pool/filter_metrics/pancreas/ids_shapeReject.json"
#       bash pancreas_arm.sh shapeRepair_rep2 113 --same-as shapeRepair
#
#   setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && bash "$PROJ/BT2026/nnunet/pancreas_arm.sh" thickSlice' \
#     > "$PROJ/logs/pancreas_arm_thickSlice_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
# (under the autopilot: autopilot/ap start pancreas_thickSlice 'bash "$REPO/nnunet/pancreas_arm.sh" thickSlice')
# Status: PANCREAS_<arm>__... (job_status.py) + NNUNET_D<id>_... (train_status.py). Eval:
# $nnUNet_results/test_eval_pancreas_<arm>/summary.md. Env: PANCREAS_WORKERS (16), PANCREAS_TEST_PROBS (0; 1 = also
# save test probabilities in the same prediction pass).
set -euo pipefail
ARM="${1:?arm name}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
: "${nnUNet_raw:?source nnunet/env.sh first}" "${nnUNet_preprocessed:?}" "${nnUNet_results:?}"
W="${PANCREAS_WORKERS:-16}"
NSYN=999 NVAL=43 NTEST=238
export nnUNet_compile="${nnUNet_compile:-f}" nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-24}"
TS="$(date +%Y%m%d_%H%M%S)"
step() { echo; echo "=== $(date '+%F %T') $*"; }

case "$ARM" in
  thickSlice)      DID=112; REG=(--all-synth) ;;
  thickSlice_rep2) DID=112; REG=(--same-as thickSlice) ;;
  *) DID="${2:?unknown arm $ARM: give <dataset id> <make_splits args...>}"; REG=("${@:3}")
     [ ${#REG[@]} -gt 0 ] || { echo "make_splits args needed (--all-synth | --ids <file> | --same-as <arm>)"; exit 2; } ;;
esac
REG=(--dataset-id "$DID" --arm "$ARM" "${REG[@]}")
found=("$nnUNet_raw"/Dataset"$DID"_*)
[ ${#found[@]} -eq 1 ] && [ -d "${found[0]}" ] || { echo "Dataset$DID not found once in $nnUNet_raw"; exit 1; }
DS="$(basename "${found[0]}")"
RAW="$nnUNet_raw/$DS"
PRE="$nnUNet_preprocessed/$DS"
RES="$nnUNet_results/$DS/nnUNetTrainer__nnUNetPlans__3d_fullres"

step "1 dataset $DS"
if [ "$DID" = 112 ]; then
  python "$HERE/build_thick_dataset.py" --workers "$W" --plans
  npre=$(find "$PRE/nnUNetPlans_3d_fullres" -name '*.pkl' 2>/dev/null | wc -l || true)
  if [ "$npre" -eq $((NSYN + NVAL)) ]; then echo "already preprocessed ($npre cases)"
  else nnUNetv2_preprocess -d $DID -plans_name nnUNetPlans -c 3d_fullres -np "$W"; fi
else
  [ -f "$PRE/nnUNetPlans.json" ] || { echo "$DS is not preprocessed: build + preprocess it first"; exit 1; }
fi
npre=$(find "$PRE/nnUNetPlans_3d_fullres" -name '*.pkl' | wc -l)
[ "$npre" -eq $((NSYN + NVAL)) ] || { echo "expected $((NSYN + NVAL)) preprocessed cases in $PRE, found $npre"; exit 1; }

step "2 register $ARM"
fold_of() { python -c "import json,sys; a=[i for i,x in enumerate(json.load(open('$PRE/arms.json'))) if x['arm']=='$1']; print(a[0] if a else '')" 2>/dev/null || true; }
F=$(fold_of "$ARM")
if [ -z "$F" ]; then python "$HERE/make_splits.py" --target pancreas "${REG[@]}"; F=$(fold_of "$ARM"); fi
[ -n "$F" ] || { echo "registering $ARM failed"; exit 1; }
python "$HERE/make_splits.py" --target pancreas --dataset-id "$DID" --list
FD="$RES/fold_$F"
echo "$ARM = D$DID fold $F -> $FD"

setsid nohup python "$REPO/job_status.py" --name "PANCREAS_$ARM" \
  --glob "$FD/checkpoint_final.pth" "$FD/test_final/msd_*.nii.gz" --total $((1 + NTEST)) --pid $$ --tail-min 10 \
  --what "pancreas arm $ARM (D$DID fold $F): train, predict, eval (training ETA: NNUNET_D$DID file)" \
  > "$PROJ/logs/status_pancreas_${ARM}_$TS.log" 2>&1 < /dev/null &

step "3 train D$DID fold $F"
if [ -f "$FD/checkpoint_final.pth" ] && [ -f "$FD/validation/summary.json" ]; then
  echo "already trained"
else
  mkdir -p "$FD"
  setsid nohup python "$HERE/train_status.py" --fold-dir "$FD" \
    > "$PROJ/logs/status_train_d${DID}_f${F}_$TS.log" 2>&1 < /dev/null &
  if [ -f "$FD/checkpoint_final.pth" ]; then mode=--val; else mode=--c; fi
  echo "nnUNetv2_train $DID 3d_fullres $F $mode (compile=$nnUNet_compile, DA workers $nnUNet_n_proc_DA)"
  nnUNetv2_train $DID 3d_fullres "$F" $mode
fi
python -c "import json; m=json.load(open('$FD/validation/summary.json'))['mean']; print('VAL Dice (43 real MSD07) $ARM:', {k: round(v['Dice'], 4) for k, v in m.items()})"

step "4 predict $NTEST test cases"
n=$(find "$FD/test_final" -name 'msd_*.nii.gz' 2>/dev/null | wc -l || true)
if [ "$n" -eq $NTEST ]; then echo "test predictions already there"
else
  # PANCREAS_TEST_PROBS=1: the same pass also saves probabilities (fold_X/test_probs/, for frozen rules that need
  # them, e.g. grow0.1 in post_rules_test.py); the segmentations are copied into test_final/
  probs=(); [ "${PANCREAS_TEST_PROBS:-0}" = 1 ] && probs=(--probs)
  python "$HERE/predict_test.py" predict --target pancreas --dataset $DID --fold "$F" "${probs[@]}"; fi

step "5 eval vs baseline (102:0 + 102:1)"
python "$HERE/predict_test.py" eval --target pancreas --runs "$DID:$F" --family "$ARM" \
  --out "$nnUNet_results/test_eval_pancreas_$ARM"
step "done: $nnUNet_results/test_eval_pancreas_$ARM/summary.md"
