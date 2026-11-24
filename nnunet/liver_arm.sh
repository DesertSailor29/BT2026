#!/usr/bin/env bash
# One liver arm end to end (autopilot, notes/AUTOPILOT.md s11; the liver twin of pancreas_arm.sh): check its
# dataset is preprocessed (build it beforehand, e.g. build_rule_repair_dataset.py), register its fold, train (1000
# epochs, default trainer, as `all`), predict the 111 LiTS test cases (checkpoint_final), eval against the `all` +
# `all_rep2` baseline (101:0 + 101:1). Strictly sequential. Resumable: every step skips what is done (training --c).
#
#   bash liver_arm.sh <arm> <dataset id> <make_splits args...>
#       bash liver_arm.sh posRepair 113 --all-synth
#       bash liver_arm.sh posRepair_rep2 113 --same-as posRepair
# (under the autopilot: autopilot/ap start liver_posRepair 'bash "$REPO/nnunet/liver_arm.sh" posRepair 113 --all-synth')
# Status: LIVER_<arm>__... (job_status.py) + NNUNET_D<id>_... (train_status.py). Eval:
# $nnUNet_results/test_eval_liver_<arm>/summary.md. Env: LIVER_WORKERS (16).
set -euo pipefail
ARM="${1:?arm name}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
: "${nnUNet_raw:?source nnunet/env.sh first}" "${nnUNet_preprocessed:?}" "${nnUNet_results:?}"
W="${LIVER_WORKERS:-16}"
NSYN=999 NVAL=20 NTEST=111
export nnUNet_compile="${nnUNet_compile:-f}" nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-24}"
TS="$(date +%Y%m%d_%H%M%S)"
step() { echo; echo "=== $(date '+%F %T') $*"; }

DID="${2:?give <arm> <dataset id> <make_splits args...>}"; REG=("${@:3}")
[ ${#REG[@]} -gt 0 ] || { echo "make_splits args needed (--all-synth | --ids <file> | --same-as <arm>)"; exit 2; }
REG=(--dataset-id "$DID" --arm "$ARM" "${REG[@]}")
found=("$nnUNet_raw"/Dataset"$DID"_*)
[ ${#found[@]} -eq 1 ] && [ -d "${found[0]}" ] || { echo "Dataset$DID not found once in $nnUNet_raw"; exit 1; }
DS="$(basename "${found[0]}")"
RAW="$nnUNet_raw/$DS"
PRE="$nnUNet_preprocessed/$DS"
RES="$nnUNet_results/$DS/nnUNetTrainer__nnUNetPlans__3d_fullres"

step "1 dataset $DS"
[ -f "$PRE/nnUNetPlans.json" ] || { echo "$DS is not preprocessed: build + preprocess it first"; exit 1; }
npre=$(find "$PRE/nnUNetPlans_3d_fullres" -name '*.pkl' | wc -l)
[ "$npre" -eq $((NSYN + NVAL)) ] || { echo "expected $((NSYN + NVAL)) preprocessed cases in $PRE, found $npre"; exit 1; }

step "2 register $ARM"
fold_of() { python -c "import json,sys; a=[i for i,x in enumerate(json.load(open('$PRE/arms.json'))) if x['arm']=='$1']; print(a[0] if a else '')" 2>/dev/null || true; }
F=$(fold_of "$ARM")
if [ -z "$F" ]; then python "$HERE/make_splits.py" --target liver "${REG[@]}"; F=$(fold_of "$ARM"); fi
[ -n "$F" ] || { echo "registering $ARM failed"; exit 1; }
python "$HERE/make_splits.py" --target liver --dataset-id "$DID" --list
FD="$RES/fold_$F"
echo "$ARM = D$DID fold $F -> $FD"

setsid nohup python "$REPO/job_status.py" --name "LIVER_$ARM" \
  --glob "$FD/checkpoint_final.pth" "$FD/test_final/lits_*.nii.gz" --total $((1 + NTEST)) --pid $$ --tail-min 10 \
  --what "liver arm $ARM (D$DID fold $F): train, predict, eval (training ETA: NNUNET_D$DID file)" \
  > "$PROJ/logs/status_liver_${ARM}_$TS.log" 2>&1 < /dev/null &

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
python -c "import json; m=json.load(open('$FD/validation/summary.json'))['mean']; print('VAL Dice (20 real LiTS val) $ARM:', {k: round(v['Dice'], 4) for k, v in m.items()})"

step "4 predict $NTEST test cases"
n=$(find "$FD/test_final" -name 'lits_*.nii.gz' 2>/dev/null | wc -l || true)
if [ "$n" -eq $NTEST ]; then echo "test predictions already there"
else python "$HERE/predict_test.py" predict --target liver --dataset $DID --fold "$F"; fi

step "5 eval vs baseline (101:0 + 101:1)"
python "$HERE/predict_test.py" eval --target liver --runs "$DID:$F" --family "$ARM" \
  --out "$nnUNet_results/test_eval_liver_$ARM"
step "done: $nnUNet_results/test_eval_liver_$ARM/summary.md"
