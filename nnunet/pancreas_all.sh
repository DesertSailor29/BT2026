#!/usr/bin/env bash
# Pancreas baseline: arm `all` (fold 0) + `all_rep2` (fold 1, same 999 cases: run-to-run noise) of
# Dataset102_PancreasSynth, end to end. Resumable: every step skips what is done, so the same command continues
# after a session end (training resumes with --c from checkpoint_latest).
#   1 build_dataset.py --target pancreas   999 synthetic pancreas pairs (padding cropped) + the 43 MSD07 val cases
#                                          (maisi/msd07_split.json) -> $nnUNet_raw/Dataset102_PancreasSynth
#   2 nnUNetv2_plan_and_preprocess -d 102 -c 3d_fullres (default nnUNetPlans, as liver)
#   3 make_splits.py: fold 0 = all, fold 1 = all_rep2 (val = the 43 real MSD07 val cases)
#   4 train fold 0, then fold 1 (default trainer, 1000 epochs, nnUNet_compile=f, nnUNet_n_proc_DA=24, as liver);
#     each fold's test prediction (238 MSD07 test cases, checkpoint_final) runs in the background during the next
#     fold's training, as the liver queue did
#   5 predict_test.py eval --target pancreas -> $nnUNet_results/test_eval_pancreas/summary.md (+ confusion.md),
#     progress plots + training curves -> reports/data/test_eval_pancreas/progress/
# Status: PANCREAS_all__<n>of<N>__ETA_...txt (job_status.py, outputs of all steps) + one NNUNET_D102_... file per
# training (train_status.py, epoch + ETA). Settings via env: PANCREAS_FOLDS ("0 1"), PANCREAS_WORKERS (16).
#
#   setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && bash "$PROJ/BT2026/nnunet/pancreas_all.sh"' \
#     > "$PROJ/logs/pancreas_all_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
: "${nnUNet_raw:?source nnunet/env.sh first}" "${nnUNet_preprocessed:?}" "${nnUNet_results:?}"
W="${PANCREAS_WORKERS:-16}"
FOLDS="${PANCREAS_FOLDS:-0 1}"
DID=102
DS=Dataset102_PancreasSynth
RAW="$nnUNet_raw/$DS"
PRE="$nnUNet_preprocessed/$DS"
RES="$nnUNet_results/$DS/nnUNetTrainer__nnUNetPlans__3d_fullres"
NSYN=999 NVAL=43 NTEST=238
export nnUNet_compile="${nnUNet_compile:-f}" nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-24}"
TS="$(date +%Y%m%d_%H%M%S)"
step() { echo; echo "=== $(date '+%F %T') $*"; }
nfolds=$(wc -w <<< "$FOLDS")

step "status file"
globs=("$RAW/labelsTr/*.nii.gz" "$PRE/nnUNetPlans_3d_fullres/*.pkl")
for f in $FOLDS; do globs+=("$RES/fold_$f/checkpoint_final.pth" "$RES/fold_$f/test_final/msd_*.nii.gz"); done
setsid nohup python "$REPO/job_status.py" --name PANCREAS_all --glob "${globs[@]}" \
  --total $(( 2 * (NSYN + NVAL) + nfolds * (1 + NTEST) )) --pid $$ --tail-min 10 \
  --what "pancreas arm all + all_rep2: build, preprocess, train, predict, eval (training ETA: NNUNET_D102 files)" \
  > "$PROJ/logs/status_pancreas_all_$TS.log" 2>&1 < /dev/null &

step "1 build Dataset102 (workers $W)"
python "$HERE/build_dataset.py" --target pancreas --workers "$W"
n=$(find "$RAW/labelsTr" -name '*.nii.gz' | wc -l)
[ "$n" -eq $((NSYN + NVAL)) ] || { echo "expected $((NSYN + NVAL)) labels in $RAW/labelsTr, found $n"; exit 1; }
[ -f "$RAW/dataset.json" ] || { echo "no dataset.json (real val cases missing?)"; exit 1; }

step "2 plan + preprocess (3d_fullres)"
npre=$(find "$PRE/nnUNetPlans_3d_fullres" -name '*.pkl' 2>/dev/null | wc -l || true)
if [ "$npre" -eq $((NSYN + NVAL)) ] && [ -f "$PRE/nnUNetPlans.json" ]; then
  echo "already preprocessed ($npre cases)"
else
  nnUNetv2_plan_and_preprocess -d $DID -c 3d_fullres -np "$W" --verify_dataset_integrity
fi

step "3 register folds"
registered() { [ -f "$PRE/arms.json" ] && python -c "import json,sys; sys.exit(0 if any(a['arm']=='$1' for a in json.load(open('$PRE/arms.json'))) else 1)"; }
for arm in all all_rep2; do
  if registered $arm; then echo "$arm already registered"; else python "$HERE/make_splits.py" --target pancreas --arm $arm; fi
done
python "$HERE/make_splits.py" --target pancreas --list
python - "$PRE/arms.json" <<'EOF'
import json, sys
arms = [a["arm"] for a in json.load(open(sys.argv[1]))]
assert arms[:2] == ["all", "all_rep2"], f"fold 0/1 must be all/all_rep2, arms.json has {arms}"
EOF

step "3b copy the $NTEST MSD07 test cases (once, before predictions may run in parallel)"
python "$HERE/predict_test.py" prepare --target pancreas

predict() {  # test prediction + nothing else; logs to its own file
  local f=$1
  local n
  n=$(find "$RES/fold_$f/test_final" -name 'msd_*.nii.gz' 2>/dev/null | wc -l || true)
  if [ "$n" -eq $NTEST ]; then echo "fold $f: test predictions already there"; return 0; fi
  python "$HERE/predict_test.py" predict --target pancreas --fold "$f"
}

step "4 train + predict, folds: $FOLDS"
pred_pids=()
for f in $FOLDS; do
  fd="$RES/fold_$f"
  if [ -f "$fd/checkpoint_final.pth" ] && [ -f "$fd/validation/summary.json" ]; then
    echo "fold $f: already trained"
  else
    mkdir -p "$fd"
    setsid nohup python "$HERE/train_status.py" --fold-dir "$fd" \
      > "$PROJ/logs/status_train_d102_f${f}_$TS.log" 2>&1 < /dev/null &
    if [ -f "$fd/checkpoint_final.pth" ]; then mode=--val; else mode=--c; fi
    echo "nnUNetv2_train $DID 3d_fullres $f $mode (compile=$nnUNet_compile, DA workers $nnUNet_n_proc_DA)"
    nnUNetv2_train $DID 3d_fullres "$f" $mode
  fi
  python -c "import json; m=json.load(open('$fd/validation/summary.json'))['mean']; print('fold $f val Dice (43 real MSD07):', {k: round(v['Dice'], 3) for k, v in m.items()})"
  step "predict fold $f (background)"
  predict "$f" > "$PROJ/logs/pancreas_predict_f${f}_$TS.log" 2>&1 &
  pred_pids+=($!)
done
step "waiting for the test predictions (logs $PROJ/logs/pancreas_predict_f*_$TS.log)"
for p in "${pred_pids[@]}"; do wait "$p" || { echo "a prediction failed, see its log"; exit 1; }; done

step "5 eval"
base=(); for f in $FOLDS; do base+=("$DID:$f"); done
python "$HERE/predict_test.py" eval --target pancreas --baseline "${base[@]}"
step "done: $nnUNet_results/test_eval_pancreas/summary.md"
