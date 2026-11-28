#!/usr/bin/env bash
# Liver L1c step 1 (autopilot/MISSION.md, optional): TotalSegmentator aorta + portal/splenic vein for the 111 LiTS
# test images, CPU only (~20 s-2 min per case, ~4 cores), resumable (skips cases whose two masks exist), then the
# descriptive phase table (nnunet/phase_eval.py). Its liver output is never produced or used (--roi_subset).
#   bash nnunet/phase_segment.sh            (needs maisi/env.sh sourced, $PROJ/venv-ts + $PROJ/totalseg_weights)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export TOTALSEG_HOME_DIR="$PROJ/totalseg_weights"
OUT="$PROJ/totalseg_lits_test"; mkdir -p "$OUT"
IMG="$nnUNet_raw/Dataset101_LiverSynth/imagesTs_real"
cases=$(cd "$REPO/nnunet" && python -c "import predict_test as pt; print(' '.join(f'lits_{i:03d}' for i in pt.split()['test']))")
n=0; for c in $cases; do
  n=$((n+1))
  if [ -s "$OUT/$c/aorta.nii.gz" ] && [ -s "$OUT/$c/portal_vein_and_splenic_vein.nii.gz" ]; then continue; fi
  echo "$(date +%H:%M:%S) [$n] $c"
  rm -rf "$OUT/$c.tmp"
  nice -n 10 "$PROJ/venv-ts/bin/TotalSegmentator" -i "$IMG/${c}_0000.nii.gz" -o "$OUT/$c.tmp" \
    --roi_subset aorta portal_vein_and_splenic_vein -d cpu --nr_thr_resamp 8 --nr_thr_saving 4 -q
  rm -rf "$OUT/$c"; mv "$OUT/$c.tmp" "$OUT/$c"
done
echo "$(date +%H:%M:%S) segmentation done: $(ls "$OUT" | grep -vc tmp) case dirs"
python "$REPO/nnunet/phase_eval.py"
