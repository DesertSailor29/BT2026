#!/usr/bin/env bash
# Pancreas, measure only (no build, no training): what the liver rules (cc, ccAuc; unchanged) do to the pancreas pool.
# CPU only, niced so a running nnU-Net training keeps priority. Resumable (every step skips cached work).
#   1 msd07_split.py          fixed MSD07 split (seed 42, 60 val) -> maisi/msd07_split.json
#   2 filter_metrics.py       per tumour component touch / distance / sep on the 999 pancreas volumes; real reference =
#                             the MSD07 val cases -> AUC cut (rules.json)
#   3 cc_rules.py             main structure per volume -> ids_cc*.json, cc_summary.md
#   4 real_lesion_metrics.py  the same metrics + what cc would change, on all 281 MSD07 cases (descriptive)
#   5 cc_review_figures.py    example figures per violation -> reports/figures/pancreas_rules/
# Run from a worktree of the 'pancreas' branch (see NOTES.md), detached:
#   setsid nohup bash -c 'source "$PROJ/BT2026-pancreas/maisi/env.sh" && bash "$PROJ/BT2026-pancreas/maisi/pancreas_measure.sh"' \
#     > "$PROJ/logs/pancreas_measure_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
W="${PANCREAS_WORKERS:-10}"
POOL="$MAISI_RUNS/pool"
FM="$POOL/filter_metrics/pancreas"
REAL="$PROJ/raw_data/msd07_lesion_metrics"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N="nice -n 19"
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "status file"
setsid nohup python "$REPO/job_status.py" --name PANCREAS_measure \
  --glob "$FM/per_volume/*.json" "$FM/cc/*.json" "$FM/real_reference/*.json" "$REAL/cases/*.json" \
  --total $((999 + 999 + 60 + 281)) --pid $$ --tail-min 10 --what "pancreas measure-only (cc/ccAuc rules)" \
  > "$PROJ/logs/status_pancreas_measure_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &

step "1 split"
$N python "$HERE/msd07_split.py"
step "2 filter_metrics (workers $W)"
$N python "$HERE/filter_metrics.py" --target pancreas --pool "$POOL" --workers "$W"
step "3 cc_rules"
$N python "$HERE/cc_rules.py" --target pancreas --workers "$W"
step "4 real MSD07 cases (descriptive)"
$N python "$HERE/real_lesion_metrics.py" --target pancreas --workers "$W" --metrics "$FM"
step "5 figures"
$N python "$HERE/cc_review_figures.py" --target pancreas --metrics "$FM" --pool "$POOL" --real "$REAL" \
  --workers "$(( W < 7 ? W : 7 ))"
step "done"
du -sh "$FM" "$REAL" "$REPO/reports/figures/pancreas_rules"
