# Backlog (parked ideas, not scheduled)

New experiments are paused until the findings are presented to the supervisor (Decisions 2026-11-15).

## Experiments
- **Tumour-size test** (2026-11-15, per user): pool tumours are much smaller than real (in-liver median 2.3 ml per
  volume vs 16 ml per real case). Ids ready (`nnunet/size_split.py`, `$MAISI_RUNS/pool/filter_metrics/liver/size/`):
  sizeLarge 590 / sizeSmall 589 / sizeRand 590; no fold registered. ~7 GPU h per arm. If large tumours help → generate
  a larger-tumour pool (size control is weak: oversample high sizes, keep large outcomes; ~30 GPU h per 1000).
- **Stricter AUC cut as a training arm** (0.60 helped at test time, see RESULTS ccAucPost); choose on val.
- **Tumour-free generated volumes** against FPs (low priority).
- **Pancreas arms:** which of all / ccReject 911 / ccRepair 999 / ccRepairDel 924 / ccAucRepair; `all` + `all_rep2`
  (baseline + noise floor) are ready to run: `nnunet/pancreas_all.sh` (NOTES Next steps). Better pancreas filter options in report 3 (cc + regenerate the 75,
  TotalSegmentator organ agreement).
- **Pancreas appearance gap beyond slice thickness** (2026-11-24): fat around / inside the gland and lobulation are missing from the synthetic scans. Options: does MAISI's label set / conditioning allow fatty or lobulated glands? intensity edits around the gland (crude, hand-made); real data (against synthetic-only). Design with user + supervisor.
- **Pancreas ccReject** (dropped from the autopilot 2026-11-24, user: cc flags only 88/999 pancreas volumes).
- **Explore/confirm split of the 238 pancreas test cases** (more screening power than 43 val; amends 'test never decides').
- **Pancreas ccAucRepair**: needs a pancreas port of `build_repair_dataset.py` (liver-only); the pancreas cc caches were measured on the uncropped pool labels, D102 labels are cropped (offsets!).
- aucReject (732) + aucRepair alone (AUC without cc; mostly repeats cc); ccAucReject; random subset of the best reject
  arm's size; train on the rejects; a real-only reference model (would cost test cases).

## Contrast phase (2026-11-14, per user)
- (a) Phase of the 111 LiTS test cases (no vessel labels → TotalSegmentator aorta/portal vein + the same HU rule) →
  tumour Dice per phase, compare with the pool's phase mix (section 5 of `liver_check/summary.md`).
- (b) Phase-relative liver check: liver HU relative to the same scan's spleen/aorta (needs (a) for LiTS).
- (c) Does MAISI react to `modality` 2 (`ct_wo_contrast`) / 3 (`ct_contrast`)? Small GPU test: same mask + seed with
  modality 1/2/3, compare aorta/portal vein HU. If yes, the phase mix becomes controllable (paper idea: fill rare phases).
- (d) General: train on mixed phases + evaluate per phase; multi-phase input where data has it; phase as conditioning last.

## Tooling / small
- Tile seams in previews (64×64×48 VAE window): check.
- Email notifier (`maisi/notify.py`) never tried on UCloud (does outbound SMTP work?).
- Report the Path A half-scale bug upstream (NV-Generate-CTMR issue)?
- Where to find: the 90 thesis MAISI volumes and the old NIM scripts (not located on UCloud).
