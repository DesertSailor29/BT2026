# Paper notes (things the paper must say or defend)

Target: MIDL 2027. Full paper ~3 Dec 2026 (soft, learning first), short paper ~15 Apr 2027; which one is open.

## Design defence
- **Synthetic-only training** (Decisions 2026-11-07): (1) train-on-synthetic/test-on-real is the standard generator
  test; (2) without real data nnU-Net can't compensate for faulty volumes → strictest test of filtering; (3) leakage
  (below) is shared by every arm → caveat for absolute Dice, not for comparisons; (4) absolute Dice below real-trained
  is expected. Filtered arm = all volumes that pass, not a fixed K: beating all 999 with fewer volumes is the argument.
- **Rules fixed before looking at the pool metrics** (2026-11-12), amendments documented with reasons before any filter
  arm was trained (shell fix 11-12, lits_049 exclusion, touch → cc, arm family 11-14): `notes/DECISIONS.md`.
- **lits_049 excluded from the real reference only** (unreliable labels: no visible lesion under its least visible
  labels); cut picked on val (0.5475) is confirmed on test (0/681 real test lesions below).
- Analysis is exploratory, no predictions in advance; main metric per-case tumour Dice on 111, baseline = per-case mean
  of the two `all` runs, Holm over 4 arms + 0.01 noise floor. checkpoint_final primary, checkpoint_best secondary.
- **"Why not just postprocess?"** A reviewer will ask: ccPost gives the baseline the same gain as the cc filters, and
  ccAucPost most of the AUC gain (RESULTS.md). Open: report main results with or without ccPost.

## Caveats
- **Leakage:** MAISI's diffusion model and ControlNet were trained on MSD Task03 Liver (built from LiTS) and on
  AbdomenCT-1K (partly reuses LiTS) → LiTS test cases may have been seen by MAISI.
- **The thesis used broken LiTS headers:** volumes 28–47 (= 3D-IRCADb-01 patients 1–20) had identity affines in the
  original release; the thesis' nnU-Net saw them as 1 mm isotropic (1.1–1.8× too large in-plane, up to 4× squashed in
  z; test cases 28, 29, 35, 43). Fixed by `maisi/fix_lits_headers.py` (true LAS affines from the IRCAD DICOMs). Also
  biased the MAISI geometry choice: corrected LiTS medians 0.744 / 1.2 mm (we used 0.8 × 0.8 × 1.0).
- **MAISI Path A half-scale bug** at `dba166e` (fixed by our patch, see `notes/MAISI.md`); did the thesis NIM volumes
  (512³ @ 0.75) have it too?
- Patched masks are nearest-upsampled 1.5 → 0.8 mm, so label edges are 1.5 mm steps (image generated at 0.8 mm).
- No contrast-phase control in MAISI; the pool is mostly weakly enhanced (RESULTS.md).
