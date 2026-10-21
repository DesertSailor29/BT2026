# trial10: quality findings

Run: `$MAISI_RUNS/trial10`, 2026-10-20 16:11–16:40 on one full B200, 4 workers.
MAISI rflow-ct @ `dba166e` + our two patches (scale_factor, mask native spacing), Path A,
512³ @ 0.8×0.8×1.0 mm, 30 image steps, 100 mask steps, mask cfg 2.0, tumour cfg 0.0.
10 liver (hepatic tumour size 0.05…0.95, seeds 0–9) + 10 pancreas (pancreatic tumour size 0.05…0.95, seeds 10–19).
Previews: `$MAISI_RUNS/trial10/previews/` (`overview_liver.png`, `overview_pancreas.png`, one PNG per case).

## Summary

- **All 20 jobs ran, all 20 pass MAISI's own quality check.** MAISI's image QC would have caught **0 of 20**.
- **Pancreas is good:** 10/10 have one tumour, fully inside or touching the pancreas (100% within 15 mm), with plausible HU.
- **Liver is the problem:** **7 of 10 fail the thesis filter** (≥ 90% of tumour voxels within 15 mm of the liver).
  Three of them put the tumour in the pelvis or the chest, nowhere near the liver. MAISI's QC doesn't look at this.
- The **tumour size control barely works:** the size value and the volume we got are almost unrelated.
- A **redo-until-pass** filter on liver tumour placement would cost about **3.3× generation time for liver**
  if the whole job is redone, but only about **+35%** if the check runs on the mask before the image is generated
  (placement is a property of the mask). Details at the end.

## What MAISI's own filters check

MAISI has two filters at this commit (`scripts/sample.py`):

1. **Mask retry (Path A):** if a conditioned label (here the tumour) is missing from the generated mask, the mask is
   regenerated, up to 5 times. It only checks that the label *exists*, not where it is or how big it is.
2. **Image QC** (`scripts/quality_check.py`): for liver, spleen, pancreas, kidney, lung, bone and brain, the median HU
   inside the label must lie in a range precomputed from real CT (roughly mean ± 3σ, very wide: liver −22…162 HU).
   It does not check the tumour at all. In **Path A (our setup) a failed volume is still saved**, with a log line.
   Only in Path B (masks from MAISI's database) does a failure trigger regeneration from the next mask.

| Filter | Fired in trial10 | Effect |
|---|---|---|
| Mask retry (tumour label missing) | 3 of 20 (liver_0005, liver_0006, pancreas_0002; 1 retry each) | already fixed inside the job, cost ~40–60 s each, included in the timings |
| Image QC (organ median HU) | **0 of 20** | none |

Our recomputed QC (`postprocess.py`) agrees with MAISI's log for all 20. Even the closest case wasn't near
the limit: the smallest distance of any organ median to its threshold was 46 HU (pancreas in liver_0008, 104 HU vs
the 150 HU upper bound). Typical medians: liver 59–88 HU, pancreas 52–104, lung −952…−770, bone 151–235.

**Conclusion:** at this geometry (after our half-scale patch) MAISI's image QC passes everything. It checks
global intensity realism, and that wasn't the problem in this trial. Before the patch, the half-scale smoke run *did*
fail it (lung median +71 HU), so the check works for gross geometry errors.

## Liver: per case

"≤15 mm" = fraction of tumour voxels within 15 mm of the liver (thesis filter keeps ≥ 0.90).

| case | size | tumour ml | comps | ≤15 mm | tumour median HU | thesis filter | what the preview shows |
|---|---|---|---|---|---|---|---|
| liver_0000 | 0.05 | 0.5 | 2 | 0.19 | **−234** | drop | tumour posterior, in the lung base next to the spine, not in the liver |
| liver_0001 | 0.15 | 1.7 | 4 | 0.05 | 18 | drop | tumour in the pelvis, between bowel loops |
| liver_0002 | 0.25 | 0.1 | 1 | 0.00 | 49 | drop | tiny tumour in the pelvis |
| liver_0003 | 0.35 | 12.2 | 6 | 0.83 | 46 | drop (close) | main tumour inside the liver + 5 small pieces |
| liver_0004 | 0.45 | 0.2 | 2 | 0.00 | **−71** | drop | tumour in pelvic fat |
| liver_0005 | 0.55 | 17.0 | 1 | 1.00 | 38 | keep | clean round hypodense lesion in the liver |
| liver_0006 | 0.65 | 26.6 | 7 | 0.83 | 39 | drop (close) | irregular lesion at the liver edge, 7 pieces |
| liver_0007 | 0.75 | 33.0 | 2 | 1.00 | 61 | keep | round lesion in the liver |
| liver_0008 | 0.85 | 5.5 | 1 | 1.00 | 72 | keep | small lesion in the liver |
| liver_0009 | 0.95 | 252.1 | 2 | 0.85 | 53 | drop (close) | large hypodense mass filling much of the liver; realistic |

- **Kept by the thesis filter: 3/10.** Of the 7 dropped, 4 are clear failures (0.00–0.19: tumour in the wrong body region)
  and 3 are borderline (0.83–0.85: tumour mostly in the liver plus small stray pieces).
- Tumour **median HU below 0** (liver_0000: −234, liver_0004: −71) means the tumour label lies on air or fat. That
  is a cheap and very specific failure signal.
- **Several tumour pieces** (liver_0003: 6, liver_0006: 7) go with the borderline cases. Largest-piece share 0.72–0.83.
- The **liver itself** is fine: 765–2388 ml, 1 component (3 cases have a second piece but the largest piece holds ≥ 99.5%).
  2388 ml (liver_0005) is large but within a realistic range.

## Pancreas: per case

| case | size | tumour ml | comps | ≤15 mm | pancreas ml | tumour median HU |
|---|---|---|---|---|---|---|
| pancreas_0000 | 0.05 | 1.1 | 1 | 1.00 | 77 | −8 |
| pancreas_0001 | 0.15 | 1.4 | 1 | 1.00 | 39 | 83 |
| pancreas_0002 | 0.25 | 6.3 | 1 | 1.00 | 58 | 47 |
| pancreas_0003 | 0.35 | 2.3 | 1 | 1.00 | 70 | 41 |
| pancreas_0004 | 0.45 | 5.6 | 1 | 1.00 | 59 | 101 |
| pancreas_0005 | 0.55 | 6.5 | 1 | 1.00 | 123 | 62 |
| pancreas_0006 | 0.65 | 7.0 | 1 | 1.00 | 55 | 77 |
| pancreas_0007 | 0.75 | 8.6 | 1 | 1.00 | 75 | 79 |
| pancreas_0008 | 0.85 | 13.6 | 1 | 1.00 | 65 | 51 |
| pancreas_0009 | 0.95 | 8.5 | 1 | 1.00 | 63 | 74 |

- All 10 pass the 15 mm check; the tumour sits in the pancreas in every preview.
- Pancreas volume 39–123 ml (typical adult range ~50–100 ml).
- Several tumours are **iso-dense** with the pancreas (pancreas_0001: 83 vs 77 HU, pancreas_0004: 101 vs 95,
  pancreas_0007: 79 vs 79). The label then marks a tumour that isn't visible in the image. Real PDAC is usually
  hypodense. Label/image mismatch like this would teach nnU-Net the wrong thing. MAISI QC doesn't check it,
  so it's a candidate metric for Option B (tumour–organ contrast / CNR).

## Other observations

- **Size control is weak.** Liver: size 0.25 → 0.1 ml, 0.45 → 0.2 ml, 0.85 → 5.5 ml, 0.95 → 252 ml. Pancreas: 1.1–13.6 ml with
  no clear trend above 0.25. One seed per size can't separate noise from trend, but the size value clearly does not
  set the volume reliably. The tumour cfg was 0.0 (no guidance on the size condition); raising
  `cfg_guidance_scale` for the tumour is the obvious next thing to try (Next steps item 4).
- **Stray "brain" labels** in abdominal scans: liver_0009 (2.0 ml), pancreas_0005 (1.8 ml), pancreas_0000 (0.04 ml),
  plus pancreas_0001 and pancreas_0009 (brain median HU present). These are small mask artefacts. They don't affect
  our 0/1/2 labels (only liver/pancreas + tumour are kept), but they are a sign of mask-model noise and
  could be counted as a cheap whole-mask plausibility metric.
- **Field of view:** labelled content covers z 64–447 (384 mm) in every case, with 64 black slices top and bottom
  (expected from the native-spacing patch).
- **Tile seams** (AE window 64×64×48): none visible in the overview previews at this zoom; not yet checked at full
  resolution.

## How much would each filter have caught

| Filter | Liver dropped | Pancreas dropped |
|---|---|---|
| MAISI image QC (organ median HU) | **0/10** | **0/10** |
| MAISI mask retry (label missing) | 0 left (2 fixed inside the job) | 0 left (1 fixed inside the job) |
| Thesis filter (≥ 90% within 15 mm) | **7/10** | 0/10 |
| Tumour median HU < 0 | 2/10 (both also dropped by the thesis filter) | 0/10 |
| Tumour not visible (tumour–organ HU difference < 10) | not computed | 3/10 (0001, 0004, 0007) |

## Time cost of redo-until-pass

Measured throughput with 4 workers: 20 volumes in 29.3 min = **88 s per volume** (about 97 s liver, 79 s pancreas,
from the per-job times; a liver job takes longer). Without any filter, 1000 volumes take ≈ 24.5 h.

With a filter that regenerates until a volume passes, the expected number of attempts per kept volume is 1 / pass rate.

**MAISI's image QC with redo:** pass rate 20/20, so **no extra time** in this trial. With 20 samples the true failure
rate could still be up to ~15% (95% upper bound), i.e. at most ~+18% time. In practice ~0.

**Thesis 15 mm filter with redo, whole job redone (liver):**

| | pass rate | attempts per kept volume | s per kept liver volume | 1000 kept liver volumes |
|---|---|---|---|---|
| no filter | — | 1 | 97 | 27 h |
| redo whole job, point estimate | 0.30 | 3.3 | 323 | **90 h** |
| redo whole job, 95% range for the pass rate (0.11–0.60) | | 1.7–9.1 | 162–880 | 45–245 h |

For this trial: 10 kept liver volumes would have needed ~23 extra jobs, about +37 min on the GPU.
Pancreas: 10/10 pass, so ~no extra time (worst case at the 95% bound, +40%).

**Thesis 15 mm filter on the mask only (recommended):** tumour placement is decided by the mask model, before the
image diffusion and VAE decode, which take ~80% of a job. The mask stage took a median ~60 s per liver job (under
4-way sharing). If the 15 mm check runs right after the mask and only the mask is regenerated (MAISI already has
this retry loop for missing labels; it would be one extra condition there), the expected cost is
2.3 extra masks × ~60 s / 4 workers ≈ **+35 s per kept liver volume: ~132 s instead of 97 s (+36%)**,
i.e. **~37 h instead of 27 h per 1000 liver volumes** (range for the pass rate 0.11–0.60: +10% to +125%).

Two cautions:

- A placement check on the mask is a *mask* filter: it can't catch image-side failures (e.g. the iso-dense pancreas
  tumours above). Those still need an image-side metric and a whole-job redo, or just dropping.
- Rejecting and regenerating changes the distribution of the kept data (e.g. if small tumours fail more often, the kept
  pool is biased towards large ones; in this trial all 3 kept liver cases have size ≥ 0.55). For Option B we probably
  want to **generate without filtering, keep everything, and record the metrics**, then compare filters offline on the
  same pool. A redo filter is the production choice once we know which filter helps.
