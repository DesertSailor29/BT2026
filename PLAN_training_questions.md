# Answer to hand back to the user (written 2026-11-07 16:50, UCloud)

> **Answered 2026-11-07** (NOTES.md Decisions); the current plan is `PLAN_training.md`. Kept for the record.

Show this to the user verbatim at the start of the next session, then wait for their answers.

## CPU in this session

**Yes for the light Phase 1 steps, not for nnU-Net preprocessing until the pool is done.** This session is capped at 48 cores and 288 GB RAM. The 4 MAISI workers use 27–32 cores on average, but when a worker stitches its decoded volume together on the CPU it briefly grabs ~7 cores on its own. That's the 15–20 you see at peak, plus short spikes.

- **Can run now:** the LiTS header fix and the conversion to RAS, installing nnU-Net, and the filter metrics on the finished liver volumes. I'd run them at the lowest CPU priority with ~8 parallel workers, so MAISI barely slows down.
- **Wait until the pool finishes (~21:00):** nnU-Net preprocessing. It's the heavy step, and after 21:00 it can have all 48 cores in this same session. A separate CPU session isn't needed.

## D1–D3

- **D1:** Agreed. Each filter writes a JSON with the volumes that pass plus a score for each, so filters can also rank. The script that builds a training set has a setting at the top: either the full list, or a random K with a fixed seed.
- **D2:** Agreed, training uses synthetic data only. All 131 LiTS cases become test data (see question 1).
- **D3:** I recommend **the whole generated volume, with no anatomy crop.** Since you have the hours, it removes one design choice we'd have to defend. nnU-Net already makes sure about a third of its training patches contain liver or tumour, so the extra torso doesn't starve it of examples. The one exception is padding. The mask covers 480×480×384 voxels of the 512³ volume, and I'll check what the image contains in the 64 slices above and below that. If they're blank, cropping them loses nothing. I'll document this in the plan.

## D4, rephrased

Each training run is a little random: the starting weights, which patches it sees, and in random-K arms which volumes are drawn. So two runs of the *same* arm can differ by maybe 1–2 Dice points. If filter X beats random by 1.5 points, we can't tell whether that's the filter or luck unless we know that run-to-run spread.

**Question:** should every arm be trained 2–3 times (costing 2–3× GPU time), only the key arms (all, random-K, the best filter), or each arm once? With once, we accept that small differences can't be called real.

## Questions to pin down the problem

1. **The research question.** Is it "with the same number of synthetic volumes, does selecting them with filter X give a better synthetic-only nnU-Net on real LiTS than picking them at random"? Or is it also about how many volumes, for example whether a filtered 300 beats all 999?
2. **Test set and leakage.** MAISI was trained on MSD Task03, which is the 131 LiTS training cases. So every LiTS test case may have been seen by the generator. This hurts absolute scores more than comparisons between filters, since every arm shares the problem. Do you want an extra liver-tumour test set that MAISI hasn't seen (I'd check MAISI's training list), or is LiTS alone enough with the caveat stated?
3. **Validation.** nnU-Net expects a validation set. Options:
   - held-out synthetic volumes, which are biased by the filter;
   - a few real cases used only to watch training, never to choose checkpoints;
   - no validation, just the final checkpoint.

   I'd go with the final checkpoint plus a fixed set of synthetic volumes for monitoring.
4. **Filter thresholds.** How do we set them without tuning on the test set?
   - fixed in advance, like the thesis 15 mm rule;
   - from real-data statistics, e.g. the liver HU range in LiTS (this uses real images, but not for training);
   - by percentile, e.g. keep the best 30% by score.
5. **Which filters.** Here are the families. Which ones are in, and should each be tested alone, or also in combinations?
   - **Geometry:** the 15 mm rule, number of tumour pieces, liver in one piece, stray labels.
   - **Intensity:** tumour–liver contrast, HU sanity checks, liver HU compared with real LiTS.
   - **Image artefacts:** seams at the edges of the 64×64×48 tiles MAISI decodes in.
   - **Model agreement:** a model segments each synthetic volume and we keep the volumes where it agrees with MAISI's mask. Since we train on no real data, the judge must be either a model trained on the whole synthetic pool (circular) or TotalSegmentator (liver only, no tumour). Which, or drop this family?
   - **Inverse control:** train on the volumes each filter rejects. If the rejects do worse, that's strong evidence. Worth an arm?
6. **Pancreas.** Is it in the first paper with the same design, tested on MSD07, or later?
7. **Relation to the thesis.** The thesis added synthetic data to real data. Switching to synthetic-only changes the claim from "synthetic data helps" to "which synthetic data is good". Is that the paper you want? Should one real-only nnU-Net still be trained, only as a reference upper bound?

Once these are answered I'll count the arms, estimate GPU time per epoch count (that settles D5), and rewrite `PLAN_training.md`.
