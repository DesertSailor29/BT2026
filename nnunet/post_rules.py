#!/usr/bin/env python
"""A2 (autopilot/MISSION.md): screen anatomical rules on the predictions of the real **val** cases (pancreas).

For both pancreas baseline runs (102:0 `all`, 102:1 `all_rep2`) and the 43 real MSD07 val cases (nnU-Net's own
fold_X/validation predictions, checkpoint_final), every rule below is applied to the saved label map (labels 0 bg,
1 pancreas, 2 tumour) and scored with the same per-case metrics as predict_test.py. Decisions use the image, the
prediction and (grow rules) the softmax probabilities of predict_val_probs.sh only; the reference is read afterwards.
Never run on test (no --split option on purpose: test is for frozen rules, MISSION A3).

Rules (anatomical fact -> parameter source; ledger: reports/data/post_rules/ledger.md):
  ccPost         one organ: keep the 26-connected piece of pancreas ∪ tumour holding the largest pancreas piece
  ccTol<d>       one organ, split by thick slices / missed gland: keep every piece of pancreas ∪ tumour within d mm
                 of the ccPost main structure (d 5 / 10 / 20 mm: 1-4 thick slices)
  ccK<k>         one organ in at most k pieces: keep the pieces holding the k largest pancreas pieces (k 2 / 3)
  tumTouch       tumours of MSD07 are pancreatic: drop tumour pieces with no 26-neighbour predicted pancreas voxel
  tumNear<d>     the same with a tolerance: drop tumour pieces farther than d mm from predicted pancreas (d 10)
  tissueTxt      gland and tumour are soft tissue: predicted voxels < -50 HU (fat) or > 300 HU (bone) -> bg (textbook)
  tissueVal      the same with the window [p0.5, p99.5] of the real val labels' pancreas ∪ tumour HU (val labels)
  minLes         drop predicted tumour pieces smaller than the smallest real val lesion (val labels)
  grow<t>        under-segmentation: voxels within 10 mm of the predicted organ, 26-connected to it, in the
                 tissueTxt window, with p(pancreas) + p(tumour) >= t -> argmax(pancreas, tumour) (t 0.3 / 0.2 / 0.1)
  grow0.2+ccTol10  growing first, then the tolerant one-organ rule
second screen (s5, after the first: no rule promising; grow lifts gland Dice, minLes / tumNear help f0 only):
  grow<t>d<mm>   grow with t 0.05 / 0.02 and the distance window 10 / 20 mm (d10 is the default above)
  growT<t>       the tumour continues where the model was nearly sure: voxels within 10 mm of, 26-connected to a
                 predicted tumour piece, in the tissueTxt window, with p(tumour) >= t -> tumour (t 0.2 / 0.1)
  minLes<ml>     minLes sensitivity: 0.5 ml and 1.75 ml (p5 of the real val lesions)
  pos            position: pancreas lies retroperitoneally just anterior to the spine. Per axial slice the spine
                 centre is found from the image alone (bone > 200 HU inside the body, within 40 mm of the body's
                 L-R centre, posterior half); the prior is the 2D (L-R, A-P) occupancy, relative to that centre, of
                 the *other* 42 val labels (leave-one-out, 2 mm bins, + 10 mm margin); pieces of pancreas ∪ tumour
                 with < 50 % of their voxels in the prior -> background
  A+B            rules applied in sequence (grow0.1+minLes, grow0.1+tumNear10, grow0.05+minLes,
                 grow0.1+growT0.2+minLes, minLes+tumNear10, pos+minLes, grow0.1+pos+minLes)
controls (not anatomy, same predictions):
  rand_<rule>    remove as many random tumour pieces per case as <rule> removed (ccPost, ccTol10, tumTouch; seeded)
  open1          binary opening (3x3x3) of pancreas ∪ tumour, labels kept where the opened mask survives
  dilate1        one 26-neighbour dilation of pancreas ∪ tumour (nearest label): growth without probabilities
  thr<t>         the grow threshold without anatomy: every voxel with p(pancreas) + p(tumour) >= t is foreground
                 (argmax of the two), no distance, connectivity or HU limit (t 0.1 / 0.05): is grow just a lower
                 threshold?
  rand_minLes / rand_pos  random tumour-piece removal matched to minLes / pos

  python nnunet/post_rules.py --workers 12          # cache under $nnUNet_results/test_eval_pancreas/post_rules_val
  python nnunet/post_rules.py --no-probs            # skip the grow rules (val_probs not there yet)
  python nnunet/post_rules.py --tables-only
liver (s7, MISSION A2 lower priority; --target liver): the same rules on the 20 real LiTS val cases of the liver
baseline runs (101:0 `all`, 101:1 `all_rep2`), no probabilities (none saved for liver: --no-probs is implied), the
position prior from the other 19 val liver labels, minLes from the smallest real LiTS val lesion; plus combinations
with the liver's known winner ccPost (ccPost+minLes, ccPost+pos, ccPost+tissueTxt: does anatomy add to the cc rule?).
  python nnunet/post_rules.py --target liver --workers 12
-> reports/data/post_rules/liver_val/ (cache $nnUNet_results/test_eval_liver/post_rules_val). "pancreas" in the rule
descriptions above reads "liver" for this target.
-> reports/data/post_rules/pancreas_val/{summary.md, per_case.csv, val_label_params.json}. Exploratory.
"""
import argparse
import csv
import json
import os
import sys
import zlib
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "maisi"))
import predict_test as pt  # noqa: E402
import fp_location as fl  # noqa: E402
from cc_rules import main_structure  # noqa: E402
from build_dataset import real_case  # noqa: E402

VERSION = "postrules2"
TARGET = "pancreas"
RUNS = ["102:0", "102:1"]
S26 = np.ones((3, 3, 3), bool)
TXT_HU = (-50.0, 300.0)
GROW_T = (0.3, 0.2, 0.1)
GROW_MM = 10.0
OUT = f"{os.environ['nnUNet_results']}/test_eval_pancreas/post_rules_val"
PUB = f"{REPO}/reports/data/post_rules/pancreas_val"
BONE_HU, SPINE_HALF_MM, POS_BIN, POS_MARGIN, POS_MIN_IN = 200.0, 40.0, 2.0, 10.0, 0.5
PROB_RULES = tuple(f"grow{t:g}" for t in GROW_T) + ("grow0.2+ccTol10", "grow0.05", "grow0.02", "grow0.1d20",
                                                   "grow0.05d20", "growT0.2", "growT0.1", "grow0.1+minLes",
                                                   "grow0.1+tumNear10", "grow0.05+minLes", "grow0.1+growT0.2+minLes",
                                                   "grow0.1+pos+minLes", "thr0.1", "thr0.05")
RULES = ("none", "ccPost", "ccTol5", "ccTol10", "ccTol20", "ccK2", "ccK3", "tumTouch", "tumNear10", "tissueTxt",
         "tissueVal", "minLes", "minLes0.5", "minLes1.75", "minLes+tumNear10", "pos", "pos+minLes") + PROB_RULES + (
         "rand_ccPost", "rand_ccTol10", "rand_tumTouch", "rand_minLes", "rand_pos", "open1", "dilate1")
LIVER_EXTRA = ("ccPost+minLes", "ccPost+pos", "ccPost+tissueTxt")
CONTROLS = ("rand_ccPost", "rand_ccTol10", "rand_tumTouch", "rand_minLes", "rand_pos", "open1", "dilate1", "thr0.1",
            "thr0.05")


def set_target(target):
    """Switch the module to 'liver' (paths, runs); the default 'pancreas' leaves everything as before."""
    global TARGET, RUNS, OUT, PUB
    TARGET = target
    RUNS = {"pancreas": ["102:0", "102:1"], "liver": ["101:0", "101:1"]}[target]
    OUT = f"{os.environ['nnUNet_results']}/test_eval_{target}/post_rules_val"
    PUB = f"{REPO}/reports/data/post_rules/{target}_val"


def spacing_of(aff):
    return np.sqrt((aff[:3, :3] ** 2).sum(axis=0))


def dist_mm(mask, spacing):
    """Distance (mm) of every voxel to the mask (inf if the mask is empty)."""
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return ndimage.distance_transform_edt(~mask, sampling=spacing).astype(np.float32)


def keep_pieces(p, keep_mask):
    """p restricted to the 26-connected pieces of p > 0 that touch keep_mask."""
    lab, n = ndimage.label(p > 0, S26)
    if not n:
        return p.copy()
    ids = np.unique(lab[keep_mask & (lab > 0)])
    return np.where(np.isin(lab, ids), p, 0).astype(np.uint8)


def cc_tol(p, spacing, d):
    main = main_structure(p)
    if not main.any():
        return np.zeros_like(p)
    near = dist_mm(main, spacing) <= d
    return keep_pieces(p, near)


def cc_k(p, k):
    pieces, n = ndimage.label(p == 1, S26)
    if not n:
        return np.zeros_like(p)
    sizes = np.bincount(pieces.ravel(), minlength=n + 1)
    sizes[0] = 0
    top = np.argsort(-sizes, kind="stable")[:k]
    top = top[sizes[top] > 0]
    return keep_pieces(p, np.isin(pieces, top))


def drop_tumour_pieces(p, drop_fn):
    """Tumour pieces for which drop_fn(piece mask) is True -> background; (new p, n dropped)."""
    comp, n = ndimage.label(p == 2, S26)
    out, k = p.copy(), 0
    for i, sl in enumerate(ndimage.find_objects(comp), 1):
        if sl is None:
            continue
        if drop_fn(i, comp, sl):
            out[comp == i] = 0
            k += 1
    return out, k


def tum_touch(p):
    panc_near = ndimage.binary_dilation(p == 1, S26)
    return drop_tumour_pieces(p, lambda i, comp, sl: not panc_near[sl][comp[sl] == i].any())


def tum_near(p, spacing, d):
    near = dist_mm(p == 1, spacing) <= d
    return drop_tumour_pieces(p, lambda i, comp, sl: not near[sl][comp[sl] == i].any())


def min_les(p, spacing, min_ml):
    vml = float(np.prod(spacing)) / 1000.0
    comp, n = ndimage.label(p == 2, S26)
    if not n:
        return p.copy()
    sizes = np.bincount(comp.ravel(), minlength=n + 1) * vml
    small = sizes < min_ml
    small[0] = False
    return np.where(small[comp], 0, p).astype(np.uint8)


def tissue(p, img, lo, hi):
    bad = (p > 0) & ((img < lo) | (img > hi))
    return np.where(bad, 0, p).astype(np.uint8)


def grow(p, img, prob, spacing, t, mm=GROW_MM):
    """Accept voxels near and connected to the predicted organ where the model was almost sure (see the docstring)."""
    org = p > 0
    if not org.any():
        return p.copy()
    fg = prob[1] + prob[2]
    cand = (dist_mm(org, spacing) <= mm) & (img >= TXT_HU[0]) & (img <= TXT_HU[1]) & (fg >= t)
    lab, _ = ndimage.label(cand | org, S26)
    ids = np.unique(lab[org])
    new = cand & ~org & np.isin(lab, ids[ids > 0])
    out = p.copy()
    out[new] = np.where(prob[2][new] > prob[1][new], 2, 1).astype(np.uint8)
    return out


def grow_tumour(p, img, prob, spacing, t):
    """growT<t>: extend predicted tumour pieces into nearby soft tissue where p(tumour) >= t (docstring)."""
    tum = p == 2
    if not tum.any():
        return p.copy()
    cand = (dist_mm(tum, spacing) <= GROW_MM) & (img >= TXT_HU[0]) & (img <= TXT_HU[1]) & (prob[2] >= t)
    lab, _ = ndimage.label(cand | tum, S26)
    ids = np.unique(lab[tum])
    new = cand & ~tum & np.isin(lab, ids[ids > 0])
    out = p.copy()
    out[new] = 2
    return out


def threshold(prob, t):
    """thr<t> control: foreground wherever p(pancreas) + p(tumour) >= t, label = argmax of the two."""
    fg = (prob[1] + prob[2]) >= t
    return np.where(fg, np.where(prob[2] > prob[1], 2, 1), 0).astype(np.uint8)


def spine_centre(img, spacing, aff):
    """Per axial slice (x, y) voxel coordinates of the spine centre, from the image alone (docstring `pos`)."""
    body = img > -300
    lab, n = ndimage.label(body[::4, ::4, :])  # coarse: the body is the largest air-separated piece
    if n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        coarse = lab == np.argmax(sizes)
        body &= np.repeat(np.repeat(coarse, 4, 0), 4, 1)[:img.shape[0], :img.shape[1]]
    nz = img.shape[2]
    cx, cy = np.full(nz, np.nan), np.full(nz, np.nan)
    post_low = aff[1, 1] > 0  # RAS: index along axis 1 grows anterior -> posterior = low index
    for z in range(nz):
        b = body[:, :, z]
        if b.sum() < 1000:
            continue
        xs, ys = np.nonzero(b)
        bx, ymid = xs.mean(), (ys.min() + ys.max()) / 2
        xi = np.arange(img.shape[0])[:, None]
        yi = np.arange(img.shape[1])[None, :]
        win = b & (np.abs(xi - bx) * spacing[0] <= SPINE_HALF_MM) & ((yi < ymid) if post_low else (yi > ymid))
        bone = win & (img[:, :, z] > BONE_HU)
        if bone.sum() < 20:
            continue
        bxs, bys = np.nonzero(bone)
        cx[z], cy[z] = bxs.mean(), bys.mean()
    ok = ~np.isnan(cx)
    if not ok.any():
        return None
    zz = np.arange(nz)
    cx, cy = np.interp(zz, zz[ok], cx[ok]), np.interp(zz, zz[ok], cy[ok])
    k = max(1, int(round(10.0 / spacing[2]))) * 2 + 1  # median over ~20 mm along z
    return ndimage.median_filter(cx, k, mode="nearest"), ndimage.median_filter(cy, k, mode="nearest")


def offsets_mm(mask, centre, spacing, aff):
    """World-oriented (R+, A+) mm offsets of the mask voxels from the slice's spine centre."""
    xs, ys, zs = np.nonzero(mask)
    sx, sy = np.sign(aff[0, 0]) or 1.0, np.sign(aff[1, 1]) or 1.0
    return (xs - centre[0][zs]) * spacing[0] * sx, (ys - centre[1][zs]) * spacing[1] * sy


def bins_of(dx, dy):
    return np.unique(np.round(dx / POS_BIN).astype(np.int32) * 100000 + np.round(dy / POS_BIN).astype(np.int32))


def prior_support(bin_sets):
    """2D boolean grid (+ origin) of the union of bin sets, dilated by POS_MARGIN."""
    allb = np.unique(np.concatenate(bin_sets))
    bx, by = np.round(allb / 100000).astype(int), allb - np.round(allb / 100000).astype(int) * 100000
    m = int(POS_MARGIN / POS_BIN) + 1
    ox, oy = bx.min() - m, by.min() - m
    g = np.zeros((bx.max() - ox + m + 1, by.max() - oy + m + 1), bool)
    g[bx - ox, by - oy] = True
    r = int(round(POS_MARGIN / POS_BIN))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    return ndimage.binary_dilation(g, disk), ox, oy


def position(p, centre, spacing, aff, support):
    if centre is None or not (p > 0).any():
        return p.copy()
    g, ox, oy = support
    dx, dy = offsets_mm(p > 0, centre, spacing, aff)
    ix, iy = np.round(dx / POS_BIN).astype(int) - ox, np.round(dy / POS_BIN).astype(int) - oy
    inside = (ix >= 0) & (iy >= 0) & (ix < g.shape[0]) & (iy < g.shape[1])
    inside[inside] = g[ix[inside], iy[inside]]
    inmask = np.zeros(p.shape, bool)
    inmask[p > 0] = inside
    lab, n = ndimage.label(p > 0, S26)
    tot = np.bincount(lab.ravel(), minlength=n + 1)
    ins = np.bincount(lab.ravel(), weights=inmask.ravel(), minlength=n + 1)
    drop = ins < POS_MIN_IN * tot
    drop[0] = False
    return np.where(drop[lab], 0, p).astype(np.uint8)


def rand_remove(p, n_remove, seed):
    comp, n = ndimage.label(p == 2, S26)
    if not n or not n_remove:
        return p.copy()
    rng = np.random.default_rng(seed)
    drop = rng.choice(np.arange(1, n + 1), size=min(n_remove, n), replace=False)
    return np.where(np.isin(comp, drop), 0, p).astype(np.uint8)


def opening(p):
    o = ndimage.binary_opening(p > 0, S26)
    return np.where(o, p, 0).astype(np.uint8)


def dilate(p):
    org = p > 0
    ring = ndimage.binary_dilation(org, S26) & ~org
    near = ndimage.grey_dilation(p, footprint=S26)  # max label in the neighbourhood: tumour wins over pancreas
    out = p.copy()
    out[ring] = near[ring]
    return out


def n_tumour_pieces(p):
    return ndimage.label(p == 2, S26)[1]


def apply_rules(p, img, prob, spacing, params, seed, aff=None, support=None):
    v = {"none": p, "ccPost": np.where(main_structure(p), p, 0).astype(np.uint8)}
    for d in (5, 10, 20):
        v[f"ccTol{d}"] = cc_tol(p, spacing, d)
    for k in (2, 3):
        v[f"ccK{k}"] = cc_k(p, k)
    v["tumTouch"], _ = tum_touch(p)
    v["tumNear10"], _ = tum_near(p, spacing, 10.0)
    v["tissueTxt"] = tissue(p, img, *TXT_HU)
    v["tissueVal"] = tissue(p, img, params["hu_lo"], params["hu_hi"])
    v["minLes"] = min_les(p, spacing, params["min_lesion_ml"])
    v["minLes0.5"] = min_les(p, spacing, 0.5)
    v["minLes1.75"] = min_les(p, spacing, params["lesion_ml_p5"])
    v["minLes+tumNear10"] = tum_near(v["minLes"], spacing, 10.0)[0]
    v["ccPost+minLes"] = min_les(v["ccPost"], spacing, params["min_lesion_ml"])
    v["ccPost+tissueTxt"] = tissue(v["ccPost"], img, *TXT_HU)
    centre = spine_centre(img, spacing, aff) if support is not None else None
    if support is not None:
        v["pos"] = position(p, centre, spacing, aff, support)
        v["pos+minLes"] = min_les(v["pos"], spacing, params["min_lesion_ml"])
        v["ccPost+pos"] = position(v["ccPost"], centre, spacing, aff, support)
    if prob is not None:
        for t in GROW_T + (0.05, 0.02):
            v[f"grow{t:g}"] = grow(p, img, prob, spacing, t)
        v["grow0.2+ccTol10"] = cc_tol(v["grow0.2"], spacing, 10.0)
        for t in (0.1, 0.05):
            v[f"grow{t:g}d20"] = grow(p, img, prob, spacing, t, 20.0)
            v[f"thr{t:g}"] = threshold(prob, t)
        for t in (0.2, 0.1):
            v[f"growT{t:g}"] = grow_tumour(p, img, prob, spacing, t)
        v["grow0.1+minLes"] = min_les(v["grow0.1"], spacing, params["min_lesion_ml"])
        v["grow0.1+tumNear10"] = tum_near(v["grow0.1"], spacing, 10.0)[0]
        v["grow0.05+minLes"] = min_les(v["grow0.05"], spacing, params["min_lesion_ml"])
        v["grow0.1+growT0.2+minLes"] = min_les(grow_tumour(v["grow0.1"], img, prob, spacing, 0.2), spacing,
                                               params["min_lesion_ml"])
        if support is not None:
            v["grow0.1+pos+minLes"] = min_les(position(v["grow0.1"], centre, spacing, aff, support), spacing,
                                              params["min_lesion_ml"])
    n0 = n_tumour_pieces(p)
    for r in ("ccPost", "ccTol10", "tumTouch", "minLes") + (("pos",) if "pos" in v else ()):
        v[f"rand_{r}"] = rand_remove(p, n0 - n_tumour_pieces(v[r]), seed)
    v["open1"] = opening(p)
    v["dilate1"] = dilate(p)
    return v


def load_probs(f, shape):
    """nnU-Net npz (C, z, y, x) -> (C, x, y, z) as the NIfTI label map."""
    prob = np.load(f)["probabilities"].transpose(0, 3, 2, 1)
    assert prob.shape[1:] == shape, (f, prob.shape, shape)
    return prob


def measure(args):
    run, case, img_f, ref_f, pred_f, prob_f, cache_f, params, bin_sets = args
    pnii = nib.load(pred_f)
    p = np.asarray(pnii.dataobj).astype(np.uint8)
    spacing = spacing_of(pnii.affine)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    img = nib.load(img_f).get_fdata(dtype=np.float32)
    assert img.shape == p.shape, (img_f, img.shape, p.shape)
    prob, agree = None, None
    if prob_f:
        prob = load_probs(prob_f, p.shape)
        agree = float((prob.argmax(0) == p).mean())
        assert agree > 0.995, (prob_f, "probabilities do not match the saved prediction", agree)
    seed = zlib.crc32(f"{run}/{case}".encode())
    support = prior_support([b for c, b in bin_sets.items() if c != case]) if bin_sets else None  # leave-one-out
    variants = apply_rules(p, img, prob, spacing, params, seed, pnii.affine, support)
    del img, prob
    ref = nib.load(ref_f)
    r = np.asarray(ref.dataobj).astype(np.uint8)
    assert r.shape == p.shape and np.allclose(ref.affine, pnii.affine, atol=1e-3), ref_f
    out = {"version": VERSION, "run": run, "case": case, "voxel_ml": voxel_ml, "prob_agree": agree,
           "metrics": {k: {**fl.metrics(r, x, voxel_ml), "changed_vox": int((x != p).sum())}
                       for k, x in variants.items()}}
    tmp = cache_f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, cache_f)
    return run, case


def label_params(cases, workers):
    """HU window and smallest lesion of the real val labels (parameter source 'val labels')."""
    f = f"{PUB}/val_label_params.json"
    if os.path.exists(f):
        return json.load(open(f))
    with Pool(workers) as pool:
        res = pool.map(_label_stats, cases)
    hu = np.concatenate([h for h, _ in res])
    les = [m for _, ms in res for m in ms]
    out = {"cases": len(cases), "voxels": int(hu.size), "hu_lo": round(float(np.percentile(hu, 0.5)), 1),
           "hu_hi": round(float(np.percentile(hu, 99.5)), 1),
           "hu_share_below_-50": round(float((hu < -50).mean()), 4), "hu_share_above_300": round(float((hu > 300).mean()), 4),
           "lesions": len(les), "min_lesion_ml": round(float(min(les)), 4) if les else 0.0,
           "lesion_ml_p5": round(float(np.percentile(les, 5)), 4) if les else 0.0}
    os.makedirs(PUB, exist_ok=True)
    json.dump(out, open(f, "w"), indent=1)
    return out


def _label_stats(case):
    ref = nib.load(f"{pt.RAW}/labelsTr/{case}.nii.gz")
    r = np.asarray(ref.dataobj).astype(np.uint8)
    img = nib.load(f"{pt.RAW}/imagesTr/{case}_0000.nii.gz").get_fdata(dtype=np.float32)
    hu = img[r > 0][::7]  # every 7th voxel: enough for percentiles, keeps memory small
    vml = float(np.prod(spacing_of(ref.affine))) / 1000.0
    comp, n = ndimage.label(r == 2, S26)
    ml = (np.bincount(comp.ravel(), minlength=n + 1)[1:] * vml).tolist()
    return hu, ml


def label_bins(cases, workers):
    """Spine-relative occupancy bins of every real val label (for the leave-one-out position prior)."""
    f = f"{OUT}/pos_bins.npz"
    if os.path.exists(f):
        d = np.load(f)
        return {c: d[c] for c in d.files}
    with Pool(workers) as pool:
        res = dict(zip(cases, pool.map(_label_bins, cases)))
    tmp = f + ".tmp.npz"
    np.savez(tmp, **res)
    os.replace(tmp, f)
    return res


def _label_bins(case):
    ref = nib.load(f"{pt.RAW}/labelsTr/{case}.nii.gz")
    r = np.asarray(ref.dataobj).astype(np.uint8)
    img = nib.load(f"{pt.RAW}/imagesTr/{case}_0000.nii.gz").get_fdata(dtype=np.float32)
    sp = spacing_of(ref.affine)
    centre = spine_centre(img, sp, ref.affine)
    if centre is None:
        return np.zeros(0, np.int32)
    return bins_of(*offsets_mm(r > 0, centre, sp, ref.affine))


def tables(runs, cases, rules):
    data = {}
    for run in runs:
        for c in cases:
            d = json.load(open(f"{OUT}/cache/{run.replace(':', '_')}/{c}.json"))
            assert d["version"] == VERSION
            data[(run, c)] = d
    rules = [r for r in rules if all(r in data[(run, c)]["metrics"] for run in runs for c in cases)]
    os.makedirs(PUB, exist_ok=True)
    with open(f"{PUB}/per_case.csv", "w", newline="") as fh:
        keys = list(data[(runs[0], cases[0])]["metrics"]["none"].keys())
        w = csv.writer(fh)
        w.writerow(["run", "case", "rule"] + keys)
        for (run, c), d in data.items():
            for r in rules:
                w.writerow([run, c, r] + [d["metrics"][r][k] for k in keys])
    names = {r: f"D{r.split(':')[0]} f{r.split(':')[1]} {pt.arm_name(*map(int, r.split(':')))}" for r in runs}
    lines = [f"# A2: anatomical rules on the val predictions ({TARGET})", "",
             f"{len(cases)} real {pt.REAL_NAME} val cases, checkpoint_final predictions, runs {', '.join(names.values())}. "
             "Each rule vs the saved prediction (`none`), paired per case (bootstrap 95 % CI, Wilcoxon p). "
             "Rule definitions: `nnunet/post_rules.py` docstring; ledger `reports/data/post_rules/ledger.md`. "
             "**Promising** (MISSION A2) = tumour Dice gain > 0 in both runs, mean gain >= 0.01, <= 5 % of the "
             "real val lesions lost. Val screen only, exploratory.", "",
             "| rule | run | tumour Dice | vs none [95% CI] | p | pancreas Dice | vs none | lesion recall | lesions lost | FP pieces / case | promising |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for rule in rules:
        gains, lost_all, ref_all, rows = [], 0, 0, []
        for run in runs:
            m = [data[(run, c)]["metrics"] for c in cases]
            t0 = [x["none"]["tumour_dice"] for x in m]
            t1 = [x[rule]["tumour_dice"] for x in m]
            g0 = [x["none"][f"{TARGET}_dice"] for x in m]
            g1 = [x[rule][f"{TARGET}_dice"] for x in m]
            st, sg = pt.paired(t0, t1), pt.paired(g0, g1)
            ref = sum(x["none"]["les_ref"] for x in m)
            det = sum(x[rule]["les_det"] for x in m)
            lost = sum(max(0, x["none"]["les_det"] - x[rule]["les_det"]) for x in m)
            fp = np.mean([x[rule]["les_fp"] for x in m])
            gains.append(st["diff"])
            lost_all += lost
            ref_all += ref
            rows.append((run, st, sg, det / ref if ref else float("nan"), lost, fp))
        prom = rule != "none" and min(gains) > 0 and np.mean(gains) >= 0.01 and lost_all <= 0.05 * ref_all
        verdict[rule] = {"gain_f0": gains[0], "gain_f1": gains[-1], "lost": lost_all, "promising": bool(prom)}
        for run, st, sg, rec, lost, fp in rows:
            diff = "–" if rule == "none" else pt.fmt_diff(st, 3)
            gd = "–" if rule == "none" else f"{sg['diff']:+.3f}"
            p = "–" if rule == "none" else f"{st['p']:.2g}"
            lines.append(f"| {rule} | {names[run]} | {st['y']:.3f} | {diff} | {p} | {sg['y']:.3f} | {gd} | {rec:.3f} | "
                         f"{lost} | {fp:.1f} | {'**yes**' if prom else ''} |")
    ag = [data[k]["prob_agree"] for k in data if data[k]["prob_agree"] is not None]
    if ag:
        lines += ["", f"Probabilities vs saved label maps: argmax agreement min {min(ag):.4f} over {len(ag)} cases."]
    lines += ["", "Controls: " + ", ".join(CONTROLS) + "."]
    open(f"{PUB}/summary.md", "w").write("\n".join(lines) + "\n")
    json.dump(verdict, open(f"{PUB}/verdict.json", "w"), indent=1)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("pancreas", "liver"), default="pancreas")
    ap.add_argument("--runs", nargs="+", help="dataset:fold (default: the target's two baseline runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="+", help="subset (fake test); default all val cases")
    ap.add_argument("--no-probs", action="store_true", help="skip the grow rules")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    set_target(a.target)
    a.runs = a.runs or RUNS
    a.no_probs = a.no_probs or TARGET == "liver"
    pt.set_target(TARGET)
    fl.ORG = TARGET
    val = [real_case(TARGET, i) for i in pt.split()["val"]]
    cases = a.cases or val
    rules = [r for r in RULES if not (a.no_probs and r in PROB_RULES)] + (list(LIVER_EXTRA) if TARGET == "liver" else [])
    if not a.tables_only:
        os.makedirs(OUT, exist_ok=True)
        params = label_params(val, a.workers)
        print("val label params", params, flush=True)
        bin_sets = label_bins(val, a.workers)
        print("position bins per val label:", {c: len(b) for c, b in bin_sets.items()}, flush=True)
        jobs = []
        for run in a.runs:
            did, fold = map(int, run.split(":"))
            fdir = f"{pt.res_dir(did)}/fold_{fold}"
            cdir = f"{OUT}/cache/{run.replace(':', '_')}"
            os.makedirs(cdir, exist_ok=True)
            for c in cases:
                pred, cache = f"{fdir}/validation/{c}.nii.gz", f"{cdir}/{c}.json"
                prob = None if a.no_probs else f"{fdir}/val_probs/{c}.npz"
                if prob and not os.path.exists(prob):
                    sys.exit(f"missing {prob} (predict_val_probs.sh, or --no-probs)")
                if os.path.exists(cache):
                    d = json.load(open(cache))
                    if d.get("version") == VERSION and all(r in d["metrics"] for r in rules):
                        continue
                jobs.append((run, c, f"{pt.RAW}/imagesTr/{c}_0000.nii.gz", f"{pt.RAW}/labelsTr/{c}.nii.gz", pred,
                             prob, cache, params, bin_sets))
        print(f"{len(jobs)} (run, case) pairs to measure, {a.workers} workers", flush=True)
        with Pool(a.workers) as pool:
            for k, (run, c) in enumerate(pool.imap_unordered(measure, jobs, chunksize=1), 1):
                print(f"  {k}/{len(jobs)} {run} {c}", flush=True)
    tables(a.runs, cases, rules)


if __name__ == "__main__":
    main()
