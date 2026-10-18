"""Summarise a run: table of all jobs (summary.md) and optional PNG previews for a visual check.

    python summarize.py --run-dir $MAISI_RUNS/trial10 --previews

Writes <run-dir>/summary.md, <run-dir>/previews/<job_id>.png (3 planes through the tumour) and
<run-dir>/previews/overview_<target>.png (one axial slice per job), small enough to download from UCloud.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

WINDOW = (-160, 240)  # abdominal soft-tissue window in HU


def load_results(run_dir):
    """Latest record per job from results.jsonl."""
    latest = {}
    path = run_dir / "results.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["job_id"]] = r
    return [latest[k] for k in sorted(latest)]


def fmt(v, spec=""):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    return format(v, spec)


def summary_table(results):
    head = "| job | size | seed | status | time s | peak GB | MAISI QC | tumour ml | tumour comps | tumour ≤15 mm | organ ml | organ comps | MAISI axcodes |"
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in results:
        m = r.get("metrics") or {}
        qc = "pass" if r.get("maisi_qc_pass") else "FAIL " + ",".join(f["label"] for f in r.get("maisi_qc_failed", []))
        rows.append(
            f"| {r['job_id']} | {r['size']:.2f} | {r['seed']} | {r['status']} | {fmt(r.get('wall_s'), '.0f')} | {fmt(r.get('peak_gpu_gb'))} | {qc} "
            f"| {fmt(m.get('tumor_ml'), '.1f')} | {fmt(m.get('tumor_components'))} | {fmt(m.get('tumor_frac_within_15mm'), '.2f')} "
            f"| {fmt(m.get('organ_ml'), '.0f')} | {fmt(m.get('organ_components'))} | {m.get('raw_axcodes', '-')} |"
        )
    return "\n".join(rows)


def totals(results):
    ok = [r for r in results if r["status"] == "ok"]
    times = [r["wall_s"] + (r.get("postprocess_s") or 0) for r in ok if r.get("wall_s")]
    peaks = [r["peak_gpu_gb"] for r in ok if r.get("peak_gpu_gb")]
    lines = [
        f"- jobs recorded: {len(results)}, ok: {len(ok)}, failed: {len(results) - len(ok)}",
        f"- MAISI QC pass: {sum(bool(r.get('maisi_qc_pass')) for r in ok)}/{len(ok)}",
        f"- tumour present in label: {sum(bool((r.get('metrics') or {}).get('tumor_present')) for r in ok)}/{len(ok)}",
    ]
    if times:
        lines.append(f"- time per job (MAISI + postprocess): median {statistics.median(times):.0f} s, min {min(times):.0f}, max {max(times):.0f}")
        lines.append(f"- estimated time for 1000 volumes (1 worker): {statistics.median(times) * 1000 / 3600:.1f} h")
    if peaks:
        lines.append(f"- peak GPU memory: max {max(peaks):.1f} GB")
    sized = [r for r in ok if r.get("raw_image_bytes") and (r.get("metrics") or {}).get("pair_bytes")]
    if sized:
        gb = 1024**3
        raw_img = statistics.mean(r["raw_image_bytes"] for r in sized) / gb
        raw_lab = statistics.mean(r["raw_label_bytes"] for r in sized) / gb
        pair = statistics.mean(r["metrics"]["pair_bytes"] for r in sized) / gb
        lines.append(f"- disk per volume: raw image {raw_img:.2f} GB, raw 132-class label {raw_lab:.3f} GB, final pair {pair:.2f} GB")
        lines.append(
            f"- estimated disk for 1000 volumes: {(raw_img + raw_lab + pair) * 1000:.0f} GB keeping everything, "
            f"{(raw_lab + pair) * 1000:.0f} GB with --drop-raw-image"
        )
    return "\n".join(lines)


def window(img):
    lo, hi = WINDOW
    return np.clip((img - lo) / (hi - lo), 0, 1)


def center_slice(label):
    """Per axis, the slice with the most tumour voxels (else organ voxels, else the volume centre).
    Works for detached tumours, where the centroid can fall between the pieces."""
    for value in (2, 1):
        mask = label == value
        if mask.any():
            return tuple(int(mask.sum(axis=tuple(a for a in range(3) if a != ax)).argmax()) for ax in range(3))
    return tuple(s // 2 for s in label.shape)


def planes(image, label, c):
    # nibabel arrays are (x, y, z); transpose so rows run along the second axis, shown with origin="lower".
    return [
        ("axial", image[:, :, c[2]].T, label[:, :, c[2]].T),
        ("coronal", image[:, c[1], :].T, label[:, c[1], :].T),
        ("sagittal", image[c[0], :, :].T, label[c[0], :, :].T),
    ]


def draw(ax, img, lab, title):
    ax.imshow(window(img), cmap="gray", origin="lower", interpolation="nearest")
    if (lab == 1).any():
        ax.contour(lab == 1, levels=[0.5], colors="lime", linewidths=0.6, origin="lower")
    if (lab == 2).any():
        ax.contour(lab == 2, levels=[0.5], colors="red", linewidths=0.8, origin="lower")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def make_previews(run_dir, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    out = run_dir / "previews"
    out.mkdir(exist_ok=True)
    by_target = {}
    for r in results:
        m = r.get("metrics") or {}
        if r["status"] != "ok" or not m.get("pair_image"):
            continue
        image = np.asarray(nib.load(m["pair_image"]).dataobj, dtype=np.float32)
        label = np.asarray(nib.load(m["pair_label"]).dataobj, dtype=np.uint8)
        c = center_slice(label)
        views = planes(image, label, c)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), dpi=110)
        for ax, (name, img, lab) in zip(axes, views):
            draw(ax, img, lab, name)
        fig.suptitle(
            f"{r['job_id']}  size {r['size']:.2f}  seed {r['seed']}  tumour {fmt(m.get('tumor_ml'), '.1f')} ml  "
            f"QC {'pass' if r.get('maisi_qc_pass') else 'FAIL'}  {m.get('axcodes')}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out / f"{r['job_id']}.png")
        plt.close(fig)
        by_target.setdefault(r["target"], []).append((r, views[0]))
        print("preview", r["job_id"], flush=True)

    for target, items in by_target.items():
        cols = min(5, len(items))
        rows = (len(items) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), dpi=110, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (r, (_, img, lab)) in zip(axes.ravel(), items):
            draw(ax, img, lab, f"{r['job_id']} s={r['size']:.2f} {fmt((r.get('metrics') or {}).get('tumor_ml'), '.1f')} ml")
        fig.tight_layout()
        fig.savefig(out / f"overview_{target}.png")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--previews", action="store_true")
    args = p.parse_args()

    results = load_results(args.run_dir)
    text = f"# {args.run_dir.name}\n\n{totals(results)}\n\n{summary_table(results)}\n"
    (args.run_dir / "summary.md").write_text(text)
    print(text)
    if args.previews:
        make_previews(args.run_dir, results)


if __name__ == "__main__":
    main()
