"""Create one JSON file per MAISI-v2 (rflow-ct, Path A) generation job.

Each job fixes everything needed to reproduce one image/label pair: target, tumour size, seed and the
full config_infer.json that run_jobs.py hands to MAISI. Seeds are unique within a run.

Trial (10 liver + 10 pancreas, tumour size 0.05..0.95):
    python make_jobs.py --run-dir $MAISI_RUNS/trial10
Smoke test (one liver volume):
    python make_jobs.py --run-dir $MAISI_RUNS/smoke --targets liver --sizes 0.5
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Label ids in MAISI's 132-class mask (configs/label_dict.json) and the size-control name per target.
# After postprocessing: organ -> 1, tumour -> 2, everything else -> 0 (same as LiTS and MSD Task07).
TARGETS = {
    "liver": {"control": "hepatic tumor", "organ_label": 1, "tumor_label": 26},
    "pancreas": {"control": "pancreatic tumor", "organ_label": 4, "tumor_label": 24},
}

TRIAL_SIZES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def build_infer_config(args, control, size):
    # Based on configs/config_infer_80g_512x512x512.json; num_inference_steps set to 30 for rflow-ct.
    return {
        "num_output_samples": 1,
        "body_region": [],
        "anatomy_list": [],
        "controllable_anatomy_size": [[control, size]],
        "controllable_demographics": None,
        "num_inference_steps": args.steps,
        "mask_generation_num_inference_steps": args.mask_steps,
        "mask_generation_cfg_guidance_scale": args.mask_cfg,
        "cfg_guidance_scale": args.tumor_cfg,
        "output_size": args.output_size,
        "spacing": args.spacing,
        "image_output_ext": ".nii.gz",
        "label_output_ext": ".nii.gz",
        "autoencoder_sliding_window_infer_size": args.ae_window,
        "autoencoder_sliding_window_infer_overlap": args.ae_overlap,
        "autoencoder_tp_num_splits": args.ae_splits,
        "controlnet": "$@controlnet_def",
        "diffusion_unet": "$@diffusion_unet_def",
        "autoencoder": "$@autoencoder_def",
        "mask_generation_autoencoder": "$@mask_generation_autoencoder_def",
        "mask_generation_diffusion": "$@mask_generation_diffusion_def",
        "modality": 1,
    }


def maisi_commit(maisi_dir):
    try:
        return subprocess.run(["git", "-C", str(maisi_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--targets", nargs="+", default=["liver", "pancreas"], choices=list(TARGETS))
    p.add_argument("--sizes", nargs="+", type=float, default=TRIAL_SIZES, help="tumour size percentiles in [0, 1]")
    p.add_argument("--repeats", type=int, default=1, help="jobs per (target, size), each with its own seed")
    p.add_argument("--seed-base", type=int, default=0, help="first seed; job k gets seed-base + k")
    p.add_argument("--output-size", nargs=3, type=int, default=[512, 512, 512])
    p.add_argument("--spacing", nargs=3, type=float, default=[0.8, 0.8, 1.0])
    p.add_argument("--steps", type=int, default=30, help="image diffusion steps (rflow-ct: 30)")
    p.add_argument("--mask-steps", type=int, default=100, help="mask diffusion steps (v2 mask model: 100)")
    p.add_argument("--mask-cfg", type=float, default=2.0, help="mask_generation_cfg_guidance_scale")
    p.add_argument("--tumor-cfg", type=float, default=0.0, help="cfg_guidance_scale on tumour presence (0 = off)")
    # NVIDIA's 24 GB preset for 512^3 (config_infer_24g_512x512x512.json) with 8 instead of 2 splits, so one volume
    # fits a 23 GB B200 MIG slice. Their 80 GB preset is window 80^3, 4 splits (fewer tile seams, ~45 GB peak).
    p.add_argument("--ae-window", nargs=3, type=int, default=[64, 64, 48])
    p.add_argument("--ae-overlap", type=float, default=0.4)
    p.add_argument("--ae-splits", type=int, default=8)
    p.add_argument("--maisi-dir", type=Path, default=None, help="only used to record the MAISI commit")
    args = p.parse_args()

    for s in args.sizes:
        if not 0.0 <= s <= 1.0:
            sys.exit(f"size {s} is outside [0, 1]")

    jobs_dir = args.run_dir / "jobs"
    if jobs_dir.exists() and any(jobs_dir.glob("*.json")):
        sys.exit(f"{jobs_dir} already has jobs. Use a new --run-dir (jobs of an existing run must not change).")
    jobs_dir.mkdir(parents=True)

    k = 0
    for target in args.targets:
        t = TARGETS[target]
        n = 0
        for _ in range(args.repeats):
            for size in args.sizes:
                job = {
                    "job_id": f"{target}_{n:04d}",
                    "target": target,
                    "control": t["control"],
                    "size": size,
                    "seed": args.seed_base + k,
                    "organ_label": t["organ_label"],
                    "tumor_label": t["tumor_label"],
                    "infer": build_infer_config(args, t["control"], size),
                }
                (jobs_dir / f"{job['job_id']}.json").write_text(json.dumps(job, indent=2))
                n += 1
                k += 1

    run_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "args": {key: (str(v) if isinstance(v, Path) else v) for key, v in vars(args).items()},
        "maisi_commit": maisi_commit(args.maisi_dir) if args.maisi_dir else None,
        "n_jobs": k,
    }
    (args.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"wrote {k} jobs to {jobs_dir}")


if __name__ == "__main__":
    main()
