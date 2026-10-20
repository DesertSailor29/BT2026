"""Run the MAISI jobs of one run directory, one MAISI process per job. Resumable.

For every job: write its config_infer.json + environment.json, run `python -m scripts.inference` inside the
NV-Generate-CTMR clone, keep the raw output (failed volumes too), postprocess, and record everything in
raw/<job_id>/result.json and in the run-wide results.jsonl.

Re-running the same command skips finished jobs. A job whose result has status != ok is retried only with
--retry-failed. Several workers: --num-workers 2 --worker 0 / --worker 1. If the machine has several GPUs or MIG
slices (e.g. UCloud's 4× B200-mig.1g.23gb), each worker is pinned to its own one (worker i -> device i), otherwise
the workers share the single GPU. --gpu overrides the choice (index into `nvidia-smi -L`, or a UUID). Each MAISI
process gets CPU quota / --num-workers threads (--threads overrides).

    python run_jobs.py --run-dir $MAISI_RUNS/trial10
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import postprocess

RE_PEAK = re.compile(r"Peak GPU memory usage: ([\d.]+) GB")
RE_QC_FAIL = re.compile(r"quality check for label '([^']+)' failed: median value (\S+) is outside the acceptable range \((\S+) - (\S+)\)")
RE_MASK_RETRY = re.compile(r"absent from generated mask, retry")
RE_MASK_GAVE_UP = re.compile(r"still absent after \d+ attempts")
RE_MASK_TIME = re.compile(r"Mask preparation time: ([\d.]+) seconds")


def gpu_uuids():
    """MIG slice UUIDs if any (a CUDA process can only use one slice), else whole-GPU UUIDs."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return re.findall(r"UUID: (MIG-[^)\s]+)", out) or re.findall(r"UUID: (GPU-[^)\s]+)", out)


def cpu_quota():
    """CPUs this container may use: the cgroup quota (UCloud: 48 on a full B200) if set, else the visible CPUs.
    os.cpu_count() and torch see all 384 host CPUs, so without a cap every MAISI process starts 384 threads."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


def pick_device(gpu, worker):
    """CUDA_VISIBLE_DEVICES value for this worker, or None to leave the environment alone."""
    uuids = gpu_uuids()
    if gpu is not None:
        if not gpu.isdigit():
            return gpu
        if int(gpu) >= len(uuids):
            sys.exit(f"--gpu {gpu}: only {len(uuids)} device(s) in nvidia-smi -L")
        return uuids[int(gpu)]
    if len(uuids) > 1:
        return uuids[worker % len(uuids)]
    return None


def write_environment(maisi_dir, job_dir):
    """Shipped environment_rflow-ct.json with absolute model paths and a per-job output_dir."""
    env = json.loads((maisi_dir / "configs" / "environment_rflow-ct.json").read_text())
    for key, val in env.items():
        if "datasets/" not in val:
            env[key] = str((maisi_dir / val).resolve())
    env["output_dir"] = str(job_dir)
    path = job_dir / "environment.json"
    path.write_text(json.dumps(env, indent=2))
    return path


def parse_log(text):
    qc_fails = [
        {"label": m.group(1), "median": m.group(2), "low": m.group(3), "high": m.group(4).rstrip(").")}
        for m in RE_QC_FAIL.finditer(text)
    ]
    peak = RE_PEAK.findall(text)
    mask_time = RE_MASK_TIME.findall(text)
    return {
        "peak_gpu_gb": float(peak[-1]) if peak else None,
        "mask_time_s": float(mask_time[-1]) if mask_time else None,
        "maisi_qc_pass": not qc_fails,
        "maisi_qc_failed": qc_fails,
        "mask_retries": len(RE_MASK_RETRY.findall(text)),
        "mask_target_missing": bool(RE_MASK_GAVE_UP.search(text)),
    }


def run_one(job, run_dir, maisi_dir, drop_raw_image=False, device=None, threads=None):
    job_dir = run_dir / "raw" / job["job_id"]
    if job_dir.exists():
        shutil.rmtree(job_dir)  # leftovers of an interrupted or failed attempt
    job_dir.mkdir(parents=True)

    cfg_path = job_dir / "config_infer.json"
    cfg_path.write_text(json.dumps(job["infer"], indent=2))
    env_path = write_environment(maisi_dir, job_dir)

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "-t", "./configs/config_network_rflow.json",
        "-i", str(cfg_path),
        "-e", str(env_path),
        "--random-seed", str(job["seed"]),
        "--version", "rflow-ct",
    ]  # fmt: skip
    env = dict(os.environ)
    env["MONAI_DATA_DIRECTORY"] = str(maisi_dir / "temp_work_dir")
    env["HF_HUB_OFFLINE"] = "1"  # weights were downloaded by setup_maisi.sh; skip network calls per job
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation; matters on 23 GB slices
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    if threads is not None:  # torch CPU threads; MAISI stitches the decoded 512^3 image on the CPU
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(threads)

    record = {k: v for k, v in job.items() if k != "infer"}
    record["device"] = device
    record["threads"] = threads
    record["started"] = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    with open(job_dir / "maisi.log", "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=maisi_dir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    record["wall_s"] = round(time.time() - t0, 1)
    record["returncode"] = rc
    record.update(parse_log((job_dir / "maisi.log").read_text(errors="replace")))

    images = sorted(job_dir.glob("sample_*_image.nii.gz"))
    labels = sorted(job_dir.glob("sample_*_label.nii.gz"))
    record["raw_image"] = str(images[0]) if images else None
    record["raw_label"] = str(labels[0]) if labels else None
    record["raw_image_bytes"] = images[0].stat().st_size if images else None
    record["raw_label_bytes"] = labels[0].stat().st_size if labels else None

    if rc != 0 or len(images) != 1 or len(labels) != 1:
        record["status"] = "error"
        record["error"] = f"returncode {rc}, {len(images)} image(s), {len(labels)} label(s); see {job_dir / 'maisi.log'}"
    else:
        try:
            t1 = time.time()
            record["metrics"] = postprocess.process(job, images[0], labels[0], run_dir, maisi_dir)
            record["postprocess_s"] = round(time.time() - t1, 1)
            record["status"] = "ok"
            if drop_raw_image:  # the int16 copy in pairs/ stays; the raw 132-class label is always kept
                images[0].unlink()
                record["raw_image"] = None
        except Exception:
            record["status"] = "postprocess_error"
            record["error"] = traceback.format_exc()

    record["finished"] = datetime.now().isoformat(timespec="seconds")
    (job_dir / "result.json").write_text(json.dumps(record, indent=2))
    with open(run_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--maisi-dir", type=Path, default=os.environ.get("MAISI_DIR"), help="NV-Generate-CTMR clone (default: $MAISI_DIR)")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu", default=None, help="device for this worker: index into `nvidia-smi -L` (MIG slices first) or a UUID")
    p.add_argument("--threads", type=int, default=None, help="CPU threads per MAISI process (default: CPU quota / --num-workers)")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="stop after this many jobs (this worker)")
    p.add_argument("--only", nargs="+", default=None, help="run only these job ids")
    p.add_argument("--drop-raw-image", action="store_true", help="delete MAISI's float image after a successful postprocess (saves ~0.3-0.5 GB/volume)")
    args = p.parse_args()

    if args.maisi_dir is None:
        sys.exit("--maisi-dir not given and $MAISI_DIR not set (source BT2026/maisi/env.sh)")
    run_dir, maisi_dir = args.run_dir.resolve(), args.maisi_dir.resolve()
    if not (maisi_dir / "scripts" / "inference.py").exists():
        sys.exit(f"{maisi_dir} is not an NV-Generate-CTMR clone")

    jobs = sorted((run_dir / "jobs").glob("*.json"))
    if not jobs:
        sys.exit(f"no jobs in {run_dir / 'jobs'} (run make_jobs.py first)")
    jobs = [j for i, j in enumerate(jobs) if i % args.num_workers == args.worker]
    if args.only:
        jobs = [j for j in jobs if j.stem in set(args.only)]

    device = pick_device(args.gpu, args.worker)
    threads = args.threads or max(1, cpu_quota() // args.num_workers)
    print(f"worker {args.worker}/{args.num_workers}: {len(jobs)} job(s), device {device or 'default'}, {threads} CPU threads", flush=True)

    done = 0
    for path in jobs:
        job = json.loads(path.read_text())
        result_path = run_dir / "raw" / job["job_id"] / "result.json"
        if result_path.exists():
            status = json.loads(result_path.read_text()).get("status")
            if status == "ok" or not args.retry_failed:
                print(f"skip {job['job_id']} ({status})", flush=True)
                continue
        if args.limit is not None and done >= args.limit:
            break
        print(f"[{datetime.now():%H:%M:%S}] start {job['job_id']} ({job['control']} {job['size']}, seed {job['seed']})", flush=True)
        r = run_one(job, run_dir, maisi_dir, args.drop_raw_image, device, threads)
        m = r.get("metrics")
        line = f"[{datetime.now():%H:%M:%S}] {r['status']:>6} {job['job_id']}  {r['wall_s']:.0f}s  peak {r['peak_gpu_gb']} GB"
        if m:
            line += f"  QC {'pass' if r['maisi_qc_pass'] else 'FAIL'}  tumour {m['tumor_ml']:.1f} ml  organ {m['organ_ml']:.0f} ml"
        else:
            line += f"  {r['error'].splitlines()[-1]}"
        print(line, flush=True)
        done += 1
    print(f"worker {args.worker}: ran {done} job(s)", flush=True)


if __name__ == "__main__":
    main()
