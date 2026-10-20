"""How many MAISI workers fit on one GPU, and how much faster are they? Benchmark for a full (non-MIG) GPU.

For each worker count N: make N × --jobs-per-worker liver jobs in <run-root>/w<N>, start N run_jobs.py workers at
once on the shared GPU, sample nvidia-smi every 2 s, and record wall time, throughput, GPU memory and utilisation.
The first wave starts all workers together, so their memory peaks coincide (worst case). Results are appended to
<run-root>/bench.jsonl and tabulated in <run-root>/bench.md. Resumable: finished worker counts are skipped, an
unfinished one is moved aside and run again. Stops after the first worker count with failed jobs (e.g. OOM).

    python bench_workers.py --run-root $MAISI_RUNS/bench --workers 1 2 3 4 5 6 --maisi-dir $MAISI_DIR
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_records(run_dir):
    records = {}
    path = run_dir / "results.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["job_id"]] = r
    return list(records.values())


def bench_one(n, args):
    run_dir = args.run_root / f"w{n}"
    if run_dir.exists():
        stale = run_dir.with_name(f"w{n}_incomplete_{datetime.now():%Y%m%d_%H%M%S}")
        run_dir.rename(stale)
        print(f"moved unfinished {run_dir.name} to {stale.name}", flush=True)
    n_jobs = n * args.jobs_per_worker
    subprocess.run([sys.executable, str(HERE / "make_jobs.py"), "--run-dir", str(run_dir), "--targets", "liver",
                    "--sizes", "0.5", "--repeats", str(n_jobs), "--seed-base", str(args.seed_base)]
                   + (["--maisi-dir", str(args.maisi_dir)] if args.maisi_dir else []), check=True)

    smi_csv = run_dir / "nvidia_smi.csv"
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits", "-lms", "2000"], stdout=smi_csv.open("w"))
    t0 = time.time()
    workers = []
    for i in range(n):
        log = (run_dir / f"worker{i}.log").open("w")
        workers.append(subprocess.Popen([sys.executable, str(HERE / "run_jobs.py"), "--run-dir", str(run_dir),
                                         "--num-workers", str(n), "--worker", str(i)], stdout=log, stderr=log))
    for w in workers:
        w.wait()
    wall = time.time() - t0
    smi.terminate()
    smi.wait()

    mem, util, total = [], [], None
    for line in smi_csv.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[1].isdigit():
            mem.append(int(parts[1]) / 1024)
            total = int(parts[2]) / 1024
            util.append(int(parts[3]))
    records = load_records(run_dir)
    ok = [r for r in records if r.get("status") == "ok"]
    oom = sum("OutOfMemoryError" in (run_dir / "raw" / r["job_id"] / "maisi.log").read_text(errors="replace")
              for r in records if r.get("status") != "ok")
    res = {
        "workers": n, "threads": ok[0].get("threads") if ok else None, "jobs": n_jobs, "ok": len(ok), "failed": n_jobs - len(ok), "oom": oom,
        "wall_s": round(wall, 1),
        "s_per_volume": round(wall / len(ok), 1) if ok else None,
        "job_wall_median_s": round(statistics.median(r["wall_s"] for r in ok), 1) if ok else None,
        "job_wall_max_s": round(max(r["wall_s"] for r in ok), 1) if ok else None,
        "proc_peak_gb_max": max((r.get("peak_gpu_gb") or 0) for r in ok) if ok else None,
        "gpu_mem_max_gb": round(max(mem), 1) if mem else None,
        "gpu_mem_total_gb": round(total, 1) if total else None,
        "gpu_util_mean": round(statistics.mean(util)) if util else None,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "bench.json").write_text(json.dumps(res, indent=2))
    with (args.run_root / "bench.jsonl").open("a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def write_table(run_root):
    rows = {}
    for line in (run_root / "bench.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["workers"]] = r
    base = rows.get(1, {}).get("s_per_volume")
    out = ["# Worker benchmark", "",
           "| workers | CPU threads each | jobs ok/total | OOM | wall s | s per volume | speedup | 1000 vols h | job time median/max s "
           "| proc peak GB | GPU mem max GB | GPU util mean % |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted(rows):
        r = rows[n]
        spv = r["s_per_volume"]
        out.append(f"| {n} | {r.get('threads') or 'all'} | {r['ok']}/{r['jobs']} | {r['oom']} | {r['wall_s']} | {spv} | "
                   f"{f'{base / spv:.2f}×' if base and spv else '-'} | {f'{spv * 1000 / 3600:.1f}' if spv else '-'} | "
                   f"{r['job_wall_median_s']}/{r['job_wall_max_s']} | {r['proc_peak_gb_max']} | "
                   f"{r['gpu_mem_max_gb']} of {r['gpu_mem_total_gb']} | {r['gpu_util_mean']} |")
    (run_root / "bench.md").write_text("\n".join(out) + "\n")
    print("\n".join(out), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--workers", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--jobs-per-worker", type=int, default=2, help="2 = a synchronised first wave + a staggered second")
    p.add_argument("--seed-base", type=int, default=1000, help="seeds away from the trial's 0-19")
    p.add_argument("--maisi-dir", type=Path, default=None)
    args = p.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    for n in args.workers:
        if (args.run_root / f"w{n}" / "bench.json").exists():
            print(f"w{n}: done, skipped", flush=True)
            continue
        print(f"[{datetime.now():%H:%M:%S}] w{n}: {n} worker(s), {n * args.jobs_per_worker} job(s)", flush=True)
        res = bench_one(n, args)
        print(f"[{datetime.now():%H:%M:%S}] w{n}: {json.dumps(res)}", flush=True)
        write_table(args.run_root)
        if res["failed"]:
            print(f"w{n}: {res['failed']} failed job(s), stopping (more workers will not fit either)", flush=True)
            break


if __name__ == "__main__":
    main()
