#!/usr/bin/env python
"""Unattended queue for the liver filter arms (PLAN_training.md, 2026-11-12): prepare, then train + test each arm.

CPU session first (`--prep-only`: everything up to the preprocessed repair datasets, then look at the sanity checks),
then the GPU session with the same command without the flag (done prep steps are skipped):

    setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && python "$PROJ/BT2026/nnunet/run_queue.py"' \
      > "$PROJ/logs/queue_liver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &

Steps, in order (each one skips what is already done, so the same command resumes after a session end):
  eval-all     lesion-wise re-evaluation of the existing `all` runs (Dataset101 folds 0 + 1)       CPU, minutes
  metrics      maisi/filter_metrics.py (touch + separability, real reference, cut)
               -> ids_rejectTouch.json, ids_rejectAUC.json, rules.json                             CPU, ~15-30 min
  figures      maisi/filter_figures.py: contact sheets for the sanity checks                       CPU, minutes
  build-X      build_repair_dataset.py --criterion X --plans -> Dataset103 (touch) / 104 (auc)     CPU, ~10 min
  preprocess-X nnUNetv2_preprocess -d 103 / 104 (Dataset101 plans)                                 CPU, ~30 min
  per arm, in ARMS order (repairTouch -> rejectTouch -> repairAUC -> rejectAUC):
    split      make_splits.py (registered once; the fold number is looked up by arm name)
    train      nnUNetv2_train <d> 3d_fullres <fold> --c (+ train_status.py status file)            GPU, ~4.9 h
    predict    predict_test.py predict (111 real LiTS test cases, checkpoint_final)                GPU, ~2 h
    eval       predict_test.py eval over every finished run (reference = `all` fold 0)             CPU, minutes
A failed step skips what depends on it (e.g. a failed metrics step skips every arm) and the queue goes on.
Only the prep steps the selected arms need are run.

Status: one file at the top of $PROJ, e.g. QUEUE_liver_arms__4of15_train-repair__ETA_Fri-13-Nov_09-30_CET.txt
(-> __DONE_ / __STOPPED_ at the end; ETA from rough per-step estimates, the training watcher has the exact one).
Progress lines with times + results: $PROJ/queue_liver/progress.md (copy the key numbers into NOTES.md).
Training settings as for `all`: default nnUNetTrainer, 1000 epochs, nnUNet_compile=f, nnUNet_n_proc_DA=24.

  python run_queue.py --dry-run          # what would run, what is done
  python run_queue.py --prep-only        # CPU session: prep steps only, no training
  python run_queue.py --arms repairTouch rejectTouch   # only some arms (prep steps they need still run)
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROJ = Path(os.environ.get("PROJ", "/work/Research"))
POOL = os.environ.get("MAISI_RUNS", f"{PROJ}/maisi_runs") + "/pool"
FM = f"{POOL}/filter_metrics/liver"
TZ = ZoneInfo("Europe/Copenhagen")
PY = sys.executable

ARMS = [  # name, dataset id, ids file (None = every synthetic case), prep step it needs
    ("repairTouch", 103, None, "preprocess-touch"),
    ("rejectTouch", 101, f"{FM}/ids_rejectTouch.json", "metrics"),
    ("repairAUC", 104, None, "preprocess-auc"),
    ("rejectAUC", 101, f"{FM}/ids_rejectAUC.json", "metrics"),
]
REPAIR_DS = {"touch": (103, "Dataset103_LiverSynthRepairTouch"), "auc": (104, "Dataset104_LiverSynthRepairAUC")}
HOURS = {"eval-all": 0.1, "metrics": 0.5, "figures": 0.1, "build": 0.2, "preprocess": 0.5, "split": 0.0, "train": 4.9,
         "predict": 2.0, "eval": 0.1}
DEPENDS = {"figures": "metrics", "build-touch": "metrics", "build-auc": "metrics",
           "preprocess-touch": "build-touch", "preprocess-auc": "build-auc"}
WORKERS = int(os.environ.get("QUEUE_CPU_WORKERS", "16"))


def now():
    return datetime.now(TZ)


class Queue:
    def __init__(self, arms, dry, prep_only=False):
        self.dry = dry
        self.steps = [("eval-all", None), ("metrics", None), ("figures", None)]
        for crit in REPAIR_DS:
            if any(a[3] == f"preprocess-{crit}" for a in arms):
                self.steps += [(f"build-{crit}", None), (f"preprocess-{crit}", None)]
        if not prep_only:
            for a in arms:
                self.steps += [(s, a) for s in ("split", "train", "predict", "eval")]
        self.failed = set()
        self.dir = PROJ / "queue_liver"
        self.status = None
        if not dry:
            self.dir.mkdir(parents=True, exist_ok=True)
            done_dir = PROJ / "status_done"
            for f in PROJ.glob("*.txt"):
                if "__DONE_" in f.name or "__STOPPED_" in f.name:
                    done_dir.mkdir(exist_ok=True)
                    shutil.move(str(f), done_dir / f.name)

    # --- bookkeeping -------------------------------------------------------------------------------------------
    def log(self, msg):
        line = f"- {now():%Y-%m-%d %H:%M} {msg}"
        print(line, flush=True)
        if not self.dry:
            with open(self.dir / "progress.md", "a") as fh:
                fh.write(line + "\n")

    def set_status(self, i, label, end=None):
        if self.dry:
            return
        left = sum(HOURS[s] for s, _ in self.steps[i:])
        when = end or f"ETA_{(now() + timedelta(hours=left)):%a-%d-%b_%H-%M}_CET"
        name = f"QUEUE_liver_arms__{i + 1 if not end else i}of{len(self.steps)}_{label}__{when}.txt"
        new = PROJ / name
        body = (f"Liver filter-arm queue (nnunet/run_queue.py), started from {REPO}\n"
                f"progress: {self.dir / 'progress.md'}\nsteps: " + ", ".join(f"{s}-{a[0]}" if a else s for s, a in self.steps) +
                f"\nfailed: {sorted(self.failed) or 'none'}\nupdated {now():%Y-%m-%d %H:%M} CET\n")
        if self.status and self.status.exists() and self.status != new:
            self.status.rename(new)
        new.write_text(body)
        self.status = new

    def run(self, cmd, log_name=None, env=None):
        """Run a command; its output goes to the queue log, or to its own log in $PROJ/logs."""
        print("$ " + " ".join(map(str, cmd)), flush=True)
        if self.dry:
            return
        e = dict(os.environ, **(env or {}))
        if log_name:
            lf = PROJ / "logs" / f"{log_name}_{now():%Y%m%d_%H%M%S}.log"
            lf.parent.mkdir(exist_ok=True)
            self.log(f"`{' '.join(map(str, cmd[:6]))} ...` log `{lf}`")
            with open(lf, "w") as fh:
                subprocess.run(list(map(str, cmd)), check=True, env=e, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        else:
            subprocess.run(list(map(str, cmd)), check=True, env=e, stdin=subprocess.DEVNULL)

    # --- helpers -----------------------------------------------------------------------------------------------
    @staticmethod
    def ds_name(did):
        f = glob.glob(f"{os.environ['nnUNet_raw']}/Dataset{did:03d}_*")
        return os.path.basename(f[0]) if f else f"Dataset{did:03d}_?"

    def fold_of(self, arm, did):
        f = Path(os.environ["nnUNet_preprocessed"]) / self.ds_name(did) / "arms.json"
        arms = json.loads(f.read_text()) if f.exists() else []
        return next((i for i, a in enumerate(arms) if a["arm"] == arm), None)

    def fold_dir(self, did, fold):
        return Path(os.environ["nnUNet_results"]) / self.ds_name(did) / "nnUNetTrainer__nnUNetPlans__3d_fullres" / f"fold_{fold}"

    def n_test(self):
        return len(json.load(open(f"{os.environ['nnUNet_raw']}/Dataset101_LiverSynth/real_split.json"))["test"])

    def finished_runs(self):
        runs = [(101, 0), (101, 1)]
        for arm, did, _, _ in ARMS:
            fold = self.fold_of(arm, did)
            if fold is not None and len(glob.glob(f"{self.fold_dir(did, fold)}/test_final/lits_*.nii.gz")) == self.n_test():
                runs.append((did, fold))
        return runs

    # --- steps -------------------------------------------------------------------------------------------------
    def step(self, name, arm):
        if name == "eval-all":
            return self.eval()
        if name == "metrics":
            self.run([PY, REPO / "maisi/filter_metrics.py", "--target", "liver", "--pool", POOL, "--workers", WORKERS],
                     "filter_metrics_liver")
            if not self.dry:
                r = json.load(open(f"{FM}/rules.json"))
                self.log(f"metrics: real lesions {r['real_lesions']} ({r['real_lesions_scored']} scored), "
                         f"AUC cut {r['auc_cut']}")
                for f in ("ids_rejectTouch.json", "ids_rejectAUC.json", "ids_thesis15mm.json"):
                    d = json.load(open(f"{FM}/{f}"))
                    self.log(f"metrics: {f} passes {d['n_pass']}/{d['n_total']}")
            return
        if name == "figures":
            fig, rules = Path(FM) / "figures" / "real_least_visible.png", Path(FM) / "rules.json"
            if not self.dry and fig.exists() and fig.stat().st_mtime > rules.stat().st_mtime:
                self.log("figures: up to date")
                return
            self.run([PY, REPO / "maisi/filter_figures.py", "--metrics", FM, "--pool", POOL, "--target", "liver"],
                     "filter_figures_liver")
            return
        if name.startswith("build-"):
            crit = name.split("-", 1)[1]
            did, ds = REPAIR_DS[crit]
            self.run([PY, HERE / "build_repair_dataset.py", "--criterion", crit, "--metrics", FM, "--workers", WORKERS,
                      "--plans"], f"build_ds{did}")
            if not self.dry:
                s = open(f"{os.environ['nnUNet_raw']}/{ds}/repair_summary.md").read()
                self.log(f"build {ds}: " + " ".join(l for l in s.splitlines() if l.startswith(("Volumes changed", "Tumour-free"))))
            return
        if name.startswith("preprocess-"):
            did, ds = REPAIR_DS[name.split("-", 1)[1]]
            pre = Path(os.environ["nnUNet_preprocessed"]) / ds
            n = json.load(open(f"{os.environ['nnUNet_raw']}/{ds}/dataset.json"))["numTraining"] if not self.dry else 0
            if not self.dry and len(list((pre / "nnUNetPlans_3d_fullres").glob("*.pkl"))) == n:
                self.log(f"preprocess: {ds} already done ({n} cases)")
                return
            self.run(["nnUNetv2_preprocess", "-d", did, "-plans_name", "nnUNetPlans", "-c", "3d_fullres", "-np", WORKERS],
                     f"preprocess_ds{did}")
            if not self.dry:
                got = len(list((pre / "nnUNetPlans_3d_fullres").glob("*.pkl")))
                assert got == n, f"preprocessed {got} of {n}"
            return
        arm_name, did, ids, _ = arm
        if name == "split":
            if self.fold_of(arm_name, did) is not None:
                return
            cmd = [PY, HERE / "make_splits.py", "--target", "liver", "--arm", arm_name]
            cmd += ["--dataset-id", did] if did != 101 else []
            cmd += ["--ids", ids] if ids else ["--all-synth"]
            self.run(cmd)
            if not self.dry:
                self.log(f"split: {arm_name} = D{did} fold {self.fold_of(arm_name, did)}")
            return
        fold = self.fold_of(arm_name, did) if not self.dry else "<fold>"
        if name == "train":
            return self.train(arm_name, did, fold)
        if name == "predict":
            self.run([PY, HERE / "predict_test.py", "predict", "--dataset", did, "--fold", fold], f"predict_d{did}_{arm_name}")
            return
        if name == "eval":
            return self.eval()

    def train(self, arm_name, did, fold):
        fd = self.fold_dir(did, fold) if not self.dry else Path("<fold_dir>")
        done = (fd / "checkpoint_final.pth").exists() and (fd / "validation" / "summary.json").exists()
        if done:
            self.log(f"train: {arm_name} already done")
            return
        val_only = (fd / "checkpoint_final.pth").exists()
        cmd = ["nnUNetv2_train", did, "3d_fullres", fold] + (["--val"] if val_only else ["--c"])
        env = {"nnUNet_compile": os.environ.get("nnUNet_compile", "f"),
               "nnUNet_n_proc_DA": os.environ.get("nnUNet_n_proc_DA", "24")}
        watcher = None
        if not self.dry and not val_only:
            fd.mkdir(parents=True, exist_ok=True)
            wl = PROJ / "logs" / f"status_train_d{did}_{arm_name}_{now():%Y%m%d_%H%M%S}.log"
            watcher = subprocess.Popen([PY, HERE / "train_status.py", "--fold-dir", fd, "--name",
                                        f"D{did}_3d_fullres_f{fold}_{arm_name}"],
                                       stdout=open(wl, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            self.run(cmd, f"train_d{did}_{arm_name}_f{fold}", env)
        finally:
            if watcher:  # polls every 60 s: stop it and write the final name (DONE, or where it stopped) once
                watcher.terminate()
                watcher.wait()
                subprocess.run([PY, HERE / "train_status.py", "--fold-dir", fd, "--once", "--name",
                                f"D{did}_3d_fullres_f{fold}_{arm_name}"], stdout=subprocess.DEVNULL, check=False)
        if not self.dry:
            summ = json.load(open(fd / "validation" / "summary.json"))
            dice = {k: round(v["Dice"], 3) for k, v in summ.get("mean", {}).items()}
            self.log(f"train: {arm_name} done, val Dice (20 real) {dice}")

    def eval(self):
        runs = self.finished_runs() if not self.dry else [(101, 0), (101, 1)]
        self.run([PY, HERE / "predict_test.py", "eval", "--runs", *[f"{d}:{f}" for d, f in runs], "--workers", WORKERS], "eval_liver")
        if not self.dry:
            out = Path(os.environ["nnUNet_results"]) / "test_eval_liver" / "summary.md"
            rows = [l for l in out.read_text().splitlines() if l.startswith("| D")]
            self.log(f"eval ({len(runs)} runs): see `{out}`\n" + "\n".join("  " + r for r in rows[: 2 * len(runs)]))

    # --- main loop ---------------------------------------------------------------------------------------------
    def go(self):
        self.log(f"queue start ({len(self.steps)} steps, dry={self.dry})")
        for i, (name, arm) in enumerate(self.steps):
            label = f"{name}-{arm[0]}" if arm else name
            needs = arm[3] if arm else DEPENDS.get(name)
            if needs in self.failed or (arm and arm[0] in self.failed):
                self.log(f"skip {label} (depends on a failed step)")
                self.failed.add(arm[0] if arm else name)
                continue
            self.set_status(i, label)
            t0 = time.time()
            try:
                self.step(name, arm)
                if not self.dry:
                    self.log(f"ok {label} ({(time.time() - t0) / 3600:.2f} h)")
            except Exception as e:  # noqa: BLE001
                self.log(f"FAILED {label}: {e!r}")
                self.failed.add(arm[0] if arm else name)
        end = f"{'STOPPED' if self.failed else 'DONE'}_{now():%a-%d-%b_%H-%M}_CET"
        self.set_status(len(self.steps), "finished", end)
        self.log(f"queue finished, failed: {sorted(self.failed) or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", nargs="+", choices=[a[0] for a in ARMS])
    ap.add_argument("--prep-only", action="store_true", help="CPU session: prep steps only (no split/train/predict)")
    a = ap.parse_args()
    arms = [x for x in ARMS if not a.arms or x[0] in a.arms]
    q = Queue(arms, a.dry_run, a.prep_only)

    def stop(sig, _frame):
        q.log(f"signal {sig}: stopping")
        q.set_status(len(q.steps), "stopped", f"STOPPED_{now():%a-%d-%b_%H-%M}_CET")
        sys.exit(1)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    q.go()


if __name__ == "__main__":
    main()
