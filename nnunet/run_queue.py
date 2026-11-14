#!/usr/bin/env python
"""Unattended queue for the liver filter arms (Decisions 2026-11-14): prepare, train + test each arm, three lanes.

    setsid nohup bash -c 'export QUEUE_CPU_WORKERS=20 && source "$PROJ/BT2026/nnunet/env.sh" && \
      python "$PROJ/BT2026/nnunet/run_queue.py"' > "$PROJ/logs/queue_liver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &

Lanes (threads; each step skips what is already done, so the same command resumes after a session end):
  prep    eval-all, metrics (filter_metrics.py), cc (maisi/cc_rules.py -> ids_cc*.json), figures,
          build-cc / preprocess-cc (Dataset103_LiverSynthRepairCC), build-ccAuc / preprocess-ccAuc (Dataset104), ...
  train   per arm in ARMS order: split (make_splits.py), train (nnUNetv2_train, 1000 epochs, + train_status.py).
          An arm waits only for the prep step it needs, so ccReject (a Dataset101 fold) trains while D103/D104 are
          built + preprocessed on the remaining cores.
  predict per trained arm: predict (predict_test.py, 111 real LiTS test cases, checkpoint_final) then eval over every
          finished run. Runs at the same time as the next arm's training (pipelining; with --no-pipeline the train
          lane waits for each prediction).
Default arms: ccReject -> ccRepair -> ccRepairDel -> ccAucRepair (the Holm family). Follow-ups by name.
A failed step skips what depends on it and the queue goes on.

Status: one file at the top of $PROJ, e.g. QUEUE_liver_arms__2of4_trained__train-ccRepair__ETA_Sun-15-Nov_09-30_CET.txt
(-> __DONE_ / __STOPPED_ at the end). Progress lines with times + results: $PROJ/queue_liver/progress.md.
Training settings as for `all`: default nnUNetTrainer, 1000 epochs, nnUNet_compile=f, nnUNet_n_proc_DA=24.

  python run_queue.py --dry-run          # what would run, what is done
  python run_queue.py --prep-only        # prep lane only
  python run_queue.py --arms ccReject    # only some arms (the prep steps they need still run)
  python run_queue.py --arms aucReject aucRepair      # follow-ups
  python run_queue.py --second-run ccRepair           # a second run of an arm (same train cases)
"""
import argparse
import glob
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
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

ARMS = [  # name, dataset id, ids file (None = every synthetic case), prep step it needs  (Decisions 2026-11-14)
    ("ccReject", 101, f"{FM}/ids_ccReject.json", "cc"),
    ("ccRepair", 103, None, "preprocess-cc"),
    ("ccRepairDel", 103, f"{FM}/ids_ccRepairDel.json", "preprocess-cc"),
    ("ccAucRepair", 104, None, "preprocess-ccAuc"),
]
MAIN = [a[0] for a in ARMS]  # the Holm family of the main comparison
FOLLOW_UPS = [
    ("aucReject", 101, f"{FM}/ids_aucReject.json", "metrics"),
    ("aucRepair", 105, None, "preprocess-auc"),
    ("ccAucReject", 101, f"{FM}/ids_ccAucReject.json", "cc"),
]
BASELINE = [(101, 0), (101, 1)]  # `all` and `all_rep2`
REPAIR_DS = {"cc": (103, "Dataset103_LiverSynthRepairCC"), "ccAuc": (104, "Dataset104_LiverSynthRepairCCAUC"),
             "auc": (105, "Dataset105_LiverSynthRepairAUC")}
HOURS = {"eval-all": 0.1, "metrics": 0.1, "cc": 0.2, "figures": 0.1, "build": 0.3, "preprocess": 0.7, "split": 0.0,
         "train": 4.9, "predict": 2.0, "eval": 0.1}
DEPENDS = {"cc": "metrics", "figures": "metrics", **{f"build-{c}": ("cc" if c.startswith("cc") else "metrics")
                                                     for c in REPAIR_DS},
           **{f"preprocess-{c}": f"build-{c}" for c in REPAIR_DS}}
WORKERS = int(os.environ.get("QUEUE_CPU_WORKERS", "16"))


def now():
    return datetime.now(TZ)


class Queue:
    def __init__(self, arms, dry, prep_only=False, pipeline=True):
        self.dry, self.arms, self.prep_only, self.pipeline = dry, arms, prep_only, pipeline
        # prep order: what the first arm needs first; eval-all + figures last (nothing waits for them)
        self.prep = ["metrics", "cc"]
        for crit in REPAIR_DS:
            if any(a[3] == f"preprocess-{crit}" for a in arms):
                self.prep += [f"build-{crit}", f"preprocess-{crit}"]
        self.prep += ["eval-all", "figures"]
        self.ready = {s: threading.Event() for s in self.prep}
        self.failed = set()
        self.lock = threading.RLock()
        self.active = {}  # lane -> step label
        self.n_trained = 0
        self.train_t0 = None
        self.pq = queue.Queue()
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
        with self.lock:
            print(line, flush=True)
            if not self.dry:
                with open(self.dir / "progress.md", "a") as fh:
                    fh.write(line + "\n")

    def set_status(self, end=None):
        if self.dry:
            return
        with self.lock:
            n = len(self.arms)
            left = HOURS["train"] * (n - self.n_trained) + HOURS["predict"] + HOURS["eval"]
            if self.train_t0 and "train" in self.active and self.active["train"].startswith("train-"):
                left -= min((time.time() - self.train_t0) / 3600, HOURS["train"] * 0.95)
            if not n:
                left = 1.0
            when = end or f"ETA_{(now() + timedelta(hours=left)):%a-%d-%b_%H-%M}_CET"
            act = "+".join(self.active.get(k) for k in ("train", "predict", "prep") if self.active.get(k)) or "idle"
            name = f"QUEUE_liver_arms__{self.n_trained}of{n}_trained__{act if not end else 'finished'}__{when}.txt"
            new = PROJ / name
            body = (f"Liver filter-arm queue (nnunet/run_queue.py), started from {REPO}\n"
                    f"progress: {self.dir / 'progress.md'}\narms: {', '.join(a[0] for a in self.arms)}\n"
                    f"prep: {', '.join(self.prep)}\npipeline: {self.pipeline}\nactive: {self.active}\n"
                    f"failed: {sorted(self.failed) or 'none'}\nupdated {now():%Y-%m-%d %H:%M} CET\n")
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
        """Every arm with a complete test prediction (the baseline runs are passed separately)."""
        runs = []
        known = ARMS + FOLLOW_UPS + [a for a in self.arms if a not in ARMS + FOLLOW_UPS]
        for arm, did, _, _ in known:
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
                for f in ("ids_touchReject.json", "ids_aucReject.json", "ids_thesis15mm.json"):
                    d = json.load(open(f"{FM}/{f}"))
                    self.log(f"metrics: {f} passes {d['n_pass']}/{d['n_total']}")
            return
        if name == "cc":
            self.run([PY, REPO / "maisi/cc_rules.py", "--metrics", FM, "--workers", WORKERS], "cc_rules_liver")
            if not self.dry:
                c = json.load(open(f"{FM}/cc_rules.json"))["counts"]
                for k in ("cc", "ccAuc"):
                    self.log(f"cc: {k} unchanged {c[k]['unchanged']}, kinds {c[k]['kind']}, tumour-free "
                             f"{c[k]['tumour_free']}, components relabelled {c[k]['components_relabelled']}")
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
            if isinstance(ids, tuple):  # ("same-as", arm): a second run of a registered arm
                cmd += ["--same-as", ids[1]]
            else:
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

    @staticmethod
    def running_train(did, fold):
        r = subprocess.run(["pgrep", "-f", f"nnUNetv2_train {did} 3d_fullres {fold}( |$)"], capture_output=True, text=True)
        return [int(x) for x in r.stdout.split()]

    def train(self, arm_name, did, fold):
        fd = self.fold_dir(did, fold) if not self.dry else Path("<fold_dir>")
        done = (fd / "checkpoint_final.pth").exists() and (fd / "validation" / "summary.json").exists()
        if done:
            self.log(f"train: {arm_name} already done")
            return
        running = self.running_train(did, fold)
        if running:  # e.g. started by a queue that was restarted: wait for it instead of starting a second one
            self.log(f"train: {arm_name} already running (pid {running}), waiting for it")
            wl = PROJ / "logs" / f"status_train_d{did}_{arm_name}_{now():%Y%m%d_%H%M%S}.log"
            watcher = subprocess.Popen([PY, HERE / "train_status.py", "--fold-dir", fd, "--name",
                                        f"D{did}_3d_fullres_f{fold}_{arm_name}"],
                                       stdout=open(wl, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                while self.running_train(did, fold):
                    time.sleep(60)
            finally:
                watcher.terminate()
                watcher.wait()
                subprocess.run([PY, HERE / "train_status.py", "--fold-dir", fd, "--once", "--name",
                                f"D{did}_3d_fullres_f{fold}_{arm_name}"], stdout=subprocess.DEVNULL, check=False)
            return self.train(arm_name, did, fold)  # done now, or resume (--c) / validate (--val)
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
        runs = self.finished_runs() if not self.dry else []
        self.run([PY, HERE / "predict_test.py", "eval", "--baseline", *[f"{d}:{f}" for d, f in BASELINE],
                  *(["--runs", *[f"{d}:{f}" for d, f in runs]] if runs else []), "--family", *MAIN,
                  "--workers", WORKERS], "eval_liver")
        if not self.dry:
            out = Path(os.environ["nnUNet_results"]) / "test_eval_liver" / "summary.md"
            rows = [l for l in out.read_text().splitlines() if l.startswith("| D")]
            self.log(f"eval (baseline + {len(runs)} arms): see `{out}`\n" + "\n".join("  " + r for r in rows[: 2 + len(runs)]))

    # --- lanes -------------------------------------------------------------------------------------------------
    def run_step(self, name, arm, lane):
        label = f"{name}-{arm[0]}" if arm else name
        with self.lock:
            self.active[lane] = label
            if name == "train":
                self.train_t0 = time.time()
        self.set_status()
        t0 = time.time()
        try:
            self.step(name, arm)
            if not self.dry:
                self.log(f"ok {label} ({(time.time() - t0) / 3600:.2f} h)")
            return True
        except Exception as e:  # noqa: BLE001
            self.log(f"FAILED {label}: {e!r}")
            with self.lock:
                self.failed.add(arm[0] if arm else name)
            return False
        finally:
            with self.lock:
                self.active.pop(lane, None)

    def prep_lane(self):
        for name in self.prep:
            try:
                if DEPENDS.get(name) in self.failed:
                    self.log(f"skip {name} (depends on a failed step)")
                    self.failed.add(name)
                else:
                    self.run_step(name, None, "prep")
            finally:
                self.ready[name].set()

    def train_lane(self):
        for arm in self.arms:
            needs = arm[3]
            if needs in self.ready and not self.ready[needs].is_set():
                self.log(f"{arm[0]} waits for {needs}")
                self.ready[needs].wait()
            if needs in self.failed:
                self.log(f"skip {arm[0]} (depends on failed {needs})")
                self.failed.add(arm[0])
                continue
            if not self.run_step("split", arm, "train") or not self.run_step("train", arm, "train"):
                continue
            with self.lock:
                self.n_trained += 1
            if self.pipeline:
                self.pq.put(arm)
            else:
                self.predict_one(arm)
        self.pq.put(None)

    def predict_lane(self):
        while (arm := self.pq.get()) is not None:
            self.predict_one(arm)

    def predict_one(self, arm):
        if self.run_step("predict", arm, "predict"):
            self.run_step("eval", arm, "predict")

    def go(self):
        self.log(f"queue start (arms {[a[0] for a in self.arms]}, prep {self.prep}, pipeline={self.pipeline}, "
                 f"dry={self.dry})")
        if self.dry:  # one after the other, commands only
            self.pipeline = False
            self.prep_lane()
            if not self.prep_only:
                self.train_lane()
            return
        lanes = [threading.Thread(target=self.prep_lane, name="prep", daemon=True)]
        if not self.prep_only:
            lanes.append(threading.Thread(target=self.train_lane, name="train", daemon=True))
            if self.pipeline:
                lanes.append(threading.Thread(target=self.predict_lane, name="predict", daemon=True))
        for t in lanes:
            t.start()
        while any(t.is_alive() for t in lanes):  # join with a timeout, so SIGTERM is handled
            for t in lanes:
                t.join(5)
        end = f"{'STOPPED' if self.failed else 'DONE'}_{now():%a-%d-%b_%H-%M}_CET"
        self.set_status(end)
        self.log(f"queue finished, failed: {sorted(self.failed) or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", nargs="+", choices=[a[0] for a in ARMS + FOLLOW_UPS],
                    help="default: the 4 main arms; follow-ups only by name")
    ap.add_argument("--second-run", nargs="+", default=[], metavar="ARM",
                    help="a second run <ARM>_rep2 of these arms (same train cases, new seed of nnU-Net's randomness)")
    ap.add_argument("--prep-only", action="store_true", help="prep lane only (no split/train/predict)")
    ap.add_argument("--no-pipeline", action="store_true", help="predict each arm before training the next")
    a = ap.parse_args()
    arms = [x for x in ARMS + FOLLOW_UPS if x[0] in a.arms] if a.arms else ([] if a.second_run else list(ARMS))
    for name in a.second_run:
        base = next((x for x in ARMS + FOLLOW_UPS if x[0] == name), None)
        if base is None:
            raise SystemExit(f"--second-run {name}: unknown arm")
        arms.append((f"{name}_rep2", base[1], ("same-as", name), base[3]))
    q = Queue(arms, a.dry_run, a.prep_only, not a.no_pipeline)

    def stop(sig, _frame):
        q.log(f"signal {sig}: stopping")
        q.set_status(f"STOPPED_{now():%a-%d-%b_%H-%M}_CET")
        os._exit(1)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    q.go()


if __name__ == "__main__":
    main()
