"""Lightweight, dependency-free experiment tracking for the Data_augmentation project.

Every run (finetune / inference sweep / eval) calls write_manifest() to record its FULL
config to experiments/<run_id>/manifest.json and append one row to experiments/runs.tsv.
This is the durable source of truth and imports nothing heavy, so it can never crash a job.

W&B logging is DECOUPLED on purpose: sync_wandb.py reads these files (+ eval outputs +
sample images) and pushes them to Weights & Biases separately. Rationale in memory
[[youlab-conda-env-deps]]: modern wandb breaks this fragile pinned env, and accelerate
imports wandb unconditionally once installed — so wandb must stay OUT of the compute path.

Upload to: /hpc/group/youlab/sa603/code/Data_augmentation/utils/tracking.py
"""
import os
import json
import csv
import subprocess
import socket
from datetime import datetime

REPO = os.environ.get("AUG_REPO", "/hpc/group/youlab/sa603/code/Data_augmentation")
EXPERIMENTS_DIR = os.environ.get("EXPERIMENTS_DIR", os.path.join(REPO, "experiments"))


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(kind, hparams):
    """Record a run. `kind` in {finetune, infer_sweep, eval, ...}; `hparams` a dict that
    should include a 'tag'. Returns (run_id, run_dir)."""
    job = os.environ.get("SLURM_JOB_ID", "local")
    tag = str(hparams.get("tag", kind))
    run_id = f"{kind}_{tag}_{job}"
    run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "run_id": run_id, "kind": kind, "tag": tag,
        "slurm_job_id": job, "host": socket.gethostname(),
        "git_sha": _git_sha(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **hparams,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    reg = os.path.join(EXPERIMENTS_DIR, "runs.tsv")
    cols = ["timestamp", "run_id", "kind", "tag", "slurm_job_id", "git_sha"]
    newfile = not os.path.exists(reg)
    with open(reg, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if newfile:
            w.writerow(cols)
        w.writerow([manifest.get(c, "") for c in cols])

    print(f"[tracking] {run_id} -> {run_dir}/manifest.json (git {manifest['git_sha']})", flush=True)
    return run_id, run_dir


def record_result(run_dir, name, data):
    """Attach a result blob (e.g. eval metrics) to a run directory."""
    with open(os.path.join(run_dir, f"result_{name}.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)
