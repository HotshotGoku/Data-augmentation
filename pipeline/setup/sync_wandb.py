"""Sync file-based experiment records to Weights & Biases (DECOUPLED from compute jobs).

Reads experiments/*/manifest.json (+ any result_*.json) and the sweep summary.tsv files, and logs
them to W&B. Uses the pinned wandb 0.16 that coexists with this env (see [[youlab-conda-env-deps]]).
Compute jobs never import wandb, so this can never crash them.

  WANDB_MODE=offline python sync_wandb.py     # log locally -> `wandb sync wandb/offline-run-*` later
  WANDB_MODE=online  python sync_wandb.py     # after `wandb login`

Upload to: /hpc/group/youlab/sa603/code/Data_augmentation/sync_wandb.py
"""
import os
import json
import glob
import csv

import wandb

REPO = os.environ.get("AUG_REPO", "/hpc/group/youlab/sa603/code/Data_augmentation")
EXP = os.path.join(REPO, "experiments")
PROJECT = os.environ.get("WANDB_PROJECT", "youlab-augmenter")


def log_manifests():
    for mf in sorted(glob.glob(os.path.join(EXP, "*", "manifest.json"))):
        m = json.load(open(mf))
        run = wandb.init(project=PROJECT, name=m.get("run_id"), config=m,
                         tags=[m.get("kind", "")], job_type=m.get("kind", "run"), reinit=True)
        for rf in sorted(glob.glob(os.path.join(os.path.dirname(mf), "result_*.json"))):
            try:
                run.log({os.path.basename(rf)[:-5]: json.load(open(rf))})
            except Exception as e:
                print(f"  skip {rf}: {e}")
        run.finish()
        print(f"[sync] logged {m.get('run_id')}")


def log_summary(tsv, kind):
    if not os.path.exists(tsv):
        print(f"[sync] no summary at {tsv}")
        return
    with open(tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return
    run = wandb.init(project=PROJECT, name=f"{kind}_summary", job_type="sweep-summary", reinit=True)
    cols = list(rows[0].keys())
    tbl = wandb.Table(columns=cols)
    for r in rows:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({f"{kind}/summary": tbl})
    run.finish()
    print(f"[sync] logged {kind} summary ({len(rows)} rows)")


if __name__ == "__main__":
    log_manifests()
    log_summary(os.path.join(REPO, "infer_sweep_out", "summary.tsv"), "infer_sweep")
    log_summary(os.path.join(REPO, "sweep_eval_2sp", "summary.tsv"), "finetune_sweep")
    print("[sync] done. If offline: run `wandb sync wandb/offline-run-*` to push.")
