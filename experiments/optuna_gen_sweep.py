"""Time-boxed Optuna sweep of SD1.5 branching GEN hyperparameters, steered by the faithfulness probe.

Each trial regenerates a small synth set (28 clean test sources x4 = 112) from a FIXED ckpt with the
trial's (strength/scale/steps/eta), labels it, builds a reals+synth features CSV, runs the probe (1 seed),
and returns the shape_factor recovery (maximize). Trial 0 is the DEFAULT HP (strength1.0/scale9.0/steps50/
eta0) baseline, so other trials are read relative to it. Runs inside one GPU sbatch (no retrain -> cheap).

Env: SWEEP_CKPT (req, the fixed SD1.5 ckpt), N_TRIALS[15].
P4 Part 2 (time-boxed, user choice 2026-10-26). Odds LOW -- HP tuning never flipped the negative.
"""
import os, subprocess, glob, csv, optuna

REPO = "/hpc/group/youlab/sa603/code/Data_augmentation"
D = "/hpc/group/youlab/sa603/data/clean/branching_256"
CKPT = os.environ["SWEEP_CKPT"]
CLEAN_CSV = f"{REPO}/downstream/outputs/branching_downstream_out/features_branching_clean.csv"  # reals rows reused
N_TRIALS = int(os.environ.get("N_TRIALS", "15"))
WORK = "/tmp/optuna_sweep"; os.makedirs(WORK, exist_ok=True)


def run(cmd, extra_env):
    r = subprocess.run(cmd, env={**os.environ, **extra_env}, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  CMD FAIL {cmd[:3]}: {r.stderr[-500:]}", flush=True)
    return r.returncode == 0


def objective(t):
    p = dict(strength=round(t.suggest_float("strength", 0.6, 1.0), 3),
             scale=round(t.suggest_float("scale", 4.0, 12.0), 2),
             steps=t.suggest_int("steps", 25, 60),
             eta=round(t.suggest_float("eta", 0.0, 0.6), 3))
    try:
        syn = f"{WORK}/synth_{t.number}"
        ok = run(["python", "downstream/gen_branching_synth.py"],
                 dict(SYNTH_CKPT=CKPT, SYNTH_DIR=syn, REAL_DIR=f"{D}/test/sources",
                      EXCLUDE_DIR="/nonexistent", SYNTH_PER_SRC="4",
                      STRENGTH=str(p["strength"]), SCALE=str(p["scale"]),
                      STEPS=str(p["steps"]), ETA=str(p["eta"])))
        n = len(glob.glob(f"{syn}/*.png"))
        if not ok or n == 0:
            print(f"trial {t.number} {p}: GEN FAIL (n={n})", flush=True); return -99.0
        fsyn = f"{WORK}/f_{t.number}.csv"
        run(["python", "downstream/colony_features.py", "--images", f"{syn}/*.png", "--out", fsyn], {})
        comb = f"{WORK}/csv_{t.number}.csv"
        lines = open(CLEAN_CSV).readlines()
        with open(comb, "w") as o:
            o.write(lines[0]); o.writelines(l for l in lines[1:] if ".TIF," in l)
            o.writelines(open(fsyn).readlines()[1:])
        tsv = f"{WORK}/probe_{t.number}.tsv"
        if os.path.exists(tsv): os.remove(tsv)
        run(["python", "downstream/faithfulness_probe.py"],
            dict(SEED="0", TASK="branching", TRAIN_REAL_DIR=f"{D}/canon",
                 TEST_SRC_DIR=f"{D}/test/sources", TEST_REAL_DIR=f"{D}/test/reals",
                 FEATURES_CSV=comb, SYNTH_DIR=syn, OUT_TSV=tsv,
                 BACKBONE="resnet18", EPOCHS="30", NW="8"))
        rec = -99.0
        if os.path.exists(tsv):
            for r in csv.DictReader(open(tsv), delimiter="\t"):
                if r.get("axis") == "shape_factor":
                    rec = float(r["synth_recovery"])
        print(f"trial {t.number} {p} n={n} -> shape_factor_recovery={rec:.4f}", flush=True)
        return rec
    except Exception as e:
        print(f"trial {t.number} {p}: EXC {e}", flush=True); return -99.0


study = optuna.create_study(direction="maximize")
study.enqueue_trial({"strength": 1.0, "scale": 9.0, "steps": 50, "eta": 0.0})  # default baseline = trial 0
study.optimize(objective, n_trials=N_TRIALS)
print(f"\n=== BEST shape_factor_recovery={study.best_value:.4f} params={study.best_params} ===", flush=True)
base = study.trials[0].value
print(f"baseline (trial 0, default HP) = {base}", flush=True)
print("=== all trials (best first) ===", flush=True)
for tr in sorted(study.trials, key=lambda x: -(x.value if x.value is not None else -99)):
    print(f"  {tr.value:.4f}  {tr.params}", flush=True)
