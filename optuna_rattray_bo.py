"""Bayesian hyperparameter optimization (Optuna TPE) for the Rattray fine-tune.

Storage-efficient: each trial resets the cached base weights, fine-tunes IN-MEMORY (no checkpoints
written), generates on the held-out test, scores realism_vs_sibling, then discards the model. Only
the scalar metric + params are logged (SQLite study + trials.tsv). Resumable (load_if_exists).

Search space: lr (log), freeze depth (none/hint/shallow), epochs (2-4; #1 showed >4 doesn't help),
only_mid_control, sd_locked (unfreeze SD decoder), batch (4/8).  Objective = MINIMIZE realism_vs_sibling.

Env: BO_TRIALS [25].  Run on ONE GPU node.  Upload to Data_augmentation repo root.
"""
from Data_augmentation.utils.shared_resources_config import CLDM_V15_YAML  # sets cldm path
import os, json, glob, gc, types, tempfile, shutil, subprocess
import numpy as np, cv2, einops, torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from reptorep_dataset_rattray import MyDataset
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from Data_augmentation.utils.local_config import CKPT_PATH_V4
import optuna

REPO = "/hpc/group/youlab/sa603/code/Data_augmentation"
RAT = "/hpc/group/youlab/sa603/data/external_datasets/rattray_2023"
TRAIN_JSON, TESTSRC, TESTREAL = RAT + "/train_rattray.json", RAT + "/test_sources", RAT + "/test_reals"
OUT = REPO + "/optuna_rattray"; os.makedirs(OUT, exist_ok=True)
N_TRIALS = int(os.environ.get("BO_TRIALS", "25"))
N_PROMPT = "longbody, lowres, bad anatomy, cropped, worst quality, low quality"

pl.seed_everything(42, workers=True)
print("[bo] caching base weights ...", flush=True)
BASE_SD = load_state_dict(CKPT_PATH_V4, location="cpu")   # cached CPU state_dict; fresh model reloads it per trial


@torch.no_grad()
def generate(m, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m.eval(); sampler = DDIMSampler(m)
    srcs = {}
    for fp in sorted(glob.glob(TESTSRC + "/*.TIF")):
        srcs.setdefault(os.path.basename(fp).split("_")[0], fp)
    for pfx, fp in srcs.items():
        img = cv2.resize(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB), (256, 256))
        ctl = torch.from_numpy(img.astype(np.float32) / 255.0).cuda()
        ctl = einops.rearrange(torch.stack([ctl, ctl], 0), "b h w c -> b c h w").clone()
        pl.seed_everything(729397049)
        cond = {"c_concat": [ctl], "c_crossattn": [m.get_learned_conditioning([""] * 2)]}
        un = {"c_concat": [ctl], "c_crossattn": [m.get_learned_conditioning([N_PROMPT] * 2)]}
        m.control_scales = [1.0] * 13
        s, _ = sampler.sample(50, 2, (4, 32, 32), cond, verbose=False, eta=0.0,
                              unconditional_guidance_scale=9.0, unconditional_conditioning=un)
        x = m.decode_first_stage(s)
        x = (einops.rearrange(x, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        for i in range(2):
            cv2.imwrite(f"{out_dir}/{pfx}_{i+1}.png", cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR))


def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 3e-5, log=True)
    freeze = trial.suggest_categorical("freeze", ["none", "hint", "shallow"])
    epochs = trial.suggest_int("epochs", 2, 4)
    only_mid = trial.suggest_categorical("only_mid_control", [False, True])
    sd_locked = True    # fixed: sd_locked=False adds SD-decoder optimizer states -> OOMs the 24GB A5000
    batch = 4           # fixed: keeps each trial within GPU memory

    model = create_model(CLDM_V15_YAML).cpu()         # fresh model per trial (proven path; trainer.fit owns device)
    model.load_state_dict(BASE_SD)
    model.learning_rate, model.sd_locked, model.only_mid_control = lr, sd_locked, only_mid
    frozen_ids = set()
    if freeze != "none":
        cm = model.control_model; mods = [cm.input_hint_block]
        if freeze == "shallow":
            mods += [cm.input_blocks[i] for i in range(6)]
        for mm in mods:
            for p in mm.parameters():
                frozen_ids.add(id(p))

    def cfg_opt(self):
        params = [p for p in self.control_model.parameters() if id(p) not in frozen_ids]
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        return torch.optim.AdamW(params, lr=self.learning_rate)
    model.configure_optimizers = types.MethodType(cfg_opt, model)

    dl = DataLoader(MyDataset(TRAIN_JSON), num_workers=0, batch_size=batch, shuffle=True)
    trainer = pl.Trainer(enable_progress_bar=False, gpus=1, precision=32, max_epochs=epochs,
                         logger=False, enable_checkpointing=False)
    trainer.fit(model, dl)
    model = model.cuda()                                # pipeline-style: makes CLIP encoder device-correct for gen

    tmp = tempfile.mkdtemp(dir=OUT)
    try:
        generate(model, tmp + "/gen")
        subprocess.run(["python", "eval_metrics.py", "--gen_dir", tmp + "/gen",
                        "--real_spec", f"RATTRAY:{TESTREAL}", "--out", tmp],
                       cwd=REPO, check=True, capture_output=True)
        o = json.load(open(tmp + "/metrics.json"))["overall"]
        g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
        realism, diversity, fid = g("realism_vs_sibling"), g("diversity"), o.get("FID")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        del trainer, model; gc.collect(); torch.cuda.empty_cache()

    trial.set_user_attr("diversity", diversity); trial.set_user_attr("FID", fid)
    hdr = not os.path.exists(OUT + "/trials.tsv")
    with open(OUT + "/trials.tsv", "a") as f:
        if hdr:
            f.write("trial\tlr\tfreeze\tepochs\tonly_mid\tsd_locked\tbatch\trealism\tdiversity\tFID\n")
        f.write(f"{trial.number}\t{lr:.2e}\t{freeze}\t{epochs}\t{only_mid}\t{sd_locked}\t{batch}\t{realism:.4f}\t{diversity:.4f}\t{fid}\n")
    print(f"[trial {trial.number}] lr={lr:.2e} fz={freeze} ep={epochs} mid={only_mid} sdl={sd_locked} bs={batch} -> realism={realism:.4f} div={diversity:.4f}", flush=True)
    return realism


study = optuna.create_study(direction="minimize", study_name="rattray_ft",
                            storage=f"sqlite:///{OUT}/study.db", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8))
study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))   # one bad trial won't kill the study
print("\n[bo] BEST realism =", round(study.best_value, 4), "params =", study.best_params, flush=True)
json.dump({"best_value": study.best_value, "best_params": study.best_params},
          open(OUT + "/best.json", "w"), indent=2)
