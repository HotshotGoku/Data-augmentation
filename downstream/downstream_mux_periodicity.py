"""Downstream UTILITY test: predict mux pattern PERIODICITY (perim_roughness, n_lobes) from a colony
image -- a NEW morphology axis complementary to the aTc/IPTG decode. perim_roughness correlates with aTc
(+0.58); the thesis predicts a generative augmenter beats rotation here, because rotation copies carry
the SAME (rotation-invariant) periodicity label -> no label diversity, while each synth replicate carries
its OWN recomputed periodicity -> genuine (image,label) diversity. Same regime as the branching
feature-regression task, on the mux dataset.

Arms (matched counts): MODE=real | classical(rot+flip, SAME label) | augmenter(synth, OWN label). Held-out
REAL test reps {3,7,12}; model-select on val reps {2,6,11}; report per-feature R2 + mean. Labels from
MUX_PERIODICITY_CSV (built by mux_periodicity.py over reals+synth). Appends one row to RESULTS_TSV.

Env: K(1|2|3|all), MODE, SEED, N_PER[8], EPOCHS[40], BACKBONE[resnet18|resnet50], FEATS
     [perim_roughness,n_lobes], MUX_PERIODICITY_CSV, SYNTH_DIR, REALS_DIR, RESULTS_TSV. DCC GPU."""
import os, glob, re, csv, random
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

REALS = os.environ.get("REALS_DIR", "/hpc/group/youlab/sa603/data/clean/multiplexed_eval/reals")
SYNTH = os.environ.get("SYNTH_DIR", "")
CSV_PATH = os.environ["MUX_PERIODICITY_CSV"]          # name -> periodicity feats (reals + synth)
RESULTS = os.environ.get("RESULTS_TSV", "/hpc/group/youlab/sa603/code/Data_augmentation/mux_periodicity_out/results.tsv")
K = os.environ.get("K", "3"); MODE = os.environ.get("MODE", "real"); MODE_TAG = os.environ.get("MODE_TAG", MODE)
SEED = int(os.environ.get("SEED", "0")); N_PER = int(os.environ.get("N_PER", "8")); EPOCHS = int(os.environ.get("EPOCHS", "40"))
BACKBONE = os.environ.get("BACKBONE", "resnet18")
FEATS = os.environ.get("FEATS", "perim_roughness,n_lobes").split(",")
NW = int(os.environ.get("NW", "4"))
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SRC_REPS = [8, 9, 10]; ALL_TRAIN = [1, 4, 5, 8, 9, 10]; VAL_REPS, TEST_REPS = [2, 6, 11], [3, 7, 12]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

labels = {row["name"]: np.array([float(row[f]) for f in FEATS], np.float32) for row in csv.DictReader(open(CSV_PATH))}
print(f"[labels] {len(labels)} imgs labeled with {FEATS}", flush=True)

def load_rgb(fp):
    return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def rot_flip(img, ang, flip):
    h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.flip(out, 1) if flip else out

real_by = {}
for fp in glob.glob(REALS + "/*.TIF"):
    m = re.match(r"R(\d+)C(\d+)_Rep(\d+)\.TIF$", os.path.basename(fp))
    if m:
        real_by[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = fp
use_reps = set(ALL_TRAIN) if K == "all" else set(SRC_REPS[:int(K)])

train_items = []
for (row, col, rep), fp in real_by.items():
    if rep not in use_reps:
        continue
    y = labels.get(os.path.basename(fp))
    if y is None:
        continue
    img = load_rgb(fp); train_items.append((img, y))
    if MODE == "classical":                                   # rotated copies keep the SAME (rot-invariant) label
        rng = random.Random((row * 131 + col * 17 + rep * 7 + SEED) & 0xffffffff)
        for _ in range(N_PER):
            train_items.append((rot_flip(img, rng.uniform(0, 360), rng.random() < 0.5), y))
    elif MODE == "real_dup":                                  # matched-count control: exact copies (per-epoch hflip only)
        for _ in range(N_PER):
            train_items.append((img, y))
    elif MODE == "augmenter":                                 # each synth carries its OWN recomputed periodicity
        for s in sorted(glob.glob(f"{SYNTH}/R{row}C{col}_src{rep}_s*.png"))[:N_PER]:
            sy = labels.get(os.path.basename(s))
            if sy is not None:
                train_items.append((load_rgb(s), sy))

def eval_items(reps):
    out = []
    for (row, col, rep), fp in real_by.items():
        if rep in reps:
            y = labels.get(os.path.basename(fp))
            if y is not None:
                out.append((load_rgb(fp), y))
    return out
val_items, test_items = eval_items(VAL_REPS), eval_items(TEST_REPS)
print(f"[data] train {len(train_items)} (mode={MODE_TAG}) val {len(val_items)} test {len(test_items)}", flush=True)
if not train_items or not test_items:
    raise SystemExit("empty train/test -- check REALS_DIR / CSV name match")

TY = np.stack([y for _, y in train_items]); mu, sd = TY.mean(0), TY.std(0) + 1e-6
def z(y): return (y - mu) / sd

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORM])
EVAL_TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), NORM])

class DS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img, y = self.items[i]; return self.tf(img), torch.tensor(z(y), dtype=torch.float32)

tl = DataLoader(DS(train_items, TRAIN_TF), batch_size=32, shuffle=True, num_workers=NW, drop_last=len(train_items) > 32)
vl = DataLoader(DS(val_items, EVAL_TF), batch_size=64, num_workers=NW)
tel = DataLoader(DS(test_items, EVAL_TF), batch_size=64, num_workers=NW)

class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        if BACKBONE == "resnet50":
            ctor, weights, feat = resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048
        else:
            ctor, weights, feat = resnet18, ResNet18_Weights.IMAGENET1K_V1, 512
        try: self.b = ctor(weights=weights)
        except Exception as e: print("[warn] no pretrained weights:", e, flush=True); self.b = ctor(weights=None)
        self.b.fc = nn.Identity(); self.head = nn.Linear(feat, len(FEATS))
    def forward(self, x): return self.head(self.b(x))

net = Reg().to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
mse = nn.MSELoss()

@torch.no_grad()
def evaluate(loader):
    net.eval(); P, T = [], []
    for x, y in loader:
        P.append(net(x.to(DEV)).cpu().numpy()); T.append(y.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    ss_res = ((P - T) ** 2).sum(0); ss_tot = ((T - T.mean(0)) ** 2).sum(0) + 1e-9
    r2 = 1 - ss_res / ss_tot
    return r2, float(r2.mean())

best_val, best_test, best_vecs = -1e9, None, None
for ep in range(EPOCHS):
    net.train()
    for x, y in tl:
        opt.zero_grad(); mse(net(x.to(DEV)), y.to(DEV)).backward(); opt.step()
    sched.step()
    _, vr = evaluate(vl); tr2, tm = evaluate(tel)
    if vr > best_val:
        best_val, best_test, best_vecs = vr, tm, tr2
print(f"[done] backbone={BACKBONE} mode={MODE_TAG} K={K} seed={SEED} n={len(train_items)} | "
      f"TEST meanR2={best_test:.3f} per-feat=" + ",".join(f"{f}:{v:.2f}" for f, v in zip(FEATS, best_vecs)), flush=True)

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
new = not os.path.exists(RESULTS)
with open(RESULTS, "a", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    if new: w.writerow(["mode", "K", "seed", "n_train_imgs", "mean_r2", *[f"r2_{f}" for f in FEATS], "best_val_r2"])
    w.writerow([MODE_TAG, K, SEED, len(train_items), f"{best_test:.4f}", *[f"{v:.4f}" for v in best_vecs], f"{best_val:.4f}"])
