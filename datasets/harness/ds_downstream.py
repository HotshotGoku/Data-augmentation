"""Generic downstream utility test for DECODE/CLASSIFICATION datasets (rattray strain, kl ratio, mux decode).
Compares real / rotation / augmenter(synth) at matched image counts, paired over seeds, held-out real test.
One MODE per invocation; appends to a shared TSV so modes are directly paired by seed.

Env: REALS (glob), LABEL_RE (groups 'lab' + 'rep'), KIND [cls|reg], LOGRATIO [0|1],
     TEST_REP (held-out real test rep), K [1], N_PER [8], SEEDS [5], EPOCHS [30], BACKBONE [resnet18],
     MODE [real|rotation|augmenter], SYNTH (dir; augmenter globs '<realstem>_s*.png'), TAG, OUT (tsv)."""
import os, glob, re, random, csv
import numpy as np, torch, torch.nn as nn, cv2
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

E = os.environ
REALS = E["REALS"]; LRE = re.compile(E["LABEL_RE"]); KIND = E.get("KIND", "cls")
LOGR = E.get("LOGRATIO", "0") == "1"; TEST_REP = E["TEST_REP"]; K = int(E.get("K", "1"))
NPER = int(E.get("N_PER", "8")); SEEDS = int(E.get("SEEDS", "5")); EPOCHS = int(E.get("EPOCHS", "30"))
BB = E.get("BACKBONE", "resnet18"); MODE = E.get("MODE", "real"); SYNTH = E.get("SYNTH", "")
TAG = E.get("TAG", "ds"); OUT = E.get("OUT", "ds_results.tsv")
dev = "cuda" if torch.cuda.is_available() else "cpu"

def parse(s):
    if LOGR: a, b = s.split("-"); return float(np.log((float(a) + 1) / (float(b) + 1)))
    return s

def rec(fp):
    m = LRE.search(fp)
    if not m: return None
    g = m.groupdict()
    return dict(fp=fp, lab=parse(g["lab"]), rep=g.get("rep", "") or "", stem=os.path.splitext(os.path.basename(fp))[0])

reals = [r for r in (rec(f) for f in sorted(glob.glob(REALS))) if r]
test = [r for r in reals if r["rep"] == TEST_REP]
pool = [r for r in reals if r["rep"] != TEST_REP]
by = defaultdict(list)
for r in pool: by[r["lab"]].append(r)
labs = sorted(set(r["lab"] for r in reals))
if KIND == "cls":
    c2i = {c: i for i, c in enumerate(labs)}; nout = len(labs); ymap = lambda l: c2i[l]
else:
    tr = np.array([r["lab"] for r in pool], float); mu, sd = float(tr.mean()), float(tr.std() + 1e-8)
    nout = 1; ymap = lambda l: (l - mu) / sd
print(f"[{TAG}] {len(reals)} reals, {len(labs)} labels, {len(test)} test, pool {len(pool)}; MODE={MODE} K={K}", flush=True)

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]); RS = transforms.Resize((224, 224))
def load(fp): return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def rot_copy(im):
    im = np.rot90(im, random.choice([1, 2, 3])).copy()
    if random.random() < 0.5: im = im[:, ::-1].copy()
    return im
class DS(Dataset):
    def __init__(s, items, train): s.items, s.train = items, train
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        im, lab = s.items[i]; pil = RS(transforms.ToPILImage()(im))
        if s.train and random.random() < 0.5: pil = transforms.functional.hflip(pil)
        x = NORM(transforms.ToTensor()(pil)); y = ymap(lab)
        return x, (torch.tensor(y, dtype=torch.long) if KIND == "cls" else torch.tensor([y], dtype=torch.float))

def build_train(seed):
    random.seed(seed); items = []
    for lab, rs in by.items():
        for r in random.sample(rs, min(K, len(rs))):
            items.append((load(r["fp"]), lab))
            if MODE == "rotation":
                for _ in range(NPER): items.append((rot_copy(load(r["fp"])), lab))
            elif MODE == "augmenter":
                for sf in sorted(glob.glob(f"{SYNTH}/{r['stem']}_s*.png"))[:NPER]:
                    items.append((load(sf), lab))
    return items
test_items = [(load(r["fp"]), r["lab"]) for r in test]

def run_seed(seed):
    tl = DataLoader(DS(build_train(seed), True), batch_size=32, shuffle=True, num_workers=4)
    vl = DataLoader(DS(test_items, False), batch_size=32, shuffle=False, num_workers=4)
    ctor, w = (resnet50, ResNet50_Weights.IMAGENET1K_V2) if BB == "resnet50" else (resnet18, ResNet18_Weights.IMAGENET1K_V1)
    net = ctor(weights=w); net.fc = nn.Linear(net.fc.in_features, nout); net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4); lf = nn.CrossEntropyLoss() if KIND == "cls" else nn.MSELoss()
    for _ in range(EPOCHS):
        net.train()
        for x, y in tl: x, y = x.to(dev), y.to(dev); opt.zero_grad(); lf(net(x), y).backward(); opt.step()
    net.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in vl: P.append(net(x.to(dev)).cpu()); Y.append(y)
    P, Y = torch.cat(P), torch.cat(Y)
    if KIND == "cls": return float((P.argmax(1) == Y).float().mean())
    p, yv = P.squeeze(1).numpy(), Y.squeeze(1).numpy()
    return float(1 - ((p - yv) ** 2).sum() / (((yv - yv.mean()) ** 2).sum() + 1e-8))

rows = []
for s in range(SEEDS):
    m = run_seed(s); rows.append((TAG, MODE, K, s, m))
    print(f">>> DS {TAG} mode={MODE} K={K} seed={s}: metric={m:.4f}", flush=True)
newf = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    wr = csv.writer(f, delimiter="\t")
    if newf: wr.writerow(["tag", "mode", "K", "seed", "metric"])
    wr.writerows(rows)
print(f">>> DS {TAG} mode={MODE} K={K}: MEAN={np.mean([r[4] for r in rows]):.4f} over {SEEDS} seeds", flush=True)
