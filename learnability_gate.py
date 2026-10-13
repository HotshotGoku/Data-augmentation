"""Phase-0 learnability gate: train a ResNet on REAL images only, predict a per-image label, report
test metric vs chance. If a dataset's label is NOT learnable from reals, its aug-vs-rotation downstream
test would be pure noise -> we drop it before spending ~40 GPU-hr on a generative bake-off.
This real-only run also BECOMES the 'real' arm of that dataset's later downstream comparison.

Env: IMAGES (glob, recursive ok), LABEL_RE (regex searched over the full path; needs group 'lab',
     optional group 'rep' for a held-out-replicate split), KIND [cls|reg], LOGRATIO [0|1] (reg: parse
     'A-B' -> log((A+1)/(B+1))), TEST_REP [''=random 20%], MAX_PER_LABEL [0=all], EPOCHS [25],
     BACKBONE [resnet18|resnet50], TAG."""
import os, glob, re, random
import numpy as np, torch, torch.nn as nn, cv2
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

IMG = os.environ["IMAGES"]; RE = re.compile(os.environ["LABEL_RE"])
KIND = os.environ.get("KIND", "cls"); LOGR = os.environ.get("LOGRATIO", "0") == "1"
TEST_REP = os.environ.get("TEST_REP", ""); MAXPL = int(os.environ.get("MAX_PER_LABEL", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "25")); BB = os.environ.get("BACKBONE", "resnet18")
TAG = os.environ.get("TAG", "gate"); dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0); torch.manual_seed(0)

def parse(s):
    if LOGR:
        a, b = s.split("-"); return float(np.log((float(a) + 1) / (float(b) + 1)))
    return s

rows = []
for fp in sorted(glob.glob(IMG, recursive=True)):
    m = RE.search(fp)
    if not m: continue
    gd = m.groupdict(); rows.append((fp, parse(gd["lab"]), gd.get("rep", "") or ""))
if MAXPL > 0:
    by = defaultdict(list)
    for r in rows: by[r[1]].append(r)
    rows = []
    for v in by.values():
        random.shuffle(v); rows += v[:MAXPL]
labs = [r[1] for r in rows]
print(f"[{TAG}] {len(rows)} imgs, {len(set(labs))} distinct labels", flush=True)
assert rows, "no images matched LABEL_RE"

if TEST_REP:
    test = [r for r in rows if r[2] == TEST_REP]; train = [r for r in rows if r[2] != TEST_REP]
else:
    random.shuffle(rows); n = max(1, len(rows) // 5); test, train = rows[:n], rows[n:]
print(f"[{TAG}] train {len(train)} / test {len(test)}", flush=True)

if KIND == "cls":
    classes = sorted(set(labs)); c2i = {c: i for i, c in enumerate(classes)}
    nout = len(classes); chance = 1.0 / nout; ymap = lambda l: c2i[l]
else:
    tr = np.array([r[1] for r in train], float); mu, sd = float(tr.mean()), float(tr.std() + 1e-8)
    nout = 1; ymap = lambda l: (l - mu) / sd

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRTF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(), transforms.RandomRotation(180), transforms.ToTensor(), NORM])
EVTF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
def load(fp): return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
class DS(Dataset):
    def __init__(s, r, tf): s.r, s.tf = r, tf
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        fp, lab, _ = s.r[i]; x = s.tf(load(fp)); y = ymap(lab)
        return x, (torch.tensor(y, dtype=torch.long) if KIND == "cls" else torch.tensor([y], dtype=torch.float))
tl = DataLoader(DS(train, TRTF), batch_size=32, shuffle=True, num_workers=4)
vl = DataLoader(DS(test, EVTF), batch_size=32, shuffle=False, num_workers=4)

ctor, w = (resnet50, ResNet50_Weights.IMAGENET1K_V2) if BB == "resnet50" else (resnet18, ResNet18_Weights.IMAGENET1K_V1)
net = ctor(weights=w); net.fc = nn.Linear(net.fc.in_features, nout); net = net.to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
lossf = nn.CrossEntropyLoss() if KIND == "cls" else nn.MSELoss()
for ep in range(EPOCHS):
    net.train()
    for x, y in tl:
        x, y = x.to(dev), y.to(dev); opt.zero_grad(); loss = lossf(net(x), y); loss.backward(); opt.step()
net.eval(); P, Y = [], []
with torch.no_grad():
    for x, y in vl:
        P.append(net(x.to(dev)).cpu()); Y.append(y)
P, Y = torch.cat(P), torch.cat(Y)
if KIND == "cls":
    acc = float((P.argmax(1) == Y).float().mean())
    print(f">>> GATE {TAG}: test_acc={acc:.3f} chance={chance:.3f} ratio={acc/chance:.1f}x  "
          f"(n_test={len(test)}, {nout} classes)  -> {'PASS' if acc > 3*chance else 'WEAK/FAIL'}", flush=True)
else:
    p, yv = P.squeeze(1).numpy(), Y.squeeze(1).numpy()
    r2 = 1 - ((p - yv) ** 2).sum() / (((yv - yv.mean()) ** 2).sum() + 1e-8)
    print(f">>> GATE {TAG}: test_R2={r2:.3f}  (n_test={len(test)})  -> {'PASS' if r2 > 0.2 else 'WEAK/FAIL'}", flush=True)
