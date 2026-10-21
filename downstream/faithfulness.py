"""CHEAP DISTRIBUTIONAL DIAGNOSTIC of a generative augmenter's synth, from a colony-feature CSV.

NOT the primary faithfulness metric. The authoritative metric is probe-recovery on IMAGES
(faithfulness_probe.py, DCC): does a real-trained probe recover the synth's label? -- intended-label
recovery for decode tasks (mux/kl/rattray), manifold-consistency for the relabeled morphology task.

Why this file is only a diagnostic: the branching downstream harness RELABELS each synth with its own
colony_features value, so synth need not preserve the SOURCE's label -- "source-tracking" is not what
makes it win. (Validated 2026-10-20: base & dropout both WIN shape_factor downstream yet show ~0
source-tracking slope here.) The slope below is also attenuated by errors-in-variables (one noisy real
rep per source), so read it loosely.

ROBUST signals this file DOES give:
    bias        : mean(synth) - mean(real) across sources  -- distribution shift (e.g. base synth are
                  +0.71 on shape_factor = systematically rounder/"washed"; dropout ~+0.07 = on-dist).
    synth_std   : within-source synth spread                -- diversity / mode-collapse guard (0 = no new info).
    slope,r     : source-tracking (attenuated; informative only for decode-style fixed-label data).

Usage: python faithfulness.py features.csv [--feats shape_factor,fractal_dim,solidity,eccentricity,edge_std]
       (features.csv must hold BOTH reals '<cond>_Rep<n>.TIF' and synth '<cond>_Rep<n>_s<i>.png')
"""
import sys, csv, re, argparse
import numpy as np

FEATS = ["shape_factor", "fractal_dim", "solidity", "eccentricity", "edge_std"]
SYNTH_RE = re.compile(r"_s\d+\.(png|tif|TIF)$")


def source_stem(name):
    """ffc100_Rep1.TIF -> ffc100_Rep1 ;  ffc100_Rep1_s3.png -> ffc100_Rep1"""
    if SYNTH_RE.search(name):
        return SYNTH_RE.sub("", name)
    return re.sub(r"\.(png|tif|TIF)$", "", name)


def load(csv_path, feats):
    reals, synth = {}, {}          # source_stem -> feature vector / list of vectors
    for row in csv.DictReader(open(csv_path)):
        name = row["name"]
        vec = np.array([float(row[f]) for f in feats], float)
        stem = source_stem(name)
        if SYNTH_RE.search(name):
            synth.setdefault(stem, []).append(vec)
        else:
            reals[stem] = vec
    return reals, synth


def faithfulness(csv_path, feats):
    reals, synth = load(csv_path, feats)
    shared = sorted(set(reals) & set(synth))
    if len(shared) < 3:
        raise SystemExit(f"need >=3 sources with both real+synth, got {len(shared)}")
    real_mat = np.stack([reals[s] for s in shared])                       # [S, F]
    synth_mean = np.stack([np.mean(synth[s], 0) for s in shared])         # [S, F]
    synth_std = np.stack([np.std(synth[s], 0) for s in shared])           # [S, F]
    out = {}
    for j, f in enumerate(feats):
        x, y = real_mat[:, j], synth_mean[:, j]
        b, a = np.polyfit(x, y, 1)                                        # slope = label-tracking
        r = float(np.corrcoef(x, y)[0, 1])
        out[f] = dict(slope=float(b), pearson_r=r,
                      synth_std=float(synth_std[:, j].mean()),
                      bias=float((y - x).mean()))
    return out, len(shared)


def _selfcheck():
    rng = np.random.default_rng(0)
    r = np.linspace(-2, 0, 40)
    # faithful: synth = real + noise -> slope ~1 ; mean-regress: synth = const -> slope ~0
    xf = np.polyfit(r, np.array([ri + rng.normal(0, 0.02) for ri in r]), 1)[0]
    xm = np.polyfit(r, np.array([r.mean() + rng.normal(0, 0.02) for _ in r]), 1)[0]
    assert xf > 0.9, f"faithful slope {xf:.2f} should be ~1"
    assert abs(xm) < 0.1, f"mean-regress slope {xm:.2f} should be ~0"
    print(f"selfcheck OK (faithful slope={xf:.2f}, mean-regress slope={xm:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--feats", default=",".join(FEATS))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); sys.exit()
    feats = a.feats.split(",")
    out, n = faithfulness(a.csv, feats)
    print(f"{a.csv}  ({n} sources)")
    print(f"{'feature':<14}{'slope(b)':>10}{'pearson_r':>11}{'synth_std':>11}{'bias':>9}   distribution")
    for f, d in out.items():
        # robust read = bias vs spread: synth distribution SHIFTED off the reals if |bias| > its own spread
        shift = abs(d["bias"]) / (d["synth_std"] + 1e-9)
        tag = "SHIFTED" if shift > 1.0 else ("drift" if shift > 0.4 else "on-dist")
        print(f"{f:<14}{d['slope']:>10.3f}{d['pearson_r']:>11.3f}{d['synth_std']:>11.4f}{d['bias']:>9.3f}   {tag}")
