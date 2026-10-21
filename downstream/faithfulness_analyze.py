"""Aggregate faithfulness_probe.py results across seeds -> per-axis mean+/-std + a win/lose verdict.

The probe is applied ZERO-SHOT (real-trained, scored on synth), so absolute recovery is attenuated by
the real->synth domain gap; read it as a RANKING signal, not an absolute score. Texture/detail axes
(edge_std, eccentricity) carry a uniform domain-gap blowup unrelated to label faithfulness, so the
verdict keys on the PRIMARY SEMANTIC axis per task (the downstream headline): mux->atc, branching->
shape_factor. Higher primary-axis recovery = more faithful; the per-task ranking is the search objective.

Usage: python faithfulness_analyze.py faithfulness_out/results.tsv
       python faithfulness_analyze.py --selfcheck
"""
import sys, csv, argparse
from collections import defaultdict
import numpy as np

PRIMARY = {"mux": "atc", "branching": "shape_factor"}      # downstream headline axis per task
CONTRAST = {"mux": "iptg"}                                  # the known within-dataset lose axis


def load(tsv):
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    for r in rows:
        r["synth_recovery"] = float(r["synth_recovery"]); r["ceiling"] = float(r["ceiling"])
    return rows


def aggregate(rows):
    """(task, synth_tag, axis) -> dict(rec_mean, rec_std, ceil_mean, n, kind)."""
    g = defaultdict(list)
    for r in rows:
        g[(r["task"], r["synth_tag"], r["axis"])].append(r)
    agg = {}
    for k, rs in g.items():
        rec = np.array([r["synth_recovery"] for r in rs])
        agg[k] = dict(rec_mean=rec.mean(), rec_std=rec.std(), n=len(rs),
                      ceil_mean=np.mean([r["ceiling"] for r in rs]), kind=rs[0]["kind"])
    return agg


def report(rows):
    agg = aggregate(rows)
    tasks = sorted({t for t, _, _ in agg})
    for task in tasks:
        tags = sorted({tag for t, tag, _ in agg if t == task})
        prim = PRIMARY.get(task)
        print(f"\n=== {task} (primary axis = {prim}) ===")
        print(f"  {'synth_tag':<28}{'axis':<14}{'kind':>4}{'ceiling':>9}{'synth(mean±std)':>18}  flag")
        for tag in tags:
            axes = sorted([a for t, g_tag, a in agg if t == task and g_tag == tag])
            for a in axes:
                d = agg[(task, tag, a)]
                flag = "<- PRIMARY" if a == prim else ("<- lose-contrast" if a == CONTRAST.get(task) else "")
                print(f"  {tag:<28}{a:<14}{d['kind']:>4}{d['ceil_mean']:>9.3f}"
                      f"{d['rec_mean']:>11.3f}±{d['rec_std']:.3f}  {flag}")
        # verdict: rank synth_tags by primary-axis recovery (search objective)
        if prim:
            ranked = sorted([(tag, agg[(task, tag, prim)]["rec_mean"]) for tag in tags
                             if (task, tag, prim) in agg], key=lambda x: -x[1])
            print(f"  RANK by {prim} recovery (higher=more faithful): " +
                  " > ".join(f"{tag}({v:+.3f})" for tag, v in ranked))
            if task in CONTRAST:
                for tag in tags:
                    p = agg.get((task, tag, prim)); c = agg.get((task, tag, CONTRAST[task]))
                    if p and c:
                        print(f"  within-{tag}: {prim}({p['rec_mean']:+.3f}) vs {CONTRAST[task]}"
                              f"({c['rec_mean']:+.3f}) -> {'WIN>LOSE OK' if p['rec_mean'] > c['rec_mean'] else 'NOT separated'}")


def _selfcheck():
    rows = [  # dropout should outrank base on shape_factor; atc should beat iptg within mux
        dict(task="branching", synth_tag="dropout", axis="shape_factor", kind="reg", ceiling="0.45", synth_recovery="0.06"),
        dict(task="branching", synth_tag="dropout", axis="shape_factor", kind="reg", ceiling="0.33", synth_recovery="-0.05"),
        dict(task="branching", synth_tag="base", axis="shape_factor", kind="reg", ceiling="0.45", synth_recovery="-1.65"),
        dict(task="mux", synth_tag="ft", axis="atc", kind="reg", ceiling="0.82", synth_recovery="0.23"),
        dict(task="mux", synth_tag="ft", axis="iptg", kind="reg", ceiling="0.76", synth_recovery="-0.29"),
    ]
    for r in rows:
        r["synth_recovery"] = float(r["synth_recovery"]); r["ceiling"] = float(r["ceiling"])
    agg = aggregate(rows)
    assert abs(agg[("branching", "dropout", "shape_factor")]["rec_mean"] - 0.005) < 1e-6
    assert agg[("branching", "dropout", "shape_factor")]["rec_mean"] > agg[("branching", "base", "shape_factor")]["rec_mean"]
    assert agg[("mux", "ft", "atc")]["rec_mean"] > agg[("mux", "ft", "iptg")]["rec_mean"]
    print("selfcheck OK (dropout>base on shape_factor; atc>iptg within mux; mean aggregation correct)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="?")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); sys.exit()
    if not a.tsv:
        raise SystemExit("need results.tsv (or --selfcheck)")
    report(load(a.tsv))
