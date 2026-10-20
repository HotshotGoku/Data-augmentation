"""Re-rank the 1141-paper harvest with an image-domain filter (local, no network). Penalize off-domain
(speech/text/tabular/etc.), require image-generative relevance, boost our-domain + named candidate
methods. Print cleaner top-per-category shortlist."""
import csv, re
from collections import defaultdict

rows = list(csv.DictReader(open("/private/tmp/claude/litsearch_results.tsv"), delimiter="\t"))

OFFDOMAIN = ["speech","audio","vocoder","acoustic","voice","phoneme","asr","nlp","language model",
 "llm","sentence","tabular","skeleton","action recognition","sonar","eeg","ecg","time series",
 "recommendation","molecul","protein","drug ","traffic","intrusion","malware","fault diagnosis",
 "bearing","weather classification"," text "]
IMAGE = ["image","images","img","microscop","microstructure","texture","visual","gan","diffusion",
 "stylegan","synthesis","generative adversarial","pixel","segmentation"," ct ","mri","ultrasound",
 "histopath","fluorescence","satellite","remote sensing","pathology","colony","cell","photo","render"]
DOMAIN = {"microscop":4,"microstructure":4,"colony":5,"bacterial":5,"biological":3,"fluorescence":3,
 "pattern formation":4,"reaction":3,"reaction-diffusion":4,"histopath":2,"medical image":2,
 "segmentation":1,"few-shot":3,"few shot":3,"limited data":3,"scarce":3,"small dataset":2,
 "stylegan":3,"-ada":3,"adaptive discriminator":4,"diffaugment":4,"fastgan":4,"projected gan":3,
 "controlnet":3,"ip-adapter":4,"kontext":4,"sana":3,"pixart":2,"morphodiff":5,"cellflux":5,
 "diversity":2,"mode collapse":2,"structure-preserving":3,"sim-to-real":3,"simulation":2,
 "flow matching":2,"rectified flow":2,"autoregressive":1,"consistency model":1,"retrieval":2}

def txt(r): return (r["title"] + " " + r["abstract"]).lower()

def keep(r):
    t = txt(r)
    if not any(k in t for k in IMAGE): return False
    off = sum(k in t for k in OFFDOMAIN)
    return off == 0                       # drop anything with an off-domain marker

def rescore(r):
    t = txt(r); base = float(r["score"])
    dom = sum(w * t.count(k) for k, w in DOMAIN.items())
    return base + dom

kept = [r for r in rows if keep(r)]
for r in kept: r["rs"] = rescore(r)
kept.sort(key=lambda r: -r["rs"])

print(f"filtered {len(rows)} -> {len(kept)} image-generative papers\n")
# group by primary category for breadth
bycat = defaultdict(list)
for r in kept:
    bycat[r["cats"].split("|")[0]].append(r)

print("=== TOP 8 PER CATEGORY (breadth view) ===")
for cat in sorted(bycat):
    print(f"\n### {cat} ({len(bycat[cat])})")
    for r in bycat[cat][:8]:
        c = r["cites"]; cs = f"{c}c" if c not in ("None","") else "–"
        print(f"  [{r['rs']:.0f}|{r['year']}|{cs}] {r['title'][:88]}")

print("\n\n=== GLOBAL TOP 40 (image-generative, re-ranked) ===")
for i, r in enumerate(kept[:40], 1):
    c = r["cites"]; cs = f"{c}c" if c not in ("None","") else "–"
    print(f"{i:2}. [{r['rs']:.0f}|{r['year']}|{cs}|{r['cats']}] {r['title'][:90]}")

# save cleaned
with open("/private/tmp/claude/litsearch_filtered.tsv", "w") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["rs","year","cites","cats","venue","title","url","abstract"])
    for r in kept:
        w.writerow([f"{r['rs']:.0f}",r["year"],r["cites"],r["cats"],r["venue"],r["title"],r["url"],r["abstract"][:260]])
print(f"\ncleaned -> litsearch_filtered.tsv ({len(kept)} rows)")
