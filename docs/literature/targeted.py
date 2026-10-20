"""Targeted confirmation round on specific named candidate architectures: canonical refs, citation
counts (maturity signal), recency. Reuses arXiv + OpenAlex. Prints top 2-3 per method, compact."""
import urllib.request, urllib.parse, json, time
import xml.etree.ElementTree as ET

EMAIL = "sarthak@atp-data.com"; UA = {"User-Agent": f"backbone-litsearch/1.0 (mailto:{EMAIL})"}
def get(url, t=30):
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t) as r: return r.read()
        except Exception: time.sleep(1.5*(i+1))
    return None
def oa(q, n=4):
    u = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": q, "per_page": n,
        "sort": "relevance_score:desc", "filter": "from_publication_date:2019-01-01", "mailto": EMAIL})
    r = get(u); out = []
    if r:
        try:
            for w in json.loads(r).get("results", []):
                out.append((w.get("title","").strip(), w.get("publication_year"), w.get("cited_by_count",0),
                            (w.get("doi") or w.get("id") or "")))
        except Exception: pass
    return out

METHODS = [
 "StyleGAN2-ADA training generative adversarial networks limited data Karras",
 "StyleGAN3 alias-free generative adversarial networks",
 "FastGAN lightweight GAN few-shot image synthesis stabilize",
 "Projected GANs converge faster",
 "DiffAugment differentiable augmentation data-efficient GAN training",
 "Stable Diffusion 3 scaling rectified flow transformers",
 "SANA efficient high-resolution linear diffusion transformer",
 "PixArt-alpha fast training diffusion transformer text-to-image",
 "MorphoDiff cellular morphology diffusion perturbation",
 "CellFlux simulating cellular microscopy flow matching",
 "neural cellular automata texture synthesis self-organising",
 "SDEdit guided image synthesis editing stochastic differential equations",
 "retrieval-augmented diffusion models text-to-image",
 "few-shot diffusion model one-shot generation domain adaptation",
 "consistency models few-step generation diversity",
 "elucidating design space diffusion EDM Karras",
]
for m in METHODS:
    print(f"\n### {m[:60]}")
    for t, y, c, u in oa(m)[:3]:
        print(f"  [{y}|{c}c] {t[:92]}  {u}")
    time.sleep(0.8)
