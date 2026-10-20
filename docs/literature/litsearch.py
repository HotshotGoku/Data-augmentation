"""Algorithmic backbone/literature search. Queries arXiv + OpenAlex (free, no key) across a wide,
deliberately cross-domain query set, dedupes, scores (relevance x recency x citations), and prints a
compact ranked shortlist. Breadth is done in code; the LLM only reads the distilled top results.
Full results -> litsearch_results.tsv."""
import urllib.request, urllib.parse, json, time, re
import xml.etree.ElementTree as ET
from collections import defaultdict

EMAIL = "sarthak@atp-data.com"
UA = {"User-Agent": f"backbone-litsearch/1.0 (mailto:{EMAIL})"}

# Deliberately wide + cross-domain. (category, query)
QUERIES = [
 # --- core task framings ---
 ("augmentation", "generative data augmentation scarce limited training data"),
 ("augmentation", "diffusion model data augmentation outperform classical augmentation"),
 ("augmentation", "when does synthetic data help classifier generative augmentation"),
 ("img2img", "image to image diffusion structure preserving translation"),
 ("img2img", "reference image conditioned generation IP-Adapter"),
 ("conditioning", "ControlNet spatial conditioning image generation"),
 ("conditioning", "in-context image generation editing Flux Kontext"),
 # --- low-data / few-shot generative ---
 ("low-data", "StyleGAN2 adaptive discriminator augmentation limited data"),
 ("low-data", "few-shot image generation diffusion small dataset"),
 ("low-data", "fine-tuning diffusion model few images overfitting diversity"),
 ("low-data", "GAN limited data regime medical imaging synthesis"),
 # --- diversity (our actual gap) ---
 ("diversity", "improving sample diversity diffusion models mode coverage"),
 ("diversity", "particle guidance diverse sampling diffusion"),
 ("diversity", "diversity generative augmentation downstream classification gain"),
 # --- microscopy / cell / bio generative ---
 ("microscopy", "microscopy image synthesis diffusion model generative"),
 ("microscopy", "cell painting morphology generative diffusion perturbation"),
 ("microscopy", "virtual staining fluorescence microscopy deep learning"),
 ("microscopy", "bacterial colony image generation deep learning"),
 # --- cross-domain analogs (creative) ---
 ("materials", "microstructure generation diffusion model materials science"),
 ("materials", "microstructure GAN synthesis three dimensional SliceGAN"),
 ("texture", "neural texture synthesis generative branching pattern"),
 ("remote-sensing", "remote sensing satellite image generative augmentation diffusion"),
 ("medical", "medical image synthesis limited data diffusion augmentation segmentation"),
 ("histopath", "histopathology image generation diffusion stain"),
 # --- physics / pattern-formation hybrids ---
 ("physics", "physics informed generative model pattern formation"),
 ("physics", "reaction diffusion pattern neural network generation"),
 ("physics", "simulation templated image synthesis sim-to-real biological"),
 # --- alternative architectures ---
 ("arch", "autoregressive image generation visual autoregressive VAR scale"),
 ("arch", "masked generative image transformer MaskGIT token"),
 ("arch", "flow matching image generation SD3 rectified flow"),
 ("arch", "consistency model one step image generation"),
]

def http_get(url, headers=UA, retries=3, timeout=30):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                print(f"  [warn] fail {url[:70]}... {type(e).__name__}")
            time.sleep(1.5 * (i + 1))
    return None

def reconstruct(inv):
    if not inv: return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs: pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))

def search_openalex(q, n=25):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "search": q, "per_page": n, "sort": "relevance_score:desc",
        "filter": "from_publication_date:2019-01-01,type:article", "mailto": EMAIL})
    raw = http_get(url); out = []
    if not raw: return out
    try:
        for w in json.loads(raw).get("results", []):
            loc = (w.get("primary_location") or {}).get("source") or {}
            out.append({"title": (w.get("title") or "").strip(), "year": w.get("publication_year"),
                "cites": w.get("cited_by_count", 0), "venue": loc.get("display_name") or "",
                "url": w.get("doi") or w.get("id") or "",
                "abstract": reconstruct(w.get("abstract_inverted_index"))[:700], "src": "oa"})
    except Exception as e: print("  [oa parse]", e)
    return out

def search_arxiv(q, n=20):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": f"all:{q}", "start": 0, "max_results": n, "sortBy": "relevance"})
    raw = http_get(url); out = []
    if not raw: return out
    try:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in ET.fromstring(raw).findall("a:entry", ns):
            t = (e.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
            s = (e.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
            y = (e.findtext("a:published", "", ns) or "")[:4]
            out.append({"title": t, "year": int(y) if y.isdigit() else None, "cites": None,
                "venue": "arXiv", "url": e.findtext("a:id", "", ns) or "", "abstract": s[:700], "src": "ax"})
    except Exception as e: print("  [ax parse]", e)
    return out

# --- relevance scoring ---
HIGH = {"augment": 3, "conditional": 2, "image-to-image": 3, "image to image": 3, "few-shot": 2,
 "few shot": 2, "low-data": 3, "limited data": 3, "small dataset": 2, "scarce": 3, "diversity": 3,
 "diverse": 1, "mode collapse": 2, "structure": 1, "microscopy": 2, "microstructure": 3, "texture": 2,
 "controlnet": 3, "ip-adapter": 3, "flow matching": 2, "stylegan": 2, "adaptive discriminator": 3,
 "synthesis": 1, "generative": 1, "replicate": 2, "morphology": 2, "reaction": 2, "pattern formation": 3,
 "sim-to-real": 2, "virtual staining": 2, "autoregressive": 1, "downstream": 2, "segmentation": 1}

def norm(t): return re.sub(r"[^a-z0-9]", "", (t or "").lower())

def score(p, cats):
    tl, al = (p["title"] or "").lower(), (p["abstract"] or "").lower()
    s = sum(w * (2 * tl.count(k) + al.count(k)) for k, w in HIGH.items())
    y = p.get("year") or 0
    s += {2025: 5, 2024: 4, 2023: 3, 2022: 1}.get(y, 0)
    c = p.get("cites")
    if c: s += min(8, (c ** 0.5) / 3)        # citation signal, capped
    s += 1.5 * (len(cats) - 1)               # cross-query hits = broadly relevant
    return s

def main():
    merged = {}   # norm-title -> record
    for ci, (cat, q) in enumerate(QUERIES, 1):
        print(f"[{ci}/{len(QUERIES)}] {cat}: {q[:55]}")
        hits = search_openalex(q) + search_arxiv(q)
        for p in hits:
            k = norm(p["title"])
            if len(k) < 8: continue
            if k not in merged:
                p["cats"] = {cat}; merged[k] = p
            else:
                m = merged[k]; m["cats"].add(cat)
                if (p.get("cites") or 0) > (m.get("cites") or 0): m["cites"] = p["cites"]
                if len(p.get("abstract") or "") > len(m.get("abstract") or ""): m["abstract"] = p["abstract"]
                if m["venue"] in ("", "arXiv") and p["venue"] not in ("", "arXiv"): m["venue"] = p["venue"]
        time.sleep(1.0)
    for p in merged.values():
        p["score"] = score(p, p["cats"])
    ranked = sorted(merged.values(), key=lambda p: -p["score"])
    with open("/private/tmp/claude/litsearch_results.tsv", "w") as f:
        f.write("score\tyear\tcites\tcats\tvenue\ttitle\turl\tabstract\n")
        for p in ranked:
            f.write(f"{p['score']:.1f}\t{p.get('year')}\t{p.get('cites')}\t{'|'.join(sorted(p['cats']))}\t"
                    f"{p['venue'][:30]}\t{p['title']}\t{p['url']}\t{(p['abstract'] or '')[:300]}\n")
    print(f"\n=== {len(ranked)} unique papers. TOP 55 ===\n")
    for i, p in enumerate(ranked[:55], 1):
        c = p.get("cites"); cs = f"{c}c" if c is not None else "–"
        print(f"{i:2}. [{p['score']:.0f}|{p.get('year')}|{cs}|{','.join(sorted(p['cats']))}] {p['title'][:95]}")
    print(f"\nFull table + abstracts -> litsearch_results.tsv  ({len(ranked)} rows)")

main()
