"""Fetch PixCell's canonical load recipe so we can fix the custom-pipeline import error.
Lists repo files, dumps the pipeline file(s) imports, and prints the ControlNet model-card README
(which should contain the intended inference snippet). Run in the pixcell env (huggingface_hub)."""
from huggingface_hub import hf_hub_download, list_repo_files

for repo in ["StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
             "StonyBrook-CVLab/PixCell-pipeline-ControlNet",
             "StonyBrook-CVLab/PixCell-pipeline"]:
    try:
        print(f"=== FILES {repo} ===\n  {list_repo_files(repo)}", flush=True)
    except Exception as e:
        print(f"list {repo} failed: {e}", flush=True)

for repo in ["StonyBrook-CVLab/PixCell-pipeline-ControlNet", "StonyBrook-CVLab/PixCell-pipeline"]:
    try:
        for pf in [f for f in list_repo_files(repo) if f.endswith(".py")]:
            p = hf_hub_download(repo, pf)
            print(f"=== IMPORTS {repo}/{pf} ===", flush=True)
            for line in open(p).read().splitlines():
                s = line.strip()
                if s.startswith("import ") or s.startswith("from "):
                    print("  ", s, flush=True)
    except Exception as e:
        print(f"{repo} py fetch failed: {e}", flush=True)

try:
    r = hf_hub_download("StonyBrook-CVLab/PixCell-256-Cell-ControlNet", "README.md")
    print("=== ControlNet model-card README (first 5000 chars) ===", flush=True)
    print(open(r).read()[:5000], flush=True)
except Exception as e:
    print(f"readme fetch failed: {e}", flush=True)
