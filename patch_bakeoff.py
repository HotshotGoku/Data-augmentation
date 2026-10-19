"""Patch flux_gen_eval.py / conditional_ddpm.py / cmmd_direct.py to support SRC_PAT (glob pattern),
STEM naming (<source_stem>_s{i}.png so ds_downstream can match synth to source), and REALS_PAT.
Backward-compatible (defaults = old behavior). Idempotent-ish (warns if a target isn't found)."""
def patch(fp, subs):
    s = open(fp).read(); n = 0
    for a, b in subs:
        if a not in s:
            print(f"  WARN not found in {fp}: {a[:55]}"); continue
        s = s.replace(a, b); n += 1
    open(fp, "w").write(s); print(f"patched {fp} ({n} subs)")

patch("flux_gen_eval.py", [
    ('glob.glob(f"{SOURCES}/*.TIF")', 'glob.glob(f"{SOURCES}/" + os.environ.get("SRC_PAT", "*.TIF"))'),
    ('prefix = os.path.basename(fp).split("_")[0]',
     'prefix = (os.path.splitext(os.path.basename(fp))[0] + "_s") if os.environ.get("STEM") else (os.path.basename(fp).split("_")[0] + "_")'),
    ('img.save(f"{OUT}/{prefix}_{i+1}.png")', 'img.save(f"{OUT}/{prefix}{i+1}.png")'),
])
patch("conditional_ddpm.py", [
    ('glob.glob(f"{SRC}/*.TIF")', 'glob.glob(f"{SRC}/" + os.environ.get("SRC_PAT", "*.TIF"))'),
    ('cond = os.path.basename(fp).split("_")[0]',
     'cond = (os.path.splitext(os.path.basename(fp))[0] + "_s") if os.environ.get("STEM") else (os.path.basename(fp).split("_")[0] + "_")'),
    ('Image.fromarray(im).save(f"{OUT}/{cond}_{k+1}.png")', 'Image.fromarray(im).save(f"{OUT}/{cond}{k+1}.png")'),
])
patch("cmmd_direct.py", [
    ('reals = load(REALS, "*.TIF")', 'reals = load(REALS, os.environ.get("REALS_PAT", "*.TIF"))'),
])
