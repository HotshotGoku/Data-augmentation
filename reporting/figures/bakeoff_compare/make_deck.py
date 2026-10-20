"""Multi-slide corrected-results deck for Kinshuk. Thesis: in-domain SD1.5 is the best realism
augmenter; Flux wins only diversity; downstream is a wash. Includes an explicit correction slide.
Dark theme matching the prior one-slide. Embeds charts (chart_*.png) + montages (model_results/...)."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import os

TMP = "/private/tmp/claude"
FIG = "/Users/sarthak/Downloads/Work/You Lab Research/model_results/bakeoff_compare"
BG=RGBColor(0x0B,0x0C,0x0F); TITLE=RGBColor(0xE9,0xEA,0xEC); GRAY=RGBColor(0x9A,0xA0,0xA6)
GRN=RGBColor(0x86,0xEF,0xAC); BLUE=RGBColor(0x7D,0xD3,0xFC); CORAL=RGBColor(0xFC,0xA5,0xA5)
AMBER=RGBColor(0xFB,0xBF,0x24); DIM=RGBColor(0x6B,0x70,0x76); F="Calibri"

prs=Presentation(); prs.slide_width=Inches(13.33); prs.slide_height=Inches(7.5)
def slide():
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb=BG; return s
def tb(s,x,y,w,h,runs,align=PP_ALIGN.LEFT,anchor=MSO_ANCHOR.TOP):
    box=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=box.text_frame
    tf.word_wrap=True; tf.vertical_anchor=anchor
    for j,line in enumerate(runs):
        p=tf.paragraphs[0] if j==0 else tf.add_paragraph(); p.alignment=align; p.space_after=Pt(3)
        for (t,c,b,sz) in line:
            r=p.add_run(); r.text=t; r.font.name=F; r.font.size=Pt(sz); r.font.color.rgb=c; r.font.bold=b
    return box
def pic(s,path,x,y,w=None,h=None):
    kw={}
    if w: kw["width"]=Inches(w)
    if h: kw["height"]=Inches(h)
    s.shapes.add_picture(path,Inches(x),Inches(y),**kw)
def header(s,txt,sub=None):
    tb(s,0.4,0.22,12.6,0.6,[[(txt,TITLE,True,26)]])
    if sub: tb(s,0.4,0.82,12.6,0.4,[[(sub,GRAY,False,13)]])

# ---- Slide 1: title / TL;DR ----
s=slide()
tb(s,0.5,1.5,12.3,1.0,[[("Branching augmenter: corrected backbone comparison",TITLE,True,34)]])
tb(s,0.5,2.6,12.3,0.5,[[("In-domain SD1.5 is the best realism augmenter — Flux does not beat it even fully scaled",GRN,True,18)]])
tb(s,0.5,3.5,12.3,2.4,[
  [("Three findings from a methodology pass (prompted by your feedback):",TITLE,True,15)],
  [("1.  Image realism/CMMD — ",BLUE,True,14),("a properly in-domain SD1.5 wins decisively (CMMD 3.4 vs Flux 4.8–8.8), and holds even against Flux's fully-escalated best (bf16, 3× ControlNet capacity).",GRAY,False,14)],
  [("2.  Diversity — ",BLUE,True,14),("Flux wins, modestly (0.27 vs 0.25), partly by generating less-coherent structure.",GRAY,False,14)],
  [("3.  Downstream utility (15 seeds) — ",BLUE,True,14),("rotation ≈ SD1.5 ≈ Flux; no augmenter significantly beats plain rotation.",GRAY,False,14)]])
tb(s,0.5,6.6,12.3,0.5,[[("Sarthak · 2026-10-07 · branching domain, 410-pair split, 28 held-out test conditions",DIM,False,11)]])

# ---- Slide 2: what changed (correction) ----
s=slide(); header(s,"What changed since the last update",
  "The earlier “Flux beats SD1.5” result did not survive two fixes you flagged")
tb(s,0.5,1.5,6.1,4.6,[
  [("Prior claim (Sept):",CORAL,True,16)],
  [("“Flux augmenter Pareto-dominates SD1.5 on realism, CMMD, and diversity, and is the best downstream augmenter.”",GRAY,False,13.5)],
  [("",GRAY,False,6)],
  [("Two problems:",CORAL,True,15)],
  [("•  Baseline was out-of-domain. ",TITLE,True,13.5),("Flux was compared to a generalist SD1.5, not one trained on branching. The branching-specialized SD1.5 is far stronger.",GRAY,False,13.5)],
  [("•  Downstream was 5-seed noise. ",TITLE,True,13.5),("The “best augmenter” gap (0.019) was ~5–8× smaller than the swing from which conditions get sampled.",GRAY,False,13.5)]])
tb(s,7.0,1.5,5.8,4.6,[
  [("Corrected here:",GRN,True,16)],
  [("•  Same-split comparison. ",TITLE,True,13.5),("SD1.5 retrained from scratch on the identical 410-pair branching split as Flux; scored identically.",GRAY,False,13.5)],
  [("•  15-seed downstream. ",TITLE,True,13.5),("Paired across all arms; error bars reported.",GRAY,False,13.5)],
  [("•  Flux given its best shot. ",TITLE,True,13.5),("Escalated fp8→bf16 base, CN2→CN6 capacity, full 20k-step training — to rule out “Flux was just under-trained.”",GRAY,False,13.5)],
  [("",GRAY,False,8)],
  [("Net: the headline reverses. Your instinct that SD1.5 “usually did well on branching” was right.",GRN,True,14)]])

# ---- Slide 3: methodology / fair comparison ----
s=slide(); header(s,"The fair comparison","Both backbones see exactly the same data and the same evaluation")
tb(s,0.5,1.4,12.3,2.0,[
  [("Train:  ",BLUE,True,14),("identical 410 replicate-pairs (branching), test conditions excluded from both.  ",GRAY,False,14),
   ("Eval:  ",BLUE,True,14),("4 samples/source on 28 held-out conditions, scored vs real replicates.",GRAY,False,14)],
  [("Metrics:  ",BLUE,True,14),("realism = LPIPS(gen→real-sibling) ÷ real-vs-real floor (lower=better) · CMMD = CLIP distribution distance (lower=better) · diversity = mean pairwise LPIPS among a model's own samples (higher=better).",GRAY,False,13)]])
tb(s,0.5,3.35,12.3,0.4,[[("Addresses: “it depends how you split train/test” — now it doesn't; the split is shared and fixed.",GRN,True,13.5)]])
pic(s,f"{FIG}/sd15_vs_flux_verify_wide.png",3.20,3.80,w=6.94)   # 2.20:1 -> h~3.15, y 3.80..6.95
tb(s,0.5,7.02,12.3,0.4,[[("Sample conditions by a fixed rule (evenly spaced across sorted test set), not hand-picked — 3 of 28 shown.",DIM,False,11)]])

# ---- Slide 4: image-metric result + ladder ----
s=slide(); header(s,"Image quality: SD1.5 wins, and the win is robust to scaling Flux")
pic(s,f"{TMP}/chart_ladder.png",0.6,1.3,w=8.2)
tb(s,9.0,1.5,4.0,5.0,[
  [("The escalation ladder:",TITLE,True,15)],
  [("Each Flux lever (capacity, bf16 base, training) lowers CMMD —",GRAY,False,13)],
  [("8.78 → 5.53 → 4.84",AMBER,True,15)],
  [("— approaching but never reaching SD1.5's ",GRAY,False,13),("3.38",GRN,True,13),(".",GRAY,False,13)],
  [("",GRAY,False,8)],
  [("But realism plateaus at ~0.50, far above SD1.5's ",GRAY,False,13),("0.324",GRN,True,13),(" (near the real-vs-real floor).",GRAY,False,13)],
  [("",GRAY,False,8)],
  [("So it's not under-trained Flux — the gap is structural.",CORAL,True,13.5)]])
tb(s,0.6,6.7,12.3,0.5,[[("Flux bf16/CN6: 48GB GPU, full bf16 base (no quantization loss), 1.4B-param ControlNet, 20k steps.",DIM,False,11)]])

# ---- Slide 5: the money figure (progression) ----
s=slide(); header(s,"Why — the structure tells the story",
  "Left→right: Flux improves with every lever, but SD1.5 keeps the fullest branches")
pic(s,f"{FIG}/flux_escalation_wide.png",1.71,1.35,w=9.90)   # 1.89:1 -> h~5.25, y 1.35..6.60
tb(s,0.5,6.68,12.3,0.6,[
  [("SD1.5 ",GRN,True,12.5),("= most complete dendritic structure (slightly over-bright).  ",GRAY,False,12.5),
   ("Flux pilot→scaled→bf16 ",BLUE,True,12.5),("= steadily crisper and color closer to real, but branches stay fainter/less complete — the realism deficit.",GRAY,False,12.5)]])

# ---- Slide 6: downstream ----
s=slide(); header(s,"Downstream utility: no augmenter beats plain rotation",
  "Predicting colony morphology under scarcity — 15 seeds, paired, ±SEM")
pic(s,f"{TMP}/chart_downstream.png",0.6,1.4,w=7.9)
tb(s,8.8,1.5,4.2,5.2,[
  [("Every augmenter helps over real-only,",TITLE,True,14)],
  [("but rotation is as good as either generative model — all within error bars.",GRAY,False,13.5)],
  [("",GRAY,False,8)],
  [("Answers: “does it depend which conditions you pick?”",BLUE,True,13.5)],
  [("Yes — strongly. ",CORAL,True,13.5),("A single arm's R² swings ~0.10–0.17 across condition draws, while the best augmenter-vs-rotation gap is ~0.02. Condition choice dominates; that's why 5 seeds misled.",GRAY,False,13.5)],
  [("",GRAY,False,8)],
  [("The earlier “Flux = best augmenter @0.458” regressed to 0.431 at 15 seeds; rotation held at 0.443.",GRAY,False,12.5)]])

# ---- Slide 7: conclusion / next ----
s=slide(); header(s,"Conclusion & next steps")
tb(s,0.5,1.4,12.3,3.6,[
  [("Bottom line:",GRN,True,17)],
  [("•  For branching, a ",GRAY,False,14.5),("branching-specialized SD1.5 is the better augmenter",TITLE,True,14.5),(" — most realistic, and rotation already covers the downstream gain.",GRAY,False,14.5)],
  [("•  Flux's advantage is ",GRAY,False,14.5),("diversity only",TITLE,True,14.5),(", and it costs realism; scaling narrows but does not close the gap.",GRAY,False,14.5)],
  [("•  The contribution is the ",GRAY,False,14.5),("evaluation that caught the artifact",TITLE,True,14.5),(": in-domain baselines, condition-subset sensitivity, a capacity/precision escalation to rule out under-training.",GRAY,False,14.5)]])
tb(s,0.5,5.1,12.3,2.0,[
  [("Open / proposed:",BLUE,True,16)],
  [("•  Your bio-oriented metrics ",TITLE,True,14),("as the downstream tiebreaker — colony-feature R² may miss biology that a domain metric would catch.",GRAY,False,14)],
  [("•  Validate the multiplexed (cell-like) domain, ",TITLE,True,14),("where the backbone verdict flipped before (PixCell CMMD 5.8 < SD1.5 12.6).",GRAY,False,14)],
  [("•  If diversity is the goal, ",TITLE,True,14),("training-time conditioning-dropout on SD1.5 may buy it without Flux's realism cost.",GRAY,False,14)]])

out="/Users/sarthak/Downloads/Work/You Lab Research/Branching_Backbone_Corrected_Deck.pptx"
prs.save(out); print("saved", out, "·", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
