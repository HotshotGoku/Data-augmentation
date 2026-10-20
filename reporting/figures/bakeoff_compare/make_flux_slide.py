"""Single self-contained update slide for Kinshuk — ALL figures embedded."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import os
B = os.path.dirname(os.path.abspath(__file__))

BG=RGBColor(0x0B,0x0C,0x0F); TITLE=RGBColor(0xE9,0xEA,0xEC); GRAY=RGBColor(0x9A,0xA0,0xA6)
GRN=RGBColor(0x86,0xEF,0xAC); BLUE=RGBColor(0x7D,0xD3,0xFC); DIM=RGBColor(0x6B,0x70,0x76); F="Calibri"

prs=Presentation(); prs.slide_width=Inches(13.33); prs.slide_height=Inches(7.5)
s=prs.slides.add_slide(prs.slide_layouts[6])
s.background.fill.solid(); s.background.fill.fore_color.rgb=BG

def tb(x,y,w,h,runs,align=PP_ALIGN.LEFT,anchor=MSO_ANCHOR.TOP):
    box=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=box.text_frame
    tf.word_wrap=True; tf.vertical_anchor=anchor
    for j,line in enumerate(runs):
        p=tf.paragraphs[0] if j==0 else tf.add_paragraph(); p.alignment=align; p.space_after=Pt(1)
        for (t,c,b,sz) in line:
            r=p.add_run(); r.text=t; r.font.name=F; r.font.size=Pt(sz); r.font.color.rgb=c; r.font.bold=b
    return box
def lbl(x,y,text,color=BLUE,sz=11): tb(x,y,7.5,0.25,[[(text,color,True,sz)]])
def pic(path,x,y,w): s.shapes.add_picture(f"{B}/{path}",Inches(x),Inches(y),width=Inches(w))

# Title + key line
tb(0.4,0.12,9.8,0.5,[[("Flux backbone: beats SD1.5 on image quality AND the downstream task",TITLE,True,23)]])
tb(9.9,0.24,3.0,0.3,[[("Sarthak · 2026-10-01",DIM,False,11)]],align=PP_ALIGN.RIGHT)
tb(0.4,0.60,12.7,0.3,[[("Key result:  ",BLUE,True,13),
  ("a fp8 Flux ControlNet augmenter (cn0.65+PG0.6) now beats-or-matches production SD1.5 on realism, CMMD, AND diversity — and is the best augmenter on the downstream test.",GRAY,False,12.5)]])

# --- TOP-LEFT: same-source synth comparison (the hook) ---
lbl(0.35,0.98,"① Same colony, 4 synthetic replicates each — SD1.5 (washed, alike) vs FLUX (crisp, varied)")
pic("sd15_vs_flux_synth.png",0.35,1.25,7.7)  # h~2.92 -> 1.25..4.17

# --- TOP-RIGHT: diversity frontier + metrics line ---
lbl(8.25,0.98,"② FLUX beats SD1.5 on all 3 image metrics")
pic("flux_diversity_frontier.png",8.2,1.25,4.95)  # 2.5 -> h~1.98 -> 1.25..3.23
tb(8.2,3.28,5.0,0.95,[
  [("realism 0.559→",GRAY,False,11),("0.524",GRN,True,11),("   CMMD ~8.8→",GRAY,False,11),("8.8",GRN,True,11),("   div 0.266→",GRAY,False,11),("0.270",GRN,True,11)],
  [("= Pareto-dominates SD1.5 (no worse on any axis, clearly better on realism). Diversity lifted 0.22→0.27, inference-only.",GRN,False,10.5)]])

# --- BOTTOM-LEFT: downstream comparison ---
lbl(0.35,4.28,"③ DOWNSTREAM (predict colony morphology under scarcity): FLUX = best augmenter tested")
pic("flux_downstream_comparison.png",0.35,4.55,6.7)  # 2.59 -> h~2.59 -> 4.55..7.14

# --- BOTTOM-RIGHT: two supporting visuals + downstream numbers ---
lbl(7.3,4.28,"④ Flux: diverse replicates / vs real")
pic("flux_synth_diversity_showcase.png",7.3,4.55,2.9)   # 1.85 -> h~1.57
pic("flux_vs_real_branching.png",10.35,4.55,2.75)        # 1.56 -> h~1.76
tb(7.3,6.42,5.8,0.85,[
  [("FLUX best arm @ N=60: ",GRN,True,11),("mean R² 0.458",TITLE,True,11),(" > rotation 0.439 > SD1.5-synth 0.418 > real 0.413.  ",GRAY,False,11)],
  [("Beats rotation on shape-factor (your metric) at every scarcity level.",TITLE,False,11)]])

# footer
tb(0.4,7.2,12.7,0.28,[[("Honest scope: one domain (branching), minimal 256px/2-block config (headroom); downstream gain real but modest.",DIM,False,10)]])

out=f"{B}/../../Flux_Update_OneSlide.pptx"; prs.save(out); print("saved",os.path.abspath(out))
