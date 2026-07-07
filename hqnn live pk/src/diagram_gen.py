#!/usr/bin/env python3
"""Combined 4-lane pipeline SVG — correct orthogonal arrow routing per the sketch."""
COLW=200; ROWH=96; BW=152; BH=54; X0=250; Y0=46
def cx(c): return X0+c*COLW
def cy(r): return Y0+r*ROWH
N={}
def node(id,col,row,label,kind="box"):
    N[id]=dict(id=id,col=col,row=row,label=label,kind=kind,x=cx(col),y=cy(row))
TR=1.7; FR=6.1; DR=8.3
node("start",2,0,"Start","startend")
node("download",3,0,"Download NASA Data")
node("features",4,0,"Engineer 83 Features")
node("split",5,0,"Split Train / Forward")
node("tune_start",2,TR,"Start Tuning")
node("t_tgt",3,TR,"For Each Target \u00d73")
node("t_mdl",4,TR,"For Each Model \u00d76")
node("optuna",5,TR,"Create Optuna Study")
node("sample",6,TR,"Sample Hyper-\nParameters")
node("cv",6,TR+1,"For each CV \u00d73")
node("fit",5,TR+1,"Fit Scalar (Fold)")
node("buildseq",4,TR+1,"Build Sequence")
node("quicktrain",3,TR+1,"Quick-Train Model")
node("valmse",2,TR+1,"Validate \u2192 MSE")
node("report",2,TR+2,"Report to Optuna")
node("prune",3,TR+2,"Prune Trial?","decision")
node("morefolds",4,TR+2,"More Folds?","decision")
node("avgmse",5,TR+2,"Average Fold MSE")
node("moretrials",6,TR+2,"More Trials?","decision")
node("selbest",6,TR+3,"Select Best\nParameters")
node("moremodels",5,TR+3,"More Models?","decision")
node("moretgts",4,TR+3,"More Targets?","decision")
node("writehp",3,TR+3,"Write best_hParams")
node("f_tgt",3,FR,"For Each Target \u00d73")
node("f_mdl",4,FR,"For Each Model \u00d76")
node("epoch",4,FR+1,"Train Epoch")
node("early",3,FR+1,"Early Stop?","decision")
node("savew",2,FR+1,"Save Best Weights")
node("meta",1,FR+1,"Write Metadata")
node("beginday",1,DR,"Begin Forward Day")
node("forecast",2,DR,"Forecast next 15 days")
node("verify",3,DR,"Verify vs Source")
node("agree",4,DR,"Source Agrees?","decision")
node("seventh",5,DR,"7th Verified Day /\n1st of Month","decision")
node("finetune",6,DR,"Fine-Tune Weights")
node("skipupd",4,DR+1,"Skip Update")
node("retrain",5,DR+1,"Full Retrain \u00d718")
node("savew2",6,DR+1,"Save Updated Weights")
node("logm",6,DR+2,"Log + Metrics")
node("moredays",4,DR+2,"More Days?","decision")
node("skip",1,DR+2,"Stop","startend")
LANES=[("Data Engineering Pipeline",0),("Hyperparameter Tuning Pipeline",TR),("Full Training Pipeline",FR),("Daily Forecast Pipeline",DR)]
def hw(n): return BW/2
def hh(n): return BH/2+(7 if n["kind"]=="decision" else 0)
def A(id,side):
    n=N[id]; x,y=n["x"],n["y"]
    return {"L":(x-hw(n),y),"R":(x+hw(n),y),"T":(x,y-hh(n)),"B":(x,y+hh(n))}[side]
EDGES=[]
def E(a,sa,b,sb,via=None,label="",lpos=None,sb_pt=None):
    EDGES.append(dict(a=a,sa=sa,b=b,sb=sb,via=via or [],label=label,lpos=lpos,sb_pt=sb_pt))
E("start","R","download","L"); E("download","R","features","L"); E("features","R","split","L")
sp=A("split","B"); ts=A("tune_start","T"); midy=(sp[1]+ts[1])/2
E("split","B","tune_start","T",via=[(sp[0],midy),(ts[0],midy)])
E("tune_start","R","t_tgt","L"); E("t_tgt","R","t_mdl","L"); E("t_mdl","R","optuna","L"); E("optuna","R","sample","L")
E("sample","B","cv","T")
E("cv","L","fit","R"); E("fit","L","buildseq","R"); E("buildseq","L","quicktrain","R"); E("quicktrain","L","valmse","R")
E("valmse","B","report","T")
E("report","R","prune","L")
E("prune","R","morefolds","L",label="No"); E("morefolds","R","avgmse","L",label="No"); E("avgmse","R","moretrials","L")
sr=A("sample","R")   # Sample right edge = cx(6)+76
# Loop-back channels MUST run entirely to the RIGHT of every column-6 box
# (right edge at cx(6)+76). Space them 30px apart so no two ever touch, and give
# each its own entry height on Sample's right edge. End each arrow a few px
# OUTSIDE the box edge so the arrowhead sits clearly outside, not inside the box.
EDGE = cx(6)+76
GAP = 4               # stop arrows this far outside the box border
ch_trials = EDGE+34
ch_models = EDGE+66
ch_tgts   = EDGE+98
ch_prune  = EDGE+130
e_trials = (sr[0]+GAP, sr[1]-20)
e_models = (sr[0]+GAP, sr[1]-7)
e_tgts   = (sr[0]+GAP, sr[1]+7)
e_prune  = (sr[0]+GAP, sr[1]+20)
# More Trials? Yes (innermost): right out of box, up its channel, left into Sample
mt=A("moretrials","R")
E("moretrials","R","sample","R",via=[(ch_trials,mt[1]),(ch_trials,e_trials[1])],sb_pt=e_trials,
  label="Yes",lpos=(ch_trials+14,mt[1]-10))
E("moretrials","B","selbest","T",label="No")
# row D
E("selbest","L","moremodels","R"); E("moremodels","L","moretgts","R",label="No"); E("moretgts","L","writehp","R",label="No")
# More Models? Yes: down below row D, right past boxes, up channel, into Sample
mm=A("moremodels","B"); belowD=cy(TR+3)+hh(N["moremodels"])+22
E("moremodels","B","sample","R",via=[(mm[0],belowD),(ch_models,belowD),(ch_models,e_models[1])],sb_pt=e_models,
  label="Yes",lpos=(mm[0]+20,belowD+12))
# More Targets? Yes: down below row D (lower line), right, up channel, into Sample
mtg=A("moretgts","B"); belowD2=cy(TR+3)+hh(N["moretgts"])+40
E("moretgts","B","sample","R",via=[(mtg[0],belowD2),(ch_tgts,belowD2),(ch_tgts,e_tgts[1])],sb_pt=e_tgts,
  label="Yes",lpos=(mtg[0]+20,belowD2+12))
# Prune Trial? Yes (outermost): down to a low line under the decision row, far right,
# up the outermost channel, into Sample.
pr=A("prune","B"); lowliney=cy(TR+2)+hh(N["prune"])+16
E("prune","B","sample","R",
  via=[(pr[0],lowliney),(ch_prune,lowliney),(ch_prune,e_prune[1])],sb_pt=e_prune,
  label="Yes",lpos=(pr[0]+26,lowliney-8))
# More Folds? Yes -> For each CV x3: straight up from More Folds top into the channel,
# right, then up into CV's bottom (offset left of CV center). End just below CV edge.
mf=A("morefolds","T"); cvb=A("cv","B"); cvy=cy(TR+1)+hh(N["cv"])+18; cv_in=(cvb[0]-28,cvb[1]+GAP)
E("morefolds","T","cv","B",via=[(mf[0],cvy),(cv_in[0],cvy)],sb_pt=cv_in,
  label="Yes",lpos=(cv_in[0]+34,cvy-8))
# Write best_hParams -> For Each Target x3: straight vertical down (same column)
E("writehp","B","f_tgt","T")
# forward flow
E("f_tgt","R","f_mdl","L")
E("f_mdl","B","epoch","T")
E("epoch","L","early","R")
E("early","L","savew","R",label="done")
E("savew","L","meta","R")
# loop 1: Early Stop? No -> Train Epoch (under, "no - next epoch")
eb=A("early","B"); epb=A("epoch","B"); lowy=cy(FR+1)+hh(N["epoch"])+24
E("early","B","epoch","B",via=[(eb[0],lowy),(epb[0],lowy)],label="no - next epoch",lpos=((eb[0]+epb[0])/2,lowy+12))
# loop 2: Train Epoch -> For Each Model x6 (right channel, "next model x6")
epr=A("epoch","R"); fmr=A("f_mdl","R"); farx3=cx(4)+92
E("epoch","R","f_mdl","R",via=[(farx3,epr[1]),(farx3,fmr[1])],label="next model \u00d76",lpos=(farx3+10,(epr[1]+fmr[1])/2))
# loop 3: For Each Model x6 -> For Each Target x3 ("next target x3"): up from Model
# top, left along a channel, down into Target x3's TOP, ending just above the edge
# (offset right of the writehp arrow) so the arrowhead clearly points into the box.
fmt=A("f_mdl","T"); ftt=A("f_tgt","T")
ft_in=(ftt[0]+34, ftt[1]-4)          # 4px above the top edge -> arrowhead points in
chy=cy(FR)-hh(N["f_mdl"])-22
E("f_mdl","T","f_tgt","T",via=[(fmt[0],chy),(ft_in[0],chy)],sb_pt=ft_in,
  label="next target \u00d73",lpos=((fmt[0]+ft_in[0])/2,chy-7))
mb=A("meta","B"); bd=A("beginday","T")
E("meta","B","beginday","T",via=[(mb[0],(mb[1]+bd[1])/2),(bd[0],(mb[1]+bd[1])/2)],label="enter daily",lpos=(mb[0]+46,(mb[1]+bd[1])/2-6))
E("beginday","R","forecast","L"); E("forecast","R","verify","L"); E("verify","R","agree","L")
E("agree","R","seventh","L",label="Yes"); E("seventh","R","finetune","L",label="No")
E("agree","B","skipupd","T",label="No"); E("seventh","B","retrain","T",label="Yes")
E("retrain","R","savew2","L",label="Retrained"); E("finetune","B","savew2","T")
E("savew2","B","logm","T"); E("logm","L","moredays","R"); E("moredays","L","skip","R",label="No")
su=A("skipupd","B"); ll=A("logm","L")
# drop into the channel between row DR+1 and DR+2 (above More Days), run right,
# into Log+Metrics' LEFT side — clear of More Days and all boxes.
chy=(cy(DR+1)+hh(N["skipupd"])+cy(DR+2)-hh(N["logm"]))/2
E("skipupd","B","logm","L",via=[(su[0],chy),(ll[0]-30,chy),(ll[0]-30,ll[1])],
  label="Skip log",lpos=(su[0]+70,chy-8))
mdl=A("moredays","L"); bdl=A("beginday","B")
# channel sits below the Begin Forward Day row; run from More Days' LEFT side along it,
# then up into Begin Forward Day's bottom (avoids passing through Skip Update).
chy_day=cy(DR)+hh(N["beginday"])+34
E("moredays","L","beginday","B",
  via=[(mdl[0]-30,mdl[1]),(mdl[0]-30,chy_day),(bdl[0],chy_day)],
  label="yes \u00b7 next day \u21bb",lpos=(bdl[0]+96,chy_day-9))
def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
W=cx(6)+270; H=cy(DR+2)+96+44
LANE_OF={
  # data engineering
  "start":"data","download":"data","features":"data","split":"data",
  # tuning
  "tune_start":"tune","t_tgt":"tune","t_mdl":"tune","optuna":"tune","sample":"tune",
  "cv":"tune","fit":"tune","buildseq":"tune","quicktrain":"tune","valmse":"tune",
  "report":"tune","prune":"tune","morefolds":"tune","avgmse":"tune","moretrials":"tune",
  "selbest":"tune","moremodels":"tune","moretgts":"tune","writehp":"tune",
  # full training
  "f_tgt":"train","f_mdl":"train","epoch":"train","early":"train","savew":"train","meta":"train",
  # daily forecast
  "beginday":"daily","forecast":"daily","verify":"daily","agree":"daily","seventh":"daily",
  "finetune":"daily","skipupd":"daily","retrain":"daily","savew2":"daily","logm":"daily",
  "moredays":"daily","skip":"daily",
}
def draw_node(n):
    x,y=n["x"],n["y"]; lines=n["label"].split("\n")
    lane=LANE_OF.get(n["id"],"data")
    if n["kind"]=="decision":
        w,h=BW/2,BH/2+7; shape=f'<polygon points="{x},{y-h} {x+w},{y} {x},{y+h} {x-w},{y}" class="dia lane-{lane}"/>'
    elif n["kind"]=="startend":
        shape=f'<rect x="{x-BW/2}" y="{y-BH/2}" width="{BW}" height="{BH}" rx="25" class="se lane-{lane}"/>'
    else:
        shape=f'<rect x="{x-BW/2}" y="{y-BH/2}" width="{BW}" height="{BH}" rx="9" class="bx lane-{lane}"/>'
    ty=y-(len(lines)-1)*8
    txt="".join(f'<tspan x="{x}" y="{ty+i*15}">{esc(l)}</tspan>' for i,l in enumerate(lines))
    # id="nd_<id>" + class "fn" so the dashboard animator can glow it
    return f'<g class="fn node" id="nd_{n["id"]}" data-id="{n["id"]}">{shape}<text class="lbl">{txt}</text></g>'
def poly(e):
    pa=A(e["a"],e["sa"])
    pb=e["sb_pt"] if e.get("sb_pt") else A(e["b"],e["sb"])
    pts=[pa]+e["via"]+[pb]
    return "M "+" L ".join(f"{p[0]} {p[1]}" for p in pts)
svg=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" class="pipe">']
svg.append('<defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3.2" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,3.2 L0,6.4 Z" fill="#5b6b80"/></marker></defs>')
LANE_KEY={"Data Engineering Pipeline":"data","Hyperparameter Tuning Pipeline":"tune",
          "Full Training Pipeline":"train","Daily Forecast Pipeline":"daily"}
for name,ry in LANES:
    svg.append(f'<text class="lane lane-lbl-{LANE_KEY[name]}" x="22" y="{cy(ry)+4}">{esc(name)}</text>')
def _edge_points(e):
    pa=A(e["a"],e["sa"])
    pb=e["sb_pt"] if e.get("sb_pt") else A(e["b"],e["sb"])
    return [pa]+e["via"]+[pb]
def _label_pos(e):
    """Explicit lpos if given, else the midpoint of the edge's longest segment,
    nudged up a little so the text sits just above the line."""
    if e["lpos"]:
        return e["lpos"]
    pts=_edge_points(e)
    best=None; blen=-1
    for i in range(len(pts)-1):
        (x1,y1),(x2,y2)=pts[i],pts[i+1]
        L=abs(x2-x1)+abs(y2-y1)
        if L>blen: blen=L; best=((x1+x2)/2,(y1+y2)/2,abs(x2-x1)>=abs(y2-y1))
    mx,my,horiz=best
    # above the line for horizontal segments; just right of it for vertical
    return (mx, my-7) if horiz else (mx+14, my)
for e in EDGES:
    # id="e_<a>_<b>" so the animator can flow the connecting edge
    svg.append(f'<path id="e_{e["a"]}_{e["b"]}" class="edge" d="{poly(e)}" marker-end="url(#arr)"/>')
    if e["label"]:
        lx,ly=_label_pos(e)
        svg.append(f'<text class="elbl" x="{lx}" y="{ly}">{esc(e["label"])}</text>')
for n in N.values():
    svg.append(draw_node(n))
# ---- legend: horizontal strip UNDER the diagram ----
ly=H-30; gx=40
def lg_item(x, swatch_svg, text, tw):
    return swatch_svg, f'<text class="lgtxt" x="{x+38}" y="{ly+4}">{esc(text)}</text>', x+38+tw+34
# shapes part
items=[]
x=gx
svg.append(f'<text class="lglabel" x="{x}" y="{ly+4}">Legend:</text>'); x+=78
# process
svg.append(f'<rect x="{x}" y="{ly-8}" width="28" height="16" rx="3" class="bx lane-data"/>')
svg.append(f'<text class="lgtxt" x="{x+34}" y="{ly+4}">Process</text>'); x+=34+58
# decision
svg.append(f'<polygon points="{x+14},{ly-10} {x+28},{ly} {x+14},{ly+10} {x},{ly}" class="dia"/>')
svg.append(f'<text class="lgtxt" x="{x+34}" y="{ly+4}">Decision (yes/no)</text>'); x+=34+108
# start/end
svg.append(f'<rect x="{x}" y="{ly-8}" width="28" height="16" rx="8" class="se"/>')
svg.append(f'<text class="lgtxt" x="{x+34}" y="{ly+4}">Start / end</text>'); x+=34+76
# divider
svg.append(f'<line x1="{x}" y1="{ly-12}" x2="{x}" y2="{ly+12}" stroke="#cdd8e6" stroke-width="1.2"/>'); x+=22
# lane color swatches
for key,lab in [("data","Data engineering"),("tune","Tuning"),("train","Full training"),("daily","Daily forecast")]:
    svg.append(f'<rect x="{x}" y="{ly-8}" width="20" height="16" rx="3" class="bx lane-{key}"/>')
    svg.append(f'<text class="lgtxt" x="{x+26}" y="{ly+4}">{lab}</text>'); x+=26+len(lab)*6.0+24
svg.append('</svg>')
open("diagram.svg","w").write("\n".join(svg))
# also emit the FLOW order + node list for the dashboard
flow=[k for k in N.keys()]
print("FLOW=",flow)
print("ok",len(N),"nodes",len(EDGES),"edges",W,"x",H)
