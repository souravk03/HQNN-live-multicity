#!/usr/bin/env python3
"""QUADRANT prototype — 2x2 layout.
Flow: Data (top-left) -> Tuning (top-right) -> Training (bottom-right) -> Daily (bottom-left).
Clockwise U-shape. Handoffs between quadrants are drawn as bold labelled connectors.
"""
COLW=180; ROWH=86; BW=148; BH=50
N={}
def mk(id,x,y,label,kind="box",lane="data"):
    N[id]=dict(id=id,label=label,kind=kind,x=x,y=y,lane=lane)

GX_L=170
GX_R=170+6.7*COLW   # right column pushed further right so Full Training clears Daily
GY_T=78
GY_B=78+5.65*ROWH   # bottom row dropped so Full Training clearly clears the Tuning block
def TL(c,r): return (GX_L+c*COLW, GY_T+r*ROWH)
def TR(c,r): return (GX_R+c*COLW, GY_T+r*ROWH)
def BR(c,r): return (GX_R+c*COLW, GY_B+r*ROWH)
def BL(c,r): return (GX_L+c*COLW, GY_B+r*ROWH)

# DATA (top-left)
mk("start",*TL(0,0),"Start","startend","data")
mk("download",*TL(1,0),"Download Data","box","data")
mk("features",*TL(2,0),"Engineer 83 Features","box","data")
mk("split",*TL(3,0),"Split Train / Forward","box","data")

# ---- FEATURE BREAKDOWN (fills the empty left-middle band; the 6 families that
# make up the 83 engineered features). These hang off "Engineer 83 Features" and
# the download animation walks through them. Laid out as a 3x2 grid below Data. ----
FX0=GX_L+0.15*COLW; FY0=GY_T+1.55*ROWH; FCW=1.55*COLW; FRH=0.92*ROWH
def FB(c,r): return (FX0+c*FCW, FY0+r*FRH)
mk("feat_lag",*FB(0,0),"Lag features","feat","data")
mk("feat_roll",*FB(1,0),"Rolling statistics","feat","data")
mk("feat_diff",*FB(2,0),"Differences &\ntendencies","feat","data")
mk("feat_anom",*FB(0,1),"Anomalies","feat","data")
mk("feat_cyc",*FB(1,1),"Cyclical encodings","feat","data")
mk("feat_cross",*FB(2,1),"Cross-variable\nfeatures","feat","data")

# TUNING (top-right)
mk("tune_start",*TR(0,0),"Start Tuning","box","tune")
mk("t_tgt",*TR(1,0),"For Each Target \u00d73","box","tune")
mk("t_mdl",*TR(2,0),"For Each Model \u00d76","box","tune")
mk("optuna",*TR(3,0),"Create Optuna Study","box","tune")
mk("sample",*TR(4,0),"Sample Hyper-\nParameters","box","tune")
mk("cv",*TR(4,1),"For each CV \u00d73","box","tune")
mk("fit",*TR(3,1),"Fit Scalar (Fold)","box","tune")
mk("buildseq",*TR(2,1),"Build Sequence","box","tune")
mk("quicktrain",*TR(1,1),"Quick-Train Model","box","tune")
mk("valmse",*TR(0,1),"Validate \u2192 MSE","box","tune")
mk("report",*TR(0,2),"Report to Optuna","box","tune")
mk("prune",*TR(1,2),"Prune Trial?","decision","tune")
mk("morefolds",*TR(2,2),"More Folds?","decision","tune")
mk("avgmse",*TR(3,2),"Average Fold MSE","box","tune")
mk("moretrials",*TR(4,2),"More Trials?","decision","tune")
mk("selbest",*TR(4,3),"Select Best\nParameters","box","tune")
mk("moremodels",*TR(3,3),"More Models?","decision","tune")
mk("moretgts",*TR(2,3),"More Targets?","decision","tune")
mk("writehp",*TR(1,3),"Write best_hParams","box","tune")

# TRAINING (bottom-right)
# TRAINING (bottom-right) — shifted right one column; Write Metadata sits to the
# LEFT of Save Best Weights on the same row.
mk("f_tgt",*BR(2,0),"For Each Target \u00d73","box","train")
mk("f_mdl",*BR(3,0),"For Each Model \u00d76","box","train")
mk("epoch",*BR(3,1),"Train Epoch","box","train")
mk("early",*BR(2,1),"Early Stop?","decision","train")
mk("savew",*BR(1,1),"Save Best Weights","box","train")
mk("meta",*BR(0,1),"Write Metadata","box","train")

# DAILY (bottom-left) — EXACT original layout, mirrored horizontally.
# Original used cols 1..6; here origcol-1 -> mirrored col (5-(origcol-1)).
def DM(origcol,row): return BL(5-(origcol-1), row)   # daily-mirror placement
mk("beginday",*DM(1,0),"Begin Forward Day","box","daily")
mk("forecast",*DM(2,0),"Forecast next 15 days","box","daily")
mk("verify",*DM(3,0),"Verify vs Source","box","daily")
mk("agree",*DM(4,0),"Source Agrees?","decision","daily")
mk("seventh",*DM(5,0),"7th Verified Day /\n1st of Month","decision","daily")
mk("finetune",*DM(6,0),"Fine-Tune Weights","box","daily")
mk("skipupd",*DM(4,1),"Skip Update","box","daily")
mk("retrain",*DM(5,1),"Full Retrain \u00d718","box","daily")
mk("savew2",*DM(6,1),"Save Updated Weights","box","daily")
mk("logm",*DM(6,2),"Log + Metrics","box","daily")
mk("moredays",*DM(4,2),"More Days?","decision","daily")
mk("skip",*DM(1,2),"Stop","startend","daily")

def hw(n): return BW/2
def hh(n): return BH/2+(6 if n["kind"]=="decision" else 0)
def A(id,side):
    n=N[id]; x,y=n["x"],n["y"]
    return {"L":(x-hw(n),y),"R":(x+hw(n),y),"T":(x,y-hh(n)),"B":(x,y+hh(n))}[side]

EDGES=[]
def E(a,sa,b,sb,via=None,label="",lpos=None,sb_pt=None,cls=""):
    EDGES.append(dict(a=a,sa=sa,b=b,sb=sb,via=via or [],label=label,lpos=lpos,sb_pt=sb_pt,cls=cls))

GAP=4
# DATA internal
E("start","R","download","L"); E("download","R","features","L"); E("features","R","split","L")
# feature-breakdown LOOP: Engineer 83 Features -> the 6 families (top row L->R,
# down, bottom row R->L) -> back into Engineer 83 Features. Drawn subtly (class
# "feat"); also gives the download animation one continuous path through all six.
fb_=A("features","B")
# enter the loop: features bottom -> down -> into feat_lag top
ftl=A("feat_lag","T")
E("features","B","feat_lag","T",via=[(fb_[0],ftl[1]-20),(ftl[0],ftl[1]-20)],
  sb_pt=(ftl[0],ftl[1]-3),cls="feat")
# top row left -> right
E("feat_lag","R","feat_roll","L",cls="feat"); E("feat_roll","R","feat_diff","L",cls="feat")
# down the right side: feat_diff -> feat_cross
E("feat_diff","B","feat_cross","T",cls="feat")
# bottom row right -> left
E("feat_cross","L","feat_cyc","R",cls="feat"); E("feat_cyc","L","feat_anom","R",cls="feat")
# close the loop: feat_anom back up into Engineer 83 Features
fa=A("feat_anom","L"); ff=A("features","B")
loopx=N["feat_anom"]["x"]-FCW*0.42
E("feat_anom","L","features","B",via=[(loopx,fa[1]),(loopx,ff[1]+24),(ff[0]-30,ff[1]+24)],
  sb_pt=(ff[0]-30,ff[1]+GAP),cls="feat")
# HANDOFF 1
sp=A("split","R"); ts=A("tune_start","L"); mx=(sp[0]+ts[0])/2
E("split","R","tune_start","L",via=[(mx,sp[1]),(mx,ts[1])],label="→ tuning",lpos=(mx,sp[1]-12),cls="handoff")
# TUNING internal
E("tune_start","R","t_tgt","L"); E("t_tgt","R","t_mdl","L"); E("t_mdl","R","optuna","L"); E("optuna","R","sample","L")
E("sample","B","cv","T")
E("cv","L","fit","R"); E("fit","L","buildseq","R"); E("buildseq","L","quicktrain","R"); E("quicktrain","L","valmse","R")
E("valmse","B","report","T")
E("report","R","prune","L")
E("prune","R","morefolds","L",label="No"); E("morefolds","R","avgmse","L",label="No"); E("avgmse","R","moretrials","L")
sr=A("sample","R"); EDGE=sr[0]
ch1,ch2,ch3,ch4=EDGE+28,EDGE+54,EDGE+80,EDGE+106
e1=(sr[0]+GAP,sr[1]-18); e2=(sr[0]+GAP,sr[1]-6); e3=(sr[0]+GAP,sr[1]+6); e4=(sr[0]+GAP,sr[1]+18)
mt=A("moretrials","R")
E("moretrials","R","sample","R",via=[(ch1,mt[1]),(ch1,e1[1])],sb_pt=e1,label="Yes",lpos=(ch1+12,mt[1]-8))
E("moretrials","B","selbest","T",label="No")
E("selbest","L","moremodels","R"); E("moremodels","L","moretgts","R",label="No"); E("moretgts","L","writehp","R",label="No")
mm=A("moremodels","B"); belowD=N["moremodels"]["y"]+hh(N["moremodels"])+16
E("moremodels","B","sample","R",via=[(mm[0],belowD),(ch2,belowD),(ch2,e2[1])],sb_pt=e2,label="Yes",lpos=(mm[0]+16,belowD+11))
mtg=A("moretgts","B"); belowD2=N["moretgts"]["y"]+hh(N["moretgts"])+30
E("moretgts","B","sample","R",via=[(mtg[0],belowD2),(ch3,belowD2),(ch3,e3[1])],sb_pt=e3,label="Yes",lpos=(mtg[0]+12,(N["moretgts"]["y"]+hh(N["moretgts"])+belowD2)/2))
pr=A("prune","B"); lowliney=N["prune"]["y"]+hh(N["prune"])+14
E("prune","B","sample","R",via=[(pr[0],lowliney),(ch4,lowliney),(ch4,e4[1])],sb_pt=e4,label="Yes",lpos=(pr[0]+22,lowliney-7))
mf=A("morefolds","T"); cvb=A("cv","B"); cvy=N["cv"]["y"]+hh(N["cv"])+16; cv_in=(cvb[0]-26,cvb[1]+GAP)
E("morefolds","T","cv","B",via=[(mf[0],cvy),(cv_in[0],cvy)],sb_pt=cv_in,label="Yes",lpos=(cv_in[0]+30,cvy-7))
# HANDOFF 2
wb=A("writehp","B"); ftt=A("f_tgt","T"); my=(wb[1]+ftt[1])/2
E("writehp","B","f_tgt","T",via=[(wb[0],my),(ftt[0],my)],label="→ training",lpos=(ftt[0],my-12),cls="handoff")
# TRAINING internal
E("f_tgt","R","f_mdl","L")
E("f_mdl","B","epoch","T")
E("epoch","L","early","R")
E("early","L","savew","R",label="done")
E("savew","L","meta","R")
eb=A("early","B"); epb=A("epoch","B"); lowy=N["epoch"]["y"]+hh(N["epoch"])+20
E("early","B","epoch","B",via=[(eb[0],lowy),(epb[0],lowy)],label="no - next epoch",lpos=((eb[0]+epb[0])/2,lowy+11))
epr=A("epoch","R"); fmr=A("f_mdl","R"); farx=N["f_mdl"]["x"]+hw(N["f_mdl"])+44
E("epoch","R","f_mdl","R",via=[(farx,epr[1]),(farx,fmr[1])],label="next model \u00d76",lpos=(farx-46,(epr[1]+fmr[1])/2))
fmt=A("f_mdl","T"); ft_in=(ftt[0]+30,ftt[1]-4); chy=N["f_mdl"]["y"]-hh(N["f_mdl"])-18
E("f_mdl","T","f_tgt","T",via=[(fmt[0],chy),(ft_in[0],chy)],sb_pt=ft_in,label="next target \u00d73",lpos=((fmt[0]+ft_in[0])/2,chy-7))
# HANDOFF 3 (Training meta -> Daily beginday). Start from meta's TOP, go up, then
# left into beginday's RIGHT side — a single 90 degree bend. beginday is in the
# Daily quadrant (left), meta in Training (right), and beginday's right edge faces
# the gap between them, so this is clean and clears the Stop box.
mt_=A("meta","T"); bdr=A("beginday","R")
E("meta","T","beginday","R",via=[(mt_[0],bdr[1])],
  sb_pt=(bdr[0]+GAP,bdr[1]),label="\u2192 daily",lpos=((mt_[0]+bdr[0])/2,bdr[1]-9),cls="handoff")
# DAILY internal — pure horizontal mirror of the approved daily lane.
# Original sides L<->R swapped; custom-via shapes flipped about their anchors.
# row-0 chain (orig: beginday.R->forecast.L ...). mirrored:
E("beginday","L","forecast","R"); E("forecast","L","verify","R"); E("verify","L","agree","R")
E("agree","L","seventh","R",label="Yes"); E("seventh","L","finetune","R",label="No")
E("agree","B","skipupd","T",label="No"); E("seventh","B","retrain","T",label="Yes")
rl=A("retrain","L"); s2r=A("savew2","R")
E("retrain","L","savew2","R",label="Retrained",lpos=((rl[0]+s2r[0])/2,rl[1]-30)); E("finetune","B","savew2","T")
E("savew2","B","logm","T")
E("logm","R","moredays","L"); E("moredays","R","skip","L",label="No")
# skipupd -> logm "Skip log": orig dropped into a mid channel and ran into logm's
# LEFT; mirrored -> runs into logm's RIGHT (shape flipped about x).
su=A("skipupd","B"); lr=A("logm","R")
chy=(N["skipupd"]["y"]+hh(N["skipupd"])+N["logm"]["y"]-hh(N["logm"]))/2
E("skipupd","B","logm","R",via=[(su[0],chy),(lr[0]+30,chy),(lr[0]+30,lr[1])],
  label="Skip log",lpos=(su[0]-66,chy-8))
# moredays -> beginday "next day": orig ran from More Days' LEFT along a low channel
# up into beginday's bottom; mirrored to the RIGHT side.
mdr=A("moredays","R"); bdb=A("beginday","B")
chy_day=N["beginday"]["y"]+hh(N["beginday"])+34
E("moredays","R","beginday","B",
  via=[(mdr[0]+30,mdr[1]),(mdr[0]+30,chy_day),(bdb[0],chy_day)],
  label="yes \u00b7 next day \u21bb",lpos=(bdb[0]-96,chy_day-9))

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
# canvas must include the tuning loop-back channels that extend RIGHT of Sample
# (to EDGE+106) plus the quadrant box right margin — otherwise the viewBox clips them.
_chan_right = A("sample","R")[0] + 106
maxx=max([n["x"]+BW/2 for n in N.values()]+[_chan_right]) + 175
maxy=max(n["y"] for n in N.values())+BH/2+110
W=maxx; H=maxy

def draw_node(n):
    x,y=n["x"],n["y"]; lines=n["label"].split("\n"); lane=n["lane"]
    if n["kind"]=="decision":
        w,h=BW/2,BH/2+6; shape=f'<polygon points="{x},{y-h} {x+w},{y} {x},{y+h} {x-w},{y}" class="dia lane-{lane}"/>'
    elif n["kind"]=="startend":
        shape=f'<rect x="{x-BW/2}" y="{y-BH/2}" width="{BW}" height="{BH}" rx="24" class="se lane-{lane}"/>'
    elif n["kind"]=="feat":
        fw,fh=FCW-26,FRH-22
        shape=f'<rect x="{x-fw/2}" y="{y-fh/2}" width="{fw}" height="{fh}" rx="8" class="featbx lane-{lane}"/>'
        ty=y-(len(lines)-1)*7
        txt="".join(f'<tspan x="{x}" y="{ty+i*13}">{esc(l)}</tspan>' for i,l in enumerate(lines))
        return f'<g class="fn node feat-node" id="nd_{n["id"]}" data-id="{n["id"]}">{shape}<text class="featlbl">{txt}</text></g>'
    else:
        shape=f'<rect x="{x-BW/2}" y="{y-BH/2}" width="{BW}" height="{BH}" rx="9" class="bx lane-{lane}"/>'
    ty=y-(len(lines)-1)*8
    txt="".join(f'<tspan x="{x}" y="{ty+i*14}">{esc(l)}</tspan>' for i,l in enumerate(lines))
    return f'<g class="fn node" id="nd_{n["id"]}" data-id="{n["id"]}">{shape}<text class="lbl">{txt}</text></g>'
def poly(e):
    pa=A(e["a"],e["sa"]); pb=e["sb_pt"] if e.get("sb_pt") else A(e["b"],e["sb"])
    pts=[pa]+e["via"]+[pb]
    return "M "+" L ".join(f"{p[0]} {p[1]}" for p in pts)
def _label_pos(e):
    if e["lpos"]: return e["lpos"]
    pa=A(e["a"],e["sa"]); pb=e["sb_pt"] if e.get("sb_pt") else A(e["b"],e["sb"])
    pts=[pa]+e["via"]+[pb]; best=None; blen=-1
    for i in range(len(pts)-1):
        (x1,y1),(x2,y2)=pts[i],pts[i+1]; L=abs(x2-x1)+abs(y2-y1)
        if L>blen: blen=L; best=((x1+x2)/2,(y1+y2)/2,abs(x2-x1)>=abs(y2-y1))
    mx,my,horiz=best
    return (mx,my-7) if horiz else (mx+13,my)

svg=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" class="pipe">']
svg.append('<defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3.2" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,3.2 L0,6.4 Z" fill="#5b6b80"/></marker>'
           '<marker id="arrH" markerWidth="13" markerHeight="13" refX="8" refY="3.8" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,3.8 L0,7.6 Z" fill="#b8531a"/></marker></defs>')

# ---- bounding boxes computed from real node extents, padded to enclose the
# loop-back channels that sit OUTSIDE the node columns (right of Sample, above
# the decision rows, etc.). margins differ per side per lane. ----
LANE_TITLE={"data":"Data Engineering","tune":"Hyperparameter Tuning",
            "train":"Full Training","daily":"Daily Forecast"}
# extra room each lane needs beyond its node bounding box: (left,right,top,bottom)
LANE_MARGIN={"data":(40,64,52,40),
             "tune":(40,155,52,46),
             "train":(40,70,52,54),   # +bottom so Train Epoch / no-next-epoch label clear the border
             "daily":(66,46,52,40)}   # +left so Daily's left edge lines up with Data's
for lane in ["data","tune","train","daily"]:
    ns=[n for n in N.values() if n["lane"]==lane]
    # half-width per node: feat nodes are wider than normal boxes
    def _hwn(n): return (FCW-26)/2 if n["kind"]=="feat" else BW/2
    def _hhn(n): return (FRH-22)/2 if n["kind"]=="feat" else BH/2
    ml,mr,mt,mb=LANE_MARGIN[lane]
    qx=min(n["x"]-_hwn(n) for n in ns)-ml; qy=min(n["y"]-_hhn(n) for n in ns)-mt
    qw=(max(n["x"]+_hwn(n) for n in ns)+mr)-qx; qh=(max(n["y"]+_hhn(n) for n in ns)+mb)-qy
    svg.append(f'<rect class="quad quad-{lane}" x="{qx}" y="{qy}" width="{qw}" height="{qh}" rx="16"/>')
    svg.append(f'<text class="qlabel qlabel-{lane}" x="{qx+16}" y="{qy+26}">{esc(LANE_TITLE[lane])}</text>')
for e in EDGES:
    if e["cls"]=="feat":
        svg.append(f'<path id="e_{e["a"]}_{e["b"]}" class="edge feat-link" d="{poly(e)}"/>')
        continue
    marker="url(#arr)"
    cls="edge handoff" if e["cls"]=="handoff" else "edge"
    svg.append(f'<path id="e_{e["a"]}_{e["b"]}" class="{cls}" d="{poly(e)}" marker-end="{marker}"/>')
    if e["label"]:
        lx,ly=_label_pos(e)
        lcls="elbl handoff-lbl" if e["cls"]=="handoff" else "elbl"
        svg.append(f'<text class="{lcls}" x="{lx}" y="{ly}">{esc(e["label"])}</text>')
for n in N.values():
    svg.append(draw_node(n))
ly=H-26; x=40
svg.append(f'<text class="lglabel" x="{x}" y="{ly+4}">Legend:</text>'); x+=78
svg.append(f'<rect x="{x}" y="{ly-8}" width="26" height="15" rx="3" class="bx lane-data"/>')
svg.append(f'<text class="lgtxt" x="{x+32}" y="{ly+4}">Process</text>'); x+=32+54
svg.append(f'<polygon points="{x+13},{ly-9} {x+26},{ly} {x+13},{ly+9} {x},{ly}" class="dia"/>')
svg.append(f'<text class="lgtxt" x="{x+32}" y="{ly+4}">Decision</text>'); x+=32+62
svg.append(f'<rect x="{x}" y="{ly-8}" width="26" height="15" rx="7" class="se"/>')
svg.append(f'<text class="lgtxt" x="{x+32}" y="{ly+4}">Start / end</text>'); x+=32+72
svg.append(f'<line x1="{x+24}" y1="{ly}" x2="{x+50}" y2="{ly}" class="edge handoff" marker-end="url(#arr)"/>')
svg.append(f'<text class="lgtxt" x="{x+58}" y="{ly+4}">Stage handoff</text>')
svg.append('</svg>')
open("diagramq.svg","w").write("\n".join(svg))
print("ok",len(N),"nodes",len(EDGES),"edges","%.0f x %.0f"%(W,H))
