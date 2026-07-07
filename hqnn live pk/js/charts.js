// charts.js — forecast charts (+cursor line), horizon accuracy, live-training charts/table
// ---- charts ----
// render charts at >=2x backing store so lines stay crisp (fixes blurry canvases)
try{ if(window.Chart) Chart.defaults.devicePixelRatio = Math.max(2, window.devicePixelRatio||1); }catch(e){}
// chart colours pulled from the active theme (see theme.css)
function cssvar(n){ try{ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }catch(e){ return ''; } }
function CK(){ return {
  ink:  cssvar('--chart-ink')||CK().ink,
  grid: cssvar('--chart-grid')||CK().grid,
  axis: cssvar('--chart-axis')||CK().axis,
  tip:  cssvar('--tip-bg')||CK().tip,
  tipln:cssvar('--tip-line')||CK().tipln
};}

const CHARTS={};
function chartFor(t){
  if(CHARTS[t])return CHARTS[t];
  const ctx=document.getElementById('ch_'+t);
  // "today" marker: vertical line at the cursor day dividing verified (left) from
  // forecast (right), labelled below; a small "today" tag sits at the top.
  const cursorLine={
    id:'cursorLine',
    afterDraw(chart){
      const cur=window._cursor; if(!cur)return;
      const lbl=cur.slice(5);
      const labels=chart.data.labels||[];
      let idx=labels.indexOf(lbl);
      if(idx<0){ for(let i=0;i<labels.length;i++){ if(labels[i]<=lbl) idx=i; } }
      if(idx<0)return;
      const x=chart.scales.x.getPixelForValue(idx);
      const top=chart.chartArea.top, bot=chart.chartArea.bottom, cx=chart.ctx;
      cx.save();
      cx.beginPath(); cx.moveTo(x,top); cx.lineTo(x,bot);
      cx.lineWidth=1.5; cx.strokeStyle='#d63a3a'; cx.setLineDash([5,4]); cx.stroke(); cx.setLineDash([]);
      // "today" pill at the top of the line
      cx.font='700 9px Inter,sans-serif'; cx.textBaseline='middle';
      const tw=cx.measureText('today').width, px=6, ty=top+1;
      let bx=x-(tw/2)-px; if(bx<chart.chartArea.left)bx=chart.chartArea.left;
      if(bx+tw+px*2>chart.chartArea.right)bx=chart.chartArea.right-tw-px*2;
      cx.fillStyle='#d63a3a'; cx.beginPath();
      (cx.roundRect?cx.roundRect(bx,ty,tw+px*2,14,7):cx.rect(bx,ty,tw+px*2,14)); cx.fill();
      cx.fillStyle='#fff'; cx.textAlign='left'; cx.fillText('today',bx+px,ty+7);
      // zone labels below the plot
      cx.fillStyle='#d63a3a'; cx.textBaseline='top'; const ly=bot+4;
      cx.textAlign='right'; if(x-6>chart.chartArea.left+30) cx.fillText('verified', x-6, ly);
      cx.textAlign='left';  if(x+6<chart.chartArea.right-36) cx.fillText('forecast', x+6, ly);
      cx.restore();
    }
  };
  // crosshair: a faint dashed vertical line following the hovered point
  const crosshair={
    id:'crosshair',
    afterDraw(chart){
      const a=chart.tooltip&&chart.tooltip._active;
      if(!a||!a.length)return;
      const x=a[0].element.x, {top,bottom}=chart.chartArea, cx=chart.ctx;
      cx.save(); cx.beginPath(); cx.moveTo(x,top); cx.lineTo(x,bottom);
      cx.lineWidth=1; cx.strokeStyle=CK().grid; cx.setLineDash([3,3]); cx.stroke(); cx.restore();
    }
  };
  // soft vertical gradient under the NASA ground-truth line
  const nasaFill=(c)=>{ const{ctx,chartArea}=c.chart; if(!chartArea)return 'rgba(58,168,143,.10)';
    const g=ctx.createLinearGradient(0,chartArea.top,0,chartArea.bottom);
    g.addColorStop(0,'rgba(58,168,143,.30)'); g.addColorStop(1,'rgba(58,168,143,.02)'); return g; };
  CHARTS[t]=new Chart(ctx,{type:'line',
    data:{labels:[],datasets:[
      {label:'NASA actual',data:[],borderColor:'#3aa88f',backgroundColor:nasaFill,tension:.3,pointRadius:0,borderWidth:2.5,fill:true,order:2},
      {label:'Forecast',data:[],borderColor:'#3b82c4',backgroundColor:'#3b82c4',showLine:false,pointRadius:3.5,pointHoverRadius:6,order:1},
      {label:'Open-Meteo actual',data:[],borderColor:'#c0641d',backgroundColor:'#c0641d',showLine:false,pointRadius:4,pointHoverRadius:6,pointStyle:'triangle',order:1}
    ]},
    plugins:[cursorLine,crosshair],
    options:{responsive:true,maintainAspectRatio:false,
      layout:{padding:{top:6,bottom:16}},
      interaction:{mode:'index',intersect:false,axis:'x'},
      hover:{mode:'index',intersect:false},
      plugins:{legend:{labels:{font:{family:'Inter',size:14,weight:'700'},color:CK().ink,boxWidth:14,padding:14,usePointStyle:true}},
        tooltip:{mode:'index',intersect:false,axis:'x',backgroundColor:CK().tip,titleColor:CK().ink,bodyColor:CK().ink,
          borderColor:CK().tipln,borderWidth:1,padding:10,cornerRadius:8,usePointStyle:true,
          titleFont:{family:'Inter',weight:'700',size:12},bodyFont:{family:'Inter',size:12}}},
      scales:{x:{grid:{color:CK().grid},ticks:{font:{family:'Inter',size:9},color:CK().ink,maxRotation:0,autoSkip:true,maxTicksLimit:6}},
              y:{grid:{color:CK().grid},ticks:{font:{family:'Inter',size:10},color:CK().ink}}}}});
  return CHARTS[t];
}
function setChart(t,pts){const c=chartFor(t);
  const scale=t==='PRmsl'?(v=>v==null?null:Math.round(v/100)):(v=>v);
  // sort by date so the line is drawn left-to-right in time order — otherwise
  // out-of-order points make the line jump back and draw a weird loop on reload.
  const sorted=(pts||[]).slice().sort((a,b)=>(a.date<b.date?-1:a.date>b.date?1:0));
  c.data.labels=sorted.map(p=>p.date.slice(5));
  c._dates=sorted.map(p=>p.date);
  // dataset 0 = NASA continuous line (prefer explicit nasa, fall back to actual)
  c.data.datasets[0].data=sorted.map(p=>scale(p.nasa!=null?p.nasa:p.actual));
  // dataset 1 = our forecast (dots)
  c.data.datasets[1].data=sorted.map(p=>scale(p.pred));
  c.data.datasets[2].data=sorted.map(p=>scale(p.meteo));
  c.update('none');}
function pushChart(t,date,pred,actual,meteo){const c=chartFor(t);
  const scale=t==='PRmsl'?(v=>v==null?null:Math.round(v/100)):(v=>v);
  const lbl=date.slice(5);
  // keep a parallel list of full dates so we can keep everything sorted by time;
  // appending blindly makes the line jump back to an earlier date (a loop).
  if(!c._dates)c._dates=[];
  let i=c.data.labels.indexOf(lbl);
  if(i<0){
    c.data.labels.push(lbl); c._dates.push(date);
    c.data.datasets[0].data.push(scale(actual));
    c.data.datasets[1].data.push(scale(pred));
    c.data.datasets[2].data.push(scale(meteo));
  }else{
    if(actual!=null)c.data.datasets[0].data[i]=scale(actual);
    if(pred!=null)c.data.datasets[1].data[i]=scale(pred);
    if(meteo!=null)c.data.datasets[2].data[i]=scale(meteo);
  }
  // re-sort all series by the real date so the line is always left-to-right
  const order=c._dates.map((d,idx)=>idx).sort((a,b)=>(c._dates[a]<c._dates[b]?-1:c._dates[a]>c._dates[b]?1:0));
  c.data.labels=order.map(k=>c.data.labels[k]);
  c._dates=order.map(k=>c._dates[k]);
  c.data.datasets[0].data=order.map(k=>c.data.datasets[0].data[k]);
  c.data.datasets[1].data=order.map(k=>c.data.datasets[1].data[k]);
  c.data.datasets[2].data=order.map(k=>c.data.datasets[2].data[k]);
  // keep the chart within the display window even as points stream in during a
  // cycle (the backend already windows on restore; this bounds live growth).
  const MAXPTS=30;
  if(c.data.labels.length>MAXPTS){
    const drop=c.data.labels.length-MAXPTS;
    c.data.labels.splice(0,drop); c._dates.splice(0,drop);
    c.data.datasets.forEach(ds=>ds.data.splice(0,drop));
  }
  c.update('none');}

// ---- accuracy by lead time (horizon) ----
let _hzChart=null;
const HZCOLORS={TMP2m:'#c0641d',RH2m:'#3b82c4',PRmsl:'#1c7a63'};
function refreshHorizon(){
  fetch(q('/horizon_accuracy?model=ensemble')).then(r=>r.json()).then(d=>{
    const tgts=d.targets||{};
    const have=Object.keys(tgts).some(t=>(tgts[t]||[]).length>0);
    document.getElementById('horizonEmpty').style.display=have?'none':'block';
    if(!have){if(_hzChart){_hzChart.destroy();_hzChart=null;}return;}
    const N=d.horizon||15;
    const labels=Array.from({length:N},(_,i)=>'+'+(i+1)+'d');
    // PRmsl error shown in hPa (÷100) so it shares a readable scale with the others
    const datasets=Object.keys(tgts).map(t=>{
      const byH={}; (tgts[t]||[]).forEach(r=>{byH[r.horizon]=r.mae;});
      const scale=(t==='PRmsl')?(v=>v==null?null:v/100):(v=>v);
      return {label:(TLABEL[t]||t)+(t==='PRmsl'?' (hPa)':''),
        data:labels.map((_,i)=>byH[i+1]!=null?scale(byH[i+1]):null),
        borderColor:HZCOLORS[t]||CK().axis,backgroundColor:'transparent',
        tension:.3,pointRadius:3,spanGaps:true};
    });
    const ctx=document.getElementById('ch_horizon');
    const existing = (Chart.getChart ? Chart.getChart(ctx) : null) || _hzChart;
    if(existing){try{existing.destroy();}catch(e){}}
    _hzChart=new Chart(ctx,{type:'line',data:{labels,datasets},
      options:{responsive:true,maintainAspectRatio:false,
        interaction:{mode:'index',intersect:false,axis:'x'},
        hover:{mode:'index',intersect:false},
        plugins:{legend:{labels:{font:{family:'Inter',size:14,weight:'700'},color:CK().ink,boxWidth:14,padding:14}},
          tooltip:{mode:'index',intersect:false,axis:'x',callbacks:{title:items=>'Lead time '+items[0].label}}},
        scales:{x:{title:{display:true,text:'days ahead',font:{family:'Inter',size:10}},
                   ticks:{font:{family:'Inter',size:9},color:'#1a2230'}},
                y:{title:{display:true,text:'mean abs error',font:{family:'Inter',size:10}},
                   beginAtZero:true,ticks:{font:{family:'Inter',size:10},color:'#1a2230'}}}}});
  }).catch(()=>{});
}

// ---- live training metrics (chart + table) ----
const MCOLORS={lstm:'#1f5d96',gru:'#1c7a63',qlstm:'#7c5cc4',qgru:'#c0641d',ann:'#c41d6f',hqnn:'#0f8a8a'};
let _ltmMetric='rmse';                 // which metric the chart shows
let _ltmTarget='TMP2m';                // which target the chart shows
let _ltmSeries={};                     // key "target|model" -> {rmse:[{x,y}],mae:[],r2:[]}
let _ltmChart=null;
let _ltmRowsInit=[], _ltmRowsFT=[], _ltmRowsRT=[];   // initial-training / fine-tune / retrain rows (latest first)
function _ltmEnsureChart(){
  if(_ltmChart)return _ltmChart;
  const ctx=document.getElementById('ltmCanvas');if(!ctx)return null;
  _ltmChart=new Chart(ctx,{type:'line',
    data:{datasets:MODELS.map(m=>({label:m.toUpperCase(),data:[],
      borderColor:MCOLORS[m],backgroundColor:MCOLORS[m],
      borderWidth:2.5,pointRadius:0,pointHoverRadius:4,tension:.35,
      pointBackgroundColor:MCOLORS[m],pointBorderColor:'#fff',pointBorderWidth:1}))},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:450,easing:'easeInOutQuart'},
      layout:{padding:{top:8,right:12,bottom:4,left:4}},
      interaction:{mode:'index',intersect:false,axis:'x'},
      hover:{mode:'index',intersect:false},
      scales:{
        x:{type:'linear',
           title:{display:true,text:'Epoch',font:{family:'Inter',size:13,weight:'700'},color:CK().ink,padding:{top:6}},
           grid:{color:CK().grid,drawTicks:false},
           border:{color:CK().axis},
           ticks:{font:{family:'Inter',size:12,weight:'600'},color:CK().ink,padding:6,maxTicksLimit:8,precision:0}},
        y:{title:{display:true,text:'RMSE',font:{family:'Inter',size:13,weight:'700'},color:CK().ink},
           grid:{color:CK().grid,drawTicks:false},
           border:{color:CK().axis},
           ticks:{font:{family:'Inter',size:12,weight:'600'},color:CK().ink,padding:8,maxTicksLimit:6}}},
      plugins:{
        legend:{position:'top',align:'end',
          labels:{font:{family:'Inter',size:13,weight:'700'},color:CK().ink,
            boxWidth:14,boxHeight:14,padding:16,usePointStyle:true,pointStyle:'rectRounded'}},
        tooltip:{mode:'index',intersect:false,axis:'x',
          backgroundColor:CK().tip,titleColor:CK().ink,bodyColor:CK().ink,
          borderColor:CK().tipln,borderWidth:1,titleFont:{family:'Inter',size:12,weight:'700'},
          bodyFont:{family:'Inter',size:12},padding:10,cornerRadius:8,displayColors:true,
          callbacks:{title:items=>'Epoch '+(items[0]&&items[0].parsed?items[0].parsed.x:'')}}}}});
  return _ltmChart;
}
function _ltmAxisTitle(){
  const c=_ltmChart;if(!c)return;
  const L={rmse:'RMSE',mae:'MAE',r2:'R²'};
  c.options.scales.y.title.text=L[_ltmMetric]||'';
}

// ---- right pane: LIVE updates chart (fine-tune / retrain over time) ----
const _LTM_WINDOW=60;   // show only the most recent N points so old runs don't
                        // pile up and crush the detail (data is kept, view is windowed)
let _ltmLiveChart=null;
let _ltmLiveSeries={};   // "target|model" -> {rmse:[{x,y}],mae:[],r2:[]}, x = update #
let _ltmLiveCount={};    // "target|model" -> running update index
function _ltmLiveEnsure(){
  if(_ltmLiveChart)return _ltmLiveChart;
  const ctx=document.getElementById('ltmCanvasLive');if(!ctx)return null;
  _ltmLiveChart=new Chart(ctx,{type:'line',
    data:{datasets:MODELS.map(m=>({label:m.toUpperCase(),data:[],
      borderColor:MCOLORS[m],backgroundColor:MCOLORS[m],
      borderWidth:2.5,pointRadius:2.5,pointHoverRadius:4,tension:.35,
      pointBackgroundColor:MCOLORS[m],pointBorderColor:'#fff',pointBorderWidth:1}))},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:450,easing:'easeInOutQuart'},
      layout:{padding:{top:8,right:12,bottom:4,left:4}},
      interaction:{mode:'index',intersect:false,axis:'x'},
      hover:{mode:'index',intersect:false},
      scales:{
        x:{type:'linear',
           title:{display:true,text:'Update #',font:{family:'Inter',size:13,weight:'700'},color:CK().ink,padding:{top:6}},
           grid:{color:CK().grid,drawTicks:false},border:{color:CK().axis},
           ticks:{font:{family:'Inter',size:12,weight:'600'},color:CK().ink,padding:6,maxTicksLimit:8,precision:0}},
        y:{title:{display:true,text:'RMSE',font:{family:'Inter',size:13,weight:'700'},color:CK().ink},
           grid:{color:CK().grid,drawTicks:false},border:{color:CK().axis},
           ticks:{font:{family:'Inter',size:12,weight:'600'},color:CK().ink,padding:8,maxTicksLimit:6}}},
      plugins:{
        legend:{position:'top',align:'end',
          labels:{font:{family:'Inter',size:13,weight:'700'},color:CK().ink,
            boxWidth:14,boxHeight:14,padding:16,usePointStyle:true,pointStyle:'rectRounded'}},
        tooltip:{mode:'index',intersect:false,axis:'x',
          backgroundColor:CK().tip,titleColor:CK().ink,bodyColor:CK().ink,
          borderColor:CK().tipln,borderWidth:1,titleFont:{family:'Inter',size:12,weight:'700'},
          bodyFont:{family:'Inter',size:12},padding:10,cornerRadius:8,
          callbacks:{title:items=>'Update '+(items[0]&&items[0].parsed?items[0].parsed.x:'')}}}}});
  return _ltmLiveChart;
}
function _ltmLiveAdd(target,model,rmse,mae,r2){
  const key=target+'|'+model;
  if(!_ltmLiveSeries[key])_ltmLiveSeries[key]={rmse:[],mae:[],r2:[]};
  _ltmLiveCount[key]=(_ltmLiveCount[key]||0)+1; const x=_ltmLiveCount[key];
  if(rmse!=null)_ltmLiveSeries[key].rmse.push({x,y:rmse});
  if(mae!=null)_ltmLiveSeries[key].mae.push({x,y:mae});
  if(r2!=null)_ltmLiveSeries[key].r2.push({x,y:r2});
  ['rmse','mae','r2'].forEach(function(k){var a=_ltmLiveSeries[key][k]; if(a.length>600)a.splice(0,a.length-600);});
}
function _ltmWindow(arr){ // keep only the most recent _LTM_WINDOW points
  if(!arr||arr.length<=_LTM_WINDOW)return arr||[];
  return arr.slice(arr.length-_LTM_WINDOW);
}
function _lockY(c,animate){
  // On an explicit switch/load (animate=true) we fix the y-range to the visible
  // data so the new graph settles cleanly. During streaming (animate=false) we
  // RELEASE the lock so incoming points stay on-screen — but because streaming
  // redraws use update('none'), the rescale is instant (no jarring animated jump
  // on every update, which was the real annoyance).
  if(!animate){ c.options.scales.y.min=undefined; c.options.scales.y.max=undefined; return; }
  let lo=Infinity,hi=-Infinity;
  c.data.datasets.forEach(ds=>ds.data.forEach(p=>{const y=p&&p.y!=null?p.y:p;if(y!=null&&isFinite(y)){if(y<lo)lo=y;if(y>hi)hi=y;}}));
  if(lo===Infinity){ c.options.scales.y.min=undefined; c.options.scales.y.max=undefined; return; }
  const pad=(hi-lo)*0.15||0.5;
  c.options.scales.y.min=lo-pad; c.options.scales.y.max=hi+pad;
}
function _ltmLiveRedraw(animate){
  const c=_ltmLiveEnsure();if(!c)return;
  MODELS.forEach((m,i)=>{const key=_ltmTarget+'|'+m;
    const s=_ltmLiveSeries[key]?_ltmLiveSeries[key][_ltmMetric]:[];
    c.data.datasets[i].data=_ltmWindow(s||[]);});
  const L={rmse:'RMSE',mae:'MAE',r2:'R²'};c.options.scales.y.title.text=L[_ltmMetric]||'';
  _lockY(c,animate);
  c.update(animate?undefined:'none');   // animate on switch/load, silent on stream
}
function _ltmRedraw(animate){
  const c=_ltmEnsureChart();if(!c)return;
  MODELS.forEach((m,i)=>{const key=_ltmTarget+'|'+m;
    const s=_ltmSeries[key]?_ltmSeries[key][_ltmMetric]:[];
    c.data.datasets[i].data=_ltmWindow(s||[]);});
  _ltmAxisTitle();
  _lockY(c,animate);
  c.update(animate?undefined:'none');
}
function _ltmAddPoint(target,model,epoch,rmse,mae,r2){
  const key=target+'|'+model;
  if(!_ltmSeries[key])_ltmSeries[key]={rmse:[],mae:[],r2:[]};
  const s=_ltmSeries[key];
  // If this epoch is <= the last recorded epoch, training RESTARTED for this
  // model/target (new run). Clear the series so the line never goes backward /
  // loops to an earlier epoch.
  const lastX=s.rmse.length?s.rmse[s.rmse.length-1].x:(s.mae.length?s.mae[s.mae.length-1].x:(s.r2.length?s.r2[s.r2.length-1].x:null));
  if(lastX!=null && epoch<=lastX){ s.rmse=[]; s.mae=[]; s.r2=[]; }
  // replace if this exact epoch already exists, else append; then keep sorted by epoch
  const put=(arr,v)=>{ if(v==null)return; const i=arr.findIndex(p=>p.x===epoch);
    if(i>=0)arr[i]={x:epoch,y:v}; else arr.push({x:epoch,y:v}); arr.sort((a,b)=>a.x-b.x); };
  put(s.rmse,rmse); put(s.mae,mae); put(s.r2,r2);
}
function _ltmCards(rows, withToggle){
  // build the 6 model cards for one table (initial / fine-tune / retrain).
  let html='';
  MODELS_ALL.forEach(m=>{
    const ready=(MODEL_READY[m]!==false);
    const on=modelEnabled(m);              // train-toggle state
    // heading: model name (left) + (only on the initial table) a train on/off toggle.
    let hd=`<span>${mUP(m)} ${modelBadges(m)}</span>`;
    if(withToggle){
      hd+=`<button class="mdtoggle${on?' on':''}" data-model="${m}" `+
          `title="${on?'training enabled — click to skip this model':'skipped — click to train this model'}">`+
          `${on?'on':'off'}</button>`;
    }
    html+=`<div class="ltmcard${on?'':' mdoff'}"><div class="ltmhd">${hd}</div>`;
    const latest={};
    rows.forEach(r=>{ if(r.model===m && !(r.target in latest)) latest[r.target]=r; });
    const have=TARGETS.some(t=>latest[t]);
    if(!have){ html+=`<div class="ltmnone">${ready?(on?'no data yet':'skipped'):'not added yet'}</div></div>`; return; }
    const fmt=(v,d)=>v==null?'—':(typeof v==='number'?v.toFixed(d):v);
    html+='<table class="ltmtable"><thead><tr><th>target</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>';
    TARGETS.forEach(t=>{const r=latest[t];
      html+=`<tr><td class="${r?r.cls:''}"><span style="font-weight:700">${TLABEL[t]||t}</span> `+
            `<span style="color:var(--mute);font-size:10px">${r?r.when:''}</span></td>`+
            `<td>${r?fmt(r.rmse,3):'—'}</td><td>${r?fmt(r.mae,3):'—'}</td><td>${r?fmt(r.r2,3):'—'}</td></tr>`;});
    html+='</tbody></table></div>';
  });
  return html;
}
function _ltmTable(){
  // three independent live tables: initial training, fine-tune, retrain.
  const wi=document.getElementById('ltmtablesInit'); if(wi) wi.innerHTML=_ltmCards(_ltmRowsInit,true);
  const wf=document.getElementById('ltmtablesFT');   if(wf) wf.innerHTML=_ltmCards(_ltmRowsFT,false);
  const wr=document.getElementById('ltmtablesRT');   if(wr) wr.innerHTML=_ltmCards(_ltmRowsRT,false);
}
function _ltmSelTarget(t,animate){
  if(animate===undefined)animate=true;   // user-initiated switch animates by default
  _ltmTarget=t;
  document.querySelectorAll('.ltmtgt').forEach(x=>x.classList.toggle('active',x.dataset.target===t));
  _ltmRedraw(animate);_ltmLiveRedraw(animate);}
// Renders (chart.update + rebuilding 3 card tables) are expensive. Events can
// arrive every epoch for 6 models — so we COALESCE renders to a few per second
// instead of one per event. Data is still ingested on every event (no points lost);
// only the painting is batched, which keeps the UI smooth during fast training.
let _ltmRenderTimer=null;
function _ltmScheduleRender(){
  if(_ltmRenderTimer)return;
  _ltmRenderTimer=setTimeout(function(){
    _ltmRenderTimer=null;
    try{ _ltmRedraw(false); }catch(e){}
    try{ _ltmLiveRedraw(false); }catch(e){}
    try{ _ltmTable(); }catch(e){}
  }, 160);
}
function _ltmFollowTarget(t){           // switch the displayed target without an animated double-redraw
  if(_ltmTarget===t)return;
  _ltmTarget=t;
  document.querySelectorAll('.ltmtgt').forEach(x=>x.classList.toggle('active',x.dataset.target===t));
}
function ltmTrain(ev){           // per-epoch during initial training
  document.getElementById('ltmphase').textContent='training · '+(TLABEL[ev.target]||ev.target);
  _ltmAddPoint(ev.target,ev.model,ev.epoch,ev.rmse,ev.mae,ev.r2);
  _ltmFollowTarget(ev.target);
  _ltmRowsInit.unshift({model:ev.model,target:ev.target,when:'ep '+ev.epoch,cls:'',rmse:ev.rmse,mae:ev.mae,r2:ev.r2});
  if(_ltmRowsInit.length>400)_ltmRowsInit.length=400;     // keep recent; cards only show latest per model/target
  _ltmScheduleRender();
}
function ltmLive(ev){            // per fine-tune / retrain during the daily cycle
  document.getElementById('ltmphase').textContent=ev.kind==='retrain'?'retrain':'fine-tune';
  const row={model:ev.model,target:ev.target,when:(ev.date||''),
             cls:ev.kind==='retrain'?'rt':'ft',rmse:ev.rmse,mae:ev.mae,r2:ev.r2};
  if(ev.kind==='retrain'){ _ltmRowsRT.unshift(row); if(_ltmRowsRT.length>400)_ltmRowsRT.length=400; }
  else { _ltmRowsFT.unshift(row); if(_ltmRowsFT.length>400)_ltmRowsFT.length=400; }
  _ltmLiveAdd(ev.target,ev.model,ev.rmse,ev.mae,ev.r2);
  _ltmFollowTarget(ev.target);
  _ltmScheduleRender();
}
document.addEventListener('click',e=>{
  const md=e.target.closest('.mdtoggle');
  if(md){const m=md.dataset.model; setModelEnabled(m,!modelEnabled(m)); _ltmTable(); return;}
  const mt=e.target.closest('.ltmtab');
  if(mt){document.querySelectorAll('.ltmtab').forEach(x=>x.classList.remove('active'));mt.classList.add('active');
    _ltmMetric=mt.dataset.metric;_ltmRedraw(true);_ltmLiveRedraw(true);return;}
  const tt=e.target.closest('.ltmtgt');
  if(tt){_ltmSelTarget(tt.dataset.target,true);}
});

// recolour every live chart instance when the theme toggles
function applyChartTheme(){
  var c=CK(), insts=[];
  try{ for(var k in CHARTS){ if(CHARTS[k]) insts.push(CHARTS[k]); } }catch(e){}
  try{ if(typeof _hzChart!=='undefined' && _hzChart) insts.push(_hzChart); }catch(e){}
  try{ if(typeof _ltmChart!=='undefined' && _ltmChart) insts.push(_ltmChart); }catch(e){}
  try{ if(typeof _ltmLiveChart!=='undefined' && _ltmLiveChart) insts.push(_ltmLiveChart); }catch(e){}
  insts.forEach(function(ch){
    var o=ch.options||{};
    if(o.scales){ ['x','y'].forEach(function(ax){ var sc=o.scales[ax]; if(!sc)return;
      if(sc.ticks){ sc.ticks.color=c.ink; }
      if(sc.title){ sc.title.color=c.ink; }
      if(sc.grid){ sc.grid.color=c.grid; }
      if(sc.border){ sc.border.color=c.axis; }
    });}
    if(o.plugins){
      if(o.plugins.legend&&o.plugins.legend.labels) o.plugins.legend.labels.color=c.ink;
      if(o.plugins.title) o.plugins.title.color=c.ink;
      if(o.plugins.tooltip){ var tp=o.plugins.tooltip; tp.backgroundColor=c.tip; tp.titleColor=c.ink; tp.bodyColor=c.ink; tp.borderColor=c.tipln; }
    }
    try{ ch.update('none'); }catch(e){}
  });
}
try{ window.addEventListener('themechange', applyChartTheme); }catch(e){}