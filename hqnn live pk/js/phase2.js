// phase2.js — fills the three Phase-2 panels:
//   #ch_spread   (model agreement / spread)   <- /model_spread
//   #mvuvCompare (multivariate vs univariate)  <- /mvuv_compare
//   #trainTime   (training time per model)     <- /train_times
// All read real backend data; each shows a clear empty state until data exists.

const TLAB_P2 = {TMP2m:'Temp', RH2m:'Humidity', PRmsl:'Pressure'};
let _spreadChart = null;

function refreshPhase2(){
  try{ loadModelSpread(); }catch(e){}
  try{ loadMvUv(); }catch(e){}
  try{ loadTrainTimes(); }catch(e){}
}

// ---- Model spread ----
function loadModelSpread(){
  fetch(q('/model_spread')).then(r=>r.json()).then(renderSpread).catch(()=>{});
}
function renderSpread(d){
  const empty=document.getElementById('spreadEmpty');
  const tg=(d&&d.targets)||{};
  const targets=Object.keys(tg);
  const canvas=document.getElementById('ch_spread'); if(!canvas) return;
  if(!targets.length){ if(empty)empty.style.display=''; if(_spreadChart){_spreadChart.destroy();_spreadChart=null;} return; }
  if(empty) empty.style.display='none';
  // x = union of dates; one line per target (std-dev of model preds that day)
  const dates=[...new Set([].concat(...targets.map(t=>tg[t].map(p=>p.date))))].sort();
  const colors={TMP2m:'#c4503a',RH2m:'#2f6fb0',PRmsl:'#1c7a63'};
  const ds=targets.map(t=>{
    const byDate={}; tg[t].forEach(p=>byDate[p.date]=p.spread);
    return {label:TLAB_P2[t]||t, data:dates.map(dt=>byDate[dt]??null),
            borderColor:colors[t]||'#888', backgroundColor:'transparent',
            borderWidth:2, pointRadius:2, spanGaps:true, tension:.25};
  });
  if(_spreadChart) _spreadChart.destroy();
  _spreadChart=new Chart(canvas.getContext('2d'),{type:'line',
    data:{labels:dates,datasets:ds},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{boxWidth:12,font:{size:11}}}},
      scales:{y:{title:{display:true,text:'std-dev across models'},beginAtZero:true},
              x:{ticks:{maxTicksLimit:8,font:{size:10}}}}}});
}

// ---- MV vs UV ----
function loadMvUv(){
  fetch('/mvuv_compare').then(r=>r.json()).then(renderMvUv).catch(()=>{});
}
function renderMvUv(d){
  const host=document.getElementById('mvuvCompare'); if(!host) return;
  const targets=(d&&d.targets)||[];
  const mv=(d&&d.multivariate)||{}, uv=(d&&d.univariate)||{};
  const hasAny = targets.some(t=>(mv[t]&&Object.keys(mv[t]).length)||(uv[t]&&Object.keys(uv[t]).length));
  if(!hasAny){ host.innerHTML='<div class="empty soon">Side-by-side MV vs UV accuracy appears once BOTH modes have verified forecasts (train + run cycles in each mode).</div>'; return; }
  // union of models per target, show RMSE for MV and UV next to each other
  let h='<table><thead><tr><th>Target</th><th>Model</th><th>MV RMSE</th><th>UV RMSE</th><th>Better</th></tr></thead><tbody>';
  targets.forEach(t=>{
    const models=[...new Set([...Object.keys(mv[t]||{}),...Object.keys(uv[t]||{})])]
      .sort((a,b)=>(a==='ensemble'?-1:b==='ensemble'?1:a.localeCompare(b)));
    models.forEach((m,i)=>{
      const a=(mv[t]||{})[m], b=(uv[t]||{})[m];
      const ar=a?a.rmse:null, br=b?b.rmse:null;
      let better='—';
      if(ar!=null&&br!=null) better=ar<br?'<span class="mvtag">MV</span>':(br<ar?'<span class="uvtag">UV</span>':'tie');
      h+=`<tr><td>${i===0?(TLAB_P2[t]||t):''}</td><td>${mUP(m)}</td>`+
         `<td>${ar!=null?ar:'—'}</td><td>${br!=null?br:'—'}</td><td>${better}</td></tr>`;
    });
  });
  h+='</tbody></table>';
  host.innerHTML=h;
}

// ---- Training time ----
function loadTrainTimes(){
  fetch(q('/train_times')).then(r=>r.json()).then(renderTrainTimes).catch(()=>{});
}
function renderTrainTimes(d){
  const host=document.getElementById('trainTime'); if(!host) return;
  const data=(d&&d.data)||{};
  const targets=Object.keys(data);
  if(!targets.length){ host.innerHTML='<div class="empty soon">Per-model training time appears after a fresh train with this build (it records wall-clock per model).</div>'; return; }
  // collect models, sum/avg time across targets
  const models={};
  targets.forEach(t=>Object.keys(data[t]).forEach(m=>{models[m]=(models[m]||0)+data[t][m];}));
  const rows=Object.keys(models).map(m=>({m,total:models[m]})).sort((a,b)=>b.total-a.total);
  const max=Math.max(...rows.map(r=>r.total),1);
  let h='<table><thead><tr><th>Model</th><th>Total train (s)</th><th></th></tr></thead><tbody>';
  rows.forEach(r=>{
    const pct=Math.round(r.total/max*100);
    h+=`<tr><td>${mUP(r.m)}</td><td>${r.total.toFixed(1)}</td>`+
       `<td style="width:45%"><div style="background:#dceaf7;border-radius:5px;overflow:hidden"><div style="width:${pct}%;background:#3b82c4;height:9px"></div></div></td></tr>`;
  });
  h+='</tbody></table><div class="ph" style="padding-top:8px">Sum across the 3 targets. Quantum models (qlstm/qgru/hqnn) are expected to cost more — weigh against their accuracy in "Accuracy by model".</div>';
  host.innerHTML=h;
}