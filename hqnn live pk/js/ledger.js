// ledger.js — accuracy-by-model, forecast ledger, weight bars, log, hparams, history, restoreTrainMetrics
// ---- CSV export ----
function _downloadCSV(filename, rows){
  const csv=rows.map(r=>r.map(c=>{
    const s=(c==null?'':String(c));
    return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;
  }).join(',')).join('\n');
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download=filename;
  document.body.appendChild(a); a.click();
  setTimeout(()=>{URL.revokeObjectURL(a.href); a.remove();},100);
}
function exportAccuracyCSV(){
  fetch(q('/live_data')).then(r=>r.json()).then(d=>{
    const m=d.metrics||{}; const out=[['model','target','RMSE','MAE','R2','MAPE','n']];
    MODELS.concat(['ensemble']).forEach(model=>TARGETS.forEach(t=>{
      const c=m&&m[t]&&m[t][model]; if(c)out.push([model,t,c.RMSE,c.MAE,c.R2,c.MAPE,c.n!=null?c.n:'']);
    }));
    _downloadCSV('accuracy_'+CITY+'_'+MODE+'.csv', out);
  }).catch(()=>{});
}
function exportHistoryCSV(){
  fetch(q('/live_data')).then(r=>r.json()).then(d=>{
    const rows=d.rows||[]; const out=[['forecast_date','made_on','target','model','horizon','prediction','actual_nasa','actual_meteo','error']];
    rows.forEach(r=>{
      const err=(r.actual!=null&&r.prediction!=null)?Math.round((r.prediction-r.actual)*1000)/1000:'';
      out.push([r.forecast_date,r.made_on||'',r.target,r.model,r.horizon!=null?r.horizon:'',r.prediction,
                r.actual_nasa!=null?r.actual_nasa:(r.actual!=null?r.actual:''),r.actual_meteo!=null?r.actual_meteo:'',err]);
    });
    _downloadCSV('forecast_history_'+CITY+'_'+MODE+'.csv', out);
  }).catch(()=>{});
}
// ---- Daily forecast vs NASA: one row per verified day, ensemble pred vs actual ----
function renderDailyVs(rows){
  const host=document.getElementById('dailyvs'); if(!host) return;
  const ens=(rows||[]).filter(r=>r.model==='ensemble' && r.actual!=null && r.actual===r.actual);
  if(!ens.length){ host.innerHTML='<div class="empty">As each day is verified, its predicted vs NASA values appear here.</div>'; return; }
  ens.sort((a,b)=>a.forecast_date<b.forecast_date?1:(a.forecast_date>b.forecast_date?-1:(a.target<b.target?-1:1)));
  const conv=(t,v)=>v==null?null:(t==='PRmsl'?Math.round(v/100):(t==='TMP2m'?tNum(v):Math.round(v*10)/10));
  let h='<table><thead><tr><th>Date</th><th>Target</th><th>Forecast</th><th>NASA</th><th>Error</th></tr></thead><tbody>';
  ens.forEach(r=>{const t=r.target;
    const p=conv(t,r.prediction);
    const na=conv(t,(r.actual_nasa!=null?r.actual_nasa:r.actual));
    let err=Math.round((r.prediction-r.actual)*100)/100;
    if(t==='TMP2m'&&TUNIT==='F')err=Math.round((r.prediction-r.actual)*9/5*100)/100;
    const cls=Math.abs(r.prediction-r.actual)<=TOL[t]?'good':'off';
    h+=`<tr><td>${r.forecast_date}</td><td>${TLABEL[t]}</td><td>${p}</td>`+
       `<td>${na!=null?na:'<span style="color:var(--mute)">—</span>'}</td>`+
       `<td class="ferr ${cls}">${err>=0?'+':''}${err}</td></tr>`;});
  h+='</tbody></table>';
  host.innerHTML=h;
}
function exportDailyCSV(){
  fetch(q('/live_data')).then(r=>r.json()).then(d=>{
    const rows=(d.rows||[]).filter(r=>r.model==='ensemble'&&r.actual!=null&&r.actual===r.actual);
    const out=[['forecast_date','target','forecast','nasa','meteo','error']];
    rows.forEach(r=>{const na=r.actual_nasa!=null?r.actual_nasa:r.actual;
      out.push([r.forecast_date,r.target,r.prediction,na,r.actual_meteo!=null?r.actual_meteo:'',
                Math.round((r.prediction-r.actual)*1000)/1000]);});
    _downloadCSV('daily_vs_nasa_'+CITY+'_'+MODE+'.csv', out);
  }).catch(()=>{});
}
function renderMetrics(table){
  let best={}; TARGETS.forEach(t=>{let b=1e9,bm=null;MODELS.concat(['ensemble']).forEach(m=>{const c=table?.[t]?.[m];if(c&&c.RMSE!=null&&c.RMSE<b){b=c.RMSE;bm=m;}});best[t]=bm;});
  let h='<table><thead><tr><th>Model</th><th>Target</th><th>RMSE</th><th>MAE</th><th>R²</th><th>MAPE</th><th>n</th></tr></thead><tbody>';
  MODELS.concat(['ensemble']).forEach(m=>{TARGETS.forEach(t=>{
    const c=table?.[t]?.[m]; const isBest=(best[t]===m); const cls=isBest?'best':'';
    const badge=isBest?' <span class="bestbadge" title="best RMSE for this target">★ best</span>':'';
    h+=`<tr${isBest?' class="bestrow"':''}><td class="m">${mUP(m)}${badge}</td><td class="tg">${TLABEL[t]}</td>`+
       (c?`<td class="${cls}">${c.RMSE}</td><td>${c.MAE}</td><td>${c.R2}</td><td>${c.MAPE}</td><td class="ncol">${c.n!=null?c.n:'—'}</td>`:
          `<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>`)+`</tr>`;});});
  h+='</tbody></table>';
  document.getElementById('metrics').innerHTML=h;
}

const ledgerRows=[];
function addLedger(o){
  // only keep the ensemble row per (date,target) so the ledger is one clean line each
  if(o.model!=='ensemble') return;
  // replace existing row for same date+target+kind if present
  const i=ledgerRows.findIndex(r=>r.date===o.date&&r.target===o.target&&r.kind===o.kind);
  if(i>=0)ledgerRows[i]=o; else ledgerRows.unshift(o);
  // keep all rows (no culling) so nothing disappears from the ledger
  renderLedger();
}
function renderLedger(){
  if(!ledgerRows.length){document.getElementById('ledger').innerHTML='<div class="empty"><span class="eicon"><svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg></span>No forecasts yet.<span class="ehint">Press &ldquo;Run today&rsquo;s cycle&rdquo; to forecast forward.</span></div>';return;}
  // sort newest date first, verified above pending within a date
  const sorted=[...ledgerRows].sort((a,b)=> a.date<b.date?1:a.date>b.date?-1:(a.kind==='vf'?-1:1));
  let h='';
  sorted.forEach(r=>{
    const isT=r.target==='TMP2m';
    const val=r.target==='PRmsl'?Math.round(r.pred/100):(isT?tNum(r.pred):r.pred);
    const unit=isT?tUnitShort():(r.target==='RH2m'?'%':' hPa');
    if(r.kind==='fc'){
      h+=`<div class="lrow"><div><div class="ld">${r.date}</div><div class="lp">${TLABEL[r.target]} · forecast +${r.hz}d</div></div>
      <div style="text-align:right"><div class="lv">${val}${unit}</div><span class="tag pending">pending</span></div></div>`;
    }else{
      const av=r.target==='PRmsl'?Math.round(r.actual/100):(isT?tNum(r.actual):r.actual);
      const pv=r.target==='PRmsl'?Math.round(r.pred/100):(isT?tNum(r.pred):r.pred);
      const errDisp=isT&&TUNIT==='F'?Math.round(r.error*9/5*10)/10:r.error;
      const cls=Math.abs(r.error)<=TOL[r.target]?'good':'off';
      h+=`<div class="lrow"><div><div class="ld">${r.date}</div><div class="lp">${TLABEL[r.target]} · pred ${pv} vs actual ${av}</div></div>
      <div style="text-align:right"><div class="lv ferr ${cls}">${errDisp>=0?'+':''}${errDisp}</div><span class="tag ok">verified</span></div></div>`;
    }
  });
  document.getElementById('ledger').innerHTML=h;
}

// ---- live weights (now folded into each hyperparameter card) ----
// ev carries target+model; update that card's weight bar + value (+ q for quantum)
function updateWeight(model, mean, q, retrained, target){
  const base = (target?target:'TMP2m') + '_' + model;
  const wb=document.getElementById('hpwb_'+base);
  const wd=document.getElementById('hpwd_'+base);
  const card=document.getElementById('hp_'+base);
  if(wb){
    const off=Math.max(-46,Math.min(46,(mean||0)*6000));
    const dir=off>=0?1:-1;
    const overshoot=Math.max(-48,Math.min(48, off + dir*Math.max(14, Math.abs(off)*0.6)));
    // push forward (overshoot) so the adjustment is visible, then settle to the real value
    wb.style.transition='left .28s cubic-bezier(.34,1.56,.64,1)';
    wb.style.left=(50+overshoot)+'%';
    setTimeout(()=>{ wb.style.transition='left .55s cubic-bezier(.22,.61,.36,1)';
                     wb.style.left=(50+off)+'%'; }, 300);
  }
  if(wd)wd.textContent=`${retrained?'retrained · ':''}Δw ${mean>=0?'+':''}${mean}`;
  if(card){card.classList.remove('wpulse');void card.offsetWidth;card.classList.add('wpulse');}
  if(q!=null){const qv=document.getElementById('hpq_'+base);if(qv)qv.textContent=`q = ${q}`;}
}

// ---- activity log ----
function logLine(html){const wrap=document.getElementById('log');
  if(wrap.querySelector('.empty'))wrap.innerHTML='';
  const d=document.createElement('div');d.innerHTML=html;wrap.prepend(d);
  while(wrap.children.length>120)wrap.removeChild(wrap.lastChild);}


function restoreTrainMetrics(){
  fetch(q('/train_metrics')).then(r=>r.json()).then(d=>{
    // This rebuilds the full state from disk, so it must START from empty —
    // otherwise it re-appends the whole history every cycle and the arrays grow
    // without bound (the main cause of long-run lag).
    _ltmRowsInit=[]; _ltmRowsFT=[]; _ltmRowsRT=[];
    _ltmLiveSeries={}; _ltmLiveCount={};
    // per-epoch training history -> chart series (per target+model) + table
    const tr=d.train||{};
    Object.keys(tr).forEach(t=>Object.keys(tr[t]).forEach(m=>{
      (tr[t][m]||[]).forEach(p=>_ltmAddPoint(t,m,p.epoch,p.rmse,p.mae,p.r2));
      // add the final epoch of each to the table
      const last=(tr[t][m]||[]).slice(-1)[0];
      if(last)_ltmRowsInit.push({model:m,target:t,when:'ep '+last.epoch,cls:'',rmse:last.rmse,mae:last.mae,r2:last.r2});
    }));
    // live fine-tune/retrain rows + right-pane chart (cap to the most recent entries)
    (d.live||[]).slice(-400).forEach(e=>{
      const row={model:e.model,target:e.target,when:(e.date||''),
        cls:e.kind==='retrain'?'rt':'ft',rmse:e.rmse,mae:e.mae,r2:e.r2};
      if(e.kind==='retrain') _ltmRowsRT.unshift(row); else _ltmRowsFT.unshift(row);
      _ltmLiveAdd(e.target,e.model,e.rmse,e.mae,e.r2);
    });
    _ltmRedraw(true); _ltmLiveRedraw(true); _ltmTable();
  }).catch(()=>{});
}



// ---- hyperparameters panel ----
const HP_KEYS={seq_len:'seq len',hidden:'hidden',lr:'learn rate',wd:'weight decay',dropout:'dropout',q_depth:'q depth',n_qubits:'qubits'};
let HP_BASE={}; // "target|model" -> {key:oldValue}
const TLABELS={TMP2m:'Temperature',RH2m:'Humidity',PRmsl:'Pressure'};
function fmtHP(k,v){if(v==null)return '—';if(k==='lr'||k==='wd')return (+v).toExponential(1);if(k==='dropout')return (+v).toFixed(2);return v;}
function renderHParams(payload){
  const src=payload.source, data=payload.data;
  const srcEl=document.getElementById('hpsrc');
  srcEl.textContent=src==='tuned'?'tuned (Optuna)':'defaults';
  srcEl.className='hpsrc'+(src==='tuned'?' tuned':'');
  if(src==='tuned'){const bt=document.getElementById('b_tune');if(bt)bt.classList.add('done');}
  HP_BASE={};
  let h='';
  TARGETS.forEach(tgt=>{
    h+=`<div class="hpgroup"><div class="hpgh">${TLABELS[tgt]||tgt}</div><div class="hpgrid">`;
    MODELS_ALL.forEach(m=>{
      const key=tgt+'|'+m;
      const entry=data?.[tgt]?.[m]; const p=entry?entry.params:{}; HP_BASE[key]={};
      const q=(m[0]==='q'||m==='hqnn');
      const ready=(MODEL_READY[m]!==false);
      const modePill=`<span class="modepill ${MODE}">${MODE==='univariate'?'univariate':'multivariate'}</span>`;
      h+=`<div class="hpcard${ready?'':' notready'}" id="hp_${tgt}_${m}"><h3>${mUP(m)}${modelBadges(m)}${modePill}${ready?'':'<span class="soon">not available yet</span>'}</h3>`;
      Object.keys(HP_KEYS).forEach(k=>{ if((k==='q_depth'||k==='n_qubits')&&!q) return;
        HP_BASE[key][k]=p[k];
        h+=`<div class="hprow"><span class="k">${HP_KEYS[k]}</span>`+
           `<span class="v"><span class="vold" id="hpold_${tgt}_${m}_${k}"></span>`+
           `<span class="vnew" id="hpv_${tgt}_${m}_${k}">${ready?fmtHP(k,p[k]):'—'}</span>`+
           `<span class="hptag" id="hptag_${tgt}_${m}_${k}"></span></span></div>`;});
      h+=`<div class="hpval" id="hpval_${tgt}_${m}">${!ready?'model not added yet':(entry&&entry.val_mse!=null?('val MSE '+entry.val_mse):'not tuned yet')}</div>`;
      h+=`<div class="hptrial" id="hptrial_${tgt}_${m}"></div>`;
      // live weight activity, folded into the card
      h+=`<div class="hpweight"><div class="hpwtop"><span class="hpwlbl">live weight</span>`+
         `${q?`<span class="hpq" id="hpq_${tgt}_${m}">q = —</span>`:''}</div>`+
         `<div class="hpwbar"><span class="mid"></span><i id="hpwb_${tgt}_${m}"></i></div>`+
         `<div class="hpwd" id="hpwd_${tgt}_${m}">${ready?'no change yet':'—'}</div></div>`;
      h+=`</div>`;
    });
    h+=`</div></div>`;
  });
  document.getElementById('hparams').innerHTML=h;
  // if tuning happened, rebuild the old->new + changed/kept badges from the
  // saved baseline so they survive a refresh (not just during a live run)
  const base=payload.baseline;
  if(src==='tuned' && base){
    TARGETS.forEach(tgt=>MODELS.forEach(m=>{
      const cur=data?.[tgt]?.[m]?.params||{};
      const old=base?.[tgt]?.[m]?.params||{};
      Object.keys(HP_KEYS).forEach(k=>{
        if((k==='q_depth'||k==='n_qubits')&&m[0]!=='q')return;
        const nv=cur[k], ov=old[k];
        const oldEl=document.getElementById('hpold_'+tgt+'_'+m+'_'+k);
        const tagEl=document.getElementById('hptag_'+tgt+'_'+m+'_'+k);
        const newEl=document.getElementById('hpv_'+tgt+'_'+m+'_'+k);
        if(ov===undefined||nv===undefined)return;
        if(String(ov)!==String(nv)){
          if(oldEl){oldEl.textContent=fmtHP(k,ov);oldEl.style.display='inline';}
          if(tagEl){tagEl.textContent='changed';tagEl.className='hptag changed';}
          if(newEl)newEl.classList.add('isnew');
        }else{
          if(tagEl){tagEl.textContent='kept';tagEl.className='hptag kept';}
        }
      });
    }));
  }
}
function loadHParams(){fetch(q('/hparams')).then(r=>r.json()).then(renderHParams).catch(()=>{});}
function restoreLiveWeights(){
  fetch(q('/live_weights')).then(r=>r.json()).then(d=>{
    (d.latest||[]).forEach(e=>updateWeight(e.model,e.mean_change,e.qweight,e.retrained,e.target));
  }).catch(()=>{});
}
// live: during tuning, flash the value being tried on the right target+model card
function hpTrial(ev){
  const base=ev.target+'_'+ev.model;
  const card=document.getElementById('hp_'+base); if(card)card.classList.add('tuning');
  const tr=document.getElementById('hptrial_'+base);
  if(tr)tr.textContent=`trial ${ev.trial}/${ev.n_trials}${ev.pruned?' · pruned':''} · best ${ev.best_val!=null?ev.best_val:'—'}`;
  if(ev.params){Object.keys(ev.params).forEach(k=>{const el=document.getElementById('hpv_'+base+'_'+k);
    if(el){el.textContent=fmtHP(k,ev.params[k]);el.classList.remove('changed');void el.offsetWidth;el.classList.add('changed');}});}
}
// done: show old→new and a changed / kept tag per field on this target+model
function hpModelDone(ev){
  const base=ev.target+'_'+ev.model, key=ev.target+'|'+ev.model;
  const tr=document.getElementById('hptrial_'+base);
  if(tr)tr.textContent=`✓ best val ${ev.best_val}`;
  const val=document.getElementById('hpval_'+base);
  if(val)val.textContent='val MSE '+ev.best_val;
  const card=document.getElementById('hp_'+base); if(card)card.classList.remove('tuning');
  if(!ev.best_params)return;
  const baseHP=HP_BASE[key]||{};
  Object.keys(ev.best_params).forEach(k=>{
    const nv=ev.best_params[k], ov=baseHP[k];
    const newEl=document.getElementById('hpv_'+base+'_'+k);
    const oldEl=document.getElementById('hpold_'+base+'_'+k);
    const tagEl=document.getElementById('hptag_'+base+'_'+k);
    if(newEl)newEl.textContent=fmtHP(k,nv);
    const changed=(ov!==undefined && String(ov)!==String(nv));
    if(changed){
      if(oldEl){oldEl.textContent=fmtHP(k,ov);oldEl.style.display='inline';}
      if(tagEl){tagEl.textContent='changed';tagEl.className='hptag changed';}
      if(newEl){newEl.classList.add('isnew');}
    }else{
      if(oldEl)oldEl.style.display='none';
      if(tagEl){tagEl.textContent='kept';tagEl.className='hptag kept';}
    }
  });
}

function renderHistory(rows){
  window.__lastRows=rows;
  // show every model (not just ensemble) with both NASA and Open-Meteo truth
  const verified=rows.filter(r=>r.actual!=null&&r.actual===r.actual);
  const shown=verified.length?verified:rows;
  if(!shown.length){document.getElementById('history').innerHTML='<div class="empty">Forecasts will be listed here, with NASA + Open-Meteo ground truth as it arrives.</div>';return;}
  const order={lstm:0,qlstm:1,gru:2,qgru:3,ann:4,hqnn:5,ensemble:6};
  shown.sort((a,b)=>a.forecast_date<b.forecast_date?1:(a.forecast_date>b.forecast_date?-1:
    (a.target<b.target?-1:a.target>b.target?1:(order[a.model]-order[b.model]))));
  const conv=(t,v)=>v==null?null:(t==='PRmsl'?Math.round(v/100):(t==='TMP2m'?tNum(v):v));
  let h='<table><thead><tr><th>Date</th><th>Target</th><th>Model</th><th>Forecast</th><th>NASA</th><th>Open-Meteo</th><th>Error</th></tr></thead><tbody>';
  shown.forEach(r=>{const t=r.target;
    const p=conv(t,r.prediction);
    const na=conv(t,r.actual_nasa);
    const me=conv(t,r.actual_meteo);
    const hasA=r.actual!=null&&r.actual===r.actual;
    let err=hasA?Math.round((r.prediction-r.actual)*100)/100:null;
    if(hasA&&t==='TMP2m'&&TUNIT==='F')err=Math.round((r.prediction-r.actual)*9/5*100)/100;
    const cls=hasA&&Math.abs(r.prediction-r.actual)<=TOL[t]?'best':'';
    const scls=hasA?(err>0?'errpos':(err<0?'errneg':'')):'';   // warm/over vs cool/under
    h+=`<tr><td>${r.forecast_date}</td><td>${TLABEL[t]}</td><td class="m">${mUP(r.model)}</td><td>${p}</td>`+
       `<td>${na!=null?na:'<span style="color:var(--mute)">—</span>'}</td>`+
       `<td>${me!=null?me:'<span style="color:var(--mute)">—</span>'}</td>`+
       (hasA?`<td class="${cls} ${scls}">${err>=0?'+':''}${err}</td>`:`<td style="color:var(--mute)">pending</td>`)+`</tr>`;});
  h+='</tbody></table>';
  document.getElementById('history').innerHTML=h;
}