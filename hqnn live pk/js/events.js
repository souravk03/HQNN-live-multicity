// events.js — SSE event handler (handle) + small helpers
function handle(ev){
  switch(ev.type){
    case 'cycle_start':
      window._cycleVerify={};   // reset the end-of-cycle toast collector
      window._hzTotal=ev.horizon_total||15;
      document.getElementById('curDate').textContent=fmtDate(ev.cursor);
      document.getElementById('curMeta').textContent=`Cycle ${ev.cycle} · forecasting forward from ${ev.cursor}`;
      setStatus('forecast','forecasting…'); setProgress(0,`day 0/${window._hzTotal}`,'cycle');
      // clear last cycle's 14-day values so we don't briefly show stale/identical
      // numbers while the new per-horizon forecasts stream in.
      window._cursor=ev.cursor;
      Object.keys(td3).forEach(k=>delete td3[k]);
      TARGETS.forEach(t=>{const d3=document.getElementById('d3_'+t);if(d3)d3.innerHTML='';});
      _renderThreeDay();
      logLine(`<span class="lt">cycle ${ev.cycle}</span> · forecasting from ${ev.cursor}`); break;
    case 'node':
      queueNode(ev.node);
      // short status word in the pill (2-3 words max) so it never overflows
      if(ev.status==='active'){
        const SHORT={download:'downloading',features:'features',forecast:'forecasting',
          verify:'verifying',confident:'checking truth',daily_ft:'fine-tuning',
          metadata:'saving',validate:'validating',split:'splitting'};
        if(SHORT[ev.node]) liveText.textContent=SHORT[ev.node];
        const ST={forecast:'forecast',verify:'verify',confident:'verify',daily_ft:'train',
          download:'forecast',features:'forecast'};
        if(ST[ev.node]) setStatus(ST[ev.node], SHORT[ev.node]||undefined);
      }
      // during data prep, walk the 6 feature-family blocks (the loop off
      // "Engineer 83 Features") so the download animation visits each one.
      if(ev.node==='features' && !_replaying){
        ['feat_lag','feat_roll','feat_diff','feat_cross','feat_cyc','feat_anom','features']
          .forEach(n=>{ if(document.getElementById('nd_'+n)) { _mQ.push(n); } });
        if(!_mTimer)_mPump();
      }
      break;
    case 'tune_init': liveText.textContent=`tuning · ${ev.n_trials} trials × ${ev.total} models`; clearTuneNodes();
      // repaint the panel for the CURRENT mode before live trial updates start, so
      // the baseline/old values and val_mse the cards diff against belong to THIS
      // mode (not whatever mode was last shown). Prevents one mode's values bleeding
      // into the other during a live tuning run.
      try{loadHParams();}catch(e){}
      logLine(`<span class="lt">tune</span> starting · ${ev.n_trials} trials per model`); break;
    case 'tune_model':
      if(ev.status==='start'){logLine(`<span class="lt">tune</span> ${ev.model} ${TLABEL[ev.target]||ev.target} (${ev.index}/${ev.total})`);
        document.querySelectorAll('.hpcard').forEach(c=>c.classList.remove('tuning'));
        const card=document.getElementById('hp_'+ev.target+'_'+ev.model);if(card)card.classList.add('tuning');}
      if(ev.status==='done'){hpModelDone(ev);
        logLine(`<span class="lok">✓ tuned</span> ${ev.model} ${TLABEL[ev.target]||ev.target} · val ${ev.best_val}`);}
      break;
    case 'tune_trial': hpTrial(ev); setTuneNode('tune_sample'); break;
    case 'tune_done':
      logLine(`<span class="lok">■ tuning complete</span> · best params saved`);
      document.querySelectorAll('.hpcard').forEach(c=>c.classList.remove('tuning'));
      liveText.textContent='tuned · ready to train';
      loadHParams();  // refresh panel to the new tuned values
      markDone('b_tune'); _resumePhase=null;
      document.getElementById('b_pause').disabled=true;document.getElementById('b_play').disabled=true;
      enable(); break;
    case 'feels_like':
      if(ev.horizon===1){ _feelsC=ev.feels_like; _feelsMethod=ev.method; renderFeels(); }
      break;
    case 'forecast':
      // hero big number = the ENSEMBLE day+1 forecast (mean of all models) — the
      // same value the strip and the reload-restore use. (Previously the live hero
      // showed lstm-only, which is noisier and disagreed with the cards / NASA.)
      if(ev.horizon===1&&ev.model==='ensemble'){const b=document.getElementById('big_'+ev.target);
        if(b){if(ev.target==='TMP2m'){b.dataset.c=ev.prediction; animateNumber(b, ev.prediction, v=>tNum(v));}
              else if(ev.target==='PRmsl'){ animateNumber(b, Math.round(ev.prediction/100), v=>Math.round(v)); }
              else { animateNumber(b, ev.prediction, v=>Math.round(v)); }}
        const s=document.getElementById('sub_'+ev.target); if(s)s.textContent=`forecast (${ev.date})`;}
      // cycle progress bar: advance by forecast horizon (last target of each day)
      if(ev.model==='ensemble'&&ev.target==='PRmsl'){const tot=window._hzTotal||15;
        setProgress(ev.horizon/tot, `day ${ev.horizon}/${tot}`, 'cycle');}
      // 15-day outlook strip uses the ensemble value per horizon
      if(ev.model==='ensemble'){const d3=document.getElementById('d3_'+ev.target);
        if(d3){const isT=ev.target==='TMP2m';
          const val=ev.target==='PRmsl'?Math.round(ev.prediction/100):(isT?tNum(ev.prediction):ev.prediction);
          const u=isT?tUnitShort():(ev.target==='RH2m'?'%':'');
          let sp=d3.querySelector('[data-h="'+ev.horizon+'"]');
          if(!sp){sp=document.createElement('span');sp.dataset.h=ev.horizon;d3.appendChild(sp);}
          if(isT)sp.dataset.c=ev.prediction;
          sp.textContent=`+${ev.horizon}d ${val}${u}`;
          [...d3.children].sort((a,b)=>a.dataset.h-b.dataset.h).forEach(c=>d3.appendChild(c));}
        // 15-day temperature box + chart line (store raw Celsius)
        if(ev.target==='TMP2m')setThreeDay(ev.date,ev.horizon,ev.prediction);
        pushChart(ev.target,ev.date,ev.prediction,null);}
      addLedger({kind:'fc',date:ev.date,target:ev.target,model:ev.model,pred:ev.prediction,unit:ev.unit,hz:ev.horizon});
      if(ev.model==='ensemble')logLine(`<span class="lt">forecast</span> ${ev.date} ${TLABEL[ev.target]} +${ev.horizon}d → ${ev.prediction}`);
      break;
    case 'verify':
      setStatus('verify','verifying…');
      addLedger({kind:'vf',date:ev.date,target:ev.target,model:ev.model,pred:ev.prediction,actual:ev.actual,error:ev.error,unit:ev.unit});
      if(ev.model==='ensemble'){
        pushChart(ev.target,ev.date,null,ev.actual_nasa!=null?ev.actual_nasa:ev.actual,ev.actual_meteo);
        if(ev.target==='TMP2m')setThreeDayActual(ev.date,ev.actual);
        // collect for the end-of-cycle summary toast
        if(!window._cycleVerify)window._cycleVerify={};
        window._cycleVerify[ev.target]={err:ev.error,date:ev.date};
        // hero sub-line: actual + error for the ENSEMBLE (the value shown big),
        // so the numbers are consistent (big number, actual, and error all match).
        const s=document.getElementById('sub_'+ev.target);
        if(s){const cls=Math.abs(ev.error)<=TOL[ev.target]?'good':'off';
          s.innerHTML=`actual ${ev.actual} · <span class="ferr ${cls}">${ev.error>=0?'+':''}${ev.error}</span>`;}
        // refresh history from server-side persisted data after verify
        fetch(q('/live_data')).then(r=>r.json()).then(d=>{if(d.rows){renderHistory(d.rows);try{renderDailyVs(d.rows);}catch(e){}}}).catch(()=>{});
      }
      logLine(`<span class="lok">verify</span> ${ev.date} ${TLABEL[ev.target]} ${ev.model} pred ${ev.prediction} vs ${ev.actual} (err ${ev.error>=0?'+':''}${ev.error})`);
      break;
    case 'readjust':
      updateWeight(ev.model,ev.mean_change,ev.qweight,ev.retrained,ev.target);
      logLine(`${ev.retrained?'<span class="lrt">retrain</span>':'<span class="lt">adapt</span>'} ${ev.model} ${TLABEL[ev.target]} · Δw ${ev.mean_change>=0?'+':''}${ev.mean_change}${ev.qweight!=null?' · q='+ev.qweight:''}`);
      liveText.textContent=ev.retrained?'retraining':'adapting';
      break;
    case 'metrics': renderMetrics(ev.table); break;
    case 'train_meta': window._maxEpochs=ev.epochs||window._maxEpochs; break;
    case 'train_progress':
      setStatus('train');
      { const mx=window._maxEpochs||120;
        setProgress(Math.min(1,ev.epoch/mx), `${(ev.model||'').toUpperCase()} · epoch ${ev.epoch}/${mx}`, 'train'); }
      ltmTrain(ev); break;
    case 'live_metric': ltmLive(ev); break;
    case 'paused':
      _resumePhase=ev.phase||'train';
      dot.classList.remove('on'); clearNodes();
      setStatus('pause', ev.msg||'paused'); hideProgress();
      logLine(`<span class="lt">paused</span> ${ev.msg||''}`);
      enable();
      document.getElementById('b_pause').disabled=true;
      document.getElementById('b_play').disabled=false;
      break;
    case 'info': logLine(`<span class="lt">info</span> ${ev.msg}`); break;
    case 'cycle_done':
      document.getElementById('curMeta').textContent=`Next run forecasts from ${ev.next_cursor} · ${ev.verified_total} day(s) verified`;
      setStatus('idle',`idle · ${ev.verified_total} verified`); hideProgress();
      logLine(`<span class="lok">cycle done</span> · next ${ev.next_cursor} · ${ev.verified_total} verified total`);
      // one-line summary toast — ALWAYS shown when a cycle finishes. If any day was
      // verified vs NASA this cycle, show the per-target errors; otherwise show a
      // short "finished, nothing to verify yet" message so you always get feedback.
      try{
        const v=window._cycleVerify||{}; const parts=[];
        const lbl={TMP2m:'TMP',RH2m:'RH',PRmsl:'PR'};
        const unit={TMP2m:'°',RH2m:'%',PRmsl:'hPa'};
        TARGETS.forEach(t=>{ if(v[t]){const e=v[t].err; parts.push(`${lbl[t]} ${e>=0?'+':''}${e}${unit[t]}`); }});
        if(parts.length){
          const anyDate=Object.values(v)[0].date;
          showToast(`${fmtDate(anyDate)} verified vs NASA · ${parts.join(' · ')}`);
        } else {
          showToast(`Cycle done · forecasting from ${ev.next_cursor} · no day verified yet`);
        }
      }catch(e){ try{ showToast('Cycle done'); }catch(_){} }
      try{ pushToast(ev.verified_total?`Cycle complete · ${ev.verified_total} verified`:'Cycle complete','good'); }catch(_){}
      dot.classList.remove('on'); clearNodes(); document.getElementById('b_run').classList.remove('busy'); enable();
      try{ if(typeof refreshPhase2==='function') refreshPhase2(); }catch(e){}
      break;
    case 'error': setStatus('error','error'); hideProgress(); dot.classList.remove('on'); document.getElementById('b_run').classList.remove('busy'); enable();
      try{ pushToast(ev.message||'Something went wrong','err'); }catch(_){}
      document.getElementById('curMeta').textContent=ev.message; logLine(`<span class="lrt">error</span> ${ev.message}`); break;
    case 'phase':
      if(ev.phase==='download'&&ev.status==='done'){markDone('b_download');window._hasData=true;document.getElementById('b_train').disabled=false;document.getElementById('b_fresh').disabled=false;document.getElementById('b_tune').disabled=false;logLine(`<span class="lok">data ready</span>`);
        setStatus('idle','data ready'); hideProgress();
        try{ const n=ev.downloaded, f=(ev.failed||[]).length; pushToast(f?`Downloaded ${n} states · ${f} failed`:`Downloaded ${n} states`, f?'info':'good'); }catch(_){}}
      if(ev.phase==='train'){if(ev.status==='active'){setStatus('train','training models…');}
        if(ev.status==='done'){markDone('b_train');window._trained=true;document.getElementById('b_run').disabled=false;setStatus('idle','ready');hideProgress();logLine(`<span class="lok">training done</span>`);
          try{ pushToast('Training complete','good'); }catch(_){}
          _resumePhase=null;document.getElementById('b_pause').disabled=true;document.getElementById('b_play').disabled=true;}}
      break;
    case 'download': document.getElementById('curMeta').textContent=ev.msg; logLine(`<span class="lt">nasa</span> ${ev.msg}`); break;
    case 'download_progress':
      setStatus('forecast', `downloading ${ev.label||ev.city} · ${ev.index}/${ev.total}`);
      setProgress((ev.index||0)/(ev.total||1), `${ev.index}/${ev.total} states`, 'cycle');
      if(ev.status==='done') logLine(`<span class="lok">${ev.label||ev.city}</span> ${ev.rows||''} rows`);
      else if(ev.status==='error') logLine(`<span class="lrt">failed</span> ${ev.label||ev.city}: ${ev.error||''}`);
      break;
    case 'train_model':
      if(ev.status==='start')document.getElementById('curMeta').textContent=`Training ${ev.model} · ${TLABEL[ev.target]||ev.target}…`;
      if(ev.status==='saved')logLine(`<span class="lok">✓</span> trained ${ev.model} ${TLABEL[ev.target]||ev.target}`);
      break;
  }
}
function fmtDate(s){if(!s)return '—';const d=new Date(s+'T00:00:00');return d.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric',year:'numeric'});}
function markDone(id){document.getElementById(id).classList.add('done');}

function disable(){['b_download','b_tune','b_train','b_run','b_reset'].forEach(i=>document.getElementById(i).disabled=true);}
function enable(){document.getElementById('b_download').disabled=false;
  document.getElementById('b_tune').disabled=!window._hasData;
  document.getElementById('b_train').disabled=!window._hasData;document.getElementById('b_fresh').disabled=!window._hasData;
  document.getElementById('b_run').disabled=!window._trained;
  document.getElementById('b_auto').disabled=!window._trained;   // auto-run available once trained
  document.getElementById('b_reset').disabled=false;
  // pause is only active while a phase runs; play only when a paused run can resume
  document.getElementById('b_play').disabled=(typeof _resumePhase==='undefined')||!_resumePhase;}
// ---- run model: a phase runs in the backend; we WATCH it via /run_stream. ----
// Starting a phase = POST to its start endpoint (fetch), then attach the viewer.
// On reload, we just attach again — the backend keeps running and /run_stream