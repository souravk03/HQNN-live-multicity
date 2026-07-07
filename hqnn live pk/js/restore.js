// restore.js — run-stream attach/replay, phase control, reload restore
// replays everything so far, then tails live. Reload never restarts the work.
let _activePhase=null;   // 'download'|'tune'|'train'|'cycle'|null
let _replaying=false;    // true while reload backlog replays (jump, don't walk)
let _activeES=null;
function _setPauseUI(running){
  document.getElementById('b_pause').disabled=!running;
  document.getElementById('b_play').disabled=running || !_canResume();
}
function _canResume(){return _resumePhase!=null;}
let _resumePhase=null;

// Attach to the backend run stream (replay buffered events + tail live).
// `phase` is just for UI labelling; the stream carries whatever is running.
function attachRunStream(phase){
  if(_activeES){try{_activeES.close();}catch(e){}_activeES=null;}
  _activePhase=phase; dot.classList.add('on');
  if(phase==='tune'){liveText.textContent='tuning…';}
  else if(phase==='train'){liveText.textContent='training…';}
  else if(phase==='cycle'){liveText.textContent='running…'; jumpMainTo('forecast');}
  else if(phase==='download'){liveText.textContent='downloading…';}
  disable(); _setPauseUI(phase==='tune'||phase==='train');
  const es=new EventSource(q('/run_stream'));
  _activeES=es;
  _replaying=true;          // backlog is replaying — jump nodes, don't walk paths
  es.onmessage=e=>{try{const ev=JSON.parse(e.data);
    if(ev.type==='_replay_done'){_replaying=false; return;}  // backlog done; live walking resumes
    if(ev.type==='_end'){es.close();_activeES=null;const wasPhase=_activePhase;_activePhase=null;
      dot.classList.remove('on');_setPauseUI(false);hideProgress();
      boot();                                       // re-read status/hparams/metrics
      // auto-run: if enabled and the phase that just ended was a cycle, kick off
      // the next one (small delay so the bus marks the run finished first).
      if(_autoRun && wasPhase==='cycle'){
        liveText.textContent='auto-run: next cycle in 1s…';
        setTimeout(()=>{ if(_autoRun) _startCycle(); }, 1000);
      }
      return;}
    handle(ev);
    if(ev.type==='paused'){es.close();_activeES=null;_activePhase=null;dot.classList.remove('on');_setPauseUI(false);}
  }catch(x){console.error(x);}};
  es.onerror=()=>{es.close();_activeES=null;dot.classList.remove('on');enable();_setPauseUI(false);};
}

function startPhase(startUrl, phase, qbuild){
  // tell the backend to begin (idempotent — it won't double-start), then watch.
  // qbuild lets callers pass qm (include the enabled-model selection); defaults to q.
  const build = qbuild || q;
  fetch(build(startUrl)).then(r=>r.json()).then(()=>attachRunStream(phase))
    .catch(()=>{liveText.textContent='could not start — is the server running?';enable();});
}
function streamPhase(url){ // legacy shim used by download button
  const phase = url.indexOf('/download')>=0?'download':'run';
  startPhase(url.replace(location.origin,''), phase);
}
function openTrain(){ startPhase('/train','train', qm); }
function openTune(){ clearTuneNodes(); startPhase('/tune','tune', qm); }


// ---- restore everything from the persisted ledger (survives reload) ----
function restore(){
  fetch(q('/live_data')).then(r=>r.json()).then(d=>{
    try{ if(typeof refreshPhase2==='function') refreshPhase2(); }catch(e){}
    try{ renderDailyVs((d&&d.rows)||[]); }catch(e){}
    if(d.metrics&&Object.keys(d.metrics).length)renderMetrics(d.metrics);
    if(d.series)TARGETS.forEach(t=>{if(d.series[t])setChart(t,d.series[t]);});
    if(d.state&&d.state.feels_like!=null){_feelsC=d.state.feels_like;_feelsMethod=d.state.feels_method||'heat_index';renderFeels();}
    if(d.rows&&d.rows.length){
      renderHistory(d.rows);
      // rebuild ledger (ensemble rows)
      ledgerRows.length=0;
      d.rows.filter(r=>r.model==='ensemble').forEach(r=>{
        if(r.actual!=null&&r.actual===r.actual)
          addLedger({kind:'vf',date:r.forecast_date,target:r.target,model:'ensemble',pred:r.prediction,actual:r.actual,error:Math.round((r.prediction-r.actual)*100)/100,unit:''});
        else addLedger({kind:'fc',date:r.forecast_date,target:r.target,model:'ensemble',pred:r.prediction,unit:'',hz:1});});

      // --- rebuild the 14-day outlook from the CURRENT forward forecast ---
      // Each forward forecast is a batch of rows with horizon 1..15. The batch's
      // ORIGIN day = forecast_date of its horizon-1 row, i.e. forecast_date minus
      // (horizon-1) days for any row. Grouping by this implied origin reconstructs
      // the true batches even when made_on is uniform/stale (older ledgers). We then
      // pick the batch whose origin is the current cursor (or the latest origin).
      const ensRows=d.rows.filter(r=>r.model==='ensemble');
      let last3=[];
      if(ensRows.length){
        const cursor=(d.state&&d.state.cursor)?d.state.cursor:(window._cursor||null);
        const originOf=r=>{
          const h=(r.horizon||1)-1;
          const dt=new Date(r.forecast_date+'T00:00:00'); dt.setDate(dt.getDate()-h);
          return dt.toISOString().slice(0,10);
        };
        const byOrigin={};
        ensRows.forEach(r=>{const k=originOf(r);(byOrigin[k]=byOrigin[k]||[]).push(r);});
        const origins=Object.keys(byOrigin).sort();   // ascending date
        let chosen=null;
        if(cursor && byOrigin[cursor]) chosen=cursor;          // exact match to cursor
        if(!chosen && cursor){                                  // closest origin >= cursor
          const future=origins.filter(o=>o>=cursor);
          chosen = future.length?future[0]:origins[origins.length-1];
        }
        if(!chosen) chosen=origins[origins.length-1];           // latest origin
        const batch=byOrigin[chosen].slice();
        batch.sort((a,b)=>((a.horizon||0)-(b.horizon||0))|| (a.forecast_date<b.forecast_date?-1:1));
        last3=[...new Set(batch.map(r=>r.forecast_date))];
        window.__batchRows=batch;
      }
      // per-card 14-day strips
      TARGETS.forEach(t=>{const d3=document.getElementById('d3_'+t);if(d3)d3.innerHTML='';});
      // reset td3
      Object.keys(td3).forEach(k=>delete td3[k]);
      last3.forEach((fdate,i)=>{
        const hz=i+1;
        TARGETS.forEach(t=>{
          const src=window.__batchRows||ensRows;
          const row=src.find(r=>r.forecast_date===fdate&&r.target===t);
          if(!row)return;
          const isT=t==='TMP2m';
          const val=t==='PRmsl'?Math.round(row.prediction/100):(isT?tNum(row.prediction):Math.round(row.prediction*10)/10);
          const u=isT?tUnitShort():(t==='RH2m'?'%':'');
          const d3=document.getElementById('d3_'+t);
          if(d3){let sp=document.createElement('span');sp.dataset.h=hz;if(isT)sp.dataset.c=row.prediction;sp.textContent=`+${hz}d ${val}${u}`;d3.appendChild(sp);}
          // hero big number = nearest-day (hz 1) ensemble
          if(hz===1){const b=document.getElementById('big_'+t);
            if(b){if(isT){b.dataset.c=row.prediction;b.textContent=tNum(row.prediction);}
                  else b.textContent=(t==='PRmsl')?Math.round(row.prediction/100):Math.round(row.prediction*10)/10;}
            const s=document.getElementById('sub_'+t);if(s)s.textContent=`forecast (${fdate})`;}
          if(isT){const act=(row.actual!=null&&row.actual===row.actual)?row.actual:null;
            td3[hz]={date:fdate,val:row.prediction,actual:act};}   // store raw °C
        });
      });
      if(Object.keys(td3).length)_renderThreeDay();
    }
  }).catch(()=>{});
}