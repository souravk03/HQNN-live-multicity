// main.js — global state, buttons, config load, boot, idle poll
// ---- collapsible pipeline diagram (persisted) ----
function applyDiagramState(){
  const collapsed = localStorage.getItem('diagramCollapsed')==='1';
  const sm=document.getElementById('sysmap'), btn=document.getElementById('diagramToggle');
  if(sm)sm.style.display=collapsed?'none':'';
  if(btn)btn.textContent=collapsed?'expand':'collapse';
}
function toggleDiagram(){
  const cur=localStorage.getItem('diagramCollapsed')==='1';
  try{localStorage.setItem('diagramCollapsed', cur?'0':'1');}catch(e){}
  applyDiagramState();
}
// ---- cycle summary toast ----
let _toastTimer=null;
function showToast(msg){
  // "Latest cycle" is now a PERSISTENT block under the forecast (not a transient
  // toast). Write the summary there and remember it so it survives a page reload.
  setCycleSummary(msg, true);
}
function setCycleSummary(msg, stamp){
  const el=document.getElementById('cycleSum'); if(!el)return;
  const when=stamp?new Date().toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):null;
  el.innerHTML=`<span class="cycsum-msg">${msg}</span>`+(when?`<span class="cycsum-when">updated ${when}</span>`:'');
  if(stamp){ try{ localStorage.setItem('iwf_lastCycle', JSON.stringify({msg, when})); }catch(e){} }
}
function restoreCycleSummary(){
  try{
    const s=localStorage.getItem('iwf_lastCycle'); if(!s)return;
    const o=JSON.parse(s); const el=document.getElementById('cycleSum'); if(!el||!o||!o.msg)return;
    el.innerHTML=`<span class="cycsum-msg">${o.msg}</span>`+(o.when?`<span class="cycsum-when">updated ${o.when}</span>`:'');
  }catch(e){}
}

// ---- transient toasts (top-right) — brief, non-persistent confirmations ----
function pushToast(msg, kind){
  const wrap=document.getElementById('toasts'); if(!wrap)return;
  kind=kind||'info';
  const ic = kind==='err'?'✕' : kind==='good'?'✓' : 'ℹ';
  const t=document.createElement('div'); t.className='toast '+kind;
  t.innerHTML='<span class="tic"></span><span class="tmsg"></span>';
  t.querySelector('.tic').textContent=ic; t.querySelector('.tmsg').textContent=msg;
  wrap.appendChild(t);
  requestAnimationFrame(()=>t.classList.add('in'));
  setTimeout(()=>{ t.classList.remove('in'); setTimeout(()=>t.remove(),320); }, 4200);
}

// ---- colour-coded run-status chip (reuses the live-pill) ----
const _STATUS={ idle:['st-idle',false], train:['st-train',true], forecast:['st-forecast',true],
  verify:['st-verify',true], pause:['st-pause',false], error:['st-error',false], done:['st-done',false] };
function setStatus(state, text){
  const p=document.getElementById('livePill'); if(!p)return;
  const d=_STATUS[state]||_STATUS.idle;
  p.className='live-pill '+d[0]+(d[1]?' busy':'');
  if(text!=null){ const lt=document.getElementById('liveText'); if(lt)lt.textContent=text; }
}

// ---- thin progress bar (epoch X/Y during training, day x/N during a cycle) ----
function setProgress(frac, label, kind){
  const w=document.getElementById('progwrap'), f=document.getElementById('progfill'), l=document.getElementById('proglbl');
  if(!w)return;
  w.classList.add('on'); w.classList.remove('train','cycle'); if(kind)w.classList.add(kind);
  if(f)f.style.width=(Math.max(0,Math.min(1,frac||0))*100).toFixed(1)+'%';
  if(l)l.textContent=label||'';
}
function hideProgress(){ const w=document.getElementById('progwrap'); if(!w)return;
  w.classList.remove('on'); const f=document.getElementById('progfill'); if(f)f.style.width='0%'; }

// ---- count-up animation for the forecast big numbers ----
function animateNumber(el, to, fmt){
  if(!el)return; fmt=fmt||(v=>v);
  const target=parseFloat(to); const from=parseFloat(el.dataset.cur);
  if(isNaN(target)){ el.textContent=fmt(to); return; }
  if(isNaN(from) || Math.abs(from-target)<0.05){ el.dataset.cur=target; el.textContent=fmt(target); return; }
  const t0=performance.now(), dur=520;
  (function step(now){ let p=Math.min(1,(now-t0)/dur); p=1-Math.pow(1-p,3);
    el.textContent=fmt(from+(target-from)*p);
    if(p<1) requestAnimationFrame(step); else { el.dataset.cur=target; el.textContent=fmt(target); }
  })(performance.now());
}
document.getElementById('b_download').onclick=()=>{window._hasData=true;
  setStatus('forecast','downloading all states…'); setProgress(0,'starting…','cycle');
  startPhase('/download_all','download');};
document.getElementById('b_tune').onclick=()=>{_resumePhase=null;openTune();};
document.getElementById('b_train').onclick=()=>{_resumePhase=null;
  // RESUME: do NOT pass fresh=1, so the backend keeps its train_state and skips
  // models that already finished — picks up where it stopped (e.g. after temp +
  // humidity were done, it continues with pressure). Use the "fresh" control to
  // force a full restart from scratch.
  setStatus('train','starting training…');
  fetch(qm('/train')).then(()=>attachRunStream('train'))
    .catch(()=>{setStatus('error','could not start training');enable();});};

document.getElementById('b_fresh').onclick=()=>{
  if(!confirm('Fresh start will WIPE all training checkpoints and retrain every model from scratch (epoch 1). This discards any in-progress training. Continue?'))return;
  _resumePhase=null;
  setStatus('train','fresh start…');
  fetch(qm('/train')+'&fresh=1').then(()=>attachRunStream('train'))
    .catch(()=>{setStatus('error','could not start training');enable();});};

document.getElementById('b_pause').onclick=()=>{
  fetch(q('/pause')).then(()=>{liveText.textContent='pausing… (finishing current checkpoint)';
    document.getElementById('b_pause').disabled=true;});};
document.getElementById('b_play').onclick=()=>{
  if(!_resumePhase)return;
  const ph=_resumePhase;
  fetch(q('/resume')).then(()=>{ if(ph==='tune')openTune(); else openTrain(); });};
let _autoRun=false;
function _startCycle(){
  if(_activePhase){return;}   // respect the event bus — never start while one runs
  document.getElementById('b_run').classList.add('busy');
  setStatus('forecast','running '+(MODE==='multivariate'?'multivariate':'univariate')+' cycle…');
  startPhase('/live_cycle','cycle', qm);   // qm() = the CURRENTLY SELECTED mode + state only
}
document.getElementById('b_run').onclick=()=>{ _startCycle(); };
document.getElementById('b_auto').onclick=()=>{
  _autoRun=!_autoRun;
  const btn=document.getElementById('b_auto');
  btn.classList.toggle('on',_autoRun);
  btn.textContent=_autoRun?'⏸ Auto':'↻ Auto';
  if(_autoRun){
    logLine('<span class="lt">auto-run on</span> cycles will repeat until you turn it off');
    if(!_activePhase) _startCycle();   // start immediately if idle
  }else{
    logLine('<span class="lt">auto-run off</span> will stop after the current cycle');
  }
};
document.getElementById('b_reset').onclick=()=>{fetch(q('/live_reset')).then(()=>{ledgerRows.length=0;renderLedger();
  TARGETS.forEach(t=>{document.getElementById('big_'+t).textContent='—';document.getElementById('sub_'+t).textContent='awaiting forecast';
    const d3=document.getElementById('d3_'+t);if(d3)d3.innerHTML='';
    if(CHARTS[t]){CHARTS[t].data.labels=[];CHARTS[t].data.datasets[0].data=[];CHARTS[t].data.datasets[1].data=[];CHARTS[t].update('none');}});
  Object.keys(td3).forEach(k=>delete td3[k]);
  Object.keys(td3).forEach(k=>delete td3[k]); _renderThreeDay();
  document.getElementById('history').innerHTML='<div class="empty">Verified forecasts will be listed here as ground truth arrives.</div>';
  document.getElementById('metrics').innerHTML='<div class="empty">Run a cycle to verify forecasts and build accuracy.</div>';refreshLive();});};

function refreshLive(){fetch(q('/live_status')).then(r=>r.json()).then(s=>{
  document.getElementById('curDate').textContent=fmtDate(s.cursor);
  document.getElementById('curMeta').textContent=s.cycles>0?
    `Next run forecasts from ${s.cursor} · ${s.verified} day(s) verified`:
    `Ready · first run forecasts from ${s.cursor}`;
  if(s.cursor){window._cursor=s.cursor; _renderThreeDay();}  // show real dates on the cards
  refreshHorizon();
}).catch(()=>{});}

// ---- global selected state + mode (drives every fetch) ----
let CITY='delhi', MODE='multivariate', APPCFG=null;
let MODEL_READY={};   // model -> trainable?
let MODE_READY={};    // mode -> ready?
function q(url){const sep=url.includes('?')?'&':'?';return `${url}${sep}city=${CITY}&state=${CITY}&mode=${MODE}`;}
// same as q() but also passes the enabled-model selection, so train/tune/cycle skip
// any models the user toggled off in the live-training panel.
function qm(url){return q(url)+`&models=${encodeURIComponent(enabledModels().join(','))}`;}
function loadConfig(){
  return fetch('/config').then(r=>r.json()).then(cfg=>{
    APPCFG=cfg;
    (cfg.models||[]).forEach(m=>MODEL_READY[m.key]=!!m.trainable);
    (cfg.modes||[]).forEach(m=>MODE_READY[m.key]=!!m.ready);
    if(cfg.horizon)HORIZON=cfg.horizon;
    // models list for the UI. Both MODELS and MODELS_ALL are the FULL list (every
    // model the server reports) so charts, the accuracy table and hparams always
    // show all of them, incl. ann/hqnn. Training/skipping is controlled separately
    // by the per-model toggles (enabledModels()), NOT by this list.
    if(cfg.models&&cfg.models.length){
      const all=cfg.models.map(m=>m.key);
      MODELS_ALL=all;
      MODELS.length=0; all.forEach(k=>MODELS.push(k));
    }
    // state dropdown — only the operational cities (the map shows the full country)
    const sel=document.getElementById('stateSel');
    const enabledStates=(cfg.states||[]).filter(s=>s.enabled);
    sel.innerHTML=enabledStates.map(s=>`<option value="${s.key}">${s.label}</option>`).join('');
    const firstEnabled=(enabledStates[0]||{key:'delhi'}).key;
    // restore saved selection if it's still valid/enabled
    let savedCity=null, savedMode=null;
    try{savedCity=localStorage.getItem('iwf_city');savedMode=localStorage.getItem('iwf_mode');}catch(e){}
    const cityOk=savedCity && (cfg.states.find(s=>s.key===savedCity&&s.enabled));
    CITY=cityOk?savedCity:firstEnabled; sel.value=CITY;
    sel.onchange=()=>{CITY=sel.value; switchContext();};
    // mode toggle — all enabled modes are selectable; not-ready ones show a preview note
    const modeOk=savedMode && (cfg.modes.find(x=>x.key===savedMode&&(x.enabled)));
    MODE = modeOk ? savedMode : (cfg.default_mode||'multivariate');
    document.querySelectorAll('#modeToggle .tg').forEach(btn=>{
      const m=btn.dataset.mode, info=cfg.modes.find(x=>x.key===m);
      btn.disabled=!(info&&info.enabled);
      if(info&&!info.ready)btn.title='preview — models not trained for this mode yet';
      btn.classList.toggle('active', m===MODE);   // reflect restored mode
      btn.onclick=()=>{if(btn.disabled)return;MODE=m;
        document.querySelectorAll('#modeToggle .tg').forEach(b=>b.classList.toggle('active',b===btn));
        switchContext();};
    });
    // unit toggle (°C / °F)
    document.querySelectorAll('#unitToggle .tg').forEach(btn=>{
      btn.onclick=()=>setUnit(btn.dataset.unit);
    });
  }).catch(()=>{});
}
function switchContext(){
  // remember the selection so a reload restores the same state+mode
  try{localStorage.setItem('iwf_city',CITY);localStorage.setItem('iwf_mode',MODE);}catch(e){}
  // crossfade: fade the content out, swap underneath, fade back in — no blank flash
  document.body.classList.add('switching');
  setTimeout(()=>{
    try{ if(_activeES){_activeES.close(); _activeES=null;} _activePhase=null; }catch(e){}
    try{ clearNodes && clearNodes(); }catch(e){}
    _ltmSeries={}; _ltmRowsInit=[]; _ltmRowsFT=[]; _ltmRowsRT=[];
    try{ _ltmLiveSeries={}; _ltmLiveCount={}; }catch(e){}   // drop prior mode's live curves
    try{ if(typeof _ltmLiveChart!=='undefined' && _ltmLiveChart){_ltmLiveChart.destroy(); _ltmLiveChart=null;} }catch(e){}
    try{ if(typeof _ltmChart!=='undefined' && _ltmChart){_ltmChart.destroy(); _ltmChart=null;} }catch(e){}
    try{ Object.keys(CHARTS).forEach(t=>{CHARTS[t].destroy(); delete CHARTS[t];}); }catch(e){}  // forecast charts repaint for new mode
    boot();
    // fade back in once the new mode's (windowed, light) data has had a moment to paint
    setTimeout(()=>document.body.classList.remove('switching'), 360);
  }, 160);
}
// sticky header condenses after a little scroll so controls stay compact + reachable
window.addEventListener('scroll', function(){
  const h=document.querySelector('.stickybar'); if(!h)return;
  h.classList.toggle('condensed', window.scrollY>40);
}, {passive:true});

function boot(){
  fetch(q('/status')).then(r=>r.json()).then(s=>{
    window._hasData=s.data; window._trained=s.trained;
    try{
      if(s.data){markDone('b_download');document.getElementById('b_train').disabled=false;document.getElementById('b_fresh').disabled=false;document.getElementById('b_tune').disabled=false;
        document.getElementById('curMeta').textContent='Data ready · train or tune';}
      if(s.trained){markDone('b_train');document.getElementById('b_run').disabled=false;}
      enable();
    }catch(e){console.error('status restore',e);}
    try{refreshLive();}catch(e){console.error('refreshLive',e);}
    try{_renderThreeDay();}catch(e){}      // show the 15 forecast boxes (empty until data)
    try{_ltmTable();}catch(e){}            // render the 6 model cards + toggles immediately
    try{renderModelParams();}catch(e){}    // model parameter counts (mode-aware: MV/UV)
    try{restoreCycleSummary();}catch(e){}  // bring back the last cycle summary block
    try{applyDiagramState();}catch(e){}    // restore collapsed/expanded pipeline diagram
    try{restore();}catch(e){console.error('restore',e);}
    try{restoreTrainMetrics();}catch(e){console.error('restoreTrainMetrics',e);}
    fetch(q('/hparams')).then(r=>r.json()).then(p=>{renderHParams(p);restoreLiveWeights();}).catch(()=>{});
    fetch(q('/run_state')).then(r=>r.json()).then(rs=>{
      if(rs && rs.running){
        // a phase is running in the backend. RE-ATTACH only if it belongs to the
        // mode we're now viewing — /run_stream replays everything so far then tails
        // live. If it belongs to the OTHER mode, leave it running untouched and just
        // show this mode's static data (with a note), so switching modes never
        // interrupts or cross-contaminates an in-progress run.
        const runMode = rs.running_mode || 'multivariate';
        if(runMode === MODE){
          if(window._bgPoll){clearInterval(window._bgPoll);window._bgPoll=null;}
          attachRunStream(rs.running_phase||'run');
          return;
        } else {
          const other = runMode==='univariate'?'Univariate':'Multivariate';
          liveText.textContent=`${other} ${rs.running_phase||'run'} still running in background`;
          logLine(`<span class="lt">note</span> a ${rs.running_phase||'run'} is running in ${other} mode — switch back to watch it live`);
          // fall through: this mode's static panels are already loaded above
        }
      }
      if(rs && rs.paused){
        const note=(rs.train_state&&rs.train_state.note)||'';
        _resumePhase = note.indexOf('tune')>=0 ? 'tune' : (rs.phase==='tune'?'tune':'train');
        liveText.textContent='paused — press Resume to continue';
        document.getElementById('b_play').disabled=false;
        document.getElementById('b_pause').disabled=true;
        logLine(`<span class="lt">paused</span> resume available${note?': '+note:''}`);
      }
    }).catch(()=>{});
  }).catch(()=>{document.getElementById('curMeta').textContent='Start the server (python server.py), then reload.';});
}

// auto-refresh the live (cycle) state while idle, so the page stays current
// without needing a manual click. Skips while a phase stream is active.
let _idlePoll=null;
function startIdlePoll(){
  if(_idlePoll)return;
  _idlePoll=setInterval(()=>{
    if(_activeES) return;                 // a run is streaming — don't interfere
    try{refreshLive();}catch(e){}
    try{restore();}catch(e){}             // refresh ledger/3-day/history from disk
  }, 20000);
}

if(document.readyState==='loading')
  document.addEventListener('DOMContentLoaded',()=>{loadConfig().then(()=>{boot();startIdlePoll();});});
else { loadConfig().then(()=>{boot();startIdlePoll();}); }