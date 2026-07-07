// config.js — units, constants, model/target lists, dot+liveText handles
// MODELS = the models charts/tables iterate over. It is the FULL list (all
// trainable models incl. ann/hqnn) and is refreshed from /config at boot, so new
// models appear in the live-training graph, the accuracy table and hparams without
// code changes. MODELS_ALL is kept as an alias for the same list.
let MODELS=["lstm","qlstm","gru","qgru","ann","hqnn"], TARGETS=["TMP2m","RH2m","PRmsl"];
let MODELS_ALL=["lstm","qlstm","gru","qgru","ann","hqnn"];  // all 6 for the UI, display order LSTM,QLSTM,GRU,QGRU,ANN,HQNN
// model names are shown UPPERCASE in the UI (LSTM, QLSTM, …) while the internal keys stay lowercase.
function mUP(m){ return (m==null?'':String(m)).toUpperCase(); }
// Small C/Q badges shown next to a model name.
//   classical (lstm, gru, ann) -> green C
//   quantum   (qlstm, qgru)    -> purple Q
//   hybrid    (hqnn)           -> C + Q
function modelBadges(m){
  m=(m==null?'':String(m)).toLowerCase();
  if(m==='hqnn')              return '<span class="mbadge dual" title="quantum + classical"><b class="bc">C</b><span class="bx">+</span><b class="bq">Q</b></span>';
  if(m==='qlstm'||m==='qgru') return '<span class="mbadge q" title="quantum">Q</span>';
  return '<span class="mbadge c" title="classical">C</span>';
}

// ---- per-model TRAIN toggle (skip slow models when testing classical) ----
// MODEL_ENABLED[m] === false means: skip this model in train / fresh / tune / cycle.
// Forecasting still uses its last-trained weights if present. Persisted in
// localStorage so the choice survives reloads.
let MODEL_ENABLED={};
(function _loadModelEnabled(){
  try{ const s=localStorage.getItem('modelEnabled'); if(s)MODEL_ENABLED=JSON.parse(s)||{}; }catch(e){ MODEL_ENABLED={}; }
})();
function modelEnabled(m){ return MODEL_ENABLED[m]!==false; }   // default ON
function setModelEnabled(m,on){
  MODEL_ENABLED[m]=!!on;
  try{ localStorage.setItem('modelEnabled', JSON.stringify(MODEL_ENABLED)); }catch(e){}
}
function enabledModels(){ return MODELS_ALL.filter(m=>modelEnabled(m)); }
let HORIZON=15;  // forecast horizon (days), set from /config
// ---- temperature unit (°C / °F) ----
let TUNIT='C';
function cToDisplay(c){ if(c==null||isNaN(c))return c; return TUNIT==='F'? (c*9/5+32) : c; }
function tNum(c,dp=1){ const v=cToDisplay(c); return v==null||isNaN(v)?'—':Math.round(v*Math.pow(10,dp))/Math.pow(10,dp); }
function tUnitSym(){ return TUNIT==='F'?'°F':'°C'; }
function tUnitShort(){ return TUNIT==='F'?'°':'°'; }
let _feelsC=null, _feelsMethod=null;   // feels-like in °C + method
function renderFeels(){
  const el=document.getElementById('feels_TMP2m'); if(!el)return;
  if(_feelsC==null){el.innerHTML='';return;}
  const label=_feelsMethod==='wind_chill'?'wind chill':(_feelsMethod==='actual'?'air temp':'heat index');
  el.innerHTML=`<svg class="ficon" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" style="vertical-align:-2px"><path d="M14 14.8V5a2 2 0 0 0-4 0v9.8a4 4 0 1 0 4 0z"/></svg>feels like ${tNum(_feelsC)}${tUnitSym()} <span class="fmethod">(${label})</span>`;
}
function setUnit(u){
  TUNIT=u;
  document.querySelectorAll('#unitToggle .tg').forEach(b=>b.classList.toggle('active',b.dataset.unit===u));
  // hero temperature card
  const hb=document.getElementById('big_TMP2m');
  if(hb&&hb.dataset.c!=null)hb.textContent=tNum(+hb.dataset.c);
  document.querySelectorAll('.fcard.temp .u').forEach(e=>e.textContent=tUnitSym());
  // per-card 15-day strip (stored raw °C in data-c)
  const d3=document.getElementById('d3_TMP2m');
  if(d3)[...d3.children].forEach(sp=>{ if(sp.dataset.c!=null)
    sp.textContent=`+${sp.dataset.h}d ${tNum(+sp.dataset.c)}${tUnitShort()}`; });
  // 15-day box + chart label + history/ledger re-render
  _renderThreeDay();
  const cl=document.querySelector('#ch_TMP2m'); // label is sibling
  document.querySelectorAll('.chartbox .clabel').forEach(l=>{ if(l.textContent.startsWith('Temperature'))l.textContent='Temperature ('+tUnitSym()+')'; });
  renderLedger&&renderLedger();
  if(window.__lastRows)renderHistory(window.__lastRows);
  renderFeels();
}
const TLABEL={TMP2m:"Temp",RH2m:"Humidity",PRmsl:"Pressure"};
const TOL={TMP2m:1,RH2m:5,PRmsl:80};
const dot=document.getElementById('dot'), liveText=document.getElementById('liveText');