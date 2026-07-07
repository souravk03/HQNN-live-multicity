// strip.js — 14-day temperature strip
const td3={};
function _dateFromCursor(k){
  // date for horizon k (1-based) = cursor + (k-1) days
  if(!window._cursor)return null;
  const d=new Date(window._cursor+'T00:00:00');
  d.setDate(d.getDate()+(k-1));
  const y=d.getFullYear(),m=String(d.getMonth()+1).padStart(2,'0'),dd=String(d.getDate()).padStart(2,'0');
  return `${y}-${m}-${dd}`;
}
function _renderThreeDay(){
  // Skip horizon 1 (that's the current/cursor day already shown as the main
  // forecast) — show the NEXT 14 days, horizons 2..HORIZON, to avoid duplication.
  let h='';for(let k=2;k<=HORIZON;k++){const d=td3[k];
    const dateStr = d?d.date : _dateFromCursor(k);   // real date even before a forecast runs
    h+=`<div class="tdcard"><div class="tdd">${dateStr?fmtDate(dateStr):'+'+(k-1)+'d'}</div>
      <div class="tdv">${d?tNum(d.val):'—'}<span class="tdu">${tUnitSym()}</span></div></div>`;}
  document.getElementById('threeday').innerHTML=h;}
function setThreeDay(date,hz,val){
  if(hz>=1&&hz<=HORIZON)td3[hz]={date,val};   // only real horizons 1..HORIZON
  _renderThreeDay();}
function setThreeDayActual(date,actual){
  Object.keys(td3).forEach(k=>{if(td3[k]&&td3[k].date===date)td3[k].actual=actual;});
  _renderThreeDay();}

// ---- forecast history (model vs NASA) ----
