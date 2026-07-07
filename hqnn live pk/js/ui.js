// ui.js — shared chrome for both pages: collapsible sidebar + collapsible panels.
// Loaded by dashboard.html AND map.html (before the page-specific scripts).
// State is persisted in localStorage so collapse choices survive reloads.

// ---------------------------------------------------------------------------
// Sidebar collapse (icon-only rail <-> full)
// ---------------------------------------------------------------------------
(function sidebarInit(){
  var KEY = 'iwf_sb_collapsed';

  // ---- single source of truth for the sidebar (shared by every page) ----
  // Icons are inline SVG using currentColor, so they follow the sidebar theme.
  var ICON_SUN   = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4" fill="currentColor" stroke="none"/><line x1="12" y1="2.5" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="21.5"/><line x1="2.5" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="21.5" y2="12"/><line x1="5.4" y1="5.4" x2="7.1" y2="7.1"/><line x1="16.9" y1="16.9" x2="18.6" y2="18.6"/><line x1="18.6" y1="5.4" x2="16.9" y2="7.1"/><line x1="7.1" y1="16.9" x2="5.4" y2="18.6"/></svg>';
  var ICON_DASH  = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="5" rx="1.5"/><rect x="13" y="10" width="8" height="11" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/></svg>';
  var ICON_GLOBE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><line x1="3" y1="12" x2="21" y2="12"/><ellipse cx="12" cy="12" rx="4" ry="9"/></svg>';
  var ICON_MODELS= '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="1.5"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><line x1="9" y1="3" x2="9" y2="6"/><line x1="15" y1="3" x2="15" y2="6"/><line x1="9" y1="18" x2="9" y2="21"/><line x1="15" y1="18" x2="15" y2="21"/><line x1="3" y1="9" x2="6" y2="9"/><line x1="3" y1="15" x2="6" y2="15"/><line x1="18" y1="9" x2="21" y2="9"/><line x1="18" y1="15" x2="21" y2="15"/></svg>';

  function buildSidebar(){
    if(!document.body) return;                       // body not ready yet
    if(document.getElementById('sidebar')) return;   // already present
    var path = location.pathname;
    var isMap = /map/i.test(path);
    var isModels = /models/i.test(path);
    var isDash = !isMap && !isModels;
    var nav = document.createElement('nav');
    nav.className = 'sidebar'; nav.id = 'sidebar';
    nav.innerHTML =
      '<div class="sb-brand"><span class="sb-logo" title="Weather ML">'+ICON_SUN+'</span><span class="sb-name">Weather&nbsp;ML</span></div>'+
      '<a class="sb-item'+(isDash?' active':'')+'" href="dashboard.html" data-page="dashboard"><span class="sb-ico">'+ICON_DASH+'</span><span class="sb-label">Dashboard</span></a>'+
      '<a class="sb-item'+(isMap?' active':'')+'" href="map.html" data-page="map"><span class="sb-ico">'+ICON_GLOBE+'</span><span class="sb-label">Map</span></a>'+
      '<a class="sb-item'+(isModels?' active':'')+'" href="models.html" data-page="models"><span class="sb-ico">'+ICON_MODELS+'</span><span class="sb-label">Models</span></a>'+
      '<a class="sb-item" href="#" data-page="status" id="sbStatus"><span class="sb-ico">\u25f7</span><span class="sb-label">Status</span></a>';
    document.body.insertBefore(nav, document.body.firstChild);
  }

  function apply(collapsed){
    document.body.classList.toggle('sb-collapsed', !!collapsed);
    var ic = document.getElementById('sbToggleIco');
    if(ic) ic.textContent = collapsed ? '»' : '«';
  }
  function ensureToggle(){
    var sb = document.getElementById('sidebar'); if(!sb) return;
    if(document.getElementById('sbToggle')) return;
    var btn = document.createElement('button');
    btn.className = 'sb-toggle'; btn.id = 'sbToggle';
    btn.title = 'Collapse / expand sidebar';
    btn.innerHTML = '<span class="sb-ico" id="sbToggleIco">«</span><span class="sb-label">Collapse</span>';
    btn.addEventListener('click', function(){
      var now = !document.body.classList.contains('sb-collapsed');
      apply(now);
      try{ localStorage.setItem(KEY, now ? '1' : '0'); }catch(e){}
    });
    sb.appendChild(btn);
  }
  function getTheme(){ try{ return localStorage.getItem('iwf_theme')||'light'; }catch(e){ return 'light'; } }
  function applyTheme(t){
    document.documentElement.setAttribute('data-theme', t);
    try{ localStorage.setItem('iwf_theme', t); }catch(e){}
    var logo=document.querySelector('.sb-logo');
    if(logo) logo.setAttribute('title', t==='dark' ? 'Switch to light theme' : 'Switch to dark theme');
    try{ window.dispatchEvent(new CustomEvent('themechange',{detail:{theme:t}})); }catch(e){}
  }
  function boot(){
    buildSidebar();
    ensureToggle();
    var collapsed = false;
    try{ collapsed = localStorage.getItem(KEY) === '1'; }catch(e){}
    apply(collapsed);
    applyTheme(getTheme());                         // restore saved theme
    var logo = document.querySelector('.sb-logo');  // the sun = theme switch
    if(logo){
      logo.style.cursor='pointer';
      logo.addEventListener('click', function(e){
        e.preventDefault(); e.stopPropagation();
        applyTheme(document.documentElement.getAttribute('data-theme')==='dark' ? 'light' : 'dark');
      });
    }
    var brand = document.querySelector('.sb-brand');
    if(brand) brand.addEventListener('click', function(){ document.body.classList.toggle('sb-open'); });
  }
  buildSidebar();   // inject immediately so the rail is present before layout settles
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();

// ---------------------------------------------------------------------------
// Panel collapse  (every dashboard panel with data-collapse="key")
// markup:  <div class="panel" data-collapse="key">
//            <div class="panelhd"> ... <button class="collapse-btn" onclick="toggleCollapse(this)">collapse</button></div>
//            <div class="panelbody"> ... </div>
//          </div>
// ---------------------------------------------------------------------------
function _panelOf(btn){ return btn ? btn.closest('.panel') : null; }
function _panelBody(panel){ return panel ? panel.querySelector('.panelbody') : null; }

function _setCollapsed(panel, collapsed, animate){
  if(!panel) return;
  var body = _panelBody(panel); if(!body) return;
  var btn = panel.querySelector('.collapse-btn');
  if(collapsed){
    if(animate){ body.style.maxHeight = body.scrollHeight + 'px';
      requestAnimationFrame(function(){ body.style.maxHeight = '0px'; }); }
    else { body.style.maxHeight = '0px'; }
    panel.classList.add('collapsed');
    if(btn) btn.textContent = 'expand';
  } else {
    panel.classList.remove('collapsed');
    body.style.maxHeight = body.scrollHeight + 'px';
    if(btn) btn.textContent = 'collapse';
    // after the open animation, release the cap so dynamic content can grow,
    // and nudge any charts inside to resize to their now-visible container.
    var done = function(){ body.style.maxHeight = ''; body.removeEventListener('transitionend', done);
      try{ window.dispatchEvent(new Event('resize')); }catch(e){} };
    if(animate) body.addEventListener('transitionend', done);
    else done();
  }
}

function _collapseKey(panel){
  return 'iwf_collapse_' + (panel && panel.dataset ? (panel.dataset.collapse || panel.id || 'x') : 'x');
}

function toggleCollapse(btn){
  var panel = _panelOf(btn); if(!panel) return;
  var willCollapse = !panel.classList.contains('collapsed');
  _setCollapsed(panel, willCollapse, true);
  try{ localStorage.setItem(_collapseKey(panel), willCollapse ? '1' : '0'); }catch(e){}
}

// restore each panel's saved collapse state on load
(function restorePanelCollapse(){
  function boot(){
    document.querySelectorAll('.panel[data-collapse]').forEach(function(panel){
      var v = null; try{ v = localStorage.getItem(_collapseKey(panel)); }catch(e){}
      if(v === '1') _setCollapsed(panel, true, false);
    });
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();