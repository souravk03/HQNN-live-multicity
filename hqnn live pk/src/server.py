"""
Full-pipeline live dashboard server.

Run:
    cd src
    pip install fastapi uvicorn
    python server.py
Open http://localhost:8000 and use the Download -> Train -> Backtest buttons.

Endpoints (all Server-Sent Events except /):
    GET /                 -> dashboard.html
    GET /status           -> JSON: does cache/models exist?
    GET /download?city=   -> stream NASA download progress
    GET /train?city=      -> stream training progress (per model/epoch)
"""
import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import CITIES, STATES, DATA_DIR, MODEL_DIR

app = FastAPI()
HERE = Path(__file__).resolve().parent
BASE = HERE.parent                      # the realtime/ folder (holds dashboard.html, css/, js/)
DASHBOARD = BASE / "dashboard.html"

# serve the split CSS/JS so <link href="css/..."> and <script src="js/..."> resolve.
# (If a folder is missing we skip its mount so the server still starts.)
for _sub in ("css", "js"):
    _d = BASE / _sub
    if _d.is_dir():
        app.mount(f"/{_sub}", StaticFiles(directory=str(_d)), name=_sub)


import math

def _json_safe(obj):
    """Recursively replace NaN / +-Infinity with None so json.dumps never raises
    'Out of range float values are not JSON compliant'. Also coerces numpy
    scalars to plain Python types."""
    # numpy scalar -> python
    if hasattr(obj, "item") and not isinstance(obj, (list, dict, tuple, str, bytes)):
        try:
            obj = obj.item()
        except Exception:
            pass
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# --- run event bus -----------------------------------------------------------
# A phase (download/tune/train/cycle) runs ONCE in a background thread and
# appends every event to a shared buffer. Any number of viewers (the page that
# started it, or a page that reloaded mid-run) subscribe via /run_stream, which
# REPLAYS the buffered events then tails live. The work is decoupled from the
# browser: reloading never restarts or interrupts it.
class _RunBus:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = None
        self.mode = None          # which mode (multivariate/univariate) this run is for
        self.running = False
        self.events = []          # full event log for the current/last run
        self.subs = []            # list of queue.Queue for live tailing
        self.seq = 0

    def start(self, phase, gen_factory, mode=None):
        with self.lock:
            if self.running:
                return False       # a run is already in progress — don't start another
            self.phase = phase
            self.mode = mode
            self.running = True
            self.events = []
            self.seq += 1
        def worker():
            try:
                for event in gen_factory():
                    self._emit(event)
            except Exception as e:
                self._emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                self._emit({"type": "_end"})
                with self.lock:
                    self.running = False
        threading.Thread(target=worker, daemon=True).start()
        return True

    def _emit(self, event):
        with self.lock:
            self.events.append(event)
            subs = list(self.subs)
        for qsub in subs:
            try: qsub.put(event)
            except Exception: pass

    def subscribe(self):
        """Return (backlog, queue). Backlog is everything so far; queue tails new."""
        with self.lock:
            backlog = list(self.events)
            q = queue.Queue()
            self.subs.append(q)
            running = self.running
        return backlog, q, running

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def state(self):
        with self.lock:
            last = self.events[-1] if self.events else None
            return {"running": self.running, "phase": self.phase, "mode": self.mode,
                    "n_events": len(self.events), "last_event": last}

BUS = _RunBus()

# Headers that stop intermediaries (nginx, Cloudflare, etc.) from buffering the
# event stream — the usual reason live updates only appear after a page reload.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

def _start_phase(phase, gen_factory, mode=None):
    """Start a phase in the background (idempotent). Returns JSON status."""
    ok = BUS.start(phase, gen_factory, mode=mode)
    return JSONResponse({"started": ok, "phase": phase, "mode": mode,
                         "already_running": (not ok)})

def _sse_from_generator(gen_factory, phase=None):
    """Legacy direct-stream (used by backtest). Prefer _start_phase + /run_stream."""
    q: "queue.Queue" = queue.Queue()
    SENTINEL = object()
    def worker():
        try:
            for event in gen_factory():
                q.put(event)
        except Exception as e:
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            q.put(SENTINEL)
    threading.Thread(target=worker, daemon=True).start()
    def event_gen():
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            yield f"data: {json.dumps(item)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/", response_class=HTMLResponse)
def index():
    return DASHBOARD.read_text(encoding="utf-8")


@app.get("/dashboard.html", response_class=HTMLResponse)
def dashboard_html():
    return DASHBOARD.read_text(encoding="utf-8")


@app.get("/map.html", response_class=HTMLResponse)
def map_html():
    p = BASE / "map.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return HTMLResponse("<h3>map.html not found</h3>", status_code=404)


@app.get("/map", response_class=HTMLResponse)
def map_alias():
    return map_html()


@app.get("/models.html", response_class=HTMLResponse)
def models_html():
    p = BASE / "models.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return HTMLResponse("<h3>models.html not found</h3>", status_code=404)


@app.get("/models", response_class=HTMLResponse)
def models_alias():
    return models_html()


@app.get("/config")
def get_config():
    """States, modes, and models with enabled flags so the UI can populate the
    dropdown/toggle and grey out what isn't ready yet."""
    import config as C
    states = []
    for key, info in C.STATES.items():
        states.append({"key": key, "label": info["label"],
                       "enabled": key in C.STATES_ENABLED})
    states.sort(key=lambda s: (not s["enabled"], s["label"]))  # enabled first
    modes = [{"key": k, "label": v["label"], "enabled": k in C.MODES_ENABLED,
              "ready": v.get("ready", True)}
             for k, v in C.MODES.items()]
    models = [{"key": m, "label": C.MODEL_INFO[m]["label"],
               "kind": C.MODEL_INFO[m]["kind"], "enabled": C.MODEL_INFO[m]["enabled"],
               "trainable": C.MODEL_INFO[m].get("trainable", False)}
              for m in C.ALL_MODEL_NAMES]
    return JSONResponse({"states": states, "modes": modes, "models": models,
                         "default_mode": C.DEFAULT_MODE,
                         "targets": C.TARGETS,
                         "config_version": "6models-uv-enabled",
                         "trainable_models": [m["key"] for m in models if m["trainable"]],
                         "horizon": 15})


@app.get("/status")
def status(city: str = "delhi", state: str = None, mode: str = None):
    import paths
    st = state or city
    cache = paths.nasa_cache(st).exists()
    trained = paths.metadata_path(st, mode).exists()
    return JSONResponse({"city": st, "state": st, "mode": mode or "multivariate",
                         "data": cache, "trained": trained})


@app.get("/download")
def download(city: str = "delhi"):
    import pipeline_stream
    city = city if city in STATES else "delhi"
    return _start_phase("download", lambda: pipeline_stream.download_stream(city))


@app.get("/download_all")
def download_all():
    """Download NASA data for every state/UT (whole country), streamed."""
    import pipeline_stream
    return _start_phase("download", lambda: pipeline_stream.download_all_stream())


def _parse_models(models: str):
    """Parse the ?models=a,b,c selection into a validated list of known model
    names. Returns None when nothing valid is given (callers then fall back to the
    full MODEL_NAMES, i.e. train everything)."""
    if not models:
        return None
    import config as C
    want = [m.strip() for m in str(models).split(",") if m.strip()]
    sel = [m for m in want if m in C.ALL_MODEL_NAMES]
    return sel or None


@app.get("/train")
def train(city: str = "delhi", fresh: int = 0, models: str = None, mode: str = None):
    import pipeline_stream, control, paths, shutil
    city = city if city in STATES else "delhi"
    sel = _parse_models(models)
    control.clear_pause()                 # a (re)start always clears the pause flag
    if fresh:
        # full wipe for THIS MODE only: forget the resume point AND delete every
        # mid-training checkpoint so training truly starts from epoch 1 (no stale/
        # collapsed weights get resumed). Other mode's data is untouched.
        control.clear_train_state(city, mode)
        try:
            cdir = control.CKPT_DIR / control._canonical(city) / control._mode(mode)
            if cdir.exists():
                for sub in cdir.glob("*"):
                    if sub.is_dir():
                        for f in sub.glob("ckpt_*.pt"):
                            try: f.unlink()
                            except Exception: pass
        except Exception:
            pass
    return _start_phase("train", lambda: pipeline_stream.train_stream(city, models=sel, mode=mode), mode=(mode or "multivariate"))




@app.get("/live_status")
def live_status(mode: str = None, city: str = None, state: str = None):
    import live_engine
    city = state or city
    s = live_engine._load_state(mode, city)
    led = live_engine._load_ledger(mode, city)
    return JSONResponse({
        "cursor": s.get("cursor"), "cycles": s.get("cycles", 0),
        "verified": s.get("verified_count", 0),
        "ledger_rows": int(len(led)),
        "start_date": live_engine.START_DATE,
        "horizon": live_engine.FORECAST_HORIZON,
    })


@app.get("/horizon_accuracy")
def horizon_accuracy(model: str = "ensemble", mode: str = None, city: str = None, state: str = None):
    """Mean absolute error grouped by lead time (horizon 1..N), per target, for the
    chosen model. Powers the 'accuracy by lead time' chart — how forecast error
    grows the further ahead it was made. Verified rows only."""
    import live_engine
    import numpy as np
    city = state or city
    led = live_engine._load_ledger(mode, city)
    out = {"horizon": live_engine.FORECAST_HORIZON, "model": model, "targets": {}}
    if led is None or len(led) == 0 or "horizon" not in led.columns:
        return JSONResponse(out)
    df = led.copy()
    df = df[(df.get("verified") == True) & (df["model"] == model)]
    df = df[df["error"].notna() & df["horizon"].notna()]
    if len(df) == 0:
        return JSONResponse(out)
    df["abs_err"] = df["error"].astype(float).abs()
    df["horizon"] = df["horizon"].astype(int)
    for tgt in live_engine.TARGETS:
        sub = df[df["target"] == tgt]
        if len(sub) == 0:
            continue
        g = sub.groupby("horizon")["abs_err"].agg(["mean", "count"]).reset_index()
        out["targets"][tgt] = [
            {"horizon": int(r["horizon"]),
             "mae": round(float(r["mean"]), 3),
             "n": int(r["count"])}
            for _, r in g.sort_values("horizon").iterrows()
        ]
    return JSONResponse(_json_safe(out))


@app.get("/oos_accuracy")
def oos_accuracy(mode: str = None, city: str = None, state: str = None):
    """TRUE out-of-sample forecast accuracy, per model + target, from the verified
    ledger. Every row here is a forecast whose prediction was stored BEFORE the
    actual was known, then compared to NASA/ERA5 — so RMSE/MAE/R2/bias are honest
    forecast skill (unlike the in-sample fine-tune 'fit' panel). All verified
    horizons are pooled. 'n' is how many verified forecasts each number rests on;
    treat small-n values with caution.
    """
    import live_engine
    import numpy as np
    city = state or city
    led = live_engine._load_ledger(mode, city)
    out = {"targets": {}, "note": "out-of-sample: prediction stored before actual was known"}
    if led is None or len(led) == 0:
        return JSONResponse(out)
    df = led.copy()
    df = df[(df.get("verified") == True)]
    df = df[df["prediction"].notna() & df["actual"].notna()]
    if len(df) == 0:
        return JSONResponse(out)
    df["prediction"] = df["prediction"].astype(float)
    df["actual"] = df["actual"].astype(float)
    # models to report: every model in the ledger (incl. 'ensemble')
    models = sorted(df["model"].dropna().unique().tolist())
    for tgt in live_engine.TARGETS:
        rows = []
        for m in models:
            sub = df[(df["target"] == tgt) & (df["model"] == m)]
            if len(sub) == 0:
                continue
            yp = sub["prediction"].values
            yt = sub["actual"].values
            err = yp - yt
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            bias = float(np.mean(err))                       # signed: +over / -under
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((yt - yt.mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else None
            rows.append({"model": m, "n": int(len(sub)),
                         "rmse": round(rmse, 3), "mae": round(mae, 3),
                         "bias": round(bias, 3),
                         "r2": (round(r2, 3) if r2 is not None else None)})
        # order: ensemble first, then by RMSE ascending
        rows.sort(key=lambda r: (r["model"] != "ensemble", r["rmse"]))
        out["targets"][tgt] = rows
    return JSONResponse(_json_safe(out))



@app.get("/live_cycle")
def live_cycle(models: str = None, mode: str = None, city: str = None, state: str = None):
    """Run one forward cycle for the SELECTED state+mode, streaming progress."""
    import live_engine
    city = state or city
    sel = _parse_models(models)
    return _start_phase("cycle", lambda: live_engine.run_cycle(models=sel, mode=mode, city=city), mode=(mode or "multivariate"))


@app.get("/live_reset")
def live_reset(mode: str = None, city: str = None, state: str = None):
    import live_engine
    city = state or city
    live_engine.reset_live(mode, city)
    return JSONResponse({"ok": True})


@app.get("/live_data")
def live_data(mode: str = None, city: str = None):
    """Full persisted state for restoring the dashboard on reload:
       ledger rows, running metrics, and per-target series for charts.
       `city` lets the Map page read any state's ledger (defaults to Delhi)."""
    import live_engine
    import pandas as pd
    from datetime import datetime, timedelta
    led_full = live_engine._load_ledger(mode, city)
    state = live_engine._load_state(mode, city)
    if led_full.empty:
        return JSONResponse({"rows": [], "metrics": {}, "series": {}, "state": state})
    led_full = led_full.where(pd.notnull(led_full), None)
    # metrics are aggregate accuracy over the FULL history (full CSV is the permanent log)
    metrics = live_engine._metrics(led_full)
    # ...but only the last +/-14 days around this mode's cursor are sent for DISPLAY,
    # so the charts/tables stay light no matter how far the cursor has advanced.
    WIN = 14
    cursor = state.get("cursor")
    try:
        cdt = datetime.strptime(cursor, "%Y-%m-%d")
        lo = (cdt - timedelta(days=WIN)).strftime("%Y-%m-%d")
        hi = (cdt + timedelta(days=WIN)).strftime("%Y-%m-%d")
    except Exception:
        lo, hi = None, None
    led = led_full
    if lo is not None and "forecast_date" in led_full.columns:
        led = led_full[(led_full["forecast_date"] >= lo) & (led_full["forecast_date"] <= hi)]
    rows = led.to_dict(orient="records")
    # per-target series: ensemble prediction vs actual over forecast_date (windowed)
    series = {}
    for t in live_engine.TARGETS:
        sub = led[(led["target"] == t)].copy()
        sub = sub.sort_values("forecast_date")
        pts = []
        seen = set()
        for _, r in sub.iterrows():
            d = r["forecast_date"]
            if d in seen:
                continue
            ens = sub[(sub["forecast_date"] == d) & (sub["model"] == "ensemble")]
            if len(ens):
                rr = ens.iloc[0]
                def _f(v):
                    return None if v is None else (float(v) if v == v else None)
                pts.append({"date": d,
                            "pred": _f(rr["prediction"]),
                            "actual": _f(rr["actual"]),
                            "nasa": _f(rr.get("actual_nasa")),
                            "meteo": _f(rr.get("actual_meteo"))})
                seen.add(d)
        series[t] = pts
    payload = {"rows": rows, "metrics": metrics, "series": series, "state": state,
               "window": {"lo": lo, "hi": hi, "days": WIN}}
    return JSONResponse(_json_safe(payload))


@app.get("/pause")
def pause_run():
    import control
    control.request_pause()
    return JSONResponse({"ok": True, "paused": True})


@app.get("/resume")
def resume_run():
    """Clear the pause flag. The client then re-opens the train/tune stream,
    which resumes from the saved checkpoint."""
    import control
    control.clear_pause()
    control.clear_paused_phase()
    return JSONResponse({"ok": True, "paused": False})


@app.get("/run_stream")
def run_stream():
    """Subscribe to the current/last run: replays all buffered events, then tails
    live. Safe to open from any page (the one that started the run, or a reloaded
    one) — it never starts or interrupts the work."""
    backlog, qsub, running = BUS.subscribe()
    def gen():
        try:
            # leading comment forces proxies/browsers to open the stream immediately
            yield ": open\n\n"
            for ev in backlog:
                yield f"data: {json.dumps(ev)}\n\n"
            if not running:
                # nothing live to tail; tell the client the replay is complete
                yield f"data: {json.dumps({'type':'_replay_done'})}\n\n"
                return
            while True:
                try:
                    ev = qsub.get(timeout=10)   # wake periodically even if idle
                except queue.Empty:
                    # heartbeat: keeps the connection flushing through proxies and
                    # alive across slow epochs (quantum models). Clients ignore comments.
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") == "_end":
                    break
        finally:
            BUS.unsubscribe(qsub)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/run_state")
def run_state(city: str = "delhi", mode: str = None):
    import control
    ts = control.load_train_state(city, mode)
    phase = "train"
    pf = control.PAUSED_PHASE
    if pf.exists():
        try:
            phase = pf.read_text().strip() or "train"
        except Exception:
            pass
    bus = BUS.state()
    return JSONResponse({"paused": control.is_paused(), "phase": phase,
                         "train_state": ts,
                         "running": bus["running"],
                         "running_phase": bus["phase"],
                         "running_mode": bus["mode"],
                         "last_event": bus["last_event"]})


@app.get("/train_metrics")
def train_metrics(city: str = "delhi", state: str = None, mode: str = None):
    city = state or city
    """Per-epoch RMSE/MAE/R2 history per target+model from the last training run,
    plus the per-cycle fine-tune/retrain metric rows, for chart+table restore."""
    import json as _json, paths
    out = {"train": {}, "live": []}
    mpath = paths.train_metrics_path(city, mode)
    if mpath.exists():
        try:
            out["train"] = _json.load(open(mpath))
        except Exception:
            pass
    lpath = paths.live_metrics_path(city, mode)
    if lpath.exists():
        try:
            out["live"] = _json.load(open(lpath))
        except Exception:
            pass
    return JSONResponse(_json_safe(out))


@app.get("/live_weights")
def live_weights(city: str = "delhi", mode: str = None):
    """Latest Δweight + qweight per model+target, so the dashboard can repaint
    the per-card LIVE WEIGHT bars after a refresh."""
    import live_engine, json as _json
    wlog = live_engine._wlog_p(mode)
    latest = {}
    if wlog.exists():
        try:
            for e in _json.load(open(wlog)):
                latest[f"{e['target']}|{e['model']}"] = e   # last write wins
        except Exception:
            pass
    return JSONResponse(_json_safe({"latest": list(latest.values())}))


@app.get("/hparams")
def hparams(city: str = "delhi", mode: str = None):
    import tune
    city = city if city in STATES else "delhi"
    return JSONResponse(_json_safe(tune.load_hparams(city, mode)))


@app.get("/tune")
def tune_run(city: str = "delhi", trials: int = 0, models: str = None, mode: str = None):
    """Run Optuna tuning for the selected models, streaming live trial progress."""
    import tune, control
    city = city if city in STATES else "delhi"
    sel = _parse_models(models)
    control.clear_pause()
    nt = trials if trials > 0 else None
    return _start_phase("tune", lambda: tune.tune_stream(city, n_trials=nt, models=sel, mode=mode), mode=(mode or "multivariate"))


@app.get("/model_spread")
def model_spread(city: str = None, mode: str = None):
    """Per-day disagreement across the (non-ensemble) models: the std-dev of their
    predictions for each verified forecast_date, per target. A tighter spread =
    higher ensemble confidence. Powers the 'Model agreement / spread' chart."""
    import live_engine
    import numpy as np
    import pandas as pd
    led = live_engine._load_ledger(mode, city)
    out = {"targets": {}}
    if led is None or len(led) == 0:
        return JSONResponse(out)
    df = led.copy()
    df = df[(df.get("verified") == True) & (df["model"] != "ensemble")]
    df = df[df["prediction"].notna()]
    if len(df) == 0:
        return JSONResponse(out)
    df["prediction"] = df["prediction"].astype(float)
    # one prediction per (date, model): prefer nearest horizon, then newest run
    if "horizon" in df.columns:
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").fillna(99).astype(int)
    else:
        df["horizon"] = 1
    if "made_on" not in df.columns:
        df["made_on"] = df["forecast_date"]
    df = df.sort_values(["horizon", "made_on"], ascending=[True, False])
    df = df.drop_duplicates(subset=["forecast_date", "target", "model"], keep="first")
    for tgt in live_engine.TARGETS:
        sub = df[df["target"] == tgt]
        if len(sub) == 0:
            continue
        pts = []
        for d, g in sub.groupby("forecast_date"):
            vals = g["prediction"].values
            if len(vals) >= 2:
                pts.append({"date": str(d), "spread": round(float(np.std(vals)), 3),
                            "n": int(len(vals))})
        pts.sort(key=lambda p: p["date"])
        if pts:
            out["targets"][tgt] = pts
    return JSONResponse(_json_safe(out))


def _mode_rmse(city, mode):
    """Verified out-of-sample RMSE per target/model for one mode (for MV vs UV)."""
    import live_engine
    import numpy as np
    res = {t: {} for t in live_engine.TARGETS}
    led = live_engine._load_ledger(mode, city)
    if led is None or len(led) == 0:
        return res
    df = led[(led.get("verified") == True)]
    df = df[df["prediction"].notna() & df["actual"].notna()]
    if len(df) == 0:
        return res
    df = df.copy()
    df["prediction"] = df["prediction"].astype(float)
    df["actual"] = df["actual"].astype(float)
    for t in live_engine.TARGETS:
        dt = df[df["target"] == t]
        for m in dt["model"].dropna().unique().tolist():
            s = dt[dt["model"] == m]
            err = (s["prediction"] - s["actual"]).values
            if len(err):
                res[t][m] = {"rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
                             "n": int(len(err))}
    return res


@app.get("/mvuv_compare")
def mvuv_compare(city: str = "delhi"):
    """Head-to-head verified accuracy of Multivariate vs Univariate for one city —
    the core research comparison. Empty until BOTH modes have verified forecasts."""
    import live_engine
    mv = _mode_rmse(city, "multivariate")
    uv = _mode_rmse(city, "univariate")
    return JSONResponse(_json_safe({"targets": live_engine.TARGETS,
                                    "multivariate": mv, "univariate": uv}))


@app.get("/train_times")
def train_times(city: str = "delhi", state: str = None, mode: str = None):
    """Wall-clock training time per (target, model) recorded during the last train,
    so quantum models' cost can be weighed against accuracy."""
    import paths, json as _json
    st = state or city or "delhi"
    out = {"data": {}}
    p = paths.train_times_path(st, mode)
    if p.exists():
        try:
            out["data"] = _json.load(open(p))
        except Exception:
            out["data"] = {}
    return JSONResponse(_json_safe(out))


@app.get("/run_all_cycle")
def run_all_cycle(mode: str = None, models: str = None):
    """Run one forward cycle for EVERY trained state (whole-country map button),
    streaming per-state progress through the shared run bus / /run_stream."""
    import live_all
    sel = _parse_models(models)
    return _start_phase("run_all",
                        lambda: live_all.run_all_cycle_stream(mode=mode, models=sel),
                        mode=(mode or "multivariate"))


# ---------------------------------------------------------------------------
# ERA5 live forecast (t+1..t+horizon) — separate pipeline from live_engine's
# one-day NASA cycle above. Wraps live_forecast_era5.run_forecast() directly.
# ---------------------------------------------------------------------------
_ERA5_ALIAS = {"maharashtra": "mumbai", "tamil_nadu": "chennai"}


def _era5_stream(city_key, horizon, mode_arg, models_sel):
    import live_forecast_era5 as lf

    yield {"type": "phase", "phase": "era5_forecast", "status": "active"}
    yield {"type": "node", "node": "fetch", "status": "active"}
    try:
        modes = (lf.MODES if mode_arg in (None, "both")
                 else [("multivariate", "mv")] if mode_arg == "mv"
                 else [("univariate", "uv")])
        result = lf.run_forecast(
            city_key=city_key,
            horizon=horizon,
            modes=modes,
            model_names=models_sel or lf.MODEL_NAMES,
            verbose=False,
        )
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        yield {"type": "phase", "phase": "era5_forecast", "status": "error"}
        return

    yield {"type": "node", "node": "fetch", "status": "done"}
    yield {"type": "node", "node": "forecast", "status": "done"}

    # Persist alongside run_live_era5.py's own output convention.
    global _LAST_ERA5_RESULT
    _LAST_ERA5_RESULT[city_key] = result

    yield {"type": "era5_summary", "city": city_key, "summary": result["summary"],
           "data_through": result["data_through"], "run_date": result["run_date"]}
    yield {"type": "phase", "phase": "era5_forecast", "status": "done"}


_LAST_ERA5_RESULT: dict = {}


@app.get("/live_forecast_era5")
def live_forecast_era5_route(city: str = "delhi", state: str = None,
                             horizon: int = 5, mode: str = None,
                             models: str = None):
    """Run the t+1..t+horizon ERA5 live forecast for one city, streaming
    progress through the shared run bus / /run_stream (same pattern as
    /live_cycle and /train)."""
    key = state or city
    key = _ERA5_ALIAS.get(key, key)   # dashboard sends state=maharashtra/tamil_nadu
    sel = _parse_models(models)
    return _start_phase(
        "era5_forecast",
        lambda: _era5_stream(key, horizon, mode, sel),
        mode=(mode or "multivariate"))


@app.get("/live_forecast_era5_result")
def live_forecast_era5_result(city: str = "delhi", state: str = None):
    """Fetch the summary from the most recent /live_forecast_era5 run for this
    city (survives until the server restarts)."""
    key = state or city
    key = _ERA5_ALIAS.get(key, key)
    result = _LAST_ERA5_RESULT.get(key)
    if result is None:
        return JSONResponse({"available": False, "city": key})
    return JSONResponse({
        "available": True,
        "city": key,
        "run_date": result["run_date"],
        "data_through": result["data_through"],
        "summary": result["summary"],
    })


@app.get("/device")
def device_info():
    """Reports whether the GPU is being used for training/tuning/forecasting.
    Everything (classical layers AND the quantum circuits, via PennyLane's torch
    backprop path) runs on CUDA when a CUDA-enabled torch build is present."""
    info = {"cuda": False, "device": "cpu", "name": None, "torch": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["device"] = "cuda" if info["cuda"] else "cpu"
        if info["cuda"]:
            info["name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["error"] = str(e)
    return JSONResponse(info)


if __name__ == "__main__":
    import uvicorn
    try:
        import torch
        if torch.cuda.is_available():
            print(f"Compute device: CUDA ({torch.cuda.get_device_name(0)}) — "
                  f"training, tuning and quantum circuits will run on the GPU.")
        else:
            print("Compute device: CPU. (Install a CUDA-enabled torch build to use the GPU.)")
    except Exception:
        print("Compute device: unknown (torch not importable yet).")
    print("Dashboard -> http://localhost:8000   |   Map -> http://localhost:8000/map.html")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")