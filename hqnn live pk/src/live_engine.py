"""
PRODUCTION forward-forecasting engine (Delhi).

Unlike the backtest (which walks historical data and stops), this forecasts
GENUINELY FORWARD from a start date (default 2025-01-01, the first day after the
training data ends) and verifies each forecast against NASA POWER as the truth.

State lives in a persistent ledger CSV so the system accumulates across runs and
survives restarts — that is what makes it production rather than a script.

One "cycle" (one click of Run Today's Cycle) does:
  1. Forecast the next FORECAST_HORIZON days forward from the cursor (all 4
     models + ensemble), append to ledger as unverified.
  2. Verify the single day at the cursor: download its NASA actual, write
     actual+error into any prior forecast rows for that date, mark verified.
  3. Adapt: fine-tune each model on that verified day (gentle), and every
     RETRAIN_EVERY verified days do a full retrain.
  4. Advance the cursor one day.

The ledger schema (one row per forecast_date x target x model):
  forecast_date, made_on, target, model, prediction, actual, error,
  verified, unit, source
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import nasa
import features
from engine import (set_seed, build_sequences, finetune_model, forecast_next,
                    load_assets, save_assets, train_model, snapshot_weights,
                    weight_delta, DEVICE)
from models import build_model
from config import (TARGETS, MODEL_NAMES, UNITS, SEQ_LEN, RETRAIN_EVERY,
                    MODEL_DIR, FORECAST_DIR, LOG_DIR, DATA_DIR)
from sklearn.preprocessing import MinMaxScaler


def feels_like(temp_c, rh_pct, wind_ms, month=None):
    """'Feels like' in °C from temperature (°C), relative humidity (%), and wind
    speed (m/s). Chooses the formula by the ACTUAL conditions (NWS standard),
    not the calendar month:
       • Heat Index (Rothfusz) when it's hot (T >= 80°F),
       • Wind Chill when it's cold and breezy (T <= 50°F, wind >= 3 mph),
       • otherwise the plain air temperature.
    `month` is accepted for backward-compatibility but no longer used. Formulas are
    in °F, so we convert in and out."""
    try:
        T = temp_c * 9.0 / 5.0 + 32.0          # °C -> °F
        R = max(0.0, min(100.0, rh_pct if rh_pct is not None else 0.0))
        V = (wind_ms or 0.0) * 2.236936         # m/s -> mph
        if T >= 80.0:
            # NWS algorithm: try the simple average first; only use the full
            # Rothfusz regression (+ corrections) when the heat index is truly >= 80°F.
            simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094)
            if (simple + T) / 2.0 < 80.0:
                f = simple
            else:
                f = (-42.379 + 2.04901523 * T + 10.14333127 * R
                     - 0.22475541 * T * R - 0.00683783 * T * T
                     - 0.05481717 * R * R + 0.00122874 * T * T * R
                     + 0.00085282 * T * R * R - 0.00000199 * T * T * R * R)
                # official low/high-humidity corrections
                if R < 13.0 and 80.0 <= T <= 112.0:
                    f -= ((13.0 - R) / 4.0) * ((17.0 - abs(T - 95.0)) / 17.0) ** 0.5
                elif R > 85.0 and 80.0 <= T <= 87.0:
                    f += ((R - 85.0) / 10.0) * ((87.0 - T) / 5.0)
        elif T <= 50.0 and V >= 3.0:
            # NWS Wind Chill (2001)
            f = (35.74 + 0.6215 * T - 35.75 * (V ** 0.16)
                 + 0.4275 * T * (V ** 0.16))
        else:
            f = T                               # mild: feels like the air temp
        return round((f - 32.0) * 5.0 / 9.0, 2)  # °F -> °C
    except Exception:
        return None


def _reg_metrics(model, X, y, sc_y):
    """RMSE / MAE / R2 on the original scale for the given sequences."""
    import torch
    try:
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy().reshape(-1, 1)
        yp = sc_y.inverse_transform(pred).ravel()
        yt = sc_y.inverse_transform(y.reshape(-1, 1)).ravel()
        err = yp - yt
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1e-9
        r2 = float(1.0 - ss_res / ss_tot)
        return round(rmse, 4), round(mae, 4), round(r2, 4)
    except Exception:
        return None, None, None

START_DATE = "2026-01-01"      # first forward day (training data ends 2025-12-12)
FORECAST_HORIZON = 15          # forecast this many days ahead each cycle
VERIFY_LAG = 0                 # verify the cursor day itself (NASA already has 2025)

CITY = "delhi"
import paths as _P
from config import DEFAULT_MODE as _DEFMODE


def _latlon(city=None):
    """(lat, lon) for a city/state key, from config (works for any state, not just
    Delhi). Falls back to Delhi's coordinates if the key is unknown."""
    city = city or CITY
    try:
        from config import STATES as _STATES, CITIES as _CITIES
        cfg = _STATES.get(city) or _CITIES.get(city)
        if cfg:
            return float(cfg["lat"]), float(cfg["lon"])
    except Exception:
        pass
    return 28.61, 77.20


def _ledger_p(mode=None, city=None):
    return _P.ledger_path(city or CITY, mode)


def _state_p(mode=None, city=None):
    return _P.live_state_path(city or CITY, mode)


def _wlog_p(mode=None, city=None):
    return _P.weights_log_path(city or CITY, mode)


LEDGER_COLS = ["forecast_date", "made_on", "target", "model", "prediction",
               "actual", "error", "verified", "unit", "source",
               "actual_nasa", "actual_meteo", "horizon"]


# ---------------------------------------------------------------------------
# State + ledger persistence (all keyed by mode)
# ---------------------------------------------------------------------------
def _load_state(mode=None, city=None):
    sp = _state_p(mode, city)
    if sp.exists():
        return json.load(open(sp))
    return {"cursor": START_DATE, "verified_count": 0, "cycles": 0}


def _save_state(s, mode=None, city=None):
    json.dump(s, open(_state_p(mode, city), "w"), indent=2)


def _load_ledger(mode=None, city=None):
    lp = _ledger_p(mode, city)
    if lp.exists():
        return pd.read_csv(lp)
    return pd.DataFrame(columns=LEDGER_COLS)


def _save_ledger(df, mode=None, city=None):
    df.to_csv(_ledger_p(mode, city), index=False)


def reset_live(mode=None, city=None):
    """Wipe ledger/state to start the forward run fresh (for the given mode)."""
    for p in (_ledger_p(mode, city), _state_p(mode, city), _wlog_p(mode, city)):
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Data: history up to a given date (training cache + any downloaded 2025 days)
# ---------------------------------------------------------------------------
def _history_through(date_str, city=None):
    """
    Return engineered features using all NASA data up to and including date_str.
    Downloads/caches the 2025 tail as needed so forecasting has real inputs.
    """
    city = city or CITY
    lat, lon = _latlon(city)
    end = datetime.strptime(date_str, "%Y-%m-%d")
    # download a generous range covering training + forward period
    cache = DATA_DIR / f"{city}_live_cache.csv"
    need_end = end.strftime("%Y%m%d")
    df = nasa._fetch(lat, lon, "20210101", need_end)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df.to_csv(cache, index=False)
    feat = features.engineer_features(df)
    feat = feat[feat["date"] <= end]
    return feat


def _raw_history_through(date_str, city=None):
    """Same as _history_through but returns the RAW (pre-feature-engineering)
    daily frame, so the forecast loop can append predicted rows and recompute
    features properly for each horizon day (true iterative multi-step)."""
    city = city or CITY
    lat, lon = _latlon(city)
    end = datetime.strptime(date_str, "%Y-%m-%d")
    cache = DATA_DIR / f"{city}_live_cache.csv"
    df = nasa._fetch(lat, lon, "20210101", end.strftime("%Y%m%d"))
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= end].reset_index(drop=True)


def _meta(mode=None, city=None):
    import paths
    return json.load(open(paths.metadata_path(city or CITY, mode)))


# ---------------------------------------------------------------------------
# One production cycle
# ---------------------------------------------------------------------------
def run_cycle(stream=False, models=None, mode=None, city=None):
    """
    Execute one forward cycle. Always a generator: yields progress events as
    each step completes (so the dashboard updates live). For non-streaming use,
    call run_cycle_sync() which drains it and returns a summary.

    city defaults to Delhi. Pass a state key (e.g. "maharashtra") to run that
    state's forward cycle — used by the whole-country map runner.
    """
    set_seed()
    city = city or CITY
    mode = mode or _DEFMODE
    _fmode = "uv" if mode in ("uv", "univariate") else "mv"   # feature-set key
    # models to UPDATE this cycle (fine-tune/retrain). If the UI passed a selection,
    # only those are adapted; the rest keep their last-trained weights but ARE STILL
    # used for forecasting below (forecast loops iterate the full MODEL_NAMES).
    _UPD = [m for m in MODEL_NAMES if m in models] if models else list(MODEL_NAMES)
    if not _UPD:
        _UPD = list(MODEL_NAMES)
    meta = _meta(mode, city)
    feats_by_target = meta["features"][_fmode]
    state = _load_state(mode, city)
    cursor = state["cursor"]
    ledger = _load_ledger(mode, city)

    yield {"type": "cycle_start", "cursor": cursor, "cycle": state["cycles"] + 1,
           "horizon_total": FORECAST_HORIZON}

    # ---- 1. FORECAST forward HORIZON days ----
    yield {"type": "node", "node": "forecast", "status": "active"}
    cur_dt = datetime.strptime(cursor, "%Y-%m-%d")
    # made_on = the simulation day this forecast is issued FROM (the cursor), not the
    # wall-clock date. Using the cursor makes each cycle a distinct, time-orderable
    # batch, so the dashboard can always show the LATEST cycle's forward forecast
    # (e.g. a cycle run from Jan 10 shows Jan 11+). Using datetime.now() made every
    # cycle share the same made_on, so the strip stayed stuck on the first batch.
    made_on = cursor
    new_rows = []

    # Load every model ONCE (not inside the day/target loop). {target:{model:(m,scX,scY)}}
    loaded = {}
    for target in TARGETS:
        feats = feats_by_target[target]
        loaded[target] = {}
        for model_name in MODEL_NAMES:
            try:
                loaded[target][model_name] = load_assets(city, model_name, target, len(feats), mode=mode)
            except Exception as e:
                yield {"type": "info", "msg": f"could not load {model_name}/{target}: {e}"}

    # Raw history up to the day BEFORE the cursor; we append predicted RAW rows and
    # recompute features each horizon day so lags/rolling/anomalies stay consistent
    # (true iterative multi-step forecast, not frozen features).
    raw_hist = _raw_history_through((cur_dt + timedelta(days=-1)).strftime("%Y-%m-%d"), city)
    # carry-forward values for exogenous drivers we don't predict
    exo_cols = [c for c in ["U10m", "V10m", "WS", "WD", "QV2m", "TMP2m_max",
                            "TMP2m_min", "PRCP", "PS_Pa"] if c in raw_hist.columns]
    last_ws = None
    try:
        if "WS" in raw_hist.columns and len(raw_hist):
            last_ws = float(raw_hist["WS"].dropna().iloc[-1])
    except Exception:
        last_ws = None

    try:
        print(f"[cycle] {city}/{mode} cursor={cursor} forecasting {FORECAST_HORIZON}d", flush=True)
    except Exception:
        pass
    for h in range(FORECAST_HORIZON):
        fdate = (cur_dt + timedelta(days=h)).strftime("%Y-%m-%d")
        # recompute features on the history we have so far (incl. prior predicted days)
        feat_hist = features.engineer_features(raw_hist)
        day_ens = {}
        for target in TARGETS:
            if len(feat_hist) < SEQ_LEN:
                continue
            feats = feats_by_target[target]
            preds = {}
            for model_name in MODEL_NAMES:
                trio = loaded.get(target, {}).get(model_name)
                if trio is None:
                    continue
                model, sc_X, sc_y = trio
                pred = forecast_next(model, feat_hist, feats, target, sc_X, sc_y)
                preds[model_name] = pred
                new_rows.append({"forecast_date": fdate, "made_on": made_on,
                    "target": target, "model": model_name,
                    "prediction": round(pred, 3), "actual": np.nan,
                    "error": np.nan, "verified": False, "unit": UNITS[target],
                    "source": "", "horizon": h + 1})
                yield {"type": "forecast", "date": fdate, "target": target,
                   "model": model_name, "prediction": round(pred, 3),
                   "unit": UNITS[target], "horizon": h + 1}
            if not preds:
                continue
            ens = float(np.mean(list(preds.values())))
            day_ens[target] = ens
            new_rows.append({"forecast_date": fdate, "made_on": made_on,
                "target": target, "model": "ensemble",
                "prediction": round(ens, 3), "actual": np.nan, "error": np.nan,
                "verified": False, "unit": UNITS[target], "source": "",
                "horizon": h + 1})
            yield {"type": "forecast", "date": fdate, "target": target,
                   "model": "ensemble", "prediction": round(ens, 3),
                   "unit": UNITS[target], "horizon": h + 1}

        # append ONE predicted RAW row for this day. Carry forward EVERY raw column
        # from the previous row (so no column is NaN), then overwrite the date and
        # the predicted targets. This is the fix for the flat forecast: previously
        # the row only set date+targets+exo, leaving other raw NASA columns
        # (TMP2m_max, TMP2m_min, PRCP, PS_Pa, ...) as NaN, so engineer_features()'s
        # dropna() DROPPED every appended row — the feature window never advanced and
        # the model saw the same input every horizon.
        if day_ens:
            prev = raw_hist.iloc[-1]
            row = prev.to_dict()                      # carry ALL raw columns forward
            row["date"] = pd.to_datetime(fdate)       # advance the date
            for t in TARGETS:                          # overwrite with this day's preds
                if t in day_ens:
                    row[t] = day_ens[t]
            raw_hist = pd.concat([raw_hist, pd.DataFrame([row])], ignore_index=True)

        # ---- "feels like" for this day, from forecasted T + RH + carried wind ----
        if "TMP2m" in day_ens:
            fl = feels_like(day_ens["TMP2m"], day_ens.get("RH2m"), last_ws)
            _Tf = day_ens["TMP2m"] * 9.0 / 5.0 + 32.0
            method = "heat_index" if _Tf >= 80.0 else ("wind_chill" if _Tf <= 50.0 else "actual")
            if h == 0:
                state["feels_like"] = fl
                state["feels_method"] = method
            yield {"type": "feels_like", "date": fdate, "horizon": h + 1,
                   "feels_like": fl, "temp": round(day_ens["TMP2m"], 2),
                   "method": method}
    if new_rows:
        nd = pd.DataFrame(new_rows)
        ledger = pd.concat([ledger, nd], ignore_index=True)
        # keep one row per (made_on, forecast_date, target, model): a given calendar
        # day is forecast at many lead times across cycles — we keep each so accuracy
        # can be measured per horizon.
        ledger = ledger.drop_duplicates(
            subset=["made_on", "forecast_date", "target", "model"], keep="last")
        # DIAGNOSTIC: show the raw ensemble temperature per horizon so we can see
        # whether the multi-step forecast actually varies day to day.
        try:
            t_ens = [r for r in new_rows if r["target"] == "TMP2m" and r["model"] == "ensemble"]
            t_ens.sort(key=lambda r: r["horizon"])
            seq = ", ".join(f"h{r['horizon']}:{r['prediction']}" for r in t_ens)
            yield {"type": "info", "msg": f"TMP2m forecast by horizon → {seq}"}
            # collapse guard: if the multi-step forecast is essentially flat across
            # all horizons, the model is predicting a constant (undertrained / not
            # converged). Surface it clearly instead of silently showing one value.
            for tgt in TARGETS:
                te = [r["prediction"] for r in new_rows
                      if r["target"] == tgt and r["model"] == "ensemble"]
                if len(te) >= 3 and (max(te) - min(te)) < 1e-3:
                    yield {"type": "info",
                           "msg": f"⚠ {tgt} forecast is flat ({te[0]}) across all horizons — "
                                  f"the {tgt} model looks undertrained (predicting a constant). "
                                  f"Retrain {tgt} to convergence."}
        except Exception:
            pass
    yield {"type": "node", "node": "forecast", "status": "done"}

    # ---- 2. VERIFY the cursor day ----
    yield {"type": "node", "node": "verify", "status": "active"}
    verify_date = cursor
    # multi-source truth: NASA POWER + Open-Meteo (ERA5)
    try:
        import providers
        truth_ms = providers.fetch_truth(city, verify_date)
    except Exception:
        truth_ms = None
    truth = _nasa_actual(verify_date, city)
    verified_any = False
    confident_targets = set()      # targets where NASA & Open-Meteo agree closely
    if truth or truth_ms:
        yield {"type": "node", "node": "confident", "status": "active"}
        for target in TARGETS:
            nasa_val = (truth_ms.get(target, {}).get("nasa") if truth_ms else None)
            if nasa_val is None and truth:
                nasa_val = truth.get(target)
            meteo_val = (truth_ms.get(target, {}).get("era5") if truth_ms else None)
            # confidence flag is informational only. For the WEIGHT UPDATE we trust
            # the primary ground truth (NASA — the same source the models were
            # trained on). NASA vs Open-Meteo routinely differ on humidity by more
            # than the threshold, and blanket-skipping every day meant humidity
            # NEVER fine-tuned and degraded as the horizon grew. So: fine-tune
            # whenever a usable actual exists; only skip a target with no truth.
            ms = (truth_ms.get(target, {}) if truth_ms else {})
            both_present = ms.get("nasa") is not None and ms.get("era5") is not None
            disagree = bool(both_present and not ms.get("confident"))
            # eligible to update as long as we have any ground-truth value
            has_truth = (nasa_val is not None) or (meteo_val is not None)
            if has_truth:
                confident_targets.add(target)
            # primary actual for error = NASA if present else Open-Meteo
            actual = nasa_val if nasa_val is not None else meteo_val
            if actual is None:
                continue
            mask = ((ledger["forecast_date"] == verify_date) &
                    (ledger["target"] == target))
            for idx in ledger[mask].index:
                pred = float(ledger.at[idx, "prediction"])
                ledger.at[idx, "actual"] = round(actual, 3)
                ledger.at[idx, "error"] = round(pred - actual, 3)
                ledger.at[idx, "verified"] = True
                ledger.at[idx, "source"] = "NASA+Meteo"
                if nasa_val is not None:
                    ledger.at[idx, "actual_nasa"] = round(nasa_val, 3)
                if meteo_val is not None:
                    ledger.at[idx, "actual_meteo"] = round(meteo_val, 3)
                yield {"type": "verify", "date": verify_date, "target": target,
                   "model": ledger.at[idx, "model"], "prediction": pred,
                   "actual": round(actual, 3),
                   "actual_nasa": (round(nasa_val, 3) if nasa_val is not None else None),
                   "actual_meteo": (round(meteo_val, 3) if meteo_val is not None else None),
                   "error": round(pred - actual, 3), "unit": UNITS[target]}
                verified_any = True
        yield {"type": "node", "node": "confident", "status": "done"}
    else:
        yield {"type": "info", "msg": f"NASA actual for {verify_date} not available yet."}
    yield {"type": "node", "node": "verify", "status": "done"}

    # ---- 3. ADAPT on the verified day ----
    if verified_any:
        state["verified_count"] += 1
        # Full retrain when EITHER the rolling interval is reached (every
        # RETRAIN_EVERY verified days) OR the verified day is the 1st of a month.
        try:
            _vd = datetime.strptime(str(verify_date)[:10], "%Y-%m-%d")
            is_first_of_month = (_vd.day == 1)
        except Exception:
            is_first_of_month = False
        interval_hit = (state["verified_count"] % RETRAIN_EVERY == 0)
        do_retrain = interval_hit or is_first_of_month
        yield {"type": "node", "node": "retrain_q", "status": "active"}
        yield {"type": "node", "node": "retrain_q", "status": "done"}
        hist = _history_through(verify_date, city)
        wlog = []
        lmetrics = []

        if do_retrain:
            _reason = ("1st of month" if is_first_of_month and not interval_hit
                       else ("scheduled + 1st of month" if is_first_of_month
                             else f"day {state['verified_count']}"))
            yield {"type": "info", "msg": f"Full retrain ({_reason}) — retraining all 12 models on full history, warm-started from current weights."}
            yield {"type": "node", "node": "apply_hp", "status": "active"}
            for target in TARGETS:
                feats = feats_by_target[target]
                yield {"type": "node", "node": "model_loop", "status": "active"}
                for model_name in _UPD:
                    # WARM START: load the existing (fine-tuned) model and keep its
                    # scalers, then retrain it thoroughly on the full history.
                    model, sc_X, sc_y = load_assets(city, model_name, target, len(feats), mode=mode)
                    before = snapshot_weights(model)
                    inner = int(len(hist) * 0.85)
                    # reuse the model's existing scalers so the warm-started weights
                    # stay valid (fit=False -> transform only, no re-fit)
                    Xtr, ytr = build_sequences(hist.iloc[:inner], feats, target, sc_X, sc_y, fit=False)
                    Xva, yva = build_sequences(hist.iloc[inner:], feats, target, sc_X, sc_y, fit=False)
                    yield {"type": "node", "node": "epoch", "status": "active"}
                    # NOTE: no set_seed()/build_model here — we continue training the
                    # existing weights rather than starting from a random init.
                    # Use the model's tuned lr/wd (falls back to config defaults).
                    try:
                        import tune as _tune
                        _hp = (_tune.load_hparams(city, mode)["data"].get(target, {})
                               .get(model_name, {}).get("params", {})) or {}
                    except Exception:
                        _hp = {}
                    model = train_model(model, Xtr, ytr, Xva, yva, hparams=_hp)
                    yield {"type": "node", "node": "save_model", "status": "active"}
                    save_assets(city, model_name, target, model, sc_X, sc_y, mode=mode)
                    after = snapshot_weights(model)
                    d = weight_delta(before, after)
                    mc = round(float(np.mean([v["mean_change"] for v in d.values()])), 6)
                    qval = None
                    for pn, info in after.items():
                        if "vqc.weights" in pn and "values" in info:
                            qval = info["values"][0]; break
                    yield {"type": "readjust", "model": model_name, "target": target,
                           "retrained": True, "mean_change": mc, "qweight": qval}
                    _rmse, _mae, _r2 = _reg_metrics(model, Xva, yva, sc_y)
                    yield {"type": "live_metric", "model": model_name, "target": target,
                           "kind": "retrain", "date": verify_date,
                           "rmse": _rmse, "mae": _mae, "r2": _r2}
                    lmetrics.append({"model": model_name, "target": target, "kind": "retrain",
                                     "date": verify_date, "rmse": _rmse, "mae": _mae, "r2": _r2})
                    wlog.append({"date": verify_date, "model": model_name, "target": target,
                                 "retrained": True, "mean_change": mc, "qweight": qval})
            yield {"type": "node", "node": "metadata", "status": "active"}
        else:
            yield {"type": "node", "node": "daily_ft", "status": "active"}
            for target in TARGETS:
                # only skip a target if we have NO ground-truth value for the day
                # (handled by confident_targets membership). Source disagreement is
                # logged but does NOT block the update — NASA is the primary truth.
                if confident_targets and target not in confident_targets:
                    yield {"type": "info",
                           "msg": f"No verified value for {target} on {verify_date} — skipping update."}
                    continue
                feats = feats_by_target[target]
                for model_name in _UPD:
                    model, sc_X, sc_y = load_assets(city, model_name, target, len(feats), mode=mode)
                    before = snapshot_weights(model)
                    X, y = build_sequences(hist, feats, target, sc_X, sc_y, fit=False)
                    model = finetune_model(model, X, y)
                    save_assets(city, model_name, target, model, sc_X, sc_y, mode=mode)
                    after = snapshot_weights(model)
                    d = weight_delta(before, after)
                    mc = round(float(np.mean([v["mean_change"] for v in d.values()])), 6)
                    qval = None
                    for pn, info in after.items():
                        if "vqc.weights" in pn and "values" in info:
                            qval = info["values"][0]; break
                    yield {"type": "readjust", "model": model_name, "target": target,
                           "retrained": False, "mean_change": mc, "qweight": qval}
                    _rmse, _mae, _r2 = _reg_metrics(model, X, y, sc_y)
                    yield {"type": "live_metric", "model": model_name, "target": target,
                           "kind": "finetune", "date": verify_date,
                           "rmse": _rmse, "mae": _mae, "r2": _r2}
                    lmetrics.append({"model": model_name, "target": target, "kind": "finetune",
                                     "date": verify_date, "rmse": _rmse, "mae": _mae, "r2": _r2})
                    wlog.append({"date": verify_date, "model": model_name, "target": target,
                                 "retrained": False, "mean_change": mc, "qweight": qval})
            yield {"type": "node", "node": "daily_ft", "status": "done"}

        _wl=_wlog_p(mode, city); existing = json.load(open(_wl)) if _wl.exists() else []
        json.dump(existing + wlog, open(_wl, "w"), indent=2)
        # persist live fine-tune/retrain metric rows (capped) for table restore
        try:
            import paths as _paths
            lpath = _paths.live_metrics_path(city, mode)
            prev = json.load(open(lpath)) if lpath.exists() else []
            json.dump((prev + lmetrics)[-300:], open(lpath, "w"))
        except Exception:
            pass

    # ---- 4. metrics + advance ----
    yield {"type": "metrics", "table": _metrics(ledger)}
    _save_ledger(ledger, mode, city)
    state["cursor"] = (cur_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    state["cycles"] += 1
    _save_state(state, mode, city)
    # release the per-cycle models/optimizers and reclaim memory so long auto-runs
    # (especially across all states) don't creep upward.
    try:
        loaded.clear()
    except Exception:
        pass
    import gc as _gc
    _gc.collect()
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:
        pass
    try:
        print(f"[cycle] {city}/{mode} done · verified {cursor} "
              f"→ next {state['cursor']} · {state['verified_count']} total", flush=True)
    except Exception:
        pass
    yield {"type": "cycle_done", "next_cursor": state["cursor"],
       "verified_total": state["verified_count"], "cycles": state["cycles"]}


def run_cycle_sync(mode=None, city=None):
    """Drain run_cycle without streaming; return a summary."""
    last = None
    for ev in run_cycle(mode=mode, city=city):
        last = ev
    state = _load_state(mode, city)
    return {"cursor": state["cursor"], "verified": state["verified_count"]}


def _nasa_actual(date_str, city=None):
    lat, lon = _latlon(city)
    d = date_str.replace("-", "")
    try:
        df = nasa._fetch(lat, lon, d, d)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[0]
    out = {}
    for t in TARGETS:
        if t in df.columns and pd.notna(row[t]):
            out[t] = float(row[t])
    return out if out else None


def _metrics(ledger):
    table = {}
    v = ledger[ledger["verified"] == True]
    for target in TARGETS:
        table[target] = {}
        for model_name in MODEL_NAMES + ["ensemble"]:
            sub = v[(v.target == target) & (v.model == model_name)]
            err = pd.to_numeric(sub["error"], errors="coerce").dropna().values
            actual = pd.to_numeric(sub["actual"], errors="coerce").dropna().values
            if len(err) == 0:
                table[target][model_name] = None
                continue
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((actual - actual.mean()) ** 2)) if len(actual) > 1 else 0.0
            r2 = (1 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
            denom = np.where(np.abs(actual) < 1e-6, 1e-6, np.abs(actual))
            mape = float(np.mean(np.abs(err / denom)) * 100)
            table[target][model_name] = {"RMSE": round(rmse, 4), "MAE": round(mae, 4),
                "R2": round(r2, 4), "MAPE": round(mape, 4), "n": int(len(err))}
    return table


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_live(); print("live state reset.")
    else:
        out = run_cycle_sync()
        print("cycle done:", out)