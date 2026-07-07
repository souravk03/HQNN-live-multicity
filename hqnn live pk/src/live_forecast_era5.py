"""
live_forecast_era5.py  (v2 — multi-city, June 2026)
-----------------------------------------------------
Live t+1 → t+5 forecasting pipeline using Open-Meteo ERA5 as input.
Supports Delhi, Mumbai, and Chennai.

CHANGES FROM v1 (Delhi-only)
-----------------------------
* CITY parameter accepted by run_forecast() — no longer hard-coded to Delhi.
* MODEL_ROOT resolved per-city: models/<city_key>/multivariate|univariate/
* ERA5 fetch passes city_key so era5_openmeteo saves openmeteo_cache.csv.
* City-aware config: lat/lon looked up from CITY_REGISTRY.
* run_delhi() kept as a thin alias for backward compatibility.

UNIT FIXES (unchanged from v1)
--------------------------------
* Both MV and UV sc_X scalers trained with PRmsl in Pa.
* SLP and PS are in kPa in training; ERA5 delivers hPa → divide by 10.
* TMP2m_max/min rollout clamped to [-10, 60] °C.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent


def _find_model_root(city_key: str) -> Path:
    candidates = [
        HERE / "models",
        HERE / ".." / "models",
        HERE,
    ]
    for c in candidates:
        if (c / city_key).exists():
            p = c.resolve()
            print(f"[LiveForecast] MODEL_ROOT resolved to: {p}")
            return p
    print(f"[LiveForecast] WARNING: models/{city_key} not found under {HERE}")
    return HERE


# ---------------------------------------------------------------------------
# City registry
# ---------------------------------------------------------------------------
CITY_REGISTRY: dict[str, dict] = {
    "delhi":       {"lat": 28.61, "lon": 77.20, "label": "Delhi",
                    "timezone": "Asia/Kolkata"},
    "mumbai":      {"lat": 19.08, "lon": 72.88, "label": "Mumbai",
                    "timezone": "Asia/Kolkata"},
    "maharashtra": {"lat": 19.08, "lon": 72.88, "label": "Mumbai",
                    "timezone": "Asia/Kolkata"},   # alias
    "chennai":     {"lat": 13.08, "lon": 80.27, "label": "Chennai",
                    "timezone": "Asia/Kolkata"},
    "tamil_nadu":  {"lat": 13.08, "lon": 80.27, "label": "Chennai",
                    "timezone": "Asia/Kolkata"},   # alias
}

# Alias → canonical folder name (models/<canonical>/ must exist)
_ALIAS_MAP = {
    "maharashtra": "mumbai",
    "tamil_nadu":  "chennai",
}

def _canonical(city_key: str) -> str:
    """Resolve alias keys to the canonical folder name used under models/ and data/."""
    return _ALIAS_MAP.get(city_key, city_key)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGETS     = ["TMP2m", "RH2m", "PRmsl"]
MODEL_NAMES = ["hqnn"]
UNITS       = {"TMP2m": "°C", "RH2m": "%", "PRmsl": "Pa"}
HORIZON     = 5
SEQ_LEN     = 30
ERA5_HISTORY_YEARS = 2

MODES = [
    ("multivariate", "mv"),
    ("univariate",   "uv"),
]

# ---------------------------------------------------------------------------
# Model / asset loading
# ---------------------------------------------------------------------------
import joblib
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_dir(model_root: Path, city_key: str,
               mode_label: str, model_name: str) -> Path:
    return model_root / city_key / mode_label / model_name


def _load_model_assets(model_root: Path, city_key: str,
                       mode_label: str, model_name: str,
                       target: str, n_features: int):
    d = _model_dir(model_root, city_key, mode_label, model_name)

    pth_path = d / f"{target}.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"Weights not found: {pth_path}")
    state_dict = torch.load(pth_path, map_location=DEVICE, weights_only=False)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]

    pkl_path = d / f"scaler_{target}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Scaler not found: {pkl_path}")
    scaler_obj = joblib.load(pkl_path)
    if not isinstance(scaler_obj, dict):
        raise ValueError(
            f"Expected dict from {pkl_path}, got {type(scaler_obj).__name__}.")

    sc_X = scaler_obj["X"]
    sc_y = scaler_obj["y"]
    hparams = scaler_obj.get("hp", {})

    try:
        import models as _models
        model = _models.build_model(
            model_name, n_features, hparams, mode=mode_label
        ).to(DEVICE)
    except Exception as exc:
        raise RuntimeError(
            f"Could not build model '{model_name}' "
            f"(n_features={n_features}): {exc}") from exc

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, sc_X, sc_y


# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

def _normalise_era5_units(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Converts ERA5 columns to the units used during model training:
      WS    : m/s  (ERA5 sometimes delivers km/h)
      QV2m  : g/kg (ERA5 delivers kg/kg)
      PS_Pa : Pa   (ERA5 sometimes delivers hPa)
      SLP   : kPa  (scaler trained in kPa ~99.3–102.34; ERA5 hPa → ÷10)
      PS    : kPa  (scaler trained in kPa ~97.06–99.93; ERA5 hPa → ÷10)
    PRmsl stays in Pa — both MV and UV sc_X trained with PRmsl in Pa.
    """
    df     = df.copy()
    issues = []

    if "WS" in df.columns:
        ws_max = df["WS"].dropna().max()
        if ws_max > 30:
            df["WS"] = df["WS"] / 3.6
            if "U10m" in df.columns: df["U10m"] = df["U10m"] / 3.6
            if "V10m" in df.columns: df["V10m"] = df["V10m"] / 3.6
            issues.append(f"WS was in km/h (max={ws_max:.1f}) → m/s")

    if "QV2m" in df.columns:
        qv_max = df["QV2m"].dropna().max()
        if qv_max < 0.1:
            df["QV2m"] = df["QV2m"] * 1000.0
            issues.append(f"QV2m was in kg/kg (max={qv_max:.5f}) → g/kg")

    if "PS_Pa" in df.columns:
        pspa_med = df["PS_Pa"].dropna().median()
        if pspa_med < 2000:
            df["PS_Pa"] = df["PS_Pa"] * 100.0
            issues.append(f"PS_Pa was in hPa (median={pspa_med:.1f}) → Pa")

    if "SLP" in df.columns:
        slp_med = df["SLP"].dropna().median()
        if slp_med > 5000:
            df["SLP"] = df["SLP"] / 100.0
            issues.append(f"SLP was in Pa (median={slp_med:.1f}) → kPa")
        elif slp_med > 200:
            df["SLP"] = df["SLP"] / 10.0
            issues.append(f"SLP was in hPa (median={slp_med:.1f}) → kPa")

    if "PS" in df.columns:
        ps_med = df["PS"].dropna().median()
        if ps_med > 5000:
            df["PS"] = df["PS"] / 100.0
            issues.append(f"PS was in Pa (median={ps_med:.1f}) → kPa")
        elif ps_med > 200:
            df["PS"] = df["PS"] / 10.0
            issues.append(f"PS was in hPa (median={ps_med:.1f}) → kPa")

    if verbose:
        if issues:
            print("[UnitFix] Applied corrections to ERA5 data:")
            for msg in issues:
                print(f"  • {msg}")
        else:
            print("[UnitFix] All ERA5 units look correct — no conversion needed.")

    return df


def _prepare_mv_df(df: pd.DataFrame) -> pd.DataFrame:
    """No-op kept for backward compatibility."""
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import features as _feat
        return _feat.engineer_features(df)
    except ImportError:
        return _minimal_engineer(df)


def _minimal_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback feature builder when features.py is not importable."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    tgts  = ["TMP2m", "RH2m", "PRmsl"]
    lags  = [1, 2, 3, 7, 14, 30]
    rolls = [3, 7, 14, 30]

    for t in tgts:
        if t not in df.columns:
            df[t] = np.nan
        for lag in lags:
            df[f"{t}_lag{lag}"] = df[t].shift(lag)
        for w in rolls:
            df[f"{t}_roll{w}_mean"] = df[t].shift(1).rolling(w).mean()
            df[f"{t}_roll{w}_std"]  = df[t].shift(1).rolling(w).std().fillna(0)
        df[f"{t}_roll7_min"]   = df[t].shift(1).rolling(7).min()
        df[f"{t}_roll7_max"]   = df[t].shift(1).rolling(7).max()
        df[f"{t}_roll7_range"] = df[f"{t}_roll7_max"] - df[f"{t}_roll7_min"]
        df[f"{t}_diff1"]       = df[t].shift(1).diff(1)
        df[f"{t}_diff7"]       = df[t].shift(1).diff(7)
        df[f"{t}_anomaly30"]   = df[t].shift(1) - df[f"{t}_roll30_mean"]

    df["PRmsl_tend3"]   = df["PRmsl"].shift(1).diff(3) / 3.0
    df["PRmsl_tend7"]   = df["PRmsl"].shift(1).diff(7) / 7.0
    df["PRmsl_anomaly"] = df["PRmsl"].shift(1) - df["PRmsl_roll7_mean"]
    df["RH2m_diff3"]    = df["RH2m"].shift(1).diff(3)
    df["RH2m_anomaly7"] = df["RH2m"].shift(1) - df["RH2m_roll7_mean"]

    if "WD" in df.columns:
        df["WD_sin"] = np.sin(np.radians(df["WD"]))
        df["WD_cos"] = np.cos(np.radians(df["WD"]))
    else:
        df["WD_sin"] = 0.0
        df["WD_cos"] = 1.0

    for col in ["SLP", "PS", "PS_Pa", "WS", "QV2m", "PRCP", "TMP2m_max", "TMP2m_min"]:
        if col not in df.columns:
            df[col] = np.nan

    for lag in [1, 3, 7]:
        df[f"WS_lag{lag}"] = df["WS"].shift(lag)
    for lag in [1, 2, 3, 7]:
        df[f"QV2m_lag{lag}"] = df["QV2m"].shift(lag)

    doy = df["date"].dt.dayofyear
    df["doy_sin"]   = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * doy / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    week = df["date"].dt.isocalendar().week.astype(float)
    df["week_sin"]  = np.sin(2 * np.pi * week / 52)
    df["week_cos"]  = np.cos(2 * np.pi * week / 52)

    m = df["date"].dt.month
    df["is_monsoon"]    = ((m >= 6) & (m <= 9)).astype(float)
    df["is_summer"]     = ((m >= 4) & (m <= 6)).astype(float)
    df["is_winter"]     = ((m == 12) | (m <= 2)).astype(float)
    df["is_premonsoon"] = ((m >= 3) & (m <= 5)).astype(float)

    df = df.dropna(subset=tgts).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Single-step forecast
# ---------------------------------------------------------------------------

def _forecast_one(model, feat_hist: pd.DataFrame, feats: list[str],
                  sc_X, sc_y, seq_len: int) -> float:
    model.eval()
    sl     = min(seq_len, len(feat_hist))
    window = feat_hist.iloc[-sl:][feats].values.astype(np.float32)
    Xs     = sc_X.transform(window)
    Xs_t   = torch.from_numpy(Xs[np.newaxis]).to(DEVICE)
    with torch.no_grad():
        pred_s = model(Xs_t).cpu().numpy().reshape(-1, 1)
    return float(sc_y.inverse_transform(pred_s)[0, 0])


# ---------------------------------------------------------------------------
# ERA5 fetch
# ---------------------------------------------------------------------------

def _load_metadata(model_root: Path, city_key: str, mode_label: str) -> dict:
    mp = model_root / city_key / mode_label / "metadata.json"
    with open(mp) as f:
        return json.load(f)


def _fetch_era5_history(city_key: str, lat: float, lon: float,
                        timezone: str) -> pd.DataFrame:
    from era5_openmeteo import fetch_era5
    end_d   = date.today()
    start_d = date(end_d.year - ERA5_HISTORY_YEARS, end_d.month, end_d.day)
    print(f"[ERA5] Fetching {start_d} → {end_d} for {city_key} …", flush=True)
    df = fetch_era5(start_d.isoformat(), end_d.isoformat(),
                    lat=lat, lon=lon, city_key=city_key, timezone=timezone)
    print(f"[ERA5] Got {len(df)} days.", flush=True)
    return df


# ---------------------------------------------------------------------------
# Feels-like (NWS)
# ---------------------------------------------------------------------------

def _feels_like(temp_c: float, rh_pct: float,
                wind_ms: Optional[float]) -> Optional[float]:
    try:
        T = temp_c * 9.0 / 5.0 + 32.0
        R = max(0.0, min(100.0, rh_pct or 0.0))
        V = (wind_ms or 0.0) * 2.236936
        if T >= 80.0:
            simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094)
            if (simple + T) / 2.0 < 80.0:
                f = simple
            else:
                f = (-42.379 + 2.04901523*T + 10.14333127*R
                     - 0.22475541*T*R - 0.00683783*T*T - 0.05481717*R*R
                     + 0.00122874*T*T*R + 0.00085282*T*R*R
                     - 0.00000199*T*T*R*R)
                if R < 13.0 and 80.0 <= T <= 112.0:
                    f -= ((13.0 - R) / 4.0) * ((17.0 - abs(T - 95.0)) / 17.0)**0.5
                elif R > 85.0 and 80.0 <= T <= 87.0:
                    f += ((R - 85.0) / 10.0) * ((87.0 - T) / 5.0)
        elif T <= 50.0 and V >= 3.0:
            f = 35.74 + 0.6215*T - 35.75*(V**0.16) + 0.4275*T*(V**0.16)
        else:
            f = T
        return round((f - 32.0) * 5.0 / 9.0, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main forecast runner  (city-agnostic)
# ---------------------------------------------------------------------------

def run_forecast(
    city_key: str = "delhi",
    horizon: int = HORIZON,
    modes: Optional[list] = None,
    model_names: Optional[list] = None,
    raw_era5_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> dict:
    """
    Run t+1 → t+horizon forecasts for *any* supported city.

    Parameters
    ----------
    city_key    : "delhi", "mumbai", "chennai" (aliases: "maharashtra", "tamil_nadu")
    horizon     : number of days ahead (default 5)
    modes       : list of (mode_label, fmode_key) tuples; None = both MV + UV
    model_names : list of model names; None = all 6
    raw_era5_df : pre-fetched ERA5 DataFrame (None = fetch from Open-Meteo)
    verbose     : whether to print per-step logs

    Returns
    -------
    dict with keys: df, summary, feels_like, mode_dfs, run_date, data_through,
                    city_key, city_label
    """
    if city_key not in CITY_REGISTRY:
        raise ValueError(
            f"Unknown city_key '{city_key}'. "
            f"Supported: {list(CITY_REGISTRY.keys())}")

    city_cfg   = CITY_REGISTRY[city_key]
    city_folder = _canonical(city_key)   # canonical folder: "mumbai", "chennai", "delhi"
    lat        = city_cfg["lat"]
    lon        = city_cfg["lon"]
    timezone   = city_cfg["timezone"]
    city_label = city_cfg["label"]

    modes       = modes       or list(MODES)
    model_names = model_names or list(MODEL_NAMES)

    model_root = _find_model_root(city_folder)

    # 1. Fetch ERA5 history
    if raw_era5_df is None:
        raw_era5_df = _fetch_era5_history(city_folder, lat, lon, timezone)

    raw_era5_df["date"] = pd.to_datetime(raw_era5_df["date"])
    raw_era5_df = (raw_era5_df
                   .sort_values("date")
                   .drop_duplicates("date")
                   .reset_index(drop=True))

    raw_era5_df = _normalise_era5_units(raw_era5_df, verbose=verbose)

    last_data_date = raw_era5_df["date"].max()
    data_through   = last_data_date.strftime("%Y-%m-%d")

    today     = date.today()
    today_str = today.isoformat()

    if verbose:
        print(f"[LiveForecast] City          : {city_label} ({city_folder})")
        print(f"[LiveForecast] ERA5 data through: {data_through}")
        print(f"[LiveForecast] Forecasting t+1 … t+{horizon} from {today_str}")

    all_rows   = []
    mode_dfs   = {}
    feels_info = []

    for mode_label, fmode_key in modes:
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Mode: {mode_label.upper()}")
            print(f"{'='*55}")

        try:
            meta = _load_metadata(model_root, city_folder, mode_label)
        except FileNotFoundError as e:
            print(f"  [SKIP] metadata.json not found for {city_folder}/{mode_label}: {e}")
            continue

        feats_by_tgt = meta["features"][fmode_key]

        # Load all model assets up-front
        loaded: dict = {}
        for target in TARGETS:
            loaded[target] = {}
            feats = feats_by_tgt[target]
            for mn in model_names:
                try:
                    m, sc_X, sc_y = _load_model_assets(
                        model_root, city_folder, mode_label, mn, target, len(feats))
                    pkl_path = _model_dir(
                        model_root, city_folder, mode_label, mn) / f"scaler_{target}.pkl"
                    hp = joblib.load(pkl_path).get("hp", {})
                    sl = int(hp.get("seq_len", SEQ_LEN))
                    loaded[target][mn] = (m, sc_X, sc_y, sl)
                    if verbose:
                        print(f"  Loaded {city_folder}/{mode_label}/{mn}/{target} "
                              f"({len(feats)} feats, seq_len={sl})")
                except Exception as e:
                    print(f"  [WARN] Could not load "
                          f"{city_folder}/{mode_label}/{mn}/{target}: {e}")

        raw_hist = raw_era5_df.copy()
        last_ws  = float(raw_hist["WS"].dropna().iloc[-1]) \
                   if "WS" in raw_hist.columns and len(raw_hist["WS"].dropna()) else None

        mode_rows = []

        for h in range(1, horizon + 1):
            fdate_d = today + timedelta(days=h)
            fdate   = fdate_d.isoformat()

            feat_hist = _engineer(raw_hist)
            if len(feat_hist) < 2:
                print(f"  [h={h}] Not enough feature history — skipping.")
                continue

            day_ens: dict[str, float] = {}

            for target in TARGETS:
                feats   = feats_by_tgt[target]
                missing = [f for f in feats if f not in feat_hist.columns]
                if missing:
                    if verbose:
                        print(f"  [h={h}][{target}] Missing feats: {missing[:4]}…")
                    continue

                preds: dict[str, float] = {}
                for mn, (model, sc_X, sc_y, sl) in loaded.get(target, {}).items():
                    try:
                        pred = _forecast_one(model, feat_hist, feats, sc_X, sc_y, sl)
                        preds[mn] = pred
                        mode_rows.append({
                            "run_date":      today_str,
                            "data_through":  data_through,
                            "city":          city_key,
                            "mode":          mode_label,
                            "forecast_date": fdate,
                            "horizon":       h,
                            "target":        target,
                            "model":         mn,
                            "prediction":    round(pred, 3),
                            "unit":          UNITS[target],
                        })
                    except Exception as e:
                        if verbose:
                            print(f"  [h={h}][{target}][{mn}] error: {e}")

                if preds:
                    ens = float(np.mean(list(preds.values())))
                    day_ens[target] = ens
                    mode_rows.append({
                        "run_date":      today_str,
                        "data_through":  data_through,
                        "city":          city_key,
                        "mode":          mode_label,
                        "forecast_date": fdate,
                        "horizon":       h,
                        "target":        target,
                        "model":         "ensemble",
                        "prediction":    round(ens, 3),
                        "unit":          UNITS[target],
                    })

            # Feels-like
            if "TMP2m" in day_ens:
                fl = _feels_like(day_ens["TMP2m"], day_ens.get("RH2m"), last_ws)
                _Tf = day_ens["TMP2m"] * 9.0 / 5.0 + 32.0
                method = ("heat_index" if _Tf >= 80.0
                          else ("wind_chill" if _Tf <= 50.0 else "actual"))
                feels_info.append({
                    "city":          city_key,
                    "mode":          mode_label,
                    "forecast_date": fdate,
                    "horizon":       h,
                    "temp":          round(day_ens["TMP2m"], 2),
                    "feels_like":    fl,
                    "rh":            round(day_ens.get("RH2m", float("nan")), 1),
                    "method":        method,
                })

            # Append predicted day to raw_hist for rollout
            if day_ens:
                prev = raw_hist.iloc[-1].to_dict()
                prev["date"] = pd.Timestamp(fdate_d)
                for t in TARGETS:
                    if t in day_ens:
                        val = day_ens[t]
                        prev[t] = val
                        if t == "TMP2m":
                            val_c = float(np.clip(val, -10.0, 60.0))
                            prev["TMP2m_max"] = val_c + 2.5
                            prev["TMP2m_min"] = val_c - 2.5
                raw_hist = pd.concat(
                    [raw_hist, pd.DataFrame([prev])], ignore_index=True)

            if verbose and day_ens:
                parts = ", ".join(
                    f"{t}={round(v,2)}{UNITS[t]}" for t, v in day_ens.items())
                print(f"  t+{h} ({fdate}):  {parts}")

        mode_df = pd.DataFrame(mode_rows)
        mode_dfs[mode_label] = mode_df
        all_rows.extend(mode_rows)

    # Build outputs
    df = pd.DataFrame(all_rows)

    summary: dict = {}
    if not df.empty:
        ens_df = df[df["model"] == "ensemble"].copy()
        for h in range(1, horizon + 1):
            summary[h] = {}
            sub     = ens_df[ens_df["horizon"] == h]
            fdate_h = (today + timedelta(days=h)).isoformat()
            for target in TARGETS:
                vals = sub[sub["target"] == target]["prediction"].dropna().values
                if len(vals):
                    summary[h][target] = round(float(np.mean(vals)), 3)
            summary[h]["forecast_date"] = fdate_h

    return {
        "df":           df,
        "summary":      summary,
        "feels_like":   feels_info,
        "mode_dfs":     mode_dfs,
        "run_date":     today_str,
        "data_through": data_through,
        "city_key":     city_key,
        "city_label":   city_label,
    }


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

def run_forecast_delhi(**kwargs) -> dict:
    """Alias for run_forecast(city_key='delhi', ...)."""
    kwargs.setdefault("city_key", "delhi")
    return run_forecast(**kwargs)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_forecast(result: dict) -> None:
    city_label = result.get("city_label", result.get("city_key", "?").capitalize())
    print("\n" + "="*60)
    print(f"  {city_label.upper()} LIVE FORECAST  —  ERA5 Open-Meteo inputs")
    print(f"  Run date    : {result['run_date']}")
    print(f"  Data through: {result['data_through']}")
    print("="*60)

    summary = result["summary"]
    for h in sorted(summary.keys()):
        s     = summary[h]
        fdate = s.get("forecast_date", "")
        parts = [f"{t}={s[t]}{UNITS[t]}" for t in TARGETS if t in s]
        print(f"  t+{h}  ({fdate}):  {',  '.join(parts)}")

    city_key = result.get("city_key", "delhi")
    fl_by_h = {e["horizon"]: e for e in result["feels_like"]
               if e["mode"] == "multivariate" and e.get("city", city_key) == city_key}
    if fl_by_h:
        print()
        print(f"  Feels-like temperature (multivariate ensemble):")
        for h in sorted(fl_by_h.keys()):
            e = fl_by_h[h]
            print(f"    t+{h}  ({e['forecast_date']}):  "
                  f"{e['feels_like']}°C  [actual {e['temp']}°C, "
                  f"RH {e['rh']}%, method={e['method']}]")

    print()
    print("  Per-mode ensemble detail:")
    df = result["df"]
    if not df.empty:
        ens = (df[df["model"] == "ensemble"]
               .pivot_table(index=["mode", "horizon", "forecast_date"],
                            columns="target",
                            values="prediction",
                            aggfunc="first")
               .reset_index())
        print(ens.to_string(index=False))
    print("="*60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-city ERA5 live t+1→t+5 forecast")
    parser.add_argument("--city", default="delhi",
                        choices=list(CITY_REGISTRY.keys()),
                        help="City to forecast (default: delhi)")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_forecast(
        city_key=args.city,
        horizon=args.horizon,
        verbose=not args.quiet,
    )
    print_forecast(result)

    out_csv = HERE / f"{args.city}_live_forecast_era5.csv"
    result["df"].to_csv(out_csv, index=False)
    print(f"\n[Saved] full forecast → {out_csv}")