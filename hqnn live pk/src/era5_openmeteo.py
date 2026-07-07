"""
era5_openmeteo.py  (v3 — multi-city + correct cache paths, June 2026)
----------------------------------------------------------------------
3-tier data strategy for live HQNN forecasting:
  Tier 1: NASA cache    (data/<city>/nasa_cache.csv)     — 2021-2025, instant
  Tier 2: Open-Meteo Forecast API past_days              — last 92 days
  Tier 3: Open-Meteo Archive API  models=era5            — fallback for gaps

Cache storage (v3 fix)
----------------------
  NASA history  : data/<city>/nasa_cache.csv
  Open-Meteo live: data/<city>/openmeteo_cache.csv
  Both paths resolve relative to the PROJECT ROOT (parent of src/).
  _city_data_dir() = project_root / data / <city_key>

UNIT FIXES (confirmed against scaler data_min_/data_max_)
  WS    → m/s       (Open-Meteo returns km/h; ÷3.6)
  QV2m  → g/kg      (Magnus formula gives kg/kg; ×1000)
  PRmsl → Pa        (API returns hPa; ×100)
  SLP   → hPa       (left as hPa — scaler trained on hPa ~998–1013)
  PS    → hPa       (left as hPa — scaler trained on hPa ~974–999)
  PS_Pa → Pa        (hPa×100)
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Default coordinates (Delhi — kept for backward-compat CLI)
# ---------------------------------------------------------------------------
DELHI_LAT = 28.61
DELHI_LON = 77.20

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

FORECAST_PAST_MAX = 92

# src/ → parent = project root
_HERE         = Path(__file__).resolve().parent          # …/src
_PROJECT_ROOT = _HERE.parent                             # …/hqnn live pk

# Canonical columns stored in both nasa_cache and openmeteo_cache
_CACHE_COLS = ["date", "TMP2m", "TMP2m_max", "TMP2m_min", "RH2m",
               "PRCP", "WS", "WD", "U10m", "V10m", "QV2m",
               "PRmsl", "SLP", "PS", "PS_Pa"]

# ---------------------------------------------------------------------------
# City registry  (lat / lon / data-folder key)
# ---------------------------------------------------------------------------
CITY_COORDS: dict[str, dict] = {
    "delhi":       {"lat": 28.61, "lon": 77.20},
    "mumbai":      {"lat": 19.08, "lon": 72.88},
    "maharashtra": {"lat": 19.08, "lon": 72.88},   # alias
    "chennai":     {"lat": 13.08, "lon": 80.27},
    "tamil_nadu":  {"lat": 13.08, "lon": 80.27},   # alias
}


def _city_data_dir(city_key: str) -> Path:
    """
    Return  project_root/data/<city_key>/
    e.g.   …/hqnn live pk/data/delhi/
    Creates the directory if it does not exist.
    """
    d = (_PROJECT_ROOT / "data" / city_key).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Daily variable lists
# ---------------------------------------------------------------------------
_FORECAST_DAILY = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "surface_pressure_mean",
    "pressure_msl_mean",
]

_ARCHIVE_DAILY = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "surface_pressure",
    "pressure_msl",
]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _qv2m_gkg(temp_c, rh_pct, ps_hpa):
    """Specific humidity in g/kg — matching NASA POWER QV2M training units."""
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in [temp_c, rh_pct, ps_hpa]):
        return np.nan
    es  = 6.112 * math.exp(17.67 * temp_c / (temp_c + 243.5))
    e   = (rh_pct / 100.0) * es
    w   = 0.622 * e / max(ps_hpa - e, 1e-6)
    qv  = w / (1.0 + w)
    return qv * 1000.0


def _uv_ms(ws_ms, wd_deg):
    """Wind components (U10m, V10m) both in m/s."""
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in [ws_ms, wd_deg]):
        return np.nan, np.nan
    r = math.radians(wd_deg)
    return round(-ws_ms * math.sin(r), 4), round(-ws_ms * math.cos(r), 4)


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))

def _r(x, n=4):
    return round(x, n) if not _nan(x) else np.nan


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_rows(daily: dict, source: str = "archive") -> list[dict]:
    """
    Build NASA-POWER-compatible rows from an Open-Meteo daily dict.
    Unit conversions:
      ws_kmh  ÷ 3.6   → ws_ms      (km/h → m/s)
      qv2m            → g/kg       (Magnus formula)
      msl_hpa × 100   → PRmsl_Pa   (hPa → Pa)
      ps_hpa          → SLP / PS   (stays as hPa)
      ps_hpa × 100    → PS_Pa
    """
    dates = daily.get("time", [])
    rows  = []
    for i, d in enumerate(dates):
        def g(key, default=np.nan):
            arr = daily.get(key, [])
            v   = arr[i] if i < len(arr) else None
            return v if v is not None else default

        if source == "forecast":
            tmax    = g("temperature_2m_max")
            tmin    = g("temperature_2m_min")
            tmp     = (tmax + tmin) / 2.0 if not (_nan(tmax) or _nan(tmin)) else np.nan
            ws_kmh  = g("wind_speed_10m_max")
            ps_hpa  = g("surface_pressure_mean")
            msl_hpa = g("pressure_msl_mean")
        else:  # archive
            tmp     = g("temperature_2m_mean")
            tmax    = g("temperature_2m_max")
            tmin    = g("temperature_2m_min")
            ws_kmh  = g("wind_speed_10m_mean")
            ps_hpa  = g("surface_pressure")
            msl_hpa = g("pressure_msl")

        rh   = g("relative_humidity_2m_mean")
        prcp = g("precipitation_sum", 0.0)
        wd   = g("wind_direction_10m_dominant", 0.0)

        ws_ms    = ws_kmh / 3.6 if not _nan(ws_kmh) else np.nan
        u10, v10 = _uv_ms(ws_ms, wd) if not (_nan(ws_ms) or _nan(wd)) else (np.nan, np.nan)
        qv_gkg   = _qv2m_gkg(tmp, rh, ps_hpa) if not any(
            _nan(x) for x in [tmp, rh, ps_hpa]) else np.nan
        prmsl_pa = msl_hpa * 100.0 if not _nan(msl_hpa) else np.nan
        slp_hpa  = msl_hpa
        ps_pa    = ps_hpa  * 100.0 if not _nan(ps_hpa)  else np.nan

        rows.append({
            "date":      pd.to_datetime(d),
            "TMP2m":     _r(tmp,      4),
            "TMP2m_max": _r(tmax,     4),
            "TMP2m_min": _r(tmin,     4),
            "RH2m":      _r(rh,       4),
            "PRCP":      _r(prcp,     4),
            "WS":        _r(ws_ms,    4),   # m/s  ✓
            "WD":        _r(wd,       2),
            "U10m":      u10,               # m/s  ✓
            "V10m":      v10,               # m/s  ✓
            "QV2m":      _r(qv_gkg,   4),  # g/kg ✓
            "PRmsl":     _r(prmsl_pa, 2),  # Pa   ✓
            "SLP":       _r(slp_hpa,  4),  # hPa  ✓
            "PS":        _r(ps_hpa,   4),  # hPa  ✓
            "PS_Pa":     _r(ps_pa,    2),  # Pa   ✓
        })
    return rows


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url, params, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt < retries - 1:
                import time; time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"Open-Meteo fetch failed ({url}): {exc}") from exc
    return {}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> pd.DataFrame:
    """Load nasa_cache.csv or openmeteo_cache.csv; return empty DF if absent."""
    if not cache_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(cache_path, parse_dates=["date"])
    for col in _CACHE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[_CACHE_COLS].sort_values("date").reset_index(drop=True)


def _save_openmeteo_cache(new_rows: list[dict], city_key: str,
                          cache_path: Optional[Path] = None) -> None:
    """
    Append freshly-fetched Open-Meteo rows to:
        data/<city_key>/openmeteo_cache.csv

    These rows are the *actual* ERA5 / forecast-API values — not NASA POWER
    derived values — giving a clean record of what Open-Meteo returned.
    """
    if not new_rows:
        return

    if cache_path is None:
        data_dir   = _city_data_dir(city_key)   # creates dir if needed
        cache_path = data_dir / "openmeteo_cache.csv"

    new_df = pd.DataFrame(new_rows)
    for col in _CACHE_COLS:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[_CACHE_COLS].copy()
    new_df["date"] = pd.to_datetime(new_df["date"])

    if cache_path.exists():
        existing = pd.read_csv(cache_path, parse_dates=["date"])
        combined = (pd.concat([existing, new_df], ignore_index=True)
                    .drop_duplicates("date", keep="last")
                    .sort_values("date")
                    .reset_index(drop=True))
    else:
        combined = (new_df
                    .drop_duplicates("date", keep="last")
                    .sort_values("date")
                    .reset_index(drop=True))

    combined.to_csv(cache_path, index=False)
    print(f"[OpenMeteoCache] {city_key}: {len(new_df)} rows → "
          f"{cache_path}  "
          f"({combined['date'].min().date()} → {combined['date'].max().date()})")


# ---------------------------------------------------------------------------
# Tier 2: Forecast API  past_days
# ---------------------------------------------------------------------------

def _fetch_forecast_past(past_days, lat, lon, timezone):
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         ",".join(_FORECAST_DAILY),
        "past_days":     min(past_days, FORECAST_PAST_MAX),
        "forecast_days": 1,
        "timezone":      timezone,
    }
    data  = _get(FORECAST_URL, params)
    rows  = _build_rows(data.get("daily", {}), source="forecast")
    today = pd.Timestamp(date.today())
    return [r for r in rows if r["date"] <= today]


# ---------------------------------------------------------------------------
# Tier 3: Archive API  models=era5
# ---------------------------------------------------------------------------

def _fetch_archive(start, end, lat, lon, timezone):
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(_ARCHIVE_DAILY),
        "models":     "era5",
        "timezone":   timezone,
    }
    data = _get(ARCHIVE_URL, params)
    return _build_rows(data.get("daily", {}), source="archive")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_era5(start, end,
               lat: float = DELHI_LAT,
               lon: float = DELHI_LON,
               city_key: str = "delhi",
               timezone: str = "Asia/Kolkata",
               nasa_cache: Optional[str] = None) -> pd.DataFrame:
    """
    Return NASA-POWER-compatible DataFrame for [start, end].

    Cache paths (always under project_root/data/<city_key>/):
      nasa_cache.csv      — historical NASA POWER data (Tier 1)
      openmeteo_cache.csv — actual Open-Meteo fetched values (Tiers 2/3)

    All columns match training units (see module docstring).
    """
    if isinstance(start, date): start = start.isoformat()
    if isinstance(end,   date): end   = end.isoformat()
    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)

    # Resolve nasa_cache path — always in data/<city>/
    if nasa_cache is None:
        data_dir        = _city_data_dir(city_key)
        nasa_cache_path = data_dir / "nasa_cache.csv"
    else:
        nasa_cache_path = Path(nasa_cache)

    print(f"[ERA5] Cache path: {nasa_cache_path}")

    # ── Tier 1: NASA cache ────────────────────────────────────────────────
    cache_df = _load_cache(nasa_cache_path)
    if not cache_df.empty:
        mask       = ((cache_df["date"] >= pd.Timestamp(start_d)) &
                      (cache_df["date"] <= pd.Timestamp(end_d)))
        from_cache = cache_df[mask].copy()
        cache_end  = cache_df["date"].max().date()
    else:
        from_cache = pd.DataFrame()
        cache_end  = date(2000, 1, 1)

    frames             = [from_cache] if not from_cache.empty else []
    openmeteo_rows: list[dict] = []

    if cache_end < end_d:
        gap_start = max(start_d, cache_end + timedelta(days=1))
        gap_days  = (date.today() - gap_start).days + 1

        # ── Tier 2: Forecast past_days ────────────────────────────────────
        if gap_days > 0:
            try:
                rows = _fetch_forecast_past(
                    min(gap_days + 5, FORECAST_PAST_MAX), lat, lon, timezone)
                if rows:
                    rdf = pd.DataFrame(rows)
                    rdf = rdf[(rdf["date"] >= pd.Timestamp(gap_start)) &
                              (rdf["date"] <= pd.Timestamp(end_d))]
                    if not rdf.empty:
                        frames.append(rdf)
                        openmeteo_rows.extend(rdf.to_dict("records"))
                        gap_start = rdf["date"].max().date() + timedelta(days=1)
            except Exception as exc:
                print(f"[ERA5] Forecast past_days failed: {exc} — trying archive")

        # ── Tier 3: Archive ERA5 ──────────────────────────────────────────
        if gap_start <= end_d:
            try:
                rows = _fetch_archive(
                    gap_start.isoformat(), end_d.isoformat(), lat, lon, timezone)
                if rows:
                    rdf = pd.DataFrame(rows)
                    frames.append(rdf)
                    openmeteo_rows.extend(rdf.to_dict("records"))
            except Exception as exc:
                print(f"[ERA5] Archive also failed: {exc}")

    # ── Save fetched rows to data/<city>/openmeteo_cache.csv ─────────────
    if openmeteo_rows:
        try:
            _save_openmeteo_cache(openmeteo_rows, city_key)
        except Exception as exc:
            print(f"[ERA5] openmeteo_cache save failed: {exc}")

    if not frames:
        return pd.DataFrame()

    df = (pd.concat(frames, ignore_index=True)
          .sort_values("date")
          .drop_duplicates("date")
          .reset_index(drop=True))

    # ── Auto-update nasa_cache.csv (archive only, ~5-day lag) ────────────
    try:
        if nasa_cache_path.exists():
            existing = pd.read_csv(nasa_cache_path, parse_dates=["date"])
            core     = ["TMP2m", "RH2m", "PRmsl"]
            bad_mask = existing[core].isnull().all(axis=1)
            if bad_mask.any():
                existing = existing[~bad_mask].reset_index(drop=True)
                print(f"[ERA5] Removed {bad_mask.sum()} NaN-only rows from nasa_cache.")

            existing_end = existing["date"].max().date()
            archive_end  = date.today() - timedelta(days=5)

            if existing_end < archive_end:
                gap_s = (existing_end + timedelta(days=1)).isoformat()
                gap_e = archive_end.isoformat()
                print(f"[ERA5] Fetching archive for nasa_cache: {gap_s} → {gap_e}")
                try:
                    archive_rows = _fetch_archive(gap_s, gap_e, lat, lon, timezone)
                    if archive_rows:
                        new_df  = pd.DataFrame(archive_rows)
                        new_df  = new_df[[c for c in _CACHE_COLS if c in new_df.columns]]
                        updated = (pd.concat([existing, new_df], ignore_index=True)
                                   .drop_duplicates("date")
                                   .sort_values("date")
                                   .reset_index(drop=True))
                        updated.to_csv(nasa_cache_path, index=False)
                        print(f"[ERA5] nasa_cache updated: +{len(new_df)} rows "
                              f"({new_df['date'].min().date()} → "
                              f"{new_df['date'].max().date()}). "
                              f"Now through {updated['date'].max().date()}.")
                        try:
                            _save_openmeteo_cache(archive_rows, city_key)
                        except Exception:
                            pass
                    else:
                        print("[ERA5] Archive returned no rows for nasa_cache gap.")
                        if bad_mask.any():
                            existing.to_csv(nasa_cache_path, index=False)
                except Exception as exc:
                    print(f"[ERA5] Archive fetch for nasa_cache failed: {exc}")
                    if bad_mask.any():
                        existing.to_csv(nasa_cache_path, index=False)
            else:
                print(f"[ERA5] nasa_cache up to date through {existing_end} (archive lag ~5 days).")
                if bad_mask.any():
                    existing.to_csv(nasa_cache_path, index=False)
    except Exception as exc:
        print(f"[ERA5] nasa_cache update skipped: {exc}")

    return df


def fetch_era5_history(years: int = 2,
                       lat: float = DELHI_LAT,
                       lon: float = DELHI_LON,
                       city_key: str = "delhi") -> pd.DataFrame:
    """Convenience wrapper: fetch last `years` years for a given city."""
    end_d   = date.today()
    start_d = date(end_d.year - years, end_d.month, end_d.day)
    return fetch_era5(start_d.isoformat(), end_d.isoformat(),
                      lat=lat, lon=lon, city_key=city_key)


# ---------------------------------------------------------------------------
# CLI  (python era5_openmeteo.py [days] [city_key])
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    days     = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    city_arg = sys.argv[2]      if len(sys.argv) > 2 else "delhi"
    coords   = CITY_COORDS.get(city_arg, {"lat": DELHI_LAT, "lon": DELHI_LON})

    end_d   = date.today()
    start_d = end_d - timedelta(days=days - 1)
    df = fetch_era5(start_d.isoformat(), end_d.isoformat(),
                    lat=coords["lat"], lon=coords["lon"], city_key=city_arg)
    print(df.tail(10).to_string())
    print(f"\nShape: {df.shape} | Last date: {df['date'].max()}")

    print("\n--- Unit verification ---")
    checks = {
        "WS (m/s)":    (df["WS"],    0.0, 25.0),
        "QV2m (g/kg)": (df["QV2m"],  0.5, 35.0),
        "PRmsl (Pa)":  (df["PRmsl"], 90000, 110000),
        "SLP (hPa)":   (df["SLP"],   900,  1100),
        "TMP2m (°C)":  (df["TMP2m"], -10,   50),
    }
    for name, (col, lo, hi) in checks.items():
        mn, mx = col.min(), col.max()
        ok = mn >= lo and mx <= hi
        print(f"  {'✓' if ok else '✗'}  {name:18s}  [{mn:.3f}, {mx:.3f}]"
              + ("" if ok else f"  EXPECTED [{lo}, {hi}]"))