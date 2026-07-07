"""
run_live_era5.py  (v2 — multi-city, June 2026)
-----------------------------------------------
Entry-point script for the ERA5-based live forecasting pipeline.

Supports Delhi, Mumbai, and Chennai.  Run daily (e.g. via cron at 07:00 IST)
to produce t+1 → t+5 forecasts using Open-Meteo ERA5 reanalysis data, with
both Multivariate and Univariate model sets.

Crontab example (all three cities, 07:00 IST):
    0 7 * * *  cd /path/to/project/src && python3 run_live_era5.py --city all >> ../logs/era5_live.log 2>&1

Usage
-----
    python run_live_era5.py                        # Delhi (default), both modes
    python run_live_era5.py --city mumbai          # Mumbai only
    python run_live_era5.py --city chennai         # Chennai only
    python run_live_era5.py --city all             # Delhi + Mumbai + Chennai
    python run_live_era5.py --horizon 3            # only t+1..t+3
    python run_live_era5.py --mode mv              # multivariate only
    python run_live_era5.py --models lstm gru ann  # subset of models
    python run_live_era5.py --quiet                # suppress per-step output

Output (per city)
-----------------
* <city>_live_era5.csv      — full per-model ledger (appended, deduplicated)
* <city>_era5_summary.json  — compact JSON summary (latest run only)
* data/<city>/openmeteo_cache.csv — actual ERA5/forecast values fetched
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE         = Path(__file__).parent          # src/
PROJECT_ROOT = HERE.parent                    # project root
FORE_ROOT    = PROJECT_ROOT / "forecasts"
FORE_ROOT.mkdir(parents=True, exist_ok=True)

ALL_MODES  = [("multivariate", "mv"), ("univariate", "uv")]
ALL_MODELS = ["hqnn"]

# Supported single-city keys (and "all")
CITY_CHOICES = ["delhi", "mumbai", "maharashtra", "chennai", "tamil_nadu", "all"]
# Canonical city order when --city all
ALL_CITIES = ["delhi", "mumbai", "chennai"]


def _city_fore_dir(city_key: str) -> Path:
    """forecasts/<city_key>/  — created on demand."""
    d = FORE_ROOT / city_key
    d.mkdir(parents=True, exist_ok=True)
    return d

def _csv_path(city_key: str) -> Path:
    """forecasts/<city_key>/<city_key>_live_era5.csv"""
    return _city_fore_dir(city_key) / f"{city_key}_live_era5.csv"

def _json_path(city_key: str) -> Path:
    """forecasts/<city_key>/multiday_forecast.json"""
    return _city_fore_dir(city_key) / "multiday_forecast.json"


def _parse_args():
    p = argparse.ArgumentParser(description="Multi-city ERA5 live t+1→t+5 forecast")
    p.add_argument("--city", default="delhi",
                   choices=CITY_CHOICES,
                   help="City to forecast, or 'all' for Delhi+Mumbai+Chennai (default: delhi)")
    p.add_argument("--horizon", type=int, default=5,
                   help="Days ahead to forecast (default: 5)")
    p.add_argument("--mode", choices=["mv", "uv", "both"], default="both",
                   help="Which mode(s) to run (default: both)")
    p.add_argument("--models", nargs="+", default=None,
                   choices=ALL_MODELS,
                   help="Subset of models to use (default: hqnn)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-horizon verbose output")
    return p.parse_args()


def _select_modes(mode_arg: str) -> list:
    if mode_arg == "mv":  return [("multivariate", "mv")]
    if mode_arg == "uv":  return [("univariate",   "uv")]
    return list(ALL_MODES)


def _append_csv(new_df: pd.DataFrame, city_key: str) -> None:
    """Append new rows to the city ledger CSV, deduplicating on key columns."""
    key  = ["run_date", "city", "mode", "forecast_date", "horizon", "target", "model"]
    path = _csv_path(city_key)
    if path.exists():
        existing = pd.read_csv(path)
        # Handle old CSVs that don't have a 'city' column
        for k in key:
            if k not in existing.columns:
                existing[k] = city_key if k == "city" else None
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key, keep="last")
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    print(f"[Saved] ledger  → {path}  ({len(combined)} total rows)")


def _save_summary(result: dict) -> None:
    """Write forecasts/<city>/multiday_forecast.json — compact, latest run only."""
    city_key = result["city_key"]
    df = result["df"]

    forecasts = []
    if not df.empty:
        ens = df[df["model"] == "ensemble"].copy()
        for h in sorted(result["summary"].keys()):
            s     = result["summary"][h]
            fdate = s.get("forecast_date", "")
            fl_rec = next(
                (e for e in result["feels_like"]
                 if e["horizon"] == h and e["mode"] == "multivariate"),
                next((e for e in result["feels_like"] if e["horizon"] == h), None)
            )
            entry = {
                "horizon":       h,
                "forecast_date": fdate,
                "TMP2m":         s.get("TMP2m"),
                "RH2m":          s.get("RH2m"),
                "PRmsl":         s.get("PRmsl"),
                "feels_like_c":  fl_rec["feels_like"] if fl_rec else None,
                "feels_method":  fl_rec["method"]     if fl_rec else None,
                "modes":         {},
            }
            for mode_label in ["multivariate", "univariate"]:
                sub = ens[(ens["mode"] == mode_label) & (ens["horizon"] == h)]
                if not sub.empty:
                    entry["modes"][mode_label] = {
                        t: round(float(sub[sub["target"] == t]["prediction"].mean()), 3)
                        for t in ["TMP2m", "RH2m", "PRmsl"]
                        if not sub[sub["target"] == t].empty
                    }
            forecasts.append(entry)

    payload = {
        "city":         city_key,
        "city_label":   result.get("city_label", city_key.capitalize()),
        "run_date":     result["run_date"],
        "data_through": result["data_through"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "forecasts":    forecasts,
    }
    path = _json_path(city_key)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] forecast → {path}")


def _run_city(city_key: str, args) -> bool:
    """Run the full pipeline for one city. Returns True on success."""
    print("\n" + "=" * 60)
    print(f"  City: {city_key.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        from live_forecast_era5 import run_forecast, print_forecast
    except ImportError as e:
        print(f"[ERROR] Could not import live_forecast_era5: {e}")
        return False

    modes   = _select_modes(args.mode)
    models  = args.models or ALL_MODELS   # default: hqnn only
    verbose = not args.quiet

    result = run_forecast(
        city_key=city_key,
        horizon=args.horizon,
        modes=modes,
        model_names=models,
        verbose=verbose,
    )

    print_forecast(result)

    if not result["df"].empty:
        _append_csv(result["df"], city_key)
        _save_summary(result)
    else:
        print(f"[WARN] No forecast rows for {city_key} — "
              "check model paths and ERA5 connectivity.")
        return False

    return True


def main():
    args = _parse_args()

    cities = ALL_CITIES if args.city == "all" else [args.city]

    success, failed = [], []
    for city in cities:
        ok = _run_city(city, args)
        (success if ok else failed).append(city)

    print("\n" + "=" * 60)
    print(f"  Done.  Success: {success or 'none'}  |  Failed: {failed or 'none'}")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()