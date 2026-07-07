"""
Daily LIVE entry point (cron). Implements the full real-time cycle:

  PHASE 1 - VERIFY + READJUST
    For each past forecast in the ledger that has no actual yet, ask
    providers.fetch_truth() (ERA5 primary + NASA gate). If both sources agree
    (confident=True), record the actual + error AND fine-tune the model toward
    that day. If they disagree too much, record the actual but SKIP the weight
    update (low-confidence day).

  PHASE 2 - FORECAST
    Pull the latest NASA data, forecast the next day with all 4 models, append
    to the ledger with actual=None (to be verified on a future run).

This is the forecast -> check-against-providers -> adjust loop, on genuinely
live (not backdated) data.

Crontab (daily 06:30):
  30 6 * * *  cd /path/to/realtime/src && python3 live_run.py delhi >> ../logs/cron.log 2>&1
"""
import sys
import json
from datetime import timedelta

import numpy as np
import pandas as pd

import nasa
import features
import providers
from engine import (load_assets, finetune_model, forecast_next, save_assets,
                    build_sequences, snapshot_weights, weight_delta, set_seed)
from config import (CITIES, TARGETS, MODEL_NAMES, UNITS, MODEL_DIR,
                    FORECAST_DIR, LOG_DIR)


def _meta(city):
    with open(MODEL_DIR / city / "metadata.json") as f:
        return json.load(f)


def _ledger_path(city):
    return FORECAST_DIR / f"{city}_forecasts.csv"


# ---------------------------------------------------------------------------
# PHASE 1 — verify past forecasts + readjust on confident days
# ---------------------------------------------------------------------------
def verify_and_readjust(city, meta, df_feat):
    ledger = _ledger_path(city)
    if not ledger.exists():
        print(f"[{city}] no ledger yet — nothing to verify.")
        return

    led = pd.read_csv(ledger)
    pending = led[led["actual"].isna()]
    if pending.empty:
        print(f"[{city}] no pending forecasts to verify.")
        return

    feats_by_target = meta["features"]["mv"]
    weight_log = []

    # group pending rows by date; fetch truth once per date
    for date, grp in pending.groupby("date"):
        truth = providers.fetch_truth(city, date)
        day_entry = {"date": date, "models": {}}
        for _, r in grp.iterrows():
            tgt, mdl = r["target"], r["model"]
            t = truth.get(tgt, {})
            actual, confident = t.get("value"), t.get("confident", False)
            if actual is None:
                continue  # provider has no data for that date yet

            # record actual + error in the ledger
            mask = ((led["date"] == date) & (led["target"] == tgt) &
                    (led["model"] == mdl))
            led.loc[mask, "actual"] = round(float(actual), 4)
            led.loc[mask, "error"] = round(float(r["prediction"]) - float(actual), 4)
            led.loc[mask, "confident"] = bool(confident)
            led.loc[mask, "nasa"] = t.get("nasa")
            led.loc[mask, "era5"] = t.get("era5")

            # READJUST only on confident days
            if confident:
                feats = feats_by_target[tgt]
                model, sc_X, sc_y = load_assets(city, mdl, tgt, len(feats))
                before = snapshot_weights(model)
                X, y = build_sequences(df_feat, feats, tgt, sc_X, sc_y, fit=False)
                model = finetune_model(model, X, y)
                save_assets(city, mdl, tgt, model, sc_X, sc_y)
                after = snapshot_weights(model)
                day_entry["models"][f"{mdl}_{tgt}"] = {
                    "confident": True,
                    "disagree": t.get("disagree"),
                    "delta": weight_delta(before, after),
                }
            else:
                day_entry["models"][f"{mdl}_{tgt}"] = {
                    "confident": False, "disagree": t.get("disagree"),
                    "note": "providers disagreed — weight update skipped",
                }
        weight_log.append(day_entry)

    led.to_csv(ledger, index=False)
    if weight_log:
        wlog_path = LOG_DIR / f"{city}_live_weight_changes.json"
        existing = []
        if wlog_path.exists():
            existing = json.load(open(wlog_path))
        json.dump(existing + weight_log, open(wlog_path, "w"), indent=2)
    print(f"[{city}] verified {len(pending)} pending rows "
          f"({len(weight_log)} dates).")


# ---------------------------------------------------------------------------
# PHASE 2 — forecast the next day
# ---------------------------------------------------------------------------
def forecast_tomorrow(city, meta, df_feat):
    feats_by_target = meta["features"]["mv"]
    latest = df_feat.iloc[-1]["date"]
    forecast_date = latest + timedelta(days=1)
    print(f"[{city}] latest data {latest.date()} -> forecasting {forecast_date.date()}")

    rows = []
    for target in TARGETS:
        feats = feats_by_target[target]
        preds = {}
        for model_name in MODEL_NAMES:
            model, sc_X, sc_y = load_assets(city, model_name, target, len(feats))
            pred = forecast_next(model, df_feat, feats, target, sc_X, sc_y)
            preds[model_name] = pred
            rows.append({"date": str(forecast_date.date()), "target": target,
                         "model": model_name, "prediction": round(pred, 4),
                         "actual": np.nan, "error": np.nan, "confident": np.nan,
                         "nasa": np.nan, "era5": np.nan, "unit": UNITS[target]})
        # ensemble row (mean of 4 models)
        rows.append({"date": str(forecast_date.date()), "target": target,
                     "model": "ensemble",
                     "prediction": round(float(np.mean(list(preds.values()))), 4),
                     "actual": np.nan, "error": np.nan, "confident": np.nan,
                     "nasa": np.nan, "era5": np.nan, "unit": UNITS[target]})

    new = pd.DataFrame(rows)
    ledger = _ledger_path(city)
    if ledger.exists():
        new = pd.concat([pd.read_csv(ledger), new], ignore_index=True)
    new.to_csv(ledger, index=False)
    print(f"[{city}] forecast written -> {ledger.name}")


def live_run(city):
    set_seed()
    meta = _meta(city)
    df_feat = features.engineer_features(nasa.load_city(city))
    verify_and_readjust(city, meta, df_feat)   # PHASE 1
    forecast_tomorrow(city, meta, df_feat)     # PHASE 2


if __name__ == "__main__":
    cities = [sys.argv[1]] if len(sys.argv) > 1 else ["delhi"]
    for c in cities:
        live_run(c)
