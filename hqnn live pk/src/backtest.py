"""
Backtest walk-forward loop.

For each simulated day D in the backtest window (the last 20%):
  1. Use data up to D-1 to FORECAST day D (all 4 models + ensemble), raw units.
  2. The actual for day D already exists -> VERIFY (error per model).
  3. READJUST: fine-tune every model on the recent window now that D is known.
  4. Every RETRAIN_EVERY days: full retrain from scratch on all data up to D.

Outputs (per city):
  forecasts/<city>_forecasts.csv      flat ledger: date,target,model,prediction,actual,error,...
  verifications/<city>_metrics.json   running RMSE/MAE per model/target
  logs/<city>_weight_changes.json     daily weight/bias snapshots + deltas  <-- watch numbers move
"""
import json
import numpy as np
import pandas as pd

import nasa
import features
from engine import (set_seed, build_sequences, train_model, finetune_model,
                    forecast_next, load_assets, save_assets,
                    snapshot_weights, weight_delta)
from models import build_model
from config import (TARGETS, MODEL_NAMES, UNITS, SEQ_LEN, RETRAIN_EVERY,
                    MODEL_DIR, FORECAST_DIR, VERIFY_DIR, LOG_DIR)
from sklearn.preprocessing import MinMaxScaler


def _load_meta(city):
    with open(MODEL_DIR / city / "metadata.json") as f:
        return json.load(f)


def run_backtest(city, max_days=None, verbose=True):
    set_seed()
    meta = _load_meta(city)
    df_raw = nasa.load_city(city)
    df_feat = features.engineer_features(df_raw)
    split = meta["split_index"]

    feats_by_target = meta["features"]["mv"]

    # load all model assets once
    assets = {}   # (model_name, target) -> dict(model, sc_X, sc_y, feats)
    for target in TARGETS:
        feats = feats_by_target[target]
        for model_name in MODEL_NAMES:
            model, sc_X, sc_y = load_assets(city, model_name, target, len(feats))
            assets[(model_name, target)] = {
                "model": model, "sc_X": sc_X, "sc_y": sc_y, "feats": feats}

    forecast_rows = []
    weight_log = []
    backtest_idx = range(split, len(df_feat))
    if max_days:
        backtest_idx = list(backtest_idx)[:max_days]

    for step, D in enumerate(backtest_idx):
        day_date = df_feat.iloc[D]["date"]
        hist = df_feat.iloc[:D]          # data available up to D-1 (target of D unknown)
        actual_row = df_feat.iloc[D]

        day_entry = {"date": str(day_date.date()), "step": step, "models": {}}

        # ---- 1 & 2: forecast each model for day D, then verify ----
        for target in TARGETS:
            actual = float(actual_row[target])
            for model_name in MODEL_NAMES:
                a = assets[(model_name, target)]
                pred = forecast_next(a["model"], hist, a["feats"], target,
                                     a["sc_X"], a["sc_y"])
                forecast_rows.append({
                    "date": str(day_date.date()), "target": target,
                    "model": model_name, "prediction": round(pred, 4),
                    "actual": round(actual, 4), "error": round(pred - actual, 4),
                    "abs_error": round(abs(pred - actual), 4),
                    "unit": UNITS[target],
                    "retrained": (step % RETRAIN_EVERY == 0 and step > 0),
                })

        # ---- 3: readjust (daily fine-tune on data through D) ----
        df_through_D = df_feat.iloc[:D + 1]
        do_retrain = (step > 0 and step % RETRAIN_EVERY == 0)

        for target in TARGETS:
            feats = feats_by_target[target]
            for model_name in MODEL_NAMES:
                a = assets[(model_name, target)]
                before = snapshot_weights(a["model"])

                if do_retrain:
                    # full retrain from scratch on all data up to D
                    inner = int(len(df_through_D) * 0.85)
                    df_tr = df_through_D.iloc[:inner]
                    df_va = df_through_D.iloc[inner:]
                    sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
                    X_tr, y_tr = build_sequences(df_tr, feats, target, sc_X, sc_y, fit=True)
                    X_va, y_va = build_sequences(df_va, feats, target, sc_X, sc_y, fit=False)
                    set_seed()
                    m = build_model(model_name, len(feats))
                    m = train_model(m, X_tr, y_tr, X_va, y_va)
                    a.update({"model": m, "sc_X": sc_X, "sc_y": sc_y})
                else:
                    # daily fine-tune (scalers fixed)
                    X, y = build_sequences(df_through_D, feats, target,
                                           a["sc_X"], a["sc_y"], fit=False)
                    a["model"] = finetune_model(a["model"], X, y)

                save_assets(city, model_name, target, a["model"], a["sc_X"], a["sc_y"])
                after = snapshot_weights(a["model"])
                day_entry["models"][f"{model_name}_{target}"] = {
                    "retrained": do_retrain,
                    "delta": weight_delta(before, after),
                    "after": after,
                }

        weight_log.append(day_entry)
        if verbose and (step % 10 == 0):
            print(f"  [{city}] step {step:3d}  {day_date.date()}"
                  f"  {'(RETRAIN)' if do_retrain else ''}")

    # ---- write ledgers ----
    fdf = pd.DataFrame(forecast_rows)
    fpath = FORECAST_DIR / f"{city}_forecasts.csv"
    fdf.to_csv(fpath, index=False)

    # running metrics
    metrics = {}
    for target in TARGETS:
        metrics[target] = {}
        for model_name in MODEL_NAMES:
            sub = fdf[(fdf.target == target) & (fdf.model == model_name)]
            err = sub["error"].values
            metrics[target][model_name] = {
                "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
                "MAE": round(float(np.mean(np.abs(err))), 4),
                "n": int(len(err)),
            }
    with open(VERIFY_DIR / f"{city}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(LOG_DIR / f"{city}_weight_changes.json", "w") as f:
        json.dump(weight_log, f, indent=2)

    print(f"  [{city}] done -> {fpath.name}, metrics.json, weight_changes.json")
    return fdf, metrics, weight_log
