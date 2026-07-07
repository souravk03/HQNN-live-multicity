"""
Streaming version of the backtest for the live dashboard.

run_backtest_stream(city, max_days) is a GENERATOR: it yields dict events as the
pipeline progresses, so the web server can push them to the browser in real time.

Event types (the dashboard reacts to each):
  {"type":"init",     city, total_days, models, targets}
  {"type":"stage",    stage, status}             # flowchart node lights up
  {"type":"day_start",step, date}
  {"type":"forecast", date, target, model, prediction, unit}
  {"type":"verify",   date, target, model, prediction, actual, error}
  {"type":"readjust", date, model, target, retrained, mean_change, l2_change}
  {"type":"metrics",  table}                     # running RMSE per model/target
  {"type":"day_done", step, date}
  {"type":"done",     total_days}

Stages (for the flowchart): load -> forecast -> verify -> readjust -> retrain -> log
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
from config import (TARGETS, MODEL_NAMES, UNITS, RETRAIN_EVERY,
                    MODEL_DIR, FORECAST_DIR, VERIFY_DIR, LOG_DIR, TRAIN_RATIO)
from sklearn.preprocessing import MinMaxScaler


def _load_meta(city):
    with open(MODEL_DIR / city / "metadata.json") as f:
        return json.load(f)


def _running_metrics(rows):
    df = pd.DataFrame(rows)
    table = {}
    for target in TARGETS:
        table[target] = {}
        for model_name in MODEL_NAMES:
            sub = df[(df.target == target) & (df.model == model_name)]
            if len(sub) == 0:
                table[target][model_name] = None
                continue
            err = sub["error"].dropna().values
            actual = sub["actual"].dropna().values
            pred = sub["prediction"].dropna().values
            if len(err) == 0:
                table[target][model_name] = None
                continue
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            # R2
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((actual - np.mean(actual)) ** 2)) if len(actual) > 1 else 0.0
            r2 = (1 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
            # MAPE (guard against divide-by-zero)
            denom = np.where(np.abs(actual) < 1e-6, 1e-6, np.abs(actual))
            mape = float(np.mean(np.abs(err / denom)) * 100)
            table[target][model_name] = {
                "RMSE": round(rmse, 4), "MAE": round(mae, 4),
                "R2": round(r2, 4), "MAPE": round(mape, 4),
            }
    return table


def run_backtest_stream(city, max_days=None):
    set_seed()
    meta = _load_meta(city)
    df_feat = features.engineer_features(nasa.load_city(city))
    split = meta["split_index"]
    feats_by_target = meta["features"]["mv"]

    backtest_idx = list(range(split, len(df_feat)))
    if max_days:
        backtest_idx = backtest_idx[:max_days]

    yield {"type": "node", "node": "enter_bt", "status": "active"}
    yield {"type": "node", "node": "enter_bt", "status": "done"}
    yield {"type": "init", "city": city, "total_days": len(backtest_idx),
           "models": MODEL_NAMES, "targets": TARGETS,
           "units": UNITS}

    # load assets
    yield {"type": "stage", "stage": "load", "status": "active"}
    assets = {}
    for target in TARGETS:
        feats = feats_by_target[target]
        for model_name in MODEL_NAMES:
            model, sc_X, sc_y = load_assets(city, model_name, target, len(feats))
            assets[(model_name, target)] = {"model": model, "sc_X": sc_X,
                                             "sc_y": sc_y, "feats": feats}
    yield {"type": "stage", "stage": "load", "status": "done"}

    forecast_rows = []

    for step, D in enumerate(backtest_idx):
        day_date = df_feat.iloc[D]["date"]
        hist = df_feat.iloc[:D]
        actual_row = df_feat.iloc[D]
        yield {"type": "day_start", "step": step, "date": str(day_date.date()),
               "total": len(backtest_idx)}

        # ---- forecast ----
        yield {"type": "node", "node": "day_loop", "status": "active"}
        yield {"type": "node", "node": "forecast", "status": "active"}
        yield {"type": "stage", "stage": "forecast", "status": "active"}
        day_preds = {}
        for target in TARGETS:
            for model_name in MODEL_NAMES:
                a = assets[(model_name, target)]
                pred = forecast_next(a["model"], hist, a["feats"], target,
                                     a["sc_X"], a["sc_y"])
                day_preds[(model_name, target)] = pred
                yield {"type": "forecast", "date": str(day_date.date()),
                       "target": target, "model": model_name,
                       "prediction": round(pred, 3), "unit": UNITS[target]}
        yield {"type": "stage", "stage": "forecast", "status": "done"}
        yield {"type": "node", "node": "forecast", "status": "done"}

        # ---- verify ----
        yield {"type": "node", "node": "verify", "status": "active"}
        yield {"type": "stage", "stage": "verify", "status": "active"}
        for target in TARGETS:
            actual = float(actual_row[target])
            for model_name in MODEL_NAMES:
                pred = day_preds[(model_name, target)]
                err = pred - actual
                forecast_rows.append({
                    "date": str(day_date.date()), "target": target,
                    "model": model_name, "prediction": round(pred, 3),
                    "actual": round(actual, 3), "error": round(err, 3),
                    "unit": UNITS[target]})
                yield {"type": "verify", "date": str(day_date.date()),
                       "target": target, "model": model_name,
                       "prediction": round(pred, 3), "actual": round(actual, 3),
                       "error": round(err, 3), "unit": UNITS[target]}
        yield {"type": "stage", "stage": "verify", "status": "done"}
        yield {"type": "node", "node": "verify", "status": "done"}
        yield {"type": "node", "node": "confident", "status": "active"}
        yield {"type": "node", "node": "confident", "status": "done"}

        # ---- readjust (and weekly retrain) ----
        do_retrain = (step > 0 and step % RETRAIN_EVERY == 0)
        stage_name = "retrain" if do_retrain else "readjust"
        yield {"type": "node", "node": "retrain_q", "status": "active"}
        yield {"type": "node", "node": "retrain_q", "status": "done"}
        yield {"type": "node", "node": ("weekly_rt" if do_retrain else "daily_ft"), "status": "active"}
        yield {"type": "stage", "stage": stage_name, "status": "active"}
        df_through_D = df_feat.iloc[:D + 1]

        for target in TARGETS:
            feats = feats_by_target[target]
            for model_name in MODEL_NAMES:
                a = assets[(model_name, target)]
                before = snapshot_weights(a["model"])
                if do_retrain:
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
                    X, y = build_sequences(df_through_D, feats, target,
                                           a["sc_X"], a["sc_y"], fit=False)
                    a["model"] = finetune_model(a["model"], X, y)
                save_assets(city, model_name, target, a["model"], a["sc_X"], a["sc_y"])
                after = snapshot_weights(a["model"])
                # summarise the change for the dashboard
                deltas = weight_delta(before, after)
                mean_change = round(float(np.mean([d["mean_change"] for d in deltas.values()])), 6)
                l2_change = round(float(np.mean([d["l2_change"] for d in deltas.values()])), 6)
                # pull the actual quantum circuit weight value if present (1 number)
                qval = None
                for pname, info in after.items():
                    if "vqc.weights" in pname and "values" in info:
                        qval = info["values"][0]
                        break
                yield {"type": "readjust", "date": str(day_date.date()),
                       "model": model_name, "target": target,
                       "retrained": do_retrain,
                       "mean_change": mean_change, "l2_change": l2_change,
                       "qweight": qval}
        yield {"type": "stage", "stage": stage_name, "status": "done"}
        yield {"type": "node", "node": ("weekly_rt" if do_retrain else "daily_ft"), "status": "done"}
        yield {"type": "node", "node": "save_w", "status": "active"}
        yield {"type": "node", "node": "save_w", "status": "done"}

        # ---- log + metrics ----
        yield {"type": "stage", "stage": "log", "status": "active"}
        yield {"type": "node", "node": "metrics", "status": "active"}
        yield {"type": "metrics", "table": _running_metrics(forecast_rows)}
        yield {"type": "node", "node": "metrics", "status": "done"}
        yield {"type": "stage", "stage": "log", "status": "done"}
        yield {"type": "node", "node": "log", "status": "active"}
        yield {"type": "node", "node": "log", "status": "done"}
        yield {"type": "node", "node": "more_days", "status": "active"}
        yield {"type": "node", "node": "more_days", "status": "done"}
        yield {"type": "day_done", "step": step, "date": str(day_date.date())}

    # persist final ledgers
    pd.DataFrame(forecast_rows).to_csv(FORECAST_DIR / f"{city}_forecasts.csv", index=False)
    with open(VERIFY_DIR / f"{city}_metrics.json", "w") as f:
        json.dump(_running_metrics(forecast_rows), f, indent=2)

    yield {"type": "node", "node": "final", "status": "active"}
    yield {"type": "done", "total_days": len(backtest_idx)}
