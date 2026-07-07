"""
train_city.py
-------------
Tune + train all models for Mumbai or Chennai,
using the same pipeline that was used for Delhi.

Reads training data from data/<city>/nasa_cache.csv (already present in the zip
for both cities — 2021-01-01 to 2025-12-31, 1826 rows, ERA5-derived values).
Falls back to Open-Meteo Archive API if the cache is missing or too short.

Produces the exact same artefacts Delhi has:
  models/<city>/<mode>/metadata.json
  models/<city>/<mode>/best_hparams.json
  models/<city>/<mode>/hparams_baseline.json
  models/<city>/<mode>/train_metrics.json
  models/<city>/<mode>/train_times.json
  models/<city>/<mode>/<model>/<target>.pth
  models/<city>/<mode>/<model>/scaler_<target>.pkl

Usage
-----
    python train_city.py --city mumbai                    # MV only (default)
    python train_city.py --city chennai                   # MV only
    python train_city.py --city mumbai --mode uv          # UV only
    python train_city.py --city mumbai --mode both        # MV + UV
    python train_city.py --city mumbai --mode both --trials 20
    python train_city.py --city mumbai --skip-tune        # use default hparams
    python train_city.py --city mumbai --force            # re-train even if weights exist

Resumable: interrupted mid-training, re-run the same command — it picks up
from the last saved checkpoint (same logic as the Delhi pipeline).

Supported city keys
-------------------
  mumbai  / maharashtra   → lat 19.08, lon 72.88  (saves under models/mumbai/)
  chennai / tamil_nadu    → lat 13.08, lon 80.27  (saves under models/chennai/)
  delhi                   → lat 28.61, lon 77.20  (already trained, for testing)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from datetime import date, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from project root or from src/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# City registry (lat/lon + which folder key to use)
# ---------------------------------------------------------------------------
CITY_REGISTRY = {
    "delhi":       {"lat": 28.61, "lon": 77.20, "label": "Delhi",   "folder": "delhi"},
    "mumbai":      {"lat": 19.08, "lon": 72.88, "label": "Mumbai",  "folder": "mumbai"},
    "maharashtra": {"lat": 19.08, "lon": 72.88, "label": "Mumbai",  "folder": "mumbai"},   # alias
    "chennai":     {"lat": 13.08, "lon": 80.27, "label": "Chennai", "folder": "chennai"},
    "tamil_nadu":  {"lat": 13.08, "lon": 80.27, "label": "Chennai", "folder": "chennai"},  # alias
}

# ---------------------------------------------------------------------------
# Data loading — ERA5 nasa_cache first, then Open-Meteo fallback
# ---------------------------------------------------------------------------

def _load_training_data(city_key: str, verbose: bool = True) -> "pd.DataFrame":
    """
    Load training data for a city.

    Priority:
      1. data/<folder>/nasa_cache.csv  (already present in the zip, 2021-2025)
      2. Open-Meteo Archive API        (fetched if cache missing / < 365 rows)

    Returns a DataFrame with the standard nasa_cache schema that
    features.engineer_features() expects.
    """
    import pandas as pd
    import paths
    import features as feat_module

    cfg    = CITY_REGISTRY[city_key]
    folder = cfg["folder"]
    label  = cfg["label"]

    cache_path = paths.nasa_cache(folder)

    # Fallback: check legacy state-name folder (maharashtra→mumbai, tamil_nadu→chennai)
    if not cache_path.exists():
        _legacy = {"mumbai": "maharashtra", "chennai": "tamil_nadu"}.get(folder)
        if _legacy:
            _legacy_path = paths.nasa_cache(_legacy)
            if _legacy_path.exists():
                import shutil
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(_legacy_path, cache_path)
                if verbose:
                    print(f"[Data] Copied legacy cache {_legacy_path.name} → {cache_path}")

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["date"])
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        if verbose:
            print(f"[Data] {label}: loaded {len(df)} rows from {cache_path.name} "
                  f"({df['date'].min().date()} → {df['date'].max().date()})")
        if len(df) >= 365:
            return df
        print(f"[Data] {label}: cache too short ({len(df)} rows) — fetching from Open-Meteo")

    # Fallback: fetch from Open-Meteo ERA5 archive
    print(f"[Data] {label}: fetching ERA5 history from Open-Meteo …")
    from era5_openmeteo import fetch_era5
    end_d   = date(2025, 12, 31)
    start_d = date(2021,  1,  1)
    df = fetch_era5(
        start_d.isoformat(), end_d.isoformat(),
        lat=cfg["lat"], lon=cfg["lon"], city_key=folder,
    )
    if df.empty:
        raise RuntimeError(f"Could not fetch ERA5 data for {label}")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # Save as nasa_cache for future runs
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    if verbose:
        print(f"[Data] {label}: fetched {len(df)} rows, saved to {cache_path.name}")
    return df


# ---------------------------------------------------------------------------
# Tune one city+mode  (mirrors tune.tune_stream but headless, with prints)
# ---------------------------------------------------------------------------

def _run_tune(city_key: str, mode: str, n_trials: int,
              models: list[str] | None = None,
              verbose: bool = True) -> None:
    import paths, tune as _tune
    from config import MODEL_NAMES

    folder   = CITY_REGISTRY[city_key]["folder"]
    label    = CITY_REGISTRY[city_key]["label"]
    hp_path  = paths.best_hparams_path(folder, mode)

    _MN = [m for m in MODEL_NAMES if m in models] if models else ["hqnn"]

    if hp_path.exists():
        print(f"[Tune] {label}/{mode}: best_hparams.json already exists — skipping.")
        return

    print(f"\n[Tune] {label}/{mode}: starting Optuna ({n_trials} trials × "
          f"{len(_MN)} model(s) ({', '.join(_MN)}) × 3 targets) …")

    # Patch tune.nasa.load_city to return our ERA5 cache data
    import tune as _tune_mod
    import nasa as _nasa_mod

    _orig_load = _nasa_mod.load_city

    def _patched_load(city):
        # tune.py calls nasa.load_city(city) — redirect to our ERA5 loader
        return _load_training_data(city_key, verbose=False)

    _nasa_mod.load_city = _patched_load
    try:
        best: dict = {}
        for ev in _tune_mod.tune_stream(folder, n_trials=n_trials, models=_MN, mode=mode):
            ty = ev.get("type")
            if ty == "tune_model" and ev.get("status") == "done":
                mt  = f"{ev.get('model')}/{ev.get('target', '')}"
                val = ev.get("best_val", ev.get("val_mse"))
                if verbose:
                    print(f"  tuned {mt:20s}  best_val_mse={val}", flush=True)
            elif ty == "tune_done":
                best = ev.get("best", {})
            elif ty == "paused":
                raise RuntimeError("Tuning paused unexpectedly.")
    finally:
        _nasa_mod.load_city = _orig_load   # always restore

    print(f"[Tune] {label}/{mode}: done. hparams written → {hp_path}")


# ---------------------------------------------------------------------------
# Train one city+mode  (mirrors pipeline_stream.train_stream but headless)
# ---------------------------------------------------------------------------

def _run_train(city_key: str, mode: str, models: list[str] | None,
               force: bool = False, verbose: bool = True) -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import joblib
    import paths, features, control, tune as _tune_mod
    from engine import set_seed, build_sequences, save_assets, DEVICE
    from models import build_model
    from sklearn.preprocessing import MinMaxScaler
    from config import (TARGETS, MODEL_NAMES, TRAIN_RATIO, EPOCHS, BATCH_SIZE,
                        LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP,
                        EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR, LR_MIN)

    folder = CITY_REGISTRY[city_key]["folder"]
    label  = CITY_REGISTRY[city_key]["label"]
    _MN    = [m for m in MODEL_NAMES if m in models] if models else ["hqnn"]
    _fmode = "uv" if mode in ("uv", "univariate") else "mv"

    # Skip if already done (unless --force)
    if not force:
        all_done = all(
            paths.weights_path(folder, mode, mn, t).exists()
            for mn in _MN for t in TARGETS
        )
        if all_done:
            print(f"[Train] {label}/{mode}: all weights already present — skipping. "
                  "Use --force to retrain.")
            return

    print(f"\n[Train] {label}/{mode}: loading training data …")
    df_raw  = _load_training_data(city_key, verbose=verbose)
    df_feat = features.engineer_features(df_raw)
    n       = len(df_feat)
    split   = int(n * TRAIN_RATIO)
    df_all  = df_feat.iloc[:split]
    inner   = int(len(df_all) * 0.85)
    df_tr   = df_all.iloc[:inner]
    df_va   = df_all.iloc[inner:]
    backtest_start = str(df_feat.iloc[split]["date"].date())

    print(f"[Train] {label}/{mode}: {n} rows | train ≤ {df_all.iloc[-1]['date'].date()} "
          f"| backtest from {backtest_start}")

    # Load hyperparameters (tuned or defaults)
    HP = _tune_mod.load_hparams(folder, mode)["data"]

    def hp_for(target, model_name):
        try:
            return HP[target][model_name]["params"] or {}
        except Exception:
            return {}

    # Build metadata  (same schema as Delhi)
    meta = {
        "city": folder, "train_ratio": TRAIN_RATIO,
        "split_index": int(split), "backtest_start_date": backtest_start,
        "n_rows": int(n), "mode": mode,
        "features": {"mv": {}, "uv": {}},
    }
    for target in TARGETS:
        for fm in ("mv", "uv"):
            meta["features"][fm][target] = features.select_features(df_feat, target, fm)

    # Resume support
    resume  = control.load_train_state(folder, mode)
    start_t = resume["target_idx"] if (resume and resume.get("city") == folder) else 0
    start_m = resume["model_idx"]  if (resume and resume.get("city") == folder) else 0

    CKPT_EVERY = 5
    all_metrics: dict = {}
    all_times:   dict = {}

    for ti, target in enumerate(TARGETS):
        feats      = meta["features"][_fmode][target]
        input_size = len(feats)
        all_metrics.setdefault(target, {})
        all_times.setdefault(target, {})

        for mi, model_name in enumerate(_MN):
            # Skip models completed in a prior interrupted run
            if ti < start_t or (ti == start_t and mi < start_m):
                print(f"  [skip resume] {model_name}/{target}")
                continue

            # Skip if weight already exists and not forcing
            wp = paths.weights_path(folder, mode, model_name, target)
            if wp.exists() and not force:
                print(f"  [skip existing] {model_name}/{target}")
                continue

            control.save_train_state(folder, ti, mi,
                                     f"{target}/{model_name}", mode=mode)

            print(f"  [{target}] {model_name:6s}  ({input_size} feats) …",
                  end=" ", flush=True)

            sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
            X_tr, y_tr = build_sequences(df_tr, feats, target, sc_X, sc_y, fit=True)
            X_va, y_va = build_sequences(df_va, feats, target, sc_X, sc_y, fit=False)

            set_seed()
            hp    = hp_for(target, model_name)
            model = build_model(model_name, input_size, hp, mode=mode).to(DEVICE)
            crit  = nn.MSELoss()
            _lr   = float(hp.get("lr", LEARNING_RATE))
            _wd   = float(hp.get("wd", WEIGHT_DECAY))
            opt   = optim.Adam(model.parameters(), lr=_lr, weight_decay=_wd)
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
                min_lr=LR_MIN)

            Xtr = torch.from_numpy(X_tr).to(DEVICE)
            ytr = torch.from_numpy(y_tr).to(DEVICE)
            Xva = torch.from_numpy(X_va).to(DEVICE)
            yva = torch.from_numpy(y_va).to(DEVICE)

            best, best_state, no_imp = float("inf"), None, 0
            start_ep = 1
            ep_hist  = []

            # Resume from mid-training checkpoint if present
            cp = control.ckpt_path(folder, model_name, target, mode=mode)
            if cp.exists():
                try:
                    blob = torch.load(cp, map_location=DEVICE)
                    model.load_state_dict(blob["model"])
                    opt.load_state_dict(blob["opt"])
                    best       = blob.get("best",       best)
                    best_state = blob.get("best_state", None)
                    no_imp     = blob.get("no_imp",     0)
                    start_ep   = blob.get("epoch",      0) + 1
                    print(f"resumed from ep {start_ep - 1} …", end=" ", flush=True)
                except Exception:
                    pass

            t0 = time.time()

            for ep in range(start_ep, EPOCHS + 1):
                model.train()
                perm = torch.randperm(len(Xtr))
                for i in range(0, len(Xtr), BATCH_SIZE):
                    idx = perm[i:i + BATCH_SIZE]
                    opt.zero_grad()
                    loss = crit(model(Xtr[idx]), ytr[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()

                model.eval()
                with torch.no_grad():
                    pred_va = model(Xva)
                    vloss   = crit(pred_va, yva).item()
                sched.step(vloss)

                if vloss < best - 1e-6:
                    best, best_state, no_imp = (
                        vloss, copy.deepcopy(model.state_dict()), 0)
                else:
                    no_imp += 1

                # Per-epoch metrics (real scale)
                yp   = sc_y.inverse_transform(
                    pred_va.cpu().numpy().reshape(-1, 1)).ravel()
                yt   = sc_y.inverse_transform(
                    yva.cpu().numpy().reshape(-1, 1)).ravel()
                err  = yp - yt
                rmse = float(np.sqrt(np.mean(err ** 2)))
                mae  = float(np.mean(np.abs(err)))
                ss_r = float(np.sum(err ** 2))
                ss_t = float(np.sum((yt - yt.mean()) ** 2)) or 1e-9
                r2   = float(1.0 - ss_r / ss_t)
                ep_hist.append({"epoch": ep, "rmse": round(rmse, 4),
                                "mae": round(mae, 4), "r2": round(r2, 4)})

                # Checkpoint every CKPT_EVERY epochs
                if ep % CKPT_EVERY == 0 or no_imp >= EARLY_STOP_PATIENCE:
                    torch.save({
                        "model": model.state_dict(), "opt": opt.state_dict(),
                        "best": best, "best_state": best_state,
                        "no_imp": no_imp, "epoch": ep,
                    }, cp)

                if no_imp >= EARLY_STOP_PATIENCE:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            elapsed = round(time.time() - t0, 2)
            last    = ep_hist[-1] if ep_hist else {}
            print(f"{len(ep_hist)} ep | val_rmse={last.get('rmse','?')} | "
                  f"R²={last.get('r2','?')} | {elapsed}s")

            save_assets(folder, model_name, target, model, sc_X, sc_y,
                        mode=mode, hparams=hp)

            all_metrics[target][model_name] = ep_hist
            all_times[target][model_name]   = elapsed

            # Persist incremental metrics + times
            _save_json(paths.train_metrics_path(folder, mode), all_metrics)
            _save_json(paths.train_times_path(folder, mode),   all_times)

            if cp.exists():
                cp.unlink()  # drop mid-training checkpoint

    # Write metadata.json
    _save_json(paths.metadata_path(folder, mode), meta)
    print(f"[Train] {label}/{mode}: metadata.json written → "
          f"{paths.metadata_path(folder, mode)}")

    # Clear resume cursor
    control.clear_train_state(folder, mode)
    print(f"[Train] {label}/{mode}: complete.")


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Tune + train HQNN models for Mumbai or Chennai")
    p.add_argument("--city", required=True,
                   choices=list(CITY_REGISTRY.keys()),
                   help="City to train")
    p.add_argument("--mode", default="multivariate",
                   choices=["multivariate", "univariate", "both"],
                   help="Feature mode (default: multivariate)")
    p.add_argument("--trials", type=int, default=30,
                   help="Optuna trials per model×target (default: 30)")
    p.add_argument("--models", nargs="+", default=None,
                   choices=["lstm", "qlstm", "gru", "qgru", "ann", "hqnn"],
                   help="Subset of models (default: hqnn)")
    p.add_argument("--skip-tune", action="store_true",
                   help="Skip Optuna tuning, use default hyperparameters")
    p.add_argument("--force", action="store_true",
                   help="Re-train even if weights already exist")
    p.add_argument("--quiet", action="store_true",
                   help="Less verbose output")
    return p.parse_args()


def main():
    args   = _parse_args()
    city   = args.city
    label  = CITY_REGISTRY[city]["label"]
    folder = CITY_REGISTRY[city]["folder"]
    modes  = (["multivariate", "univariate"] if args.mode == "both"
              else [args.mode])
    verbose = not args.quiet

    print("=" * 60)
    print(f"  Training pipeline: {label} ({folder})")
    print(f"  Modes  : {', '.join(modes)}")
    print(f"  Models : {args.models or ['hqnn']}")
    print(f"  Trials : {args.trials}  |  skip-tune={args.skip_tune}  |  force={args.force}")
    print("=" * 60)

    t_total = time.time()

    for mode in modes:
        print(f"\n{'─'*55}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'─'*55}")

        # --- Tune ---
        if not args.skip_tune:
            _run_tune(city, mode, args.trials, models=args.models, verbose=verbose)
        else:
            print(f"[Tune] skipped (--skip-tune)")

        # --- Train ---
        _run_train(city, mode, args.models, force=args.force, verbose=verbose)

    elapsed = time.time() - t_total
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"  All done for {label}.  Total time: {mins}m {secs}s")
    print(f"  Weights saved under: models/{folder}/  (data cache: data/{folder}/)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()