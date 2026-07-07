"""
Streaming download + train for the full-pipeline dashboard.

Both are generators that yield dict events so the server can push live progress
to the browser. They let the user start from NULL data (no cache, no models) and
watch download -> train happen in the flowchart.

download_stream(city)  yields:
  {"type":"phase","phase":"download","status":"active"/"done"}
  {"type":"download","msg":...,"rows":N}

train_stream(city)  yields:
  {"type":"phase","phase":"train","status":"active"/"done"}
  {"type":"train_init","total":N_models}
  {"type":"train_model","target","model","input_size","status":"start"/"saved"}
  {"type":"train_progress","target","model","epoch","val"}
  {"type":"train_meta","backtest_start","n_rows","split"}
"""
import json
import numpy as np

import nasa
import features
from config import (CITIES, TARGETS, MODEL_NAMES, TRAIN_RATIO, MODEL_DIR,
                    EPOCHS)
from sklearn.preprocessing import MinMaxScaler


def download_stream(city):
    yield {"type": "phase", "phase": "download", "status": "active"}
    yield {"type": "node", "node": "download", "status": "active"}
    yield {"type": "download", "msg": f"Requesting NASA POWER for {city}…", "rows": 0}
    df = nasa.download_city(city)
    yield {"type": "node", "node": "download", "status": "done"}
    yield {"type": "node", "node": "features", "status": "active"}
    yield {"type": "node", "node": "features", "status": "done"}
    yield {"type": "node", "node": "validate", "status": "active"}
    yield {"type": "node", "node": "validate", "status": "done"}
    yield {"type": "node", "node": "split", "status": "active"}
    yield {"type": "node", "node": "split", "status": "done"}
    yield {"type": "download", "msg": f"Saved {len(df)} daily records to cache.",
           "rows": int(len(df))}
    yield {"type": "phase", "phase": "download", "status": "done"}


def download_all_stream():
    """Download NASA data for EVERY state / union territory (the "Download" button
    grabs the whole country). Streams per-state progress and continues past any
    single failure so one bad request never aborts the rest. Already-cached states
    are simply re-fetched (latest data)."""
    from config import STATES
    keys = [k for k in STATES if "lat" in STATES[k]]
    total = len(keys)
    yield {"type": "phase", "phase": "download", "status": "active"}
    yield {"type": "node", "node": "download", "status": "active"}
    done = 0
    failed = []
    for i, k in enumerate(keys, 1):
        label = STATES[k].get("label", k)
        yield {"type": "download_progress", "city": k, "label": label,
               "index": i, "total": total, "status": "start"}
        try:
            df = nasa.download_city(k)
            done += 1
            yield {"type": "download_progress", "city": k, "label": label,
                   "index": i, "total": total, "status": "done", "rows": int(len(df))}
        except Exception as e:
            failed.append(k)
            yield {"type": "download_progress", "city": k, "label": label,
                   "index": i, "total": total, "status": "error", "error": str(e)[:160]}
    yield {"type": "node", "node": "download", "status": "done"}
    msg = f"Downloaded {done}/{total} states/UTs" + (f" · {len(failed)} failed" if failed else "")
    yield {"type": "download", "msg": msg, "rows": done}
    yield {"type": "phase", "phase": "download", "status": "done",
           "downloaded": done, "failed": failed}


def train_stream(city, models=None, mode=None):
    """
    Mirrors train_initial.train_city but yields progress so the UI can animate.
    Imports torch lazily so the server can start without torch present.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import copy
    import time as _time
    from engine import (set_seed, build_sequences, save_assets, DEVICE)
    from models import build_model
    from config import (BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP,
                        EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR, LR_MIN, MODEL_DIR)
    import paths
    from config import DEFAULT_MODE
    mode = mode or DEFAULT_MODE       # "multivariate" / "univariate" (UI/paths key)
    _fmode = "uv" if mode in ("uv", "univariate") else "mv"   # feature-set key

    # --- Tuning under Train: auto-tune ONLY if best_hparams.json is missing ---
    import tune as _tune
    hp_path = paths.best_hparams_path(city, mode)
    if not hp_path.exists():
        yield {"type": "info", "msg": "No tuned hyperparameters found — tuning first (one-time)."}
        for ev in _tune.tune_stream(city, models=models, mode=mode):
            yield ev   # stream the whole tuning flow (lights the tune diagram)
    else:
        yield {"type": "info", "msg": "Using existing tuned hyperparameters."}
    HP = _tune.load_hparams(city, mode)["data"]   # {target:{model:{params, val_mse}}}

    def hp_for(target, model_name):
        try:
            return HP[target][model_name]["params"] or {}
        except Exception:
            return {}

    yield {"type": "phase", "phase": "train", "status": "active"}
    set_seed()
    df_raw = nasa.load_city(city)
    df_feat = features.engineer_features(df_raw)
    n = len(df_feat)
    split = int(n * TRAIN_RATIO)
    df_train_all = df_feat.iloc[:split]
    backtest_start = str(df_feat.iloc[split]["date"].date())
    inner = int(len(df_train_all) * 0.85)
    df_tr = df_train_all.iloc[:inner]
    df_va = df_train_all.iloc[inner:]

    yield {"type": "train_meta", "backtest_start": backtest_start,
           "n_rows": int(n), "split": int(split), "epochs": int(EPOCHS)}
    # effective model set: the UI selection if given, else all trainable models.
    # Preserve MODEL_NAMES order so indices/resume stay consistent.
    _MN = [m for m in MODEL_NAMES if m in models] if models else list(MODEL_NAMES)
    if not _MN:
        _MN = list(MODEL_NAMES)
    yield {"type": "train_init", "total": len(TARGETS) * len(_MN)}
    for nd in ("apply_hp", "fit_scalers", "build_seq"):
        yield {"type": "node", "node": nd, "status": "active"}
        yield {"type": "node", "node": nd, "status": "done"}

    meta = {"city": city, "train_ratio": TRAIN_RATIO, "split_index": int(split),
            "backtest_start_date": backtest_start, "n_rows": int(n),
            "mode": mode,
            "features": {"mv": {}, "uv": {}}}
    for target in TARGETS:
        for fm in ("mv", "uv"):
            meta["features"][fm][target] = features.select_features(df_feat, target, fm)

    import control
    resume = control.load_train_state(city, mode)
    start_t = resume["target_idx"] if (resume and resume.get("city") == city) else 0
    start_m = resume["model_idx"] if (resume and resume.get("city") == city) else 0
    CKPT_EVERY = 5

    for ti, target in enumerate(TARGETS):
        feats = meta["features"][_fmode][target]
        input_size = len(feats)
        yield {"type": "node", "node": "target_loop", "status": "active"}
        for mi, model_name in enumerate(_MN):
            # skip models already completed in a prior (resumed) run
            if ti < start_t or (ti == start_t and mi < start_m):
                continue
            control.save_train_state(city, ti, mi, f"{target}/{model_name}", mode=mode)
            yield {"type": "node", "node": "model_loop", "status": "active"}
            yield {"type": "train_model", "target": target, "model": model_name,
                   "input_size": input_size, "status": "start"}
            sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
            X_tr, y_tr = build_sequences(df_tr, feats, target, sc_X, sc_y, fit=True)
            X_va, y_va = build_sequences(df_va, feats, target, sc_X, sc_y, fit=False)

            set_seed()
            hp = hp_for(target, model_name)
            model = build_model(model_name, input_size, hp, mode=mode).to(DEVICE)
            crit = nn.MSELoss()
            _lr = float(hp.get("lr", LEARNING_RATE))
            _wd = float(hp.get("wd", WEIGHT_DECAY))
            opt = optim.Adam(model.parameters(), lr=_lr, weight_decay=_wd)
            sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                        factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
            Xtr = torch.from_numpy(X_tr).to(DEVICE); ytr = torch.from_numpy(y_tr).to(DEVICE)
            Xva = torch.from_numpy(X_va).to(DEVICE); yva = torch.from_numpy(y_va).to(DEVICE)
            best, best_state, no_imp = float("inf"), None, 0
            start_ep = 1

            # resume this model from a mid-training checkpoint if one exists
            cp = control.ckpt_path(city, model_name, target, mode=mode)
            if cp.exists():
                try:
                    blob = torch.load(cp, map_location=DEVICE)
                    model.load_state_dict(blob["model"])
                    opt.load_state_dict(blob["opt"])
                    best = blob.get("best", best)
                    best_state = blob.get("best_state", None)
                    no_imp = blob.get("no_imp", 0)
                    start_ep = blob.get("epoch", 0) + 1
                    yield {"type": "info", "msg": f"resumed {model_name} {target} from epoch {start_ep-1}"}
                except Exception:
                    pass

            nlen = len(Xtr)
            paused = False
            _t_start = _time.time()     # wall-clock train time for this model+target
            _ep_hist = []
            for ep in range(start_ep, EPOCHS + 1):
                model.train()
                perm = torch.randperm(nlen)
                for i in range(0, nlen, BATCH_SIZE):
                    idx = perm[i:i + BATCH_SIZE]
                    opt.zero_grad()
                    loss = crit(model(Xtr[idx]), ytr[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                model.eval()
                with torch.no_grad():
                    pred_va = model(Xva)
                    vloss = crit(pred_va, yva).item()
                sched.step(vloss)
                if vloss < best - 1e-6:
                    best, best_state, no_imp = vloss, copy.deepcopy(model.state_dict()), 0
                else:
                    no_imp += 1
                # Emit a live point EVERY epoch for every model, so the training
                # graph and metrics update continuously (the SSE heartbeat keeps the
                # stream flushing even for fast classical models). Quantum models were
                # already per-epoch; classical models now match.
                _emit = True
                if _emit:
                    # metrics on the ORIGINAL scale (inverse-transform the target)
                    import numpy as _np
                    yp = sc_y.inverse_transform(pred_va.cpu().numpy().reshape(-1, 1)).ravel()
                    yt = sc_y.inverse_transform(yva.cpu().numpy().reshape(-1, 1)).ravel()
                    err = yp - yt
                    rmse = float(_np.sqrt(_np.mean(err ** 2)))
                    mae = float(_np.mean(_np.abs(err)))
                    ss_res = float(_np.sum(err ** 2))
                    ss_tot = float(_np.sum((yt - yt.mean()) ** 2)) or 1e-9
                    r2 = float(1.0 - ss_res / ss_tot)
                    yield {"type": "node", "node": "epoch", "status": "active"}
                    yield {"type": "train_progress", "target": target,
                           "model": model_name, "epoch": ep,
                           "val": round(float(vloss), 5),
                           "rmse": round(rmse, 4), "mae": round(mae, 4),
                           "r2": round(r2, 4), "phase": "train"}
                    _ep_hist.append({"epoch": ep, "rmse": round(rmse, 4),
                                     "mae": round(mae, 4), "r2": round(r2, 4)})
                # periodic checkpoint + pause check
                if ep % CKPT_EVERY == 0 or no_imp >= EARLY_STOP_PATIENCE:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "best": best, "best_state": best_state,
                                "no_imp": no_imp, "epoch": ep}, cp)
                    if control.is_paused():
                        control.save_train_state(city, ti, mi, f"paused at {target}/{model_name} ep{ep}", mode=mode)
                        control.mark_paused_phase("train")
                        yield {"type": "paused", "phase": "train",
                               "msg": f"Paused at {model_name} {target}, epoch {ep}. Press Play to resume."}
                        paused = True
                        break
                if no_imp >= EARLY_STOP_PATIENCE:
                    yield {"type": "node", "node": "early_stop", "status": "active"}
                    yield {"type": "node", "node": "early_stop", "status": "done"}
                    break
            if paused:
                return  # leave checkpoint + train_state; resume picks up here
            if best_state is not None:
                model.load_state_dict(best_state)

            # concise heartbeat: one line per finished (target, model)
            import sys as _sys
            try:
                _last = _ep_hist[-1] if _ep_hist else None
                _rmse = (f"{_last['rmse']:.3f}" if _last and _last.get('rmse') is not None else "?")
                print(f"[train] {city}/{mode} {model_name}/{target} "
                      f"done · {len(_ep_hist)} ep · val_rmse={_rmse}", flush=True)
            except Exception:
                pass

            save_assets(city, model_name, target, model, sc_X, sc_y, mode=mode)
            # ---- record wall-clock training time per (target, model) ----
            try:
                _elapsed = round(float(_time.time() - _t_start), 2)
                ttp = paths.train_times_path(city, mode)
                tt = json.load(open(ttp)) if ttp.exists() else {}
                tt.setdefault(target, {})[model_name] = _elapsed
                json.dump(tt, open(ttp, "w"), indent=2)
                yield {"type": "train_time", "target": target, "model": model_name,
                       "seconds": _elapsed}
            except Exception:
                pass
            # persist this model's per-epoch metric history (per target+model) so the
            # live training chart + table can be restored after a refresh
            try:
                import json as _json
                mpath = paths.train_metrics_path(city, mode)
                allm = _json.load(open(mpath)) if mpath.exists() else {}
                allm.setdefault(target, {})[model_name] = _ep_hist
                _json.dump(allm, open(mpath, "w"))
            except Exception:
                pass
            if cp.exists():
                cp.unlink()   # model done; drop its mid-training checkpoint
            yield {"type": "node", "node": "save_model", "status": "active"}
            yield {"type": "node", "node": "save_model", "status": "done"}
            yield {"type": "train_model", "target": target, "model": model_name,
                   "status": "saved", "best_val": round(float(best), 5)}
        start_m = 0  # only the first resumed target skips models

    control.clear_train_state(city, mode)   # full training finished
    yield {"type": "node", "node": "metadata", "status": "active"}

    mpath = paths.metadata_path(city, mode)
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    yield {"type": "node", "node": "metadata", "status": "done"}
    yield {"type": "phase", "phase": "train", "status": "done"}