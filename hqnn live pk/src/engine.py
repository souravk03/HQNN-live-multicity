"""
Core engine shared by the initial-train script and the backtest notebook.

Key pieces:
  build_sequences         -> (X, y) windows
  fit_scalers / load_save -> per (city, model, target) scaler + weight persistence
  train_model             -> full training (initial + weekly retrain)
  finetune_model          -> daily online weight adjustment on a recent window
  forecast_next           -> single raw t+1 prediction (real units)
  snapshot_weights        -> compact summary of weights/biases for the JSON ledger
"""
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import MinMaxScaler

from config import (SEQ_LEN, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
                    GRAD_CLIP, EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR, LR_MIN,
                    FT_EPOCHS, FT_BATCH, FT_WINDOW, FT_LR, SEED, MODEL_DIR)
from models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Sequences & scaling
# ---------------------------------------------------------------------------
def build_sequences(df, feats, target, sc_X, sc_y, fit=False):
    """Scale and window. If fit=True, fits the scalers (train only)."""
    X_raw = df[feats].values.astype(np.float32)
    y_raw = df[[target]].values.astype(np.float32)
    if fit:
        X_s = sc_X.fit_transform(X_raw)
        y_s = sc_y.fit_transform(y_raw)
    else:
        X_s = sc_X.transform(X_raw)
        y_s = sc_y.transform(y_raw)
    X, y = [], []
    for i in range(len(X_s) - SEQ_LEN):
        X.append(X_s[i:i + SEQ_LEN])
        y.append(y_s[i + SEQ_LEN])
    return np.array(X, np.float32), np.array(y, np.float32)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _paths(city, model_name, target, mode=None):
    import paths
    return (paths.weights_path(city, mode, model_name, target),
            paths.scaler_path(city, mode, model_name, target))


def save_assets(city, model_name, target, model, sc_X, sc_y, mode=None, hparams=None):
    wp, sp = _paths(city, model_name, target, mode)
    torch.save(model.state_dict(), wp)
    # Persist the EXACT hyperparameters the weights were built with, so the model
    # is always reconstructed with a matching architecture at load time. This is
    # the fix for state_dict size-mismatch errors (e.g. gru/qgru hidden size) that
    # happened when load_assets re-derived hparams from a file that had since
    # changed (after tuning). hparams arg wins; else fall back to whatever the
    # model was built with (stashed on the module by build/load).
    hp = hparams if hparams is not None else getattr(model, "_iwf_hp", None)
    _mode = mode if mode is not None else getattr(model, "_iwf_mode", None)
    joblib.dump({"X": sc_X, "y": sc_y, "hp": hp, "mode": _mode}, sp)


def load_assets(city, model_name, target, input_size, mode=None):
    wp, sp = _paths(city, model_name, target, mode)
    sc = joblib.load(sp)
    # Prefer the hparams saved WITH the weights (exact architecture match). Only if
    # an older asset has no stored hp do we fall back to re-deriving from the tuned
    # hparams file (legacy behaviour).
    _hp = sc.get("hp") if isinstance(sc, dict) else None
    _mode = sc.get("mode") if isinstance(sc, dict) else None
    if _mode is None:
        _mode = mode
    if _hp is None:
        try:
            import tune as _tune
            _hp = (_tune.load_hparams(city, mode)["data"].get(target, {})
                   .get(model_name, {}).get("params", {})) or {}
        except Exception:
            _hp = {}
    model = build_model(model_name, input_size, _hp, mode=_mode).to(DEVICE)
    model.load_state_dict(torch.load(wp, map_location=DEVICE))
    model._iwf_hp = _hp          # so a later finetune/retrain re-saves the same arch
    return model, sc["X"], sc["y"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, X_tr, y_tr, X_val, y_val, epochs=EPOCHS, verbose=False,
                hparams=None):
    """Full training. Uses tuned lr / weight-decay when hparams are provided,
    otherwise falls back to the config defaults."""
    hp = hparams or {}
    lr = float(hp["lr"]) if hp.get("lr") is not None else LEARNING_RATE
    wd = float(hp["wd"]) if hp.get("wd") is not None else WEIGHT_DECAY
    model = model.to(DEVICE)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)

    Xtr = torch.from_numpy(X_tr).to(DEVICE)
    ytr = torch.from_numpy(y_tr).to(DEVICE)
    Xva = torch.from_numpy(X_val).to(DEVICE)
    yva = torch.from_numpy(y_val).to(DEVICE)

    best_val, best_state, no_imp = float("inf"), None, 0
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xva), yva).item()
        sched.step(vloss)
        if vloss < best_val - 1e-6:
            best_val, best_state, no_imp = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if verbose and (ep == 1 or ep % 40 == 0):
            print(f"      ep {ep:3d}  val={vloss:.5f}", flush=True)
        if no_imp >= EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def finetune_model(model, X, y, epochs=FT_EPOCHS, lr=FT_LR):
    """Daily online adjustment on the most recent FT_WINDOW windows."""
    model = model.to(DEVICE)
    if len(X) > FT_WINDOW:
        X, y = X[-FT_WINDOW:], y[-FT_WINDOW:]
    crit = nn.HuberLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y).to(DEVICE)
    n = len(Xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, FT_BATCH):
            idx = perm[i:i + FT_BATCH]
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
    return model


# ---------------------------------------------------------------------------
# Forecast (raw, no corrections)
# ---------------------------------------------------------------------------
def forecast_next(model, df, feats, target, sc_X, sc_y):
    """Predict t+1 (the day after the last row of df) in real units."""
    window = df[feats].values[-SEQ_LEN:].astype(np.float32)
    ws = sc_X.transform(window)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(ws[np.newaxis]).to(DEVICE)
        pred_s = float(model(x).cpu().numpy()[0, 0])
    return float(sc_y.inverse_transform([[pred_s]])[0, 0])


# ---------------------------------------------------------------------------
# Weight snapshot for the JSON ledger
# ---------------------------------------------------------------------------
def snapshot_weights(model):
    """Compact, JSON-friendly summary so you can watch numbers move daily."""
    snap = {}
    for name, p in model.named_parameters():
        v = p.detach().cpu().numpy().ravel()
        snap[name] = {
            "shape": list(p.shape),
            "mean": round(float(v.mean()), 6),
            "std": round(float(v.std()), 6),
            "l2": round(float(np.linalg.norm(v)), 6),
        }
        # store full quantum weights verbatim (they are tiny)
        if "vqc.weights" in name:
            snap[name]["values"] = [round(float(x), 6) for x in v]
    return snap


def weight_delta(before, after):
    """L2 of (after - before) per parameter, for the change ledger."""
    out = {}
    for name in before:
        out[name] = {
            "mean_change": round(after[name]["mean"] - before[name]["mean"], 6),
            "l2_change":   round(after[name]["l2"] - before[name]["l2"], 6),
        }
    return out