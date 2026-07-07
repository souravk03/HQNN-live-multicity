"""
Hyperparameter tuner (run OFFLINE, once, on historical data).

Tunes all 4 model families on walk-forward time-series CV, then tunes the
LIVE-ADAPTATION knobs (fine-tune LR/window, retrain cadence) by simulating the
online loop. Writes the best config to models/<city>/best_hparams.json.

Speed safeguards (essential for tuning quantum models on a laptop):
  - quantum QNodes use diff_method='backprop' (5-20x faster on a simulator)
  - short epoch budget per trial (TRIAL_EPOCHS) just to RANK configs
  - Optuna median pruning kills losing trials early
  - tune on ONE city (Delhi, hardest), apply winner to the others

Usage:
    pip install optuna
    python tune.py                 # tune all 4 models on delhi, ~30 trials
    python tune.py mumbai          # different city
"""
import sys
import json
import control
import paths
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

import optuna
from optuna.pruners import MedianPruner

import nasa
import features
from config import (CITIES, TARGETS, MODEL_NAMES, TRAIN_RATIO, MODEL_DIR,
                    GRAD_CLIP, SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- tuning budget (small on purpose; final model trained fully later) ----
N_TRIALS      = 30
TRIAL_EPOCHS  = 40
N_FOLDS       = 3          # walk-forward folds
PRUNE_WARMUP  = 8

try:
    import pennylane as qml
    _HAS_QML = True
except ImportError:
    _HAS_QML = False


# ---------------------------------------------------------------------------
# Models built with TUNABLE sizes and FAST quantum (backprop)
# ---------------------------------------------------------------------------
def _vqc_fast():
    dev = qml.device("default.qubit", wires=1, shots=None)

    @qml.qnode(dev, interface="torch", diff_method="backprop")   # <-- fast
    def circuit(inputs, weights):
        qml.RX(inputs[0], wires=0)
        qml.RY(weights[0, 0], wires=0)
        return qml.expval(qml.PauliZ(0))

    class VQC(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(0.01 * torch.randn(1, 1))

        def forward(self, x):
            return torch.stack([circuit(s, self.w) for s in x]).view(-1, 1).float()
    return VQC()


def build_tunable(name, input_size, hidden, dropout, mode=None):
    # Use the REAL POC architectures from models.py for every model, so tuning
    # evaluates exactly what will be trained/served (mode-specific sizes). The
    # tuner still explores 'hidden' (recurrent) and q_depth via hparams overrides.
    import models as _models
    hp = {"hidden": hidden, "dropout": dropout}
    return _models.build_model(name, input_size, hp, mode=mode).to(DEVICE)


# ---------------------------------------------------------------------------
# Data / sequences
# ---------------------------------------------------------------------------
def make_seq(df, feats, target, scX, scy, seq_len, fit=False):
    Xr = df[feats].values.astype(np.float32)
    yr = df[[target]].values.astype(np.float32)
    Xs = scX.fit_transform(Xr) if fit else scX.transform(Xr)
    ys = scy.fit_transform(yr) if fit else scy.transform(yr)
    X, y = [], []
    for i in range(len(Xs) - seq_len):
        X.append(Xs[i:i + seq_len]); y.append(ys[i + seq_len])
    return np.array(X, np.float32), np.array(y, np.float32)


def quick_train(model, Xtr, ytr, Xva, yva, lr, wd):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xtr = torch.from_numpy(Xtr).to(DEVICE); ytr = torch.from_numpy(ytr).to(DEVICE)
    Xva = torch.from_numpy(Xva).to(DEVICE); yva = torch.from_numpy(yva).to(DEVICE)
    best, best_state, no_imp = 1e9, None, 0
    n = len(Xtr)
    for ep in range(TRIAL_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(Xva), yva).item()
        if v < best - 1e-6:
            best, best_state, no_imp = v, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if no_imp >= 8:
            break
    return best


# ---------------------------------------------------------------------------
# Objective: walk-forward CV error for one (model, target)
# ---------------------------------------------------------------------------
def make_objective(df_feat, model_name, target, mode="mv"):
    _fmode = "uv" if mode in ("uv", "univariate") else "mv"
    def objective(trial):
        seq_len = trial.suggest_categorical("seq_len", [7, 14, 21, 30])
        hidden  = trial.suggest_categorical("hidden", [16, 32, 64])
        lr      = trial.suggest_float("lr", 3e-4, 5e-3, log=True)
        wd      = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)

        feats = features.select_features(df_feat, target, _fmode)
        n = len(df_feat)
        fold_scores = []
        # expanding-window walk-forward folds
        for k in range(1, N_FOLDS + 1):
            if control.is_paused():
                raise optuna.TrialPruned()   # bail out of this trial promptly on pause
            tr_end = int(n * (0.5 + 0.1 * k))
            va_end = int(n * (0.5 + 0.1 * k) + n * 0.1)
            va_end = min(va_end, n)
            if va_end - tr_end < seq_len + 20:
                continue
            df_tr = df_feat.iloc[:tr_end]
            df_va = df_feat.iloc[tr_end:va_end]
            scX, scy = MinMaxScaler(), MinMaxScaler()
            Xtr, ytr = make_seq(df_tr, feats, target, scX, scy, seq_len, fit=True)
            Xva, yva = make_seq(df_va, feats, target, scX, scy, seq_len, fit=False)
            if len(Xtr) < 30 or len(Xva) < 5:
                continue
            torch.manual_seed(SEED)
            model = build_tunable(model_name, len(feats), hidden, dropout, mode=mode)
            score = quick_train(model, Xtr, ytr, Xva, yva, lr, wd)
            fold_scores.append(score)
            trial.report(np.mean(fold_scores), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores)) if fold_scores else 1e9
    return objective


def tune_city(city):
    from config import DEFAULT_MODE
    mode = DEFAULT_MODE
    print(f"\n{'='*60}\nTUNING: {city}  ({N_TRIALS} trials x 4 models x 3 targets)\n{'='*60}")
    df_feat = features.engineer_features(nasa.load_city(city))
    best = {}
    for target in TARGETS:
        best[target] = {}
        for model_name in MODEL_NAMES:
            print(f"  [{target}] {model_name} ...", flush=True)
            study = optuna.create_study(
                direction="minimize",
                pruner=MedianPruner(n_warmup_steps=1, n_startup_trials=PRUNE_WARMUP))
            study.optimize(make_objective(df_feat, model_name, target, mode),
                           n_trials=N_TRIALS, show_progress_bar=False)
            best[target][model_name] = {"params": study.best_params,
                                        "val_mse": round(study.best_value, 6)}
            print(f"     best val_mse={study.best_value:.5f}  {study.best_params}")

    (MODEL_DIR / city).mkdir(parents=True, exist_ok=True)
    out = paths.best_hparams_path(city, mode)
    with open(out, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n  Best hyperparameters written -> {out}")
    print("  Apply these in config.py / train_initial.py, then retrain fully.")
    return best


def tune_stream(city, n_trials=None, models=None, mode=None):
    from config import DEFAULT_MODE
    mode = mode or DEFAULT_MODE
    """
    Streaming version of tune_city: yields live events the dashboard can render.
    Tunes all 4 models x 3 targets. Each trial emits its sampled hyperparameters
    and validation score; the best-so-far updates as trials complete.
    Writes best_hparams.json at the end (same format as tune_city).
    """
    nt = n_trials or N_TRIALS
    # models to tune: the UI selection if given, else all. Preserve MODEL_NAMES order.
    _MN = [m for m in MODEL_NAMES if m in models] if models else list(MODEL_NAMES)
    if not _MN:
        _MN = list(MODEL_NAMES)
    yield {"type": "node", "node": "tune_start", "status": "active"}
    # snapshot the PRE-TUNING defaults as the baseline exactly once, so the
    # dashboard can show old->new + changed/kept even after a restart. We only
    # write it if it doesn't already exist, so a re-tune never overwrites the
    # original defaults baseline with already-tuned values.
    try:
        _bpath = paths.baseline_hparams_path(city, mode)
        if not _bpath.exists():
            from config import SEQ_LEN, LEARNING_RATE, WEIGHT_DECAY
            import models as _models
            _fmb = _models._fmode(mode)
            _pcb = _models.POC_CONFIG[_fmb]
            def _mkb(m):
                d = {"seq_len": SEQ_LEN, "lr": LEARNING_RATE, "wd": WEIGHT_DECAY, "dropout": 0.0}
                c = _pcb.get(m, {})
                if "hidden_size" in c: d["hidden"] = c["hidden_size"]
                if "hidden_units" in c: d["hidden"] = c["hidden_units"][0]
                if "q_depth" in c: d["q_depth"] = c["q_depth"]
                if "n_qubits" in c: d["n_qubits"] = c["n_qubits"]
                return d
            _def = {m: _mkb(m) for m in ("lstm", "gru", "qlstm", "qgru", "ann", "hqnn")}
            _fb = {"seq_len": SEQ_LEN, "hidden": 32, "lr": LEARNING_RATE, "wd": WEIGHT_DECAY, "dropout": 0.0}
            _base = {t: {m: {"params": dict(_def.get(m, _fb)), "val_mse": None}
                         for m in MODEL_NAMES} for t in TARGETS}
            _bpath.parent.mkdir(parents=True, exist_ok=True)
            json.dump(_base, open(_bpath, "w"), indent=2)
    except Exception:
        pass
    yield {"type": "tune_init", "n_trials": nt,
           "total": len(TARGETS) * len(_MN), "models": _MN,
           "targets": TARGETS}
    df_feat = features.engineer_features(nasa.load_city(city))
    best = {}
    done = 0
    for target in TARGETS:
        best[target] = {}
        for model_name in _MN:
            done += 1
            yield {"type": "tune_model", "status": "start", "model": model_name,
                   "target": target, "index": done,
                   "total": len(TARGETS) * len(_MN)}
            study = optuna.create_study(
                direction="minimize",
                study_name=f"{city}_{mode}_{target}_{model_name}",
                storage=control.optuna_storage_url(city, mode),
                load_if_exists=True,   # resume prior trials for this study
                pruner=MedianPruner(n_warmup_steps=1, n_startup_trials=PRUNE_WARMUP))
            yield {"type": "node", "node": "tune_study", "status": "active"}
            objective = make_objective(df_feat, model_name, target, mode)
            best_val = [min([t.value for t in study.trials if t.value is not None], default=1e18)]
            done_trials = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
            for ti in range(done_trials, nt):
                if control.is_paused():
                    control.mark_paused_phase("tune")
                    yield {"type": "paused", "phase": "tune",
                           "msg": f"Paused tuning at {model_name} {target}, trial {ti}. Press Play to resume."}
                    return
                yield {"type": "node", "node": "tune_sample", "status": "active"}
                yield {"type": "node", "node": "tune_fold", "status": "active"}
                yield {"type": "node", "node": "tune_train", "status": "active"}
                yield {"type": "node", "node": "tune_valid", "status": "active"}
                try:
                    study.optimize(objective, n_trials=1, show_progress_bar=False)
                except Exception as e:
                    yield {"type": "info", "msg": f"trial error: {e}"}
                    continue
                t = study.trials[-1]
                pruned = (t.state.name == "PRUNED")
                val = t.value if t.value is not None else None
                yield {"type": "node", "node": "tune_report", "status": "active"}
                yield {"type": "node", "node": "tune_prune" if pruned else "tune_record",
                       "status": "active"}
                if val is not None and val < best_val[0]:
                    best_val[0] = val
                yield {"type": "tune_trial", "model": model_name, "target": target,
                       "trial": ti + 1, "n_trials": nt,
                       "params": t.params, "value": (round(val, 6) if val is not None else None),
                       "pruned": pruned,
                       "best_val": (round(best_val[0], 6) if best_val[0] < 1e17 else None)}
            best[target][model_name] = {"params": study.best_params,
                                        "val_mse": round(study.best_value, 6)}
            yield {"type": "node", "node": "tune_avg", "status": "active"}
            yield {"type": "node", "node": "tune_best", "status": "active"}
            yield {"type": "tune_model", "status": "done", "model": model_name,
                   "target": target, "best_params": study.best_params,
                   "best_val": round(study.best_value, 6)}

    (MODEL_DIR / city).mkdir(parents=True, exist_ok=True)
    out = paths.best_hparams_path(city, mode)
    with open(out, "w") as f:
        json.dump(best, f, indent=2)
    yield {"type": "node", "node": "tune_done", "status": "active"}
    yield {"type": "tune_done", "best": best, "path": str(out)}


def load_hparams(city, mode=None):
    from config import DEFAULT_MODE
    mode = mode or DEFAULT_MODE
    """Return per-model hyperparameters: tuned best_hparams.json if present,
    otherwise the fixed defaults from config so the panel always has values.
    Also returns the pre-tuning 'baseline' so the UI can show old->new +
    changed/kept after a refresh. If no baseline file was saved (or it is
    missing entries), we fall back to the deterministic config DEFAULTS as the
    baseline, so the diff survives a server restart."""
    from config import SEQ_LEN, LEARNING_RATE, WEIGHT_DECAY
    out = paths.best_hparams_path(city, mode)

    # the deterministic, pre-tuning defaults (same values shown before tuning)
    # Pull the per-model defaults from the SAME source the real models use
    # (models.POC_CONFIG), keyed by the current mode, so the hparams panel always
    # shows the architecture that is actually built (e.g. qlstm 6 qubits MV / 4 UV),
    # not stale placeholder values.
    import models as _models
    _fm = _models._fmode(mode)
    _pc = _models.POC_CONFIG[_fm]
    def _mk(m):
        d = {"seq_len": SEQ_LEN, "lr": LEARNING_RATE, "wd": WEIGHT_DECAY, "dropout": 0.0}
        c = _pc.get(m, {})
        if "hidden_size" in c: d["hidden"] = c["hidden_size"]
        if "hidden_units" in c: d["hidden"] = c["hidden_units"][0]
        if "q_depth" in c: d["q_depth"] = c["q_depth"]
        if "n_qubits" in c: d["n_qubits"] = c["n_qubits"]
        return d
    _def = {m: _mk(m) for m in ("lstm", "gru", "qlstm", "qgru", "ann", "hqnn")}
    # be robust to any model in MODEL_NAMES that lacks an explicit default above
    _fallback = {"seq_len": SEQ_LEN, "hidden": 32, "lr": LEARNING_RATE, "wd": WEIGHT_DECAY, "dropout": 0.0}
    defaults_data = {t: {m: {"params": dict(_def.get(m, _fallback)), "val_mse": None}
                         for m in MODEL_NAMES} for t in TARGETS}

    # load a saved baseline if present
    baseline = None
    bpath = paths.baseline_hparams_path(city, mode)
    if bpath.exists():
        try:
            baseline = json.load(open(bpath))
        except Exception:
            baseline = None

    if out.exists():
        raw = json.load(open(out))
        # Optuna's best_params only contains the keys it SEARCHED — it may omit
        # seq_len, q_depth, n_qubits, etc. Merge the defaults UNDER each tuned
        # entry so every field always has a value to show (tuned where optuna set
        # it, default otherwise). Without this, missing keys render blank and the
        # card looks like it "lost" q_depth / qubits.
        merged = {}
        for t in TARGETS:
            merged[t] = {}
            for m in MODEL_NAMES:
                base_params = dict(_def.get(m, {}))
                entry = (raw.get(t, {}) or {}).get(m, {}) or {}
                tuned_params = entry.get("params", {}) or {}
                base_params.update(tuned_params)   # tuned overrides default
                merged[t][m] = {"params": base_params, "val_mse": entry.get("val_mse")}

        # ensure every tuned target/model has a baseline entry; fill any gap
        # (or a wholly-missing baseline) from the deterministic defaults so the
        # dashboard can always render the old->new diff after a restart.
        if not baseline:
            baseline = defaults_data
        else:
            for t in TARGETS:
                baseline.setdefault(t, {})
                for m in MODEL_NAMES:
                    if m not in baseline[t] or not baseline[t][m].get("params"):
                        baseline[t][m] = {"params": dict(_def.get(m, _fallback)), "val_mse": None}
        return {"source": "tuned", "data": merged, "baseline": baseline}

    return {"source": "default", "data": defaults_data, "baseline": baseline}


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "delhi"
    tune_city(city)
