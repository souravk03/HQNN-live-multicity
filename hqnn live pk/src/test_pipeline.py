"""
Lightweight sanity tests for the data/feature/forecast pipeline.

These use a small SYNTHETIC daily series (no network, no NASA download) and only
the CLASSICAL models (fast on any hardware) so they run in seconds. They guard
the bugs that were fixed:

  test_no_target_leakage   - no engineered feature can reconstruct the same-day target
  test_anomaly_is_past     - the *_anomaly features use yesterday's value, not today's
  test_iterative_forecast  - a multi-step forecast actually varies across the horizon
  test_tuned_lr_is_used    - train_model honours a passed-in learning rate

Run:  python test_pipeline.py        (prints PASS/FAIL per test)
"""
import numpy as np
import pandas as pd

import features
from config import TARGETS, SEQ_LEN


def _synth(n=1095, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    doy = dates.dayofyear.values
    tmp = 25 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.randn(n) * 1.5
    rh = (55 + 20 * np.sin(2 * np.pi * (doy - 180) / 365) + rng.randn(n) * 4).clip(1, 100)
    pr = 101000 + 400 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.randn(n) * 60
    u10 = rng.randn(n) * 2
    v10 = rng.randn(n) * 2
    return pd.DataFrame({
        "date": dates, "TMP2m": tmp, "RH2m": rh, "PRmsl": pr,
        "U10m": u10, "V10m": v10, "WS": np.sqrt(u10 ** 2 + v10 ** 2),
        "WD": (np.degrees(np.arctan2(-u10, -v10)) + 360) % 360,
        "QV2m": 8 + 4 * np.sin(2 * np.pi * (doy - 180) / 365) + rng.randn(n),
    })


def test_no_target_leakage():
    feat = features.engineer_features(_synth())
    for tgt in TARGETS:
        cols = features.select_features(feat, tgt, "mv")
        for c in cols:
            r = abs(np.corrcoef(feat[c], feat[tgt])[0, 1])
            assert r < 0.999, f"{c} perfectly correlates with same-day {tgt} (leak)"
    return True


def test_anomaly_is_past():
    feat = features.engineer_features(_synth())
    # fixed anomaly must reconstruct YESTERDAY's target, not today's
    recon = feat["PRmsl_anomaly"] + feat["PRmsl_roll7_mean"]
    assert not np.allclose(recon, feat["PRmsl"]), "anomaly reconstructs TODAY (leak)"
    assert np.allclose(recon.values[1:], feat["PRmsl"].values[:-1], atol=1e-6), \
        "anomaly should reconstruct yesterday"
    return True


def test_iterative_forecast():
    import torch
    from engine import build_sequences, train_model, forecast_next
    from sklearn.preprocessing import MinMaxScaler
    from models import build_model
    df = _synth()
    feat_all = features.engineer_features(df)
    models = {}
    for tgt in TARGETS:
        feats = features.select_features(feat_all, tgt, "mv")
        scX, scY = MinMaxScaler(), MinMaxScaler()
        Xtr, ytr = build_sequences(feat_all, feats, tgt, scX, scY, fit=True)
        m = build_model("lstm", len(feats), {"hidden": 16, "dropout": 0.0})
        m = train_model(m, Xtr, ytr, Xtr[-80:], ytr[-80:], epochs=8)
        models[tgt] = (m, scX, scY, feats)
    raw = df.iloc[:-1].copy()
    seqs = {t: [] for t in TARGETS}
    for h in range(15):
        fh = features.engineer_features(raw)
        day = {}
        for tgt in TARGETS:
            m, scX, scY, feats = models[tgt]
            p = forecast_next(m, fh, feats, tgt, scX, scY)
            day[tgt] = p
            seqs[tgt].append(p)
        prev = raw.iloc[-1]
        row = {"date": prev["date"] + pd.Timedelta(days=1)}
        for t in TARGETS:
            row[t] = day[t]
        for c in ["U10m", "V10m", "WS", "WD", "QV2m"]:
            if c in raw.columns:
                row[c] = prev[c]
        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
    for t in TARGETS:
        assert len(set(np.round(seqs[t], 4))) > 1, f"{t} forecast is frozen across horizon"
    return True


def test_tuned_lr_is_used():
    import torch
    from engine import build_sequences, train_model
    from sklearn.preprocessing import MinMaxScaler
    from models import build_model
    feat = features.engineer_features(_synth())
    feats = features.select_features(feat, "TMP2m", "mv")
    scX, scY = MinMaxScaler(), MinMaxScaler()
    X, y = build_sequences(feat, feats, "TMP2m", scX, scY, fit=True)
    m = build_model("lstm", len(feats), {"hidden": 12})
    # capture the optimizer lr by monkey-patching Adam briefly
    seen = {}
    import torch.optim as optim
    orig = optim.Adam
    def spy(params, lr=1e-3, **kw):
        seen["lr"] = lr
        return orig(params, lr=lr, **kw)
    optim.Adam = spy
    try:
        train_model(m, X, y, X[-50:], y[-50:], epochs=1, hparams={"lr": 0.0042, "wd": 1e-6})
    finally:
        optim.Adam = orig
    assert abs(seen.get("lr", 0) - 0.0042) < 1e-9, f"tuned lr not used (got {seen.get('lr')})"
    return True


if __name__ == "__main__":
    tests = [test_no_target_leakage, test_anomaly_is_past,
             test_iterative_forecast, test_tuned_lr_is_used]
    ok = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            ok += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed")
