"""
Initial training (run once before the backtest).

For each city x model x target:
  - split engineered data 80/20 (chronological)
  - of the 80% train block, hold out the last 15% as validation
  - fit scalers on train only, train the model, save weights + scalers
Writes metadata.json per city recording the feature list each model expects
(so the backtest can replay the exact same input columns).

Usage:
    python train_initial.py            # all cities
    python train_initial.py delhi      # one city
"""
import sys
import json

import nasa
import features
from engine import (set_seed, build_sequences, train_model, save_assets, DEVICE)
from models import build_model
from config import (CITIES, TARGETS, MODEL_NAMES, TRAIN_RATIO, MODEL_DIR)
from sklearn.preprocessing import MinMaxScaler


def train_city(city):
    print(f"\n{'='*60}\nCITY: {city}\n{'='*60}")
    set_seed()
    df_raw = nasa.load_city(city)
    df_feat = features.engineer_features(df_raw)

    n = len(df_feat)
    split = int(n * TRAIN_RATIO)          # boundary between train and backtest
    df_train_all = df_feat.iloc[:split]
    backtest_start_date = str(df_feat.iloc[split]["date"].date())
    print(f"  {n} rows  | train<= idx{split} ({df_train_all.iloc[-1]['date'].date()})"
          f"  | backtest starts {backtest_start_date}")

    # inner train/val split (last 15% of the train block for validation)
    inner = int(len(df_train_all) * 0.85)
    df_tr = df_train_all.iloc[:inner]
    df_va = df_train_all.iloc[inner:]

    meta = {"city": city, "train_ratio": TRAIN_RATIO,
            "split_index": split, "backtest_start_date": backtest_start_date,
            "n_rows": n, "features": {"mv": {}, "uv": {}}, "test_r2": {}}

    for target in TARGETS:
        for mode in ("mv", "uv"):
            feats = features.select_features(df_feat, target, mode)
            meta["features"][mode][target] = feats

    # mv models: lstm, qlstm use mv features ; gru, qgru also use mv here
    # (you can split mv/uv per model if you like; we train every model on mv
    #  features and additionally keep uv lists in metadata for reference)
    for target in TARGETS:
        feats = meta["features"]["mv"][target]
        input_size = len(feats)
        for model_name in MODEL_NAMES:
            print(f"  [{target}] {model_name} ({input_size} feats) ...", flush=True)
            sc_X, sc_y = MinMaxScaler(), MinMaxScaler()
            X_tr, y_tr = build_sequences(df_tr, feats, target, sc_X, sc_y, fit=True)
            X_va, y_va = build_sequences(df_va, feats, target, sc_X, sc_y, fit=False)
            set_seed()
            try:
                import tune as _tune
                _hp = (_tune.load_hparams(city)["data"].get(target, {})
                       .get(model_name, {}).get("params", {})) or {}
            except Exception:
                _hp = {}
            model = build_model(model_name, input_size, _hp)
            model = train_model(model, X_tr, y_tr, X_va, y_va, verbose=False)
            save_assets(city, model_name, target, model, sc_X, sc_y)
            print(f"      saved {model_name}_{target}")

    (MODEL_DIR / city).mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / city / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata.json written for {city}")


if __name__ == "__main__":
    targets = [sys.argv[1]] if len(sys.argv) > 1 else list(CITIES)
    for c in targets:
        train_city(c)
    print("\nInitial training complete.")
