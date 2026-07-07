"""
Central path resolver. Everything that touches disk goes through here, so the
whole system is keyed consistently by:

    <state> / <mode> / <model> / <target>

Directory tree:

    data/<state>/nasa_cache.csv
    models/<state>/<mode>/metadata.json
    models/<state>/<mode>/best_hparams.json
    models/<state>/<mode>/hparams_baseline.json
    models/<state>/<mode>/train_metrics.json
    models/<state>/<mode>/<model>/<target>.pth
    models/<state>/<mode>/<model>/scaler_<target>.pkl
    forecasts/<state>/<mode>/ledger.csv
    forecasts/<state>/<mode>/live_state.json
    forecasts/<state>/<mode>/live_metrics.json
    logs/<state>/<mode>/weights.json
    checkpoints/<state>/<mode>/<model>/ckpt_<target>.pt
    checkpoints/<state>/<mode>/train_state.json
    checkpoints/<state>/<mode>/optuna_study.db
"""
from pathlib import Path
from config import DATA_DIR, MODEL_DIR, FORECAST_DIR, LOG_DIR, DEFAULT_MODE

BASE = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE / "checkpoints"

# ---------------------------------------------------------------------------
# State-key -> on-disk folder alias
#
# config.CITIES / config.STATES key some entries by *state* ("maharashtra",
# "tamil_nadu") but the trained model/data artifacts for those live under the
# *city* folder name ("mumbai", "chennai") instead. Every path helper below
# resolves through _canonical() first so a dashboard request for state=
# "maharashtra" finds models/mumbai/... instead of a nonexistent
# models/maharashtra/... and silently falling back to "no metadata -> retrain
# from scratch".
# ---------------------------------------------------------------------------
_STATE_TO_CITY_FOLDER = {
    "maharashtra": "mumbai",
    "tamil_nadu":  "chennai",
}


def _canonical(state):
    """Map a state/city key to the folder name actually used on disk."""
    return _STATE_TO_CITY_FOLDER.get(state, state)


def _mode(mode):
    return mode or DEFAULT_MODE


def ensure(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs[-1] if dirs else None


# ---------- data (per state; data is mode-independent) ----------
def data_dir(state):
    return ensure(DATA_DIR / _canonical(state))


def nasa_cache(state):
    return data_dir(state) / "nasa_cache.csv"


# ---------- models (per state/mode) ----------
def model_root(state, mode):
    return ensure(MODEL_DIR / _canonical(state) / _mode(mode))


def model_dir(state, mode, model_name):
    return ensure(MODEL_DIR / _canonical(state) / _mode(mode) / model_name)


def weights_path(state, mode, model_name, target):
    return model_dir(state, mode, model_name) / f"{target}.pth"


def scaler_path(state, mode, model_name, target):
    return model_dir(state, mode, model_name) / f"scaler_{target}.pkl"


def metadata_path(state, mode):
    return model_root(state, mode) / "metadata.json"


def best_hparams_path(state, mode):
    return model_root(state, mode) / "best_hparams.json"


def baseline_hparams_path(state, mode):
    return model_root(state, mode) / "hparams_baseline.json"


def train_metrics_path(state, mode):
    return model_root(state, mode) / "train_metrics.json"


def train_times_path(state, mode):
    return model_root(state, mode) / "train_times.json"


# ---------- forecasts / live cycle (per state/mode) ----------
def forecast_dir(state, mode):
    return ensure(FORECAST_DIR / _canonical(state) / _mode(mode))


def ledger_path(state, mode):
    return forecast_dir(state, mode) / "ledger.csv"


def live_state_path(state, mode):
    return forecast_dir(state, mode) / "live_state.json"


def live_metrics_path(state, mode):
    return forecast_dir(state, mode) / "live_metrics.json"


# ---------- logs (per state/mode) ----------
def log_dir(state, mode):
    return ensure(LOG_DIR / _canonical(state) / _mode(mode))


def weights_log_path(state, mode):
    return log_dir(state, mode) / "weights.json"


# ---------- checkpoints (per state/mode[/model]) ----------
def ckpt_root(state, mode):
    return ensure(CKPT_DIR / _canonical(state) / _mode(mode))


def ckpt_model_dir(state, mode, model_name):
    return ensure(CKPT_DIR / _canonical(state) / _mode(mode) / model_name)


def ckpt_path(state, mode, model_name, target):
    return ckpt_model_dir(state, mode, model_name) / f"ckpt_{target}.pt"


def train_state_path(state, mode):
    return ckpt_root(state, mode) / "train_state.json"


def optuna_db_url(state, mode):
    p = (ckpt_root(state, mode) / "optuna_study.db").as_posix()
    return f"sqlite:///{p}"