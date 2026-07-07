"""
Shared run-control for pause/resume + checkpointing across training and tuning.

- A simple on-disk PAUSE flag the dashboard can toggle (pause/resume buttons).
- Checkpoint save/load helpers for training (per model+target, every N epochs).
- A tuning storage URL so Optuna studies are disk-backed and resumable.

Everything is file-based so it survives process restarts and works across the
separate request threads the server spawns for SSE.
"""
from pathlib import Path
import json
import os
from config import DEFAULT_MODE

BASE = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

PAUSE_FILE = CKPT_DIR / "PAUSE"

# Same state-key -> on-disk city-folder alias as paths.py. config.CITIES /
# config.STATES key Mumbai/Chennai by state ("maharashtra", "tamil_nadu"),
# but their trained checkpoints/models live under the city folder name.
_STATE_TO_CITY_FOLDER = {
    "maharashtra": "mumbai",
    "tamil_nadu":  "chennai",
}


def _canonical(city):
    return _STATE_TO_CITY_FOLDER.get(city, city)


def _mode(mode):
    return mode or DEFAULT_MODE


def _mode_dir(city, mode):
    d = CKPT_DIR / _canonical(city) / _mode(mode)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------- pause flag (global; one run at a time) ----------------
def request_pause():
    PAUSE_FILE.write_text("1")


def clear_pause():
    if PAUSE_FILE.exists():
        PAUSE_FILE.unlink()


def is_paused() -> bool:
    return PAUSE_FILE.exists()


PAUSED_PHASE = CKPT_DIR / "paused_phase"


def mark_paused_phase(phase: str):
    PAUSED_PHASE.write_text(phase)


def clear_paused_phase():
    if PAUSED_PHASE.exists():
        PAUSED_PHASE.unlink()


# ---------------- training checkpoints (per city/mode/model) ----------------
def ckpt_path(city, model_name, target, mode=None):
    d = _mode_dir(city, mode) / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ckpt_{target}.pt"


def _train_state_file(city, mode):
    return _mode_dir(city, mode) / "train_state.json"


def save_train_state(city, target_idx, model_idx, note="", mode=None):
    _train_state_file(city, mode).write_text(json.dumps(
        {"city": city, "mode": _mode(mode), "target_idx": target_idx,
         "model_idx": model_idx, "note": note}))


def load_train_state(city=None, mode=None):
    # if a city/mode is given, read that mode's file; otherwise (legacy callers)
    # fall back to the default mode for the given/known city.
    if city is None:
        return None
    f = _train_state_file(city, mode)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None


def clear_train_state(city=None, mode=None):
    if city is None:
        return
    f = _train_state_file(city, mode)
    if f.exists():
        f.unlink()


# ---------------- tuning storage (per city/mode) ----------------
def optuna_storage_url(city, mode=None) -> str:
    d = _mode_dir(city, mode)
    # forward slashes work for sqlite URL on all platforms
    p = (d / "optuna_study.db").as_posix()
    return f"sqlite:///{p}"