"""
Central configuration for the real-time / backtest weather pipeline.
Edit ONLY this file to change cities, dates, split ratio, or hyper-parameters.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # .../realtime
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FORECAST_DIR = BASE_DIR / "forecasts"
VERIFY_DIR = BASE_DIR / "verifications"
LOG_DIR = BASE_DIR / "logs"
PLOT_DIR = BASE_DIR / "plots"

for _d in (DATA_DIR, MODEL_DIR, FORECAST_DIR, VERIFY_DIR, LOG_DIR, PLOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Cities / States  (lat/lon used for the NASA POWER point query)
#   "enabled": False  -> shown in the UI but greyed out / not selectable yet.
#   Only Delhi is active for now; the rest are seeded and ready to switch on.
# ----------------------------------------------------------------------------
CITIES = {
    "delhi":       {"lat": 28.61, "lon": 77.20, "label": "Delhi",   "enabled": True},
    "maharashtra": {"lat": 19.08, "lon": 72.88, "label": "Mumbai",  "enabled": True},
    "tamil_nadu":  {"lat": 13.08, "lon": 80.27, "label": "Chennai", "enabled": True},
}

# Full set of Indian states/UTs (representative coordinates) — all greyed for now.
# Flip "enabled": True (or just presence in STATES_ENABLED) to switch one on.
STATES = {
    "andhra_pradesh":   {"lat": 16.51, "lon": 80.65, "label": "Andhra Pradesh"},
    "arunachal_pradesh":{"lat": 27.10, "lon": 93.62, "label": "Arunachal Pradesh"},
    "assam":            {"lat": 26.14, "lon": 91.74, "label": "Assam"},
    "bihar":            {"lat": 25.59, "lon": 85.14, "label": "Bihar"},
    "chhattisgarh":     {"lat": 21.25, "lon": 81.63, "label": "Chhattisgarh"},
    "goa":              {"lat": 15.50, "lon": 73.83, "label": "Goa"},
    "gujarat":          {"lat": 23.03, "lon": 72.58, "label": "Gujarat"},
    "haryana":          {"lat": 28.46, "lon": 77.03, "label": "Haryana"},
    "himachal_pradesh": {"lat": 31.10, "lon": 77.17, "label": "Himachal Pradesh"},
    "jharkhand":        {"lat": 23.36, "lon": 85.33, "label": "Jharkhand"},
    "karnataka":        {"lat": 12.97, "lon": 77.59, "label": "Karnataka"},
    "kerala":           {"lat": 8.52,  "lon": 76.94, "label": "Kerala"},
    "madhya_pradesh":   {"lat": 23.26, "lon": 77.41, "label": "Madhya Pradesh"},
    "maharashtra":      {"lat": 19.08, "lon": 72.88, "label": "Mumbai"},
    "manipur":          {"lat": 24.82, "lon": 93.94, "label": "Manipur"},
    "meghalaya":        {"lat": 25.57, "lon": 91.88, "label": "Meghalaya"},
    "mizoram":          {"lat": 23.73, "lon": 92.72, "label": "Mizoram"},
    "nagaland":         {"lat": 25.67, "lon": 94.11, "label": "Nagaland"},
    "odisha":           {"lat": 20.30, "lon": 85.82, "label": "Odisha"},
    "punjab":           {"lat": 30.73, "lon": 76.78, "label": "Punjab"},
    "rajasthan":        {"lat": 26.91, "lon": 75.79, "label": "Rajasthan"},
    "sikkim":           {"lat": 27.33, "lon": 88.61, "label": "Sikkim"},
    "tamil_nadu":       {"lat": 13.08, "lon": 80.27, "label": "Chennai"},
    "telangana":        {"lat": 17.39, "lon": 78.49, "label": "Telangana"},
    "tripura":          {"lat": 23.84, "lon": 91.28, "label": "Tripura"},
    "uttar_pradesh":    {"lat": 26.85, "lon": 80.95, "label": "Uttar Pradesh"},
    "uttarakhand":      {"lat": 30.32, "lon": 78.03, "label": "Uttarakhand"},
    "west_bengal":      {"lat": 22.57, "lon": 88.36, "label": "West Bengal"},
    "delhi":            {"lat": 28.61, "lon": 77.20, "label": "Delhi"},
    # union territories (complete the set of 36 states + UTs)
    "jammu_and_kashmir":{"lat": 34.08, "lon": 74.80, "label": "Jammu & Kashmir"},
    "ladakh":           {"lat": 34.15, "lon": 77.58, "label": "Ladakh"},
    "puducherry":       {"lat": 11.94, "lon": 79.83, "label": "Puducherry"},
    "chandigarh":       {"lat": 30.73, "lon": 76.78, "label": "Chandigarh"},
    "andaman_nicobar":  {"lat": 11.62, "lon": 92.73, "label": "Andaman & Nicobar"},
    "lakshadweep":      {"lat": 10.57, "lon": 72.64, "label": "Lakshadweep"},
    "dnh_daman_diu":    {"lat": 20.40, "lon": 72.83, "label": "DNH & Daman & Diu"},
}
# Only Delhi is switched on in the UI (others appear greyed). Terminal scripts can
# still target any state directly by passing its key.
STATES_ENABLED = list(STATES.keys())   # all 36 states/UTs selectable in the UI

# ----------------------------------------------------------------------------
# Modes (feature sets). "multivariate" is active; "univariate" is greyed for now.
# ----------------------------------------------------------------------------
MODES = {
    "multivariate": {"label": "Multivariate", "enabled": True, "ready": True},
    "univariate":   {"label": "Univariate",   "enabled": True, "ready": True},
}
MODES_ENABLED = ["multivariate", "univariate"]
DEFAULT_MODE = "multivariate"

# ----------------------------------------------------------------------------
# NASA POWER
# ----------------------------------------------------------------------------
NASA_PARAMS = ["SLP", "RH2M", "T2M", "T2M_MAX", "T2M_MIN",
               "U10M", "V10M", "WS10M", "WD10M", "QV2M", "PRECTOTCORR", "PS"]

# Backtest window: download exactly this range, train on first part, walk the rest.
DOWNLOAD_START = "20210101"
DOWNLOAD_END   = "20251231"

# ----------------------------------------------------------------------------
# Targets
#   TMP2m  -> T2M    (deg C)
#   RH2m   -> RH2M   (%)
#   PRmsl  -> SLP*1000 (Pa)
# ----------------------------------------------------------------------------
TARGETS = ["TMP2m", "RH2m", "PRmsl"]
UNITS   = {"TMP2m": "C", "RH2m": "%", "PRmsl": "Pa"}

# ----------------------------------------------------------------------------
# Sequence / split
# ----------------------------------------------------------------------------
SEQ_LEN     = 14        # lookback window (matches your original notebooks)
TRAIN_RATIO = 0.80      # first 80% used for the initial training

# ----------------------------------------------------------------------------
# Models to run  (name -> module-level class in models.py)
#   ENABLED models are trained/tuned/used. DISABLED ones (ann, hqnn) are shown
#   in the UI greyed out and skipped by every loop until their architectures are
#   provided and registered in models.py.
# ----------------------------------------------------------------------------
MODEL_INFO = {
    "lstm":  {"label": "LSTM",  "kind": "classical", "enabled": True,  "trainable": True},
    "gru":   {"label": "GRU",   "kind": "classical", "enabled": True,  "trainable": True},
    "qlstm": {"label": "QLSTM", "kind": "quantum",   "enabled": True,  "trainable": True},
    "qgru":  {"label": "QGRU",  "kind": "quantum",   "enabled": True,  "trainable": True},
    "ann":   {"label": "ANN",   "kind": "classical", "enabled": True,  "trainable": True},
    "hqnn":  {"label": "HQNN",  "kind": "quantum + classical", "enabled": True,  "trainable": True},
}
# Full ordered list (incl. not-yet-trainable) for the UI:
ALL_MODEL_NAMES = ["lstm", "qlstm", "gru", "qgru", "ann", "hqnn"]
# Active list used by training/tuning/cycle loops (only trainable models):
MODEL_NAMES = [m for m in ALL_MODEL_NAMES if MODEL_INFO[m].get("trainable")]

# ----------------------------------------------------------------------------
# Training hyper-parameters (initial train + weekly retrain)
# ----------------------------------------------------------------------------
SEED          = 42
EPOCHS        = 120
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
GRAD_CLIP     = 1.0
EARLY_STOP_PATIENCE = 12
LR_PATIENCE   = 7
LR_FACTOR     = 0.5
LR_MIN        = 1e-6

# ----------------------------------------------------------------------------
# Daily fine-tune (online weight adjustment) hyper-parameters
# ----------------------------------------------------------------------------
FT_EPOCHS = 6
FT_BATCH  = 16
FT_WINDOW = 90          # how many recent days form the fine-tune mini-dataset
FT_LR     = 1e-4

# ----------------------------------------------------------------------------
# Backtest loop
# ----------------------------------------------------------------------------
RETRAIN_EVERY = 7       # full retrain cadence (days)
QUANTUM_BACKEND = "default.qubit"   # change to lightning.gpu locally if available
N_QUBITS = 6        # multi-qubit width for HQNN
Q_DEPTH  = 2        # StronglyEntanglingLayers depth for HQNN