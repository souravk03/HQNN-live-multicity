"""
Feature engineering.

engineer_features(df_raw)  -> df_feat  (one row per day, NaNs from warmup dropped)

Every lag / rolling / tendency feature uses .shift(1) so that the feature row for
day D only ever uses information available up to day D-1.  The target columns
themselves (TMP2m, RH2m, PRmsl) are the *current-day* values to be predicted.

select_features(df_feat, target, tr_end, mode) -> list[str]
    Picks the feature columns for a given target.  Two modes:
      'mv' : all engineered features (multivariate)
      'uv' : only this target's own derived features + seasonality (univariate)
"""

import numpy as np
import pandas as pd

from config import TARGETS

_SEASON_COLS = ["doy_sin", "doy_cos", "month_sin", "month_cos",
                "week_sin", "week_cos", "is_monsoon", "is_summer",
                "is_winter", "is_premonsoon"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    prmsl_med = df["PRmsl"].dropna().median() if "PRmsl" in df.columns else 0
    prmsl_lo, prmsl_hi = (95000, 106000) if prmsl_med > 2000 else (950, 1060)
    bounds = {"TMP2m": (-5, 55), "RH2m": (0, 100),
              "PRmsl": (prmsl_lo, prmsl_hi), "WS": (0, 50)}
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            df.loc[bad, col] = np.nan
    for col in ["TMP2m", "RH2m", "PRmsl", "WS", "QV2m"]:
        if col in df.columns:
            df[col] = df[col].interpolate("linear", limit_direction="forward").ffill().bfill()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = validate(df)
    df["date"] = pd.to_datetime(df["date"])

    # ------------------------------------------------------------------
    # Unit fixes for SLP and PS
    # Training scalers used kPa (SLP: 99.3–102.34, PS: 97.06–99.93).
    # ERA5 / Open-Meteo delivers these in hPa (~998 / ~974).
    # Detect by median and divide by 10 (hPa→kPa) or 100 (Pa→kPa).
    # ------------------------------------------------------------------
    if "SLP" in df.columns:
        mask = df["SLP"] > 200
        df.loc[mask, "SLP"] = df.loc[mask, "SLP"] / 10.0

    if "PS" in df.columns:
        mask = df["PS"] > 200
        df.loc[mask, "PS"] = df.loc[mask, "PS"] / 10.0
    # ------------------------------------------------------------------

    # wind speed / direction
    if "U10m" in df.columns and "V10m" in df.columns:
        df["WS"] = np.sqrt(df["U10m"] ** 2 + df["V10m"] ** 2)
        df["WD"] = (np.degrees(np.arctan2(-df["U10m"], -df["V10m"])) + 360) % 360
    r = np.deg2rad(df["WD"].fillna(0).values)
    df["WD_sin"] = np.sin(r)
    df["WD_cos"] = np.cos(r)

    # standard lags
    for lag in [1, 2, 3, 7, 14, 30]:
        for col in ["TMP2m", "RH2m", "PRmsl"]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for lag in [1, 3, 7]:
        df[f"WS_lag{lag}"] = df["WS"].shift(lag)
    if "QV2m" in df.columns:
        for lag in [1, 2, 3, 7]:
            df[f"QV2m_lag{lag}"] = df["QV2m"].shift(lag)

    # rolling stats
    for w in [3, 7, 14, 30]:
        for col in ["TMP2m", "RH2m", "PRmsl"]:
            df[f"{col}_roll{w}_mean"] = df[col].shift(1).rolling(w).mean()
            df[f"{col}_roll{w}_std"]  = df[col].shift(1).rolling(w).std()
    df["TMP2m_roll7_min"]   = df["TMP2m"].shift(1).rolling(7).min()
    df["TMP2m_roll7_max"]   = df["TMP2m"].shift(1).rolling(7).max()
    df["TMP2m_roll7_range"] = df["TMP2m_roll7_max"] - df["TMP2m_roll7_min"]

    # tendencies / differencing
    df["PRmsl_tend3"]  = df["PRmsl"].shift(1).diff(3) / 3.0
    df["PRmsl_tend7"]  = df["PRmsl"].shift(1).diff(7) / 7.0
    df["PRmsl_anomaly"] = df["PRmsl"].shift(1) - df["PRmsl_roll7_mean"]
    df["RH2m_diff1"]  = df["RH2m"].shift(1).diff(1)
    df["RH2m_diff3"]  = df["RH2m"].shift(1).diff(3)
    df["RH2m_anomaly7"] = df["RH2m"].shift(1) - df["RH2m_roll7_mean"]
    for col in ["TMP2m", "PRmsl"]:
        df[f"{col}_diff1"] = df[col].shift(1).diff(1)
        df[f"{col}_diff7"] = df[col].shift(1).diff(7)
    df["TMP2m_anomaly30"] = df["TMP2m"].shift(1) - df["TMP2m_roll30_mean"]

    # seasonality
    doy   = df["date"].dt.dayofyear
    month = df["date"].dt.month
    week  = df["date"].dt.isocalendar().week.astype(int)
    df["doy_sin"]   = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * doy / 365)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["week_sin"]  = np.sin(2 * np.pi * week / 52)
    df["week_cos"]  = np.cos(2 * np.pi * week / 52)
    df["is_monsoon"]    = month.between(6, 9).astype(int)
    df["is_summer"]     = month.between(3, 5).astype(int)
    df["is_winter"]     = month.isin([12, 1, 2]).astype(int)
    df["is_premonsoon"] = month.isin([4, 5]).astype(int)

    return df.dropna().reset_index(drop=True)


# univariate self-prefixes per target
_UV_PREFIX = {
    "PRmsl": ["PRmsl_lag", "PRmsl_roll", "PRmsl_tend", "PRmsl_diff", "PRmsl_anomaly"],
    "RH2m":  ["RH2m_lag", "RH2m_roll", "RH2m_diff", "RH2m_anomaly"],
    "TMP2m": ["TMP2m_lag", "TMP2m_roll", "TMP2m_diff", "TMP2m_anomaly"],
}


def select_features(df_feat: pd.DataFrame, target: str, mode: str) -> list:
    exclude = {"date", "WD", "U10m", "V10m"} | set(TARGETS)
    all_feats = [c for c in df_feat.columns if c not in exclude]
    if mode == "mv":
        return all_feats
    # uv: own derived features + seasonality
    pref = _UV_PREFIX[target]
    return [c for c in all_feats
            if any(c.startswith(p) for p in pref) or c in _SEASON_COLS]