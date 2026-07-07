"""
NASA POWER data access.

download_city(city)        -> full pull DOWNLOAD_START..DOWNLOAD_END, cached to CSV.
load_city(city)            -> returns the cached dataframe (downloads if missing).

The dataframe has a normalised schema used everywhere downstream:
    date, TMP2m, RH2m, PRmsl, U10m, V10m, WS, WD, QV2m, PRCP,
    TMP2m_max, TMP2m_min, PS_Pa
"""
import time
import requests
import numpy as np
import pandas as pd

from config import (NASA_PARAMS, CITIES, DOWNLOAD_START, DOWNLOAD_END,
                    DATA_DIR)


def _cache_path(city: str):
    import paths
    return paths.nasa_cache(city)


def _norm_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _fetch(lat: float, lon: float, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={','.join(NASA_PARAMS)}&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON&time-standard=LST"
    )
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as exc:
            if attempt < retries:
                time.sleep(10 * attempt)
            else:
                raise RuntimeError(f"NASA POWER fetch failed: {exc}")

    df = pd.DataFrame(data["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.reset_index()
    df.replace([-999.0, -999], np.nan, inplace=True)

    # Pressure: NASA SLP is in kPa -> Pa
    if "SLP" in df.columns:
        df["PRmsl"] = df["SLP"] * 1000.0
    if "PS" in df.columns:
        df["PS_Pa"] = df["PS"] * 1000.0

    df = df.rename(columns={
        "T2M": "TMP2m", "RH2M": "RH2m",
        "T2M_MAX": "TMP2m_max", "T2M_MIN": "TMP2m_min",
        "U10M": "U10m", "V10M": "V10m",
        "WS10M": "WS", "WD10M": "WD",
        "QV2M": "QV2m", "PRECTOTCORR": "PRCP",
    })
    return _norm_dates(df)


def download_city(city: str) -> pd.DataFrame:
    from config import STATES as _STATES
    cfg = _STATES.get(city) or CITIES[city]
    print(f"[{city}] downloading NASA {DOWNLOAD_START} -> {DOWNLOAD_END} ...")
    df = _fetch(cfg["lat"], cfg["lon"], DOWNLOAD_START, DOWNLOAD_END)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df.to_csv(_cache_path(city), index=False)
    print(f"[{city}] saved {len(df)} rows -> {_cache_path(city).name}")
    return df


def load_city(city: str) -> pd.DataFrame:
    p = _cache_path(city)
    if p.exists():
        return _norm_dates(pd.read_csv(p))
    return download_city(city)


if __name__ == "__main__":
    for c in CITIES:
        download_city(c)
