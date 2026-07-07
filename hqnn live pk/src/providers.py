"""
Multi-source ground truth for live fine-tuning.

Strategy (the sound version of your voting idea):
  - Open-Meteo ERA5 archive = primary truth.
  - NASA POWER             = confirmation / confidence gate.
  - If the two disagree by more than a per-variable threshold, the day is
    flagged low-confidence and the caller should SKIP the weight update.
  - We do NOT use "median of three": NASA and ERA5 share model lineage, so
    'closest two' would just reinforce a shared bias. Two independent-enough
    reanalyses + a disagreement gate is the defensible choice.

fetch_truth(city, date) -> dict per target:
    {"value": float, "nasa": float, "era5": float,
     "disagree": float, "confident": bool}
"""
import time
import requests
import numpy as np

from config import CITIES, TARGETS

# Max allowed |NASA - ERA5| before we distrust the day (real units).
DISAGREE_THRESHOLD = {"TMP2m": 2.0, "RH2m": 12.0, "PRmsl": 200.0}


def _open_meteo(lat, lon, date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon,
              "start_date": date, "end_date": date,
              "daily": "temperature_2m_mean,relative_humidity_2m_mean,pressure_msl_mean",
              "timezone": "auto"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    if not d.get("time"):
        return None
    return {"TMP2m": d["temperature_2m_mean"][0],
            "RH2m": d["relative_humidity_2m_mean"][0],
            "PRmsl": (d["pressure_msl_mean"][0] or 0) * 100.0}  # hPa -> Pa


def _nasa_day(lat, lon, date):
    ds = date.replace("-", "")
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters=T2M,RH2M,SLP&community=RE"
           f"&longitude={lon}&latitude={lat}&start={ds}&end={ds}"
           f"&format=JSON&time-standard=LST")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    if ds not in p["T2M"]:
        return None
    return {"TMP2m": p["T2M"][ds], "RH2m": p["RH2M"][ds],
            "PRmsl": p["SLP"][ds] * 1000.0}


def fetch_truth(city, date):
    try:
        from config import STATES as _STATES
        cfg = _STATES.get(city) or CITIES[city]
    except Exception:
        cfg = CITIES[city]
    era5 = _open_meteo(cfg["lat"], cfg["lon"], date)
    nasa = _nasa_day(cfg["lat"], cfg["lon"], date)
    out = {}
    for t in TARGETS:
        e = era5.get(t) if era5 else None
        n = nasa.get(t) if nasa else None
        if e is None and n is None:
            out[t] = {"value": None, "confident": False,
                      "nasa": None, "era5": None, "disagree": None}
            continue
        if e is None or n is None:
            val = e if e is not None else n
            out[t] = {"value": val, "nasa": n, "era5": e,
                      "disagree": None, "confident": False}  # single source = not confident
            continue
        disagree = abs(e - n)
        out[t] = {"value": e,                      # ERA5 is primary truth
                  "nasa": n, "era5": e,
                  "disagree": round(disagree, 3),
                  "confident": disagree <= DISAGREE_THRESHOLD[t]}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_truth("delhi", "2024-06-01"), indent=2))
