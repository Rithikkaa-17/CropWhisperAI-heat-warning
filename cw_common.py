"""CropWhisperAI - shared data loading, event definition and feature engineering.

All features at time t use readings at or before t only (causal).
"""
import numpy as np
import pandas as pd

AIR_T, RH, SOIL_T, SOIL_M, LUX, WL = (
    "DHT22_Air_Temperature_C", "DHT22_Humidity_pct", "DS18B20_Soil_Temperature_C",
    "Capacitive_Soil_Moisture_pct", "BH1750_Light_lux", "Water_Level_pct")
SENSORS = [AIR_T, RH, SOIL_T, SOIL_M, LUX, WL]
SHORT = {AIR_T: "Tair", RH: "RH", SOIL_T: "Tsoil", SOIL_M: "SM", LUX: "Lux", WL: "WL"}

AIR_THR, SOIL_THR = 35.0, 32.0     # thermal rule (project definition)
STEP_MIN = 5                        # sampling interval (min)
MERGE_GAP = 6                       # merge thermal runs separated by <= 30 min
MIN_LEN = 3                         # an episode needs >= 15 min above threshold


import os
# Quality-control mode (revision 1):
#   "sensor" (default) - flags invalidate only the affected sensor(s): DHT22_TIMEOUT -> air T and RH;
#                        MAINTENANCE / RECALIBRATION_CHECK -> all sensors (node-level events);
#                        WIFI_RETRY -> none (upload failure, measurement itself is valid).
#                        Invalid values are carried forward from the last valid reading of THAT sensor.
#   "row"              - original submission: any non-OK flag invalidates every sensor on that row.
#   "sensor_nan"       - as "sensor" but invalid values are left missing (no carry-forward).
QC_MODE = os.environ.get("CW_QC", "sensor")
FLAG_SENSORS = {
    "OK": [], "WIFI_RETRY": [],
    "DHT22_TIMEOUT": [AIR_T, RH],
    "DS18B20_FAULT": [SOIL_T], "SOIL_PROBE_FAULT": [SOIL_M], "BH1750_FAULT": [LUX],
    "MAINTENANCE": SENSORS, "RECALIBRATION_CHECK": SENSORS,
}


def load(path="data.csv", mode=None):
    mode = mode or QC_MODE
    d = pd.read_csv(path, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    st = d["Sensor_Status"].astype(str)
    if mode == "row":
        bad = {c: (st != "OK").values for c in SENSORS}
    else:
        bad = {c: st.map(lambda s: c in FLAG_SENSORS.get(s, SENSORS)).values | d[c].isna().values for c in SENSORS}
    # a time step can be scored only if both temperature sensors that define the label are valid
    d["valid"] = (~(bad[AIR_T] | bad[SOIL_T])).astype(int)
    for c in SENSORS:
        d[c + "_valid"] = (~bad[c]).astype(int)
        d.loc[bad[c], c] = np.nan
        if mode != "sensor_nan":
            d[c] = d[c].ffill()          # causal last-valid-value carry-forward, per sensor
    return d


def raw_thermal(d):
    return ((d[AIR_T] > AIR_THR) | (d[SOIL_T] > SOIL_THR)).astype(int).values


def episodes(s, gap=MERGE_GAP, min_len=MIN_LEN):
    """Debounced heat-stress episodes as (start_idx, end_idx)."""
    idx = np.where(s == 1)[0]
    if len(idx) == 0:
        return []
    eps, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b <= gap:
            b = i
        else:
            eps.append((a, b)); a = b = i
    eps.append((a, b))
    return [(a, b) for a, b in eps if s[a:b + 1].sum() >= min_len]


def make_labels(d, horizon_steps):
    """y_t = 1 if an episode starts in (t, t+H]. Eligible t: valid, not raw-thermal, not inside an episode."""
    s = raw_thermal(d)
    eps = episodes(s)
    n = len(d)
    in_ep = np.zeros(n, bool)
    onset = np.zeros(n, bool)
    for a, b in eps:
        in_ep[a:b + 1] = True
        onset[a] = True
    y = np.zeros(n, int)
    on_idx = np.where(onset)[0]
    for o in on_idx:
        y[max(0, o - horizon_steps):o] = 1
    eligible = (d["valid"].values == 1) & (s == 0) & (~in_ep)
    return y, eligible, eps


def vpd_kpa(t, rh):
    svp = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return svp * (1 - rh / 100.0)


def build_features(d):
    f = pd.DataFrame(index=d.index)
    d = d.copy()
    d["VPD"] = vpd_kpa(d[AIR_T], d[RH])
    d["dT"] = d[AIR_T] - d[SOIL_T]
    base = SENSORS + ["VPD", "dT"]
    short = dict(SHORT, VPD="VPD", dT="dTas")
    for c in base:
        k = short[c]
        x = d[c]
        f[f"{k}"] = x
        for lag in (3, 6, 12):                      # 15, 30, 60 min
            f[f"{k}_d{lag * 5}"] = x - x.shift(lag)
        r = x.rolling(12, min_periods=6)             # last 60 min
        f[f"{k}_mean60"] = r.mean()
        f[f"{k}_std60"] = r.std()
        f[f"{k}_max60"] = r.max()
    # distance to thresholds and 30-min linear extrapolation
    f["Tair_margin"] = AIR_THR - d[AIR_T]
    f["Tsoil_margin"] = SOIL_THR - d[SOIL_T]
    # physics-inspired thermal load: degree-minutes above 30 C in last 3 h, cumulative light in last 2 h
    f["Tair_dm30_3h"] = (d[AIR_T] - 30).clip(lower=0).rolling(36, min_periods=1).sum() * STEP_MIN
    f["Lux_int_2h"] = d[LUX].rolling(24, min_periods=1).sum() * STEP_MIN * 60 / 1e6   # Mlux*s
    # yesterday's max (same-time-previous-day persistence of weather regime)
    f["Tsoil_max_24h"] = d[SOIL_T].rolling(288, min_periods=12).max()
    f["Tair_max_24h"] = d[AIR_T].rolling(288, min_periods=12).max()
    h = d.Timestamp.dt.hour + d.Timestamp.dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return f


SENSOR_GROUPS = {   # feature prefixes that each physical sensor provides
    "DHT22": ["Tair", "RH", "VPD"],
    "DS18B20": ["Tsoil"],
    "Capacitive": ["SM"],
    "BH1750": ["Lux"],
    "WaterLevel": ["WL"],
}


def cols_for(sensors, all_cols):
    """Columns computable from a set of sensors (time features always allowed)."""
    pref = set()
    for s in sensors:
        pref.update(SENSOR_GROUPS[s])
    keep = []
    for c in all_cols:
        if c.startswith("hour_"):
            keep.append(c); continue
        p = c.split("_")[0]
        if p == "dTas":
            if {"DHT22", "DS18B20"} <= set(sensors):
                keep.append(c)
            continue
        if p in pref:
            keep.append(c)
    return keep


def day_blocks(d, n_folds=10):
    """Contiguous day blocks for blocked cross-validation."""
    days = d.Timestamp.dt.normalize()
    ud = np.array(sorted(days.unique()))
    chunks = np.array_split(ud, n_folds)
    fold = np.zeros(len(d), int)
    for k, ch in enumerate(chunks):
        fold[days.isin(ch).values] = k
    return fold, chunks
