import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "temp_f",
    "feels_like_f",
    "humidity_pct",
    "hdd",
    "cdd",
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
    "load_roll_24h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # heating/cooling degree days using 65F as comfortable baseline
    df["hdd"] = np.maximum(65 - df["temp_f"], 0)
    df["cdd"] = np.maximum(df["temp_f"] - 65, 0)

    # lag features give the model recent load context
    df["load_lag_1h"] = df["mw"].shift(1)
    df["load_lag_24h"] = df["mw"].shift(24)
    df["load_lag_168h"] = df["mw"].shift(168)  # same hour last week
    df["load_roll_24h"] = df["mw"].shift(1).rolling(24).mean()

    hour = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    month = df["datetime"].dt.month

    # sine/cosine encoding so the model sees time as continuous and circular
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["is_weekend"] = dow.isin([5, 6]).astype(int)

    return df