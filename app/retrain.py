import os
from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from app.features import add_features, FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "load_weather_full.csv"

_RETRAINED_DIR = Path(os.getenv("RETRAINED_MODELS_DIR", str(BASE_DIR / "models")))
_RETRAINED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT_PATH = _RETRAINED_DIR / "rf_retrained.pkl"
SCALER_X_PATH = _RETRAINED_DIR / "rf_retrained_scaler_X.pkl"
SCALER_Y_PATH = _RETRAINED_DIR / "rf_retrained_scaler_y.pkl"

RETRAIN_YEARS = 1


def retrain_model(end_time: str):
    print("Starting retraining...")

    end_dt = pd.to_datetime(end_time)
    # retrain on the past year to capture seasonal patterns without losing too much history
    start_dt = end_dt - pd.DateOffset(years=RETRAIN_YEARS)

    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])

    df = add_features(df)

    # shift target back 6 hours so each row predicts load 6h ahead
    df["target_mw_6h"] = df["mw"].shift(-6)

    df = df[
        (df["datetime"] >= start_dt)
        & (df["datetime"] <= end_dt)
    ].copy()

    df = df.dropna()

    print(f"Retraining window: {start_dt} to {end_dt}")
    print(f"Retraining rows: {len(df)}")

    if len(df) < 100:
        raise ValueError("Not enough data for retraining window.")

    X_raw = df[FEATURE_COLUMNS].values
    y_raw = df["target_mw_6h"].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X_raw)
    y = scaler_y.fit_transform(y_raw).flatten()

    # n_estimators=200 was slower with no real improvement on validation
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=2,
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_OUT_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    print("Retraining complete.")
    print(f"Model saved to: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", required=True)
    args = parser.parse_args()

    retrain_model(args.end_time)
