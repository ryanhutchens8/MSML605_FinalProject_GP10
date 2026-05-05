import gc
import os
import threading
from pathlib import Path

import joblib
import pandas as pd

from app.features import add_features, FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "rf_2018.pkl"
SCALER_X_PATH = BASE_DIR / "models" / "rf_2018_scaler_X.pkl"
SCALER_Y_PATH = BASE_DIR / "models" / "rf_2018_scaler_y.pkl"

_RETRAINED_DIR = Path(os.getenv("RETRAINED_MODELS_DIR", str(BASE_DIR / "models")))

RETRAINED_MODEL_PATH = _RETRAINED_DIR / "rf_retrained.pkl"
RETRAINED_SCALER_X_PATH = _RETRAINED_DIR / "rf_retrained_scaler_X.pkl"
RETRAINED_SCALER_Y_PATH = _RETRAINED_DIR / "rf_retrained_scaler_y.pkl"


# module-level model state shared across requests
model = None
scaler_x = None
scaler_y = None
loaded_retrained = False
_model_lock = threading.RLock()  # reentrant lock so load_model can be called from predict


def retrained_files_available():
    return (
        RETRAINED_MODEL_PATH.exists()
        and RETRAINED_SCALER_X_PATH.exists()
        and RETRAINED_SCALER_Y_PATH.exists()
    )


def load_model(use_retrained=False, require_retrained=False):
    global model
    global scaler_x
    global scaler_y
    global loaded_retrained

    if use_retrained and retrained_files_available():
        new_model = joblib.load(RETRAINED_MODEL_PATH)
        new_scaler_x = joblib.load(RETRAINED_SCALER_X_PATH)
        new_scaler_y = joblib.load(RETRAINED_SCALER_Y_PATH)
        new_loaded_retrained = True
    else:
        if require_retrained:
            raise FileNotFoundError("Missing retrained model files.")

        new_model = joblib.load(MODEL_PATH)
        new_scaler_x = joblib.load(SCALER_X_PATH)
        new_scaler_y = joblib.load(SCALER_Y_PATH)
        new_loaded_retrained = False

    with _model_lock:
        old_model = model
        old_scaler_x = scaler_x
        old_scaler_y = scaler_y

        model = new_model
        scaler_x = new_scaler_x
        scaler_y = new_scaler_y
        loaded_retrained = new_loaded_retrained

    # free old model memory before the next request hits
    del old_model, old_scaler_x, old_scaler_y
    gc.collect()

    return model, scaler_x, scaler_y


def predict_from_history(history_df: pd.DataFrame, use_retrained=False) -> float:
    with _model_lock:
        needs_load = model is None or use_retrained != loaded_retrained

    if needs_load:
        load_model(use_retrained)

    featured = add_features(history_df)
    usable = featured.dropna()

    if len(usable) == 0:
        raise ValueError("Not enough history yet. Need at least 168 hours of data.")

    latest = usable.iloc[-1]
    X = latest[FEATURE_COLUMNS].to_frame().T

    with _model_lock:
        active_model = model
        active_scaler_x = scaler_x
        active_scaler_y = scaler_y

    X_scaled = active_scaler_x.transform(X.values)

    y_scaled = active_model.predict(X_scaled).reshape(-1, 1)
    y = active_scaler_y.inverse_transform(y_scaled)

    return float(y[0][0])
