from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import os
import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
)

from app.model import predict_from_history, load_model, retrained_files_available


MODE = os.getenv("MODE", "dynamic")
DRIFT_THRESHOLD_MW = float(os.getenv("DRIFT_THRESHOLD_MW", "2000"))
INLINE_COOLDOWN_HOURS = int(os.getenv("COOLDOWN_HOURS", "336"))

USE_RETRAINED = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global USE_RETRAINED

    from app.features import FEATURE_COLUMNS
    

    USE_RETRAINED = MODE == "dynamic" and retrained_files_available()
    load_model(use_retrained=USE_RETRAINED)
    MODEL_RETRAINED_ACTIVE.labels(mode=MODE).set(1 if USE_RETRAINED else 0)
    
    from app.model import model as _model, scaler_x as _scaler_x, scaler_y as _scaler_y
    
    warmup_input = np.zeros((1, len(FEATURE_COLUMNS)))
    warmup_scaled = _scaler_x.transform(warmup_input)
    _scaler_y.inverse_transform(_model.predict(warmup_scaled).reshape(-1, 1))
    print(f"Model loaded ({MODE}, retrained={USE_RETRAINED})")
    yield


app = FastAPI(title="Load Forecasting RF Service", lifespan=lifespan)


ROLLING_MAE = Gauge(
    "rolling_mae_mw",
    "rolling MAE in MW",
    ["mode"],
)
DRIFT_ACTIVE_METRIC = Gauge(
    "drift_active",
    "1 when drift detected",
    ["mode"],
)
RETRAIN_COUNT_METRIC = Gauge(
    "retrain_count_total",
    "Number of retrains completed this session",
    ["mode"],
)
RETRAIN_STARTED = Counter(
    "retrain_started_total",
    "Number of retrains started this session",
    ["mode"],
)
MODEL_RETRAINED_ACTIVE = Gauge(
    "model_retrained_active",
    "1 if using retrained model",
    ["mode"],
)
LATEST_ACTUAL_MW = Gauge(
    "latest_actual_mw",
    "Latest actual load value in MW",
    ["mode"],
)
EVALUATED_PREDICTION_MW = Gauge(
    "evaluated_prediction_mw",
    "Prediction value evaluated against the current actual load in MW",
    ["mode"],
)
FORECAST_MW_6H = Gauge(
    "forecast_mw_6h",
    "Latest 6-hour-ahead forecast in MW",
    ["mode"],
)
PREDICTION_ERROR_MW = Gauge(
    "prediction_error_mw",
    "abs error for last evaluated prediction in MW",
    ["mode"],
)
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total predictions returned",
    ["mode"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time to generate a prediction",
    ["mode"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
RETRAIN_IN_PROGRESS = Gauge(
    "retrain_in_progress",
    "1 while inline retraining is running",
    ["mode"],
)

ROLLING_MAE.labels(mode=MODE).set(0)
DRIFT_ACTIVE_METRIC.labels(mode=MODE).set(0)
RETRAIN_COUNT_METRIC.labels(mode=MODE).set(0)
MODEL_RETRAINED_ACTIVE.labels(mode=MODE).set(0)
RETRAIN_IN_PROGRESS.labels(mode=MODE).set(0)


# 169 rows covers the 168h lag window plus current row
history_rows = deque(maxlen=169)

# predictions are queued until the target time arrives so we can evaluate error
pending_predictions = deque()
recent_errors = deque(maxlen=48)  # rolling window for MAE calculation

rolling_mae = None
retrain_count = 0
drift_active = False

_retraining = False
_retrain_lock = threading.Lock()
_last_retrain_time = None


class LoadRow(BaseModel):
    datetime: str
    mw: float
    temp_f: float
    feels_like_f: float
    humidity_pct: float


class ForceRetrainRequest(BaseModel):
    end_time: str | None = None


def normalize_time(dt) -> str:
    return pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M:%S")


def _run_inline_retrain(end_time_str):
    global USE_RETRAINED, retrain_count, rolling_mae, _retraining, _last_retrain_time

    previous_use_retrained = USE_RETRAINED

    try:
        RETRAIN_IN_PROGRESS.labels(mode=MODE).set(1)
        from app.retrain import retrain_model
        retrain_model(end_time_str)

        with _retrain_lock:
            USE_RETRAINED = True

        load_model(use_retrained=True, require_retrained=True)

        with _retrain_lock:
            retrain_count += 1
            recent_errors.clear()
            rolling_mae = None
            _last_retrain_time = datetime.now(timezone.utc)

        RETRAIN_COUNT_METRIC.labels(mode=MODE).set(retrain_count)
        ROLLING_MAE.labels(mode=MODE).set(0)
        DRIFT_ACTIVE_METRIC.labels(mode=MODE).set(0)
        MODEL_RETRAINED_ACTIVE.labels(mode=MODE).set(1)

        print(f"Inline retrain complete | retrain_count={retrain_count}")

    except Exception:
        with _retrain_lock:
            USE_RETRAINED = previous_use_retrained
        print("Inline retrain failed")

    finally:
        RETRAIN_IN_PROGRESS.labels(mode=MODE).set(0)
        with _retrain_lock:
            _retraining = False


def _start_inline_retrain(current_time, force=False):
    global _retraining

    if MODE != "inline":
        return False

    if not force and not drift_active:
        return False

    now = datetime.now(timezone.utc)
    with _retrain_lock:
        if _retraining:
            return False

        if not force:
            cooldown_ok = (
                _last_retrain_time is None
                or (now - _last_retrain_time).total_seconds() / 3600 >= INLINE_COOLDOWN_HOURS
            )
            if not cooldown_ok:
                return False

        _retraining = True
        RETRAIN_STARTED.labels(mode=MODE).inc()

    t = threading.Thread(
        target=_run_inline_retrain,
        args=(current_time,),
        daemon=True,
    )
    t.start()
    return True


def _start_inline_retrain_if_needed(current_time):
    return _start_inline_retrain(current_time, force=False)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {
        "service": "load-forecast-rf",
        "status": "running",
        "mode": MODE,
        "use_retrained": USE_RETRAINED,
        "history_rows": len(history_rows),
        "latest_datetime": history_rows[-1]["datetime"] if history_rows else None,
        "rolling_mae": rolling_mae,
        "drift_active": drift_active,
        "drift_threshold_mw": DRIFT_THRESHOLD_MW,
        "retrain_count": retrain_count,
        "retraining_in_progress": _retraining,
    }


@app.post("/predict")
def predict(row: LoadRow):
    global rolling_mae
    global drift_active

    t_start = time.time()

    row_dict = row.model_dump()
    row_dict["datetime"] = normalize_time(row_dict["datetime"])

    history_rows.append(row_dict)
    history_df = pd.DataFrame(list(history_rows))

    current_time = row_dict["datetime"]
    current_dt = pd.to_datetime(current_time)
    actual_mw = row.mw
    LATEST_ACTUAL_MW.labels(mode=MODE).set(actual_mw)

    while pending_predictions and pending_predictions[0]["target_time"] <= current_time:
        old_prediction = pending_predictions.popleft()

        if old_prediction["target_time"] == current_time:
            error = abs(old_prediction["prediction_mw"] - actual_mw)
            recent_errors.append(error)
            EVALUATED_PREDICTION_MW.labels(mode=MODE).set(old_prediction["prediction_mw"])
            PREDICTION_ERROR_MW.labels(mode=MODE).set(error)

            if len(recent_errors) > 0:
                rolling_mae = float(np.mean(recent_errors))

    drift_active = rolling_mae is not None and rolling_mae > DRIFT_THRESHOLD_MW

    ROLLING_MAE.labels(mode=MODE).set(rolling_mae if rolling_mae is not None else 0)
    DRIFT_ACTIVE_METRIC.labels(mode=MODE).set(1 if drift_active else 0)

    _start_inline_retrain_if_needed(current_time)

    try:
        prediction = predict_from_history(
            history_df,
            use_retrained=USE_RETRAINED
        )
    except ValueError:
        return {
            "status": "warming_up",
            "message": "Not enough history yet.",
            "mode": MODE,
            "use_retrained": USE_RETRAINED,
            "history_rows": len(history_rows),
            "prediction_mw_6h": None,
            "rolling_mae": rolling_mae,
            "drift_threshold_mw": DRIFT_THRESHOLD_MW,
            "drift_active": drift_active,
            "retrain_count": retrain_count,
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

    FORECAST_MW_6H.labels(mode=MODE).set(prediction)

    target_dt = current_dt + timedelta(hours=6)

    pending_predictions.append(
        {
            "made_at": current_time,
            "target_time": normalize_time(target_dt),
            "prediction_mw": prediction,
        }
    )

    PREDICTION_REQUESTS.labels(mode=MODE).inc()
    PREDICTION_LATENCY.labels(mode=MODE).observe(time.time() - t_start)

    return {
        "status": "ok",
        "mode": MODE,
        "use_retrained": USE_RETRAINED,
        "history_rows": len(history_rows),
        "prediction_mw_6h": prediction,
        "rolling_mae": rolling_mae,
        "drift_threshold_mw": DRIFT_THRESHOLD_MW,
        "drift_active": drift_active,
        "retrain_count": retrain_count,
        "retraining_in_progress": _retraining,
    }


@app.post("/reload")
def reload_model():
    global USE_RETRAINED
    global retrain_count
    global rolling_mae

    try:
        load_model(use_retrained=True, require_retrained=True)
    except Exception:
        raise HTTPException(status_code=500, detail="Model reload failed")

    USE_RETRAINED = True
    retrain_count += 1
    recent_errors.clear()
    rolling_mae = None

    RETRAIN_COUNT_METRIC.labels(mode=MODE).set(retrain_count)
    ROLLING_MAE.labels(mode=MODE).set(0)
    DRIFT_ACTIVE_METRIC.labels(mode=MODE).set(0)
    MODEL_RETRAINED_ACTIVE.labels(mode=MODE).set(1)

    print(f"Model reloaded | retrain_count={retrain_count}")

    return {
        "status": "reloaded",
        "retrain_count": retrain_count,
        "use_retrained": USE_RETRAINED,
    }


@app.post("/retrain-start")
def retrain_start():
    if MODE != "dynamic":
        raise HTTPException(status_code=400, detail="Use this on the dynamic service.")

    RETRAIN_STARTED.labels(mode=MODE).inc()

    return {
        "status": "marked",
        "mode": MODE,
    }


@app.post("/force-retrain")
def force_retrain(request: ForceRetrainRequest):
    if MODE != "inline":
        raise HTTPException(
            status_code=400,
            detail="Use this on the inline service.",
        )

    if request.end_time is None:
        if not history_rows:
            raise HTTPException(
                status_code=400,
                detail="Missing end_time and no rows have been sent yet.",
            )
        end_time = history_rows[-1]["datetime"]
    else:
        end_time = normalize_time(request.end_time)

    started = _start_inline_retrain(end_time, force=True)

    return {
        "status": "started" if started else "already_running",
        "mode": MODE,
        "end_time": end_time,
        "retraining_in_progress": _retraining,
        "retrain_count": retrain_count,
    }


@app.get("/history-size")
def history_size():
    return {"history_rows": len(history_rows)}
