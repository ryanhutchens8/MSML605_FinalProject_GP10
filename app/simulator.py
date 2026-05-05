import csv
import time
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]

STATIC_URL = "http://127.0.0.1:8001/predict"
DYNAMIC_URL = "http://127.0.0.1:8002/predict"
INLINE_URL = "http://127.0.0.1:8003/predict"

CSV_PATH = BASE_DIR / "data" / "load_weather_full.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "model_comparison_results.csv"

SECONDS_PER_ROW = 1
REQUEST_TIMEOUT_SECONDS = 10
SERVICE_RECOVERY_TIMEOUT_SECONDS = 120
SERVICE_RETRY_DELAY_SECONDS = 2


FIELDNAMES = [
    "step",
    "datetime",
    "actual_mw",
    "static_prediction",
    "dynamic_prediction",
    "inline_prediction",
    "static_rolling_mae",
    "dynamic_rolling_mae",
    "inline_rolling_mae",
    "static_error",
    "dynamic_error",
    "inline_error",
    "dynamic_retrain_count",
    "dynamic_drift_active",
    "inline_retrain_count",
    "inline_drift_active",
    "inline_retraining_in_progress",
]


def safe_error(prediction, actual):
    if prediction is None:
        return None
    return abs(float(prediction) - float(actual))


def run_simulation():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting simulation...")
    print(f"Reading data from: {CSV_PATH}")
    print(f"Static model:  {STATIC_URL}")
    print(f"Dynamic model: {DYNAMIC_URL}")
    print(f"Inline model:  {INLINE_URL}")
    print(f"Writing results to: {RESULTS_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    # 2018 is used for initial training, simulation runs on 2019 onward
    df = df[df["datetime"] >= "2019-01-01"].reset_index(drop=True)

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, row in df.iterrows():
            payload = {
                "datetime": str(row["datetime"]),
                "mw": float(row["mw"]),
                "temp_f": float(row["temp_f"]),
                "feels_like_f": float(row["feels_like_f"]),
                "humidity_pct": float(row["humidity_pct"]),
            }

            # retries if the service is temporarily unavailable (e.g. pod restart)
            def call(name, url):
                deadline = time.monotonic() + SERVICE_RECOVERY_TIMEOUT_SECONDS
                waiting_logged = False

                while True:
                    try:
                        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
                        if not resp.text:
                            raise requests.RequestException("empty response body")

                        data = resp.json()
                        if not resp.ok and isinstance(data.get("detail"), dict):
                            return data["detail"]
                        if not resp.ok:
                            resp.raise_for_status()
                        return data
                    except (requests.RequestException, ValueError):
                        if time.monotonic() >= deadline:
                            print(f"  {name} unavailable after {SERVICE_RECOVERY_TIMEOUT_SECONDS}s")
                            return {}

                        if not waiting_logged:
                            print(f"  Waiting for {name} service")
                            waiting_logged = True

                        time.sleep(SERVICE_RETRY_DELAY_SECONDS)

            res_static = call("static", STATIC_URL)
            res_dynamic = call("dynamic", DYNAMIC_URL)
            res_inline = call("inline", INLINE_URL)

            static_pred = res_static.get("prediction_mw_6h")
            dynamic_pred = res_dynamic.get("prediction_mw_6h")
            inline_pred = res_inline.get("prediction_mw_6h")

            static_error = safe_error(static_pred, payload["mw"])
            dynamic_error = safe_error(dynamic_pred, payload["mw"])
            inline_error = safe_error(inline_pred, payload["mw"])

            writer.writerow(
                {
                    "step": i,
                    "datetime": payload["datetime"],
                    "actual_mw": payload["mw"],
                    "static_prediction": static_pred,
                    "dynamic_prediction": dynamic_pred,
                    "inline_prediction": inline_pred,
                    "static_rolling_mae": res_static.get("rolling_mae"),
                    "dynamic_rolling_mae": res_dynamic.get("rolling_mae"),
                    "inline_rolling_mae": res_inline.get("rolling_mae"),
                    "static_error": static_error,
                    "dynamic_error": dynamic_error,
                    "inline_error": inline_error,
                    "dynamic_retrain_count": res_dynamic.get("retrain_count"),
                    "dynamic_drift_active": res_dynamic.get("drift_active"),
                    "inline_retrain_count": res_inline.get("retrain_count"),
                    "inline_drift_active": res_inline.get("drift_active"),
                    "inline_retraining_in_progress": res_inline.get("retraining_in_progress"),
                }
            )
            f.flush()

            print(
                f"Step {i} | "
                f"S_mae={res_static.get('rolling_mae')} | "
                f"D_mae={res_dynamic.get('rolling_mae')} | "
                f"I_mae={res_inline.get('rolling_mae')} | "
                f"I_retraining={res_inline.get('retraining_in_progress')}"
            )

            time.sleep(SECONDS_PER_ROW)


if __name__ == "__main__":
    run_simulation()
