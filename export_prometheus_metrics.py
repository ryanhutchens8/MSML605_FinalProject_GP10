import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

QUERIES = {
    "static_rolling_mae": 'rolling_mae_mw{job="load-forecast-static"}',
    "dynamic_rolling_mae": 'rolling_mae_mw{job="load-forecast-dynamic"}',
    "inline_rolling_mae": 'rolling_mae_mw{job="load-forecast-inline"}',
    "static_error": 'prediction_error_mw{job="load-forecast-static"}',
    "dynamic_error": 'prediction_error_mw{job="load-forecast-dynamic"}',
    "inline_error": 'prediction_error_mw{job="load-forecast-inline"}',
    "static_forecast": 'forecast_mw_6h{job="load-forecast-static"}',
    "dynamic_forecast": 'forecast_mw_6h{job="load-forecast-dynamic"}',
    "inline_forecast": 'forecast_mw_6h{job="load-forecast-inline"}',
    "static_actual": 'latest_actual_mw{job="load-forecast-static"}',
    "dynamic_actual": 'latest_actual_mw{job="load-forecast-dynamic"}',
    "inline_actual": 'latest_actual_mw{job="load-forecast-inline"}',
    "static_memory_bytes": 'process_resident_memory_bytes{job="load-forecast-static"}',
    "dynamic_memory_bytes": 'process_resident_memory_bytes{job="load-forecast-dynamic"}',
    "inline_memory_bytes": 'process_resident_memory_bytes{job="load-forecast-inline"}',
    "static_cpu_cores": 'rate(process_cpu_seconds_total{job="load-forecast-static"}[1m])',
    "dynamic_cpu_cores": 'rate(process_cpu_seconds_total{job="load-forecast-dynamic"}[1m])',
    "inline_cpu_cores": 'rate(process_cpu_seconds_total{job="load-forecast-inline"}[1m])',
    "static_requests_per_second": 'rate(prediction_requests_total{job="load-forecast-static"}[1m])',
    "dynamic_requests_per_second": 'rate(prediction_requests_total{job="load-forecast-dynamic"}[1m])',
    "inline_requests_per_second": 'rate(prediction_requests_total{job="load-forecast-inline"}[1m])',
    "static_latency_p95_seconds": 'histogram_quantile(0.95, sum by (le) (rate(prediction_duration_seconds_bucket{job="load-forecast-static"}[1m])))',
    "dynamic_latency_p95_seconds": 'histogram_quantile(0.95, sum by (le) (rate(prediction_duration_seconds_bucket{job="load-forecast-dynamic"}[1m])))',
    "inline_latency_p95_seconds": 'histogram_quantile(0.95, sum by (le) (rate(prediction_duration_seconds_bucket{job="load-forecast-inline"}[1m])))',
    "dynamic_retrain_count": 'retrain_count_total{job="load-forecast-dynamic"}',
    "inline_retrain_count": 'retrain_count_total{job="load-forecast-inline"}',
    "dynamic_model_retrained": 'model_retrained_active{job="load-forecast-dynamic"}',
    "inline_model_retrained": 'model_retrained_active{job="load-forecast-inline"}',
    "dynamic_drift_active": 'drift_active{job="load-forecast-dynamic"}',
    "inline_drift_active": 'drift_active{job="load-forecast-inline"}',
    "inline_retrain_in_progress": 'retrain_in_progress{job="load-forecast-inline"}',
}


def query_range(prometheus_url, query, start, end, step):
    params = urlencode(
        {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        }
    )
    url = f"{prometheus_url.rstrip('/')}/api/v1/query_range?{params}"

    with urlopen(url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("status") != "success":
        return []
    return payload.get("data", {}).get("result", [])


def export_metrics(prometheus_url, minutes, step):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"prometheus_metrics_{stamp}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "timestamp", "value", "job", "mode", "instance"],
        )
        writer.writeheader()

        for metric_name, query in QUERIES.items():
            series_list = query_range(prometheus_url, query, start, end, step)
            for series in series_list:
                labels = series.get("metric", {})
                for timestamp, value in series.get("values", []):
                    writer.writerow(
                        {
                            "metric": metric_name,
                            "timestamp": datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat(),
                            "value": value,
                            "job": labels.get("job", ""),
                            "mode": labels.get("mode", ""),
                            "instance": labels.get("instance", ""),
                        }
                    )

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--step", default="5s")
    args = parser.parse_args()

    output_path = export_metrics(args.prometheus_url, args.minutes, args.step)
    print(output_path)


if __name__ == "__main__":
    main()
