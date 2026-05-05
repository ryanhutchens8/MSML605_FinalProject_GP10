# Drift-Aware Electrical Grid Load Forecasting on Kubernetes

This repo runs the local Kubernetes demo for the drift-aware load forecasting
project. It uses Docker, Kubernetes, Prometheus, and Grafana to compare three
serving modes:

- **Static**: uses the original Random Forest model and never retrains
- **Inline**: retrains inside the same pod that handles predictions
- **Dynamic**: starts a separate Kubernetes Job for retraining and reloads the updated model

The data is replayed from the historical hourly load/weather CSV so the demo can
move faster than real time.

## Requirements

Before running the project, make sure the following are installed:

- Docker Desktop
- Kubernetes enabled in Docker Desktop
- `kubectl`
- PowerShell
- Python, or the included `.venv`

## Deploy the System

Open PowerShell and go to the project folder:

```powershell
cd "<project-folder>"
```

Then run:

```powershell
.\run_k8s.ps1
```

This script builds the Docker image, applies the Kubernetes configuration files,
starts the services, and opens local port-forwarding for the APIs, Prometheus,
and Grafana.

When it finishes, the main services should be available here:

- Static API: <http://127.0.0.1:8001/docs>
- Dynamic API: <http://127.0.0.1:8002/docs>
- Inline API: <http://127.0.0.1:8003/docs>
- Grafana dashboard: <http://127.0.0.1:3000/d/load-forecast/load-forecast-mlops-monitor>
- Prometheus: <http://127.0.0.1:9090>

Grafana login:

```text
username: admin
password: admin
```

Anonymous access is enabled, so the dashboard may open without logging in. If
Grafana opens but the dashboard does not show automatically, use the dashboard
tab on the left side and click **Load Forecast MLOps Monitor**.

## Start the Data Stream

The dashboard will not show meaningful model behavior until data is sent to the
services. In a second PowerShell window, run:

```powershell
cd "<project-folder>"
.\.venv\Scripts\python.exe -m app.simulator
```

If the virtual environment is not available, use:

```powershell
python -m app.simulator
```

The simulator sends one hourly row per second to the static, dynamic, and inline
services. The first 168 rows are mostly warmup because the model uses a 168-hour
lag feature. After about three minutes, predictions and rolling error metrics
should start showing up more clearly in Grafana.

## Manual Retraining Test

To force retraining without waiting for drift:

```powershell
.\force_retrain_both.ps1
```

This starts retraining for both the dynamic and inline approaches. It is useful
for demos because it makes the retraining behavior visible without waiting for
the drift threshold to trigger naturally.

## Pod Recovery Test

To test Kubernetes recovery after a pod failure:

```powershell
.\pod_recovery_test.ps1
```

This deletes a selected pod and measures how long Kubernetes takes to bring the
service back.

## Export Prometheus Metrics

To export collected metrics from the last hour of clock time:

```powershell
.\export_prometheus_metrics.ps1
```

The output is saved in the `results/` folder.

## Stop or Reset

To stop the Kubernetes resources:

```powershell
kubectl delete namespace load-forecast
```

To deploy again after stopping:

```powershell
.\run_k8s.ps1
```
