param(
    [string]$PrometheusUrl = "http://127.0.0.1:9090",
    [int]$Minutes = 60,
    [string]$Step = "5s"
)

$ErrorActionPreference = "Stop"

py export_prometheus_metrics.py `
    --prometheus-url $PrometheusUrl `
    --minutes $Minutes `
    --step $Step
