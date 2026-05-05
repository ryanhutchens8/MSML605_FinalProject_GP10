param(
    [string]$Namespace = "load-forecast",
    [string]$Target = "apis",
    [int]$TimeoutSec = 180
)

$ErrorActionPreference = "Stop"

if ($Target -eq "static") {
    $apps = @("load-forecast-static")
} elseif ($Target -eq "dynamic") {
    $apps = @("load-forecast-dynamic")
} elseif ($Target -eq "inline") {
    $apps = @("load-forecast-inline")
} elseif ($Target -eq "monitor") {
    $apps = @("drift-monitor")
} elseif ($Target -eq "grafana") {
    $apps = @("grafana")
} elseif ($Target -eq "prometheus") {
    $apps = @("prometheus")
} elseif ($Target -eq "apis") {
    $apps = @("load-forecast-static", "load-forecast-dynamic", "load-forecast-inline")
} else {
    $apps = @("load-forecast-static", "load-forecast-dynamic", "load-forecast-inline", "drift-monitor", "prometheus", "grafana")
}

Write-Host "Pod recovery test"
Write-Host "Namespace: $Namespace"
Write-Host "Target: $Target"
Write-Host ""

foreach ($app in $apps) {
    Write-Host "Testing $app"

    $pods = @(kubectl get pods -n $Namespace -l "app=$app" -o name)

    if ($pods.Count -eq 0) {
        Write-Host "  No pods found"
        continue
    }

    Write-Host "  Deleting:"
    foreach ($pod in $pods) {
        Write-Host "  - $pod"
    }

    $start = Get-Date
    kubectl delete $pods -n $Namespace --wait=false | Out-Host

    foreach ($pod in $pods) {
        kubectl wait --for=delete $pod -n $Namespace --timeout="$($TimeoutSec)s" | Out-Host
    }

    kubectl wait --for=condition=Ready pod -n $Namespace -l "app=$app" --timeout="$($TimeoutSec)s" | Out-Host

    $end = Get-Date
    $elapsed = [math]::Round(($end - $start).TotalSeconds, 1)
    Write-Host "  Back in $elapsed seconds"
    kubectl get pods -n $Namespace -l "app=$app" | Out-Host
    Write-Host ""
}

Write-Host "Done"
