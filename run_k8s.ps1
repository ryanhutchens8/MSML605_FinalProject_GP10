Set-Location $PSScriptRoot
$namespace = "load-forecast"
$image = "load-forecast-api:v18"

Write-Host "Building Docker image..."
docker build -t $image .

$currentContext = kubectl config current-context
if ($currentContext.StartsWith("kind-") -and (Get-Command kind -ErrorAction SilentlyContinue)) {
    $clusterName = $currentContext.Substring(5)
    Write-Host "Loading image into kind cluster: $clusterName"
    kind load docker-image $image --name $clusterName
}

Write-Host "Making sure namespace exists..."
kubectl get namespace $namespace | Out-Null
if ($LASTEXITCODE -ne 0) {
    kubectl create namespace $namespace
}

Write-Host "Applying Kubernetes manifests..."

kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/static-deployment.yaml
kubectl apply -f k8s/static-service.yaml
kubectl apply -f k8s/dynamic-deployment.yaml
kubectl apply -f k8s/dynamic-service.yaml
kubectl apply -f k8s/inline-deployment.yaml
kubectl apply -f k8s/inline-service.yaml
kubectl apply -f k8s/drift-monitor-deployment.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

kubectl rollout status deployment/load-forecast-dynamic -n $namespace

Write-Host "Clearing old retrained model files..."
kubectl exec deployment/load-forecast-dynamic -n $namespace -- rm -f /app/retrained-models/rf_retrained.pkl /app/retrained-models/rf_retrained_scaler_X.pkl /app/retrained-models/rf_retrained_scaler_y.pkl

Write-Host "Restarting deployments so the new image is used..."
kubectl rollout restart deployment/load-forecast-static -n $namespace
kubectl rollout restart deployment/load-forecast-dynamic -n $namespace
kubectl rollout restart deployment/load-forecast-inline -n $namespace
kubectl rollout restart deployment/drift-monitor -n $namespace
kubectl rollout restart deployment/prometheus -n $namespace
kubectl rollout restart deployment/grafana -n $namespace

Write-Host "Waiting for pods to be ready..."
kubectl rollout status deployment/load-forecast-static -n $namespace
kubectl rollout status deployment/load-forecast-dynamic -n $namespace
kubectl rollout status deployment/load-forecast-inline -n $namespace
kubectl rollout status deployment/drift-monitor -n $namespace
kubectl rollout status deployment/prometheus -n $namespace
kubectl rollout status deployment/grafana -n $namespace

Write-Host "Waiting for pods to finish initializing..."
Start-Sleep -Seconds 10

Write-Host "Stopping existing port-forward processes..."
Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" |
    Where-Object { $_.CommandLine -like "*port_forward_supervisor.ps1*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process kubectl -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Starting port-forwards..."
$supervisor = Join-Path $PSScriptRoot "port_forward_supervisor.ps1"

Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$supervisor`" -Namespace `"$namespace`" -Service `"static-service`" -LocalPort 8001 -RemotePort 8000 -LogPrefix `"static`""
Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$supervisor`" -Namespace `"$namespace`" -Service `"dynamic-service`" -LocalPort 8002 -RemotePort 8000 -LogPrefix `"dynamic`""
Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$supervisor`" -Namespace `"$namespace`" -Service `"inline-service`" -LocalPort 8003 -RemotePort 8000 -LogPrefix `"inline`""
Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$supervisor`" -Namespace `"$namespace`" -Service `"grafana-service`" -LocalPort 3000 -RemotePort 3000 -LogPrefix `"grafana`""
Start-Process powershell -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File `"$supervisor`" -Namespace `"$namespace`" -Service `"prometheus-service`" -LocalPort 9090 -RemotePort 9090 -LogPrefix `"prometheus`""

Start-Sleep -Seconds 3

Write-Host "Port forwarding started"
Write-Host "Static: http://127.0.0.1:8001/docs"
Write-Host "Dynamic: http://127.0.0.1:8002/docs"
Write-Host "Inline: http://127.0.0.1:8003/docs"
Write-Host "Grafana: http://127.0.0.1:3000/d/load-forecast/load-forecast-mlops-monitor"
Write-Host "Prometheus: http://127.0.0.1:9090"
