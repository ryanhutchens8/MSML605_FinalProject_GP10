param(
    [string]$EndTime = "",
    [string]$Namespace = "load-forecast",
    [string]$Image = "load-forecast-api:v18"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($EndTime)) {
    try {
        $status = Invoke-RestMethod -Uri "http://127.0.0.1:8002/" -TimeoutSec 5
        if ($status.latest_datetime) {
            $EndTime = $status.latest_datetime
        }
    } catch {
        Write-Host "Could not read latest dynamic timestamp."
    }
}

if ([string]::IsNullOrWhiteSpace($EndTime)) {
    $EndTime = "2024-12-31 18:00:00"
}

$jobName = "force-dynamic-retrain-$((Get-Date).ToString('yyyyMMddHHmmss'))"

Write-Host "Starting dynamic retrain at end_time=$EndTime"

$tmpFile = [System.IO.Path]::GetTempFileName() + ".yaml"

@"
apiVersion: batch/v1
kind: Job
metadata:
  name: $jobName
  namespace: $Namespace
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 300
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: retrain
          image: $Image
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "app.retrain", "--end-time", "$EndTime"]
          env:
            - name: RETRAINED_MODELS_DIR
              value: "/app/retrained-models"
          volumeMounts:
            - name: model-storage
              mountPath: /app/retrained-models
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-storage
"@ | Out-File -FilePath $tmpFile -Encoding utf8

kubectl apply -f $tmpFile
Remove-Item $tmpFile
try {
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8002/retrain-start" -TimeoutSec 10 | Out-Null
} catch {
    Write-Host "Could not mark dynamic retrain start."
}

kubectl wait --for=condition=complete "job/$jobName" -n $Namespace --timeout=10m
kubectl logs -n $Namespace "job/$jobName"

Write-Host "Reloading dynamic model files..."
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8002/reload" -TimeoutSec 30

Write-Host "Starting inline retrain..."
$body = @{ end_time = $EndTime } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8003/force-retrain" -Body $body -ContentType "application/json" -TimeoutSec 30

Write-Host "Done, inline may still retrianing"
