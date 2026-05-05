param(
    [string]$Namespace,
    [string]$Service,
    [int]$LocalPort,
    [int]$RemotePort,
    [string]$LogPrefix
)

$ErrorActionPreference = "Continue"

$LogPath = Join-Path $PSScriptRoot "$LogPrefix-port-forward.log"
$ErrPath = Join-Path $PSScriptRoot "$LogPrefix-port-forward.err"
$PortMap = "${LocalPort}:${RemotePort}"

while ($true) {
    $startedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$startedAt] Starting port-forward service/$Service $PortMap"

    & kubectl port-forward -n $Namespace "service/$Service" $PortMap 1>> $LogPath 2>> $ErrPath

    $exitCode = $LASTEXITCODE
    $endedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $ErrPath -Value "[$endedAt] port-forward service/$Service exited with code $exitCode; restarting in 2 seconds"
    Start-Sleep -Seconds 2
}
