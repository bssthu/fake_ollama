param(
    [string]$BaseUrl = "http://127.0.0.1:8071",
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$healthUrl = "$($BaseUrl.TrimEnd('/'))/health"
$shutdownUrl = "$($BaseUrl.TrimEnd('/'))/shutdown"

try {
    Invoke-RestMethod -Method Post -Uri $shutdownUrl -TimeoutSec 10 | Out-Null
} catch {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 2 | Out-Null
    } catch {
        Write-Output "Mage-VL adapter is already stopped."
        exit 0
    }
    throw "Could not request graceful Mage-VL shutdown: $($_.Exception.Message)"
}

$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 2 | Out-Null
    } catch {
        Write-Output "Mage-VL adapter stopped gracefully."
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

throw "Mage-VL adapter did not stop within $TimeoutSeconds seconds. No forced termination was attempted."
