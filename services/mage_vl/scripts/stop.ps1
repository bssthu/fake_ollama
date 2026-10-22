# Gracefully stop the standalone Mage-VL service.
param(
    [string]$BaseUrl = "http://127.0.0.1:8071",
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"

function Invoke-MageControlRequest {
    param([uri]$Uri, [string]$Method = "GET", [string]$Token = "", [int]$TimeoutSeconds = 2)
    # Lifecycle credentials never go to a system proxy or a redirect target.
    $request = [System.Net.HttpWebRequest]::Create($Uri)
    $request.Proxy = $null
    $request.AllowAutoRedirect = $false
    $request.Method = $Method
    $request.Timeout = $TimeoutSeconds * 1000
    if ($Token) { $request.Headers["Authorization"] = "Bearer $Token" }
    if ($Method -eq "POST") { $request.ContentLength = 0 }
    $response = $null
    try {
        try { $response = $request.GetResponse() }
        catch [System.Net.WebException] {
            if (-not $_.Exception.Response) { throw }
            $response = $_.Exception.Response
        }
        return [int]$response.StatusCode
    } finally {
        if ($response) { $response.Dispose() }
    }
}

function Test-MageRunning {
    param([uri]$Uri)
    try {
        Invoke-MageControlRequest -Uri $Uri | Out-Null
        return $true # HTTP 503 still means the process is running.
    } catch [System.Net.WebException] {
        # A timeout is not evidence of a completed shutdown.
        return $_.Exception.Status -ne [System.Net.WebExceptionStatus]::ConnectFailure
    }
}

function Stop-MageService {
    param([string]$BaseUrl, [int]$TimeoutSeconds)
    $endpoint = [uri]$BaseUrl
    if (-not $endpoint.IsAbsoluteUri -or $endpoint.Scheme -ne "http" -or
        -not $endpoint.IsLoopback -or $endpoint.UserInfo -or $endpoint.Query -or
        $endpoint.Fragment -or $endpoint.AbsolutePath -ne "/" -or $endpoint.Port -le 0) {
        throw "Mage-VL shutdown requires a local HTTP base URL."
    }
    $healthUrl = [uri]::new($endpoint, "/health")
    $shutdownUrl = [uri]::new($endpoint, "/shutdown")
    $localRoot = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else {
        [Environment]::GetFolderPath("LocalApplicationData")
    }
    $tokenPath = Join-Path $localRoot "fake-ollama\mage-vl\control\shutdown-$($endpoint.Port).token"
    $token = ""
    try {
        if (Test-Path -LiteralPath $tokenPath -PathType Leaf) {
            $token = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
            if ($token -cnotmatch '^[A-Za-z0-9_-]{43}$') {
                throw "Mage-VL shutdown credential is invalid."
            }
        }
        # An already-running pre-upgrade instance has no token. Only that old
        # server can accept this; upgraded servers always require credentials.
        $status = Invoke-MageControlRequest -Uri $shutdownUrl -Method POST -Token $token -TimeoutSeconds 10
        if ($status -lt 200 -or $status -ge 300) {
            throw "Mage-VL shutdown returned HTTP $status."
        }
    } catch {
        if (-not (Test-MageRunning -Uri $healthUrl)) {
            Write-Output "Mage-VL adapter is already stopped."
            return
        }
        throw "Could not request graceful Mage-VL shutdown: $($_.Exception.Message)"
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (-not (Test-MageRunning -Uri $healthUrl)) {
            Write-Output "Mage-VL adapter stopped gracefully."
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Mage-VL adapter did not stop within $TimeoutSeconds seconds. No forced termination was attempted."
}

Stop-MageService -BaseUrl $BaseUrl -TimeoutSeconds $TimeoutSeconds
