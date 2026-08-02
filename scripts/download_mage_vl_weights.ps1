param(
    [string]$ModelPath = "J:\Projects\LLM_Models\Mage\Mage-VL",
    [string]$ProxyUrl = "http://127.0.0.1:6268",
    [switch]$IncludeStreamingGate
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
    throw "Mage-VL metadata directory is missing: $ModelPath"
}
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe is not available."
}

$specs = @(
    [pscustomobject]@{
        Name = "model-00001-of-00002.safetensors"
        Size = [int64]4967403560
        Sha256 = "98fa203652a843650343732d2597f08c9b491cf4e310cd269eed34ca27ebce58"
    },
    [pscustomobject]@{
        Name = "model-00002-of-00002.safetensors"
        Size = [int64]4516272328
        Sha256 = "719360069004f6b7b59303bd21bc1111bcd353bb0ccdebe1fe3971b87cd7b30f"
    }
)
if ($IncludeStreamingGate) {
    $specs += [pscustomobject]@{
        Name = "streammind_gate.safetensors"
        Size = [int64]1073494728
        Sha256 = "01938c515679c1130cff2e6a2af2e4cbc3aad10ea7ccb29229e64c2cfdbf6535"
    }
}

$hubDownloadDir = Join-Path $ModelPath ".cache\huggingface\download"
foreach ($spec in $specs) {
    $finalPath = Join-Path $ModelPath $spec.Name
    $partPath = "$finalPath.part"
    if (Test-Path -LiteralPath $finalPath -PathType Leaf) {
        $existingSize = (Get-Item -LiteralPath $finalPath).Length
        if ($existingSize -eq $spec.Size) {
            Write-Output "$($spec.Name) already has the expected size; keeping it."
            continue
        }
        throw "Existing final file has the wrong size and will not be overwritten: $finalPath ($existingSize bytes)"
    }

    if (-not (Test-Path -LiteralPath $partPath -PathType Leaf)) {
        $candidate = Get-ChildItem -File -Force -LiteralPath $hubDownloadDir -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "*$($spec.Sha256)*.incomplete" -and $_.Length -gt 0 } |
            Sort-Object Length -Descending |
            Select-Object -First 1
        if ($candidate) {
            Copy-Item -LiteralPath $candidate.FullName -Destination $partPath
            Write-Output "Reused $($candidate.Length) downloaded bytes for $($spec.Name)."
        }
    }

    $url = "https://huggingface.co/microsoft/Mage-VL/resolve/main/$($spec.Name)?download=true"
    $curlArgs = @(
        "--location",
        "--fail",
        "--show-error",
        "--progress-bar",
        "--proxy", $ProxyUrl,
        "--connect-timeout", "30",
        "--speed-time", "60",
        "--speed-limit", "1024",
        "--retry", "50",
        "--retry-all-errors",
        "--retry-delay", "3",
        "--continue-at", "-",
        "--output", $partPath,
        $url
    )
    Write-Output "Downloading $($spec.Name) through $ProxyUrl ..."
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $($spec.Name) with exit code $LASTEXITCODE"
    }

    $actualSize = (Get-Item -LiteralPath $partPath).Length
    if ($actualSize -ne $spec.Size) {
        throw "Downloaded size mismatch for $($spec.Name): expected $($spec.Size), got $actualSize"
    }
    Write-Output "Verifying SHA-256 for $($spec.Name) ..."
    $actualHash = (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $spec.Sha256) {
        throw "SHA-256 mismatch for $($spec.Name): expected $($spec.Sha256), got $actualHash"
    }
    Move-Item -LiteralPath $partPath -Destination $finalPath
    Write-Output "$($spec.Name) verified and installed."
}
