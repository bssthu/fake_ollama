param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8071
)

$ErrorActionPreference = "Stop"
$adapterRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = if ($env:MAGE_VL_PYTHON) {
    $env:MAGE_VL_PYTHON
} else {
    "J:\Projects\LLM_Models\Mage\.venv\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Mage-VL Python environment is missing: $pythonPath. Run scripts\install_mage_vl.ps1 first."
}

if (-not $env:MAGE_VL_MODEL_DIR) {
    $env:MAGE_VL_MODEL_DIR = "J:\Projects\LLM_Models\Mage\Mage-VL"
}
if (-not $env:MAGE_VL_FFMPEG) {
    $env:MAGE_VL_FFMPEG = "I:\Projects\Tools\ffmpeg\ffmpeg.exe"
}
if (-not $env:MAGE_VL_TEMP_DIR) {
    $env:MAGE_VL_TEMP_DIR = "I:\Projects\fake_ollama\.tmp\mage-vl-runtime"
}
$env:PYTHONUTF8 = "1"

& $pythonPath (Join-Path $adapterRoot "mage_vl_adapter\server.py") `
    --host $HostAddress --port $Port
exit $LASTEXITCODE
