# Start the standalone Mage-VL service.
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8071,
    [string]$PythonPath = "",
    [string]$ModelDir = "",
    [string]$FfmpegPath = "",
    [string]$TempDir = ""
)

$ErrorActionPreference = "Stop"
$installRoot = if ($env:MAGE_VL_INSTALL_ROOT) {
    $env:MAGE_VL_INSTALL_ROOT
} else {
    Join-Path $env:LOCALAPPDATA "fake-ollama\mage-vl"
}
$PythonPath = if ($PythonPath) { $PythonPath } elseif ($env:MAGE_VL_PYTHON) {
    $env:MAGE_VL_PYTHON
} else { Join-Path $installRoot ".venv\Scripts\python.exe" }
$ModelDir = if ($ModelDir) { $ModelDir } elseif ($env:MAGE_VL_MODEL_DIR) {
    $env:MAGE_VL_MODEL_DIR
} else { Join-Path $installRoot "Mage-VL" }
$FfmpegPath = if ($FfmpegPath) { $FfmpegPath } elseif ($env:MAGE_VL_FFMPEG) {
    $env:MAGE_VL_FFMPEG
} else {
    $ffmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCommand) { $ffmpegCommand.Source } else { "ffmpeg" }
}
$TempDir = if ($TempDir) { $TempDir } elseif ($env:MAGE_VL_TEMP_DIR) {
    $env:MAGE_VL_TEMP_DIR
} else { Join-Path $installRoot "runtime" }

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Mage-VL Python environment is missing: $PythonPath. Run services\mage_vl\scripts\install.ps1 first."
}

$env:MAGE_VL_MODEL_DIR = $ModelDir
$env:MAGE_VL_FFMPEG = $FfmpegPath
$env:MAGE_VL_TEMP_DIR = $TempDir
$env:PYTHONUTF8 = "1"

& $PythonPath -m mage_vl_adapter --host $HostAddress --port $Port
exit $LASTEXITCODE
