param(
    [string]$InstallRoot = "J:\Projects\LLM_Models\Mage",
    [string]$UvPath = "C:\Users\a\.local\bin\uv.exe",
    [string]$BasePythonPath = "C:\Python313\python.exe",
    [string]$ProxyUrl = "http://127.0.0.1:6268",
    [switch]$IncludeStreamingGate
)

$ErrorActionPreference = "Stop"
$modelPath = Join-Path $InstallRoot "Mage-VL"
$venvPath = Join-Path $InstallRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path (Split-Path -Parent $PSScriptRoot) "mage_vl_adapter\requirements.txt"

if (-not (Test-Path -LiteralPath $UvPath -PathType Leaf)) {
    throw "uv is missing: $UvPath"
}
if (-not (Test-Path -LiteralPath $BasePythonPath -PathType Leaf)) {
    throw "Base Python is missing: $BasePythonPath"
}
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

if (-not (Test-Path -LiteralPath $modelPath)) {
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    git clone https://huggingface.co/microsoft/Mage-VL $modelPath
    if ($LASTEXITCODE -ne 0) { throw "Mage-VL metadata clone failed." }
}
$venvHealthy = $false
if (
    (Test-Path -LiteralPath $pythonPath -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $venvPath "pyvenv.cfg") -PathType Leaf)
) {
    $venvConfig = Get-Content -LiteralPath (Join-Path $venvPath "pyvenv.cfg") -Raw
    $venvHealthy = $venvConfig -like "*$BasePythonPath*"
}
if (-not $venvHealthy) {
    & $UvPath venv --python $BasePythonPath --clear $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Mage-VL virtual environment creation failed." }
}

& $UvPath pip install --python $pythonPath `
    torch==2.12.1 torchvision==0.27.1 `
    --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }

& $UvPath pip install --python $pythonPath -r $requirementsPath
if ($LASTEXITCODE -ne 0) { throw "Mage-VL adapter dependency installation failed." }

$downloadScript = Join-Path $PSScriptRoot "download_mage_vl_weights.ps1"
& $downloadScript -ModelPath $modelPath -ProxyUrl $ProxyUrl -IncludeStreamingGate:$IncludeStreamingGate
if ($LASTEXITCODE -ne 0) { throw "Mage-VL weight download failed." }

& $pythonPath -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA validation failed." }

Write-Output "Mage-VL base installation is ready at $modelPath"
