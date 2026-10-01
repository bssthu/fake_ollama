param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& $Python bootstrap.py
if ($LASTEXITCODE -ne 0) { throw 'Pinned source/model download failed' }
& uv venv --python 3.12 .venv
if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
& uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
$lock = Get-Content model-lock.json -Raw | ConvertFrom-Json
$source = Join-Path $PSScriptRoot ('runtime/laya-' + $lock.source_revision)
& uv pip install --python .venv/Scripts/python.exe --no-deps $source
if ($LASTEXITCODE -ne 0) { throw 'Pinned Laya install failed' }
& .venv/Scripts/python.exe runtime.py start
