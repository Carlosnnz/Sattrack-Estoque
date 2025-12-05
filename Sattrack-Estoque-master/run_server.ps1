param(
    [string]$BindHost = $env:UVICORN_HOST,
    [int]$Port = $env:UVICORN_PORT,
    [int]$Workers = $env:UVICORN_WORKERS
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
if (!(Test-Path $venvPath)) {
    Write-Host "Criando ambiente virtual em $venvPath"
    python -m venv $venvPath
}

$activate = Join-Path $venvPath "Scripts\Activate.ps1"
. $activate

if (-not (Get-Command uvicorn -ErrorAction SilentlyContinue)) {
    Write-Host "Instalando dependencias..."
    pip install -r requirements.txt
}

if (-not $BindHost) { $BindHost = "0.0.0.0" }
if (-not $Port) { $Port = 8000 }
if (-not $Workers) { $Workers = 1 }

Write-Host "Iniciando uvicorn em ${BindHost}:${Port} (workers=${Workers})..."
$uvicornArgs = @(
    "app.main:app",
    "--host", $BindHost,
    "--port", $Port,
    "--workers", $Workers,
    "--proxy-headers",
    "--forwarded-allow-ips=*",
    "--log-level", "info"
)
& uvicorn @uvicornArgs
