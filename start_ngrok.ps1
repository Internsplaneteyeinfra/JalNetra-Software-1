# Start JalNetra API + ngrok tunnel (port 8010 by default)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

# Load .env (simple KEY=VALUE lines; skip EE JSON block handled by python dotenv)
if (Test-Path ".env") {
    Get-Content ".env" -Raw | ForEach-Object { $_ } | Out-Null
    python -c "
from dotenv import load_dotenv
import os
load_dotenv()
token = os.getenv('NGROK_AUTHTOKEN', '')
port = os.getenv('PORT', '8010')
print(f'NGROK_TOKEN_SET={1 if token else 0}')
print(f'PORT={port}')
" | ForEach-Object {
        if ($_ -match '^NGROK_TOKEN_SET=(\d+)$') { $script:HasToken = [int]$Matches[1] -eq 1 }
        if ($_ -match '^PORT=(\d+)$') { $script:Port = [int]$Matches[1] }
    }
}
if (-not $script:Port) { $script:Port = 8010 }

$ngrok = Get-Command ngrok -ErrorAction SilentlyContinue
if (-not $ngrok) {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
        [System.Environment]::GetEnvironmentVariable("Path", "User")
    $ngrok = Get-Command ngrok -ErrorAction SilentlyContinue
}
if (-not $ngrok) {
    Write-Host "[error] ngrok not found. Install: winget install Ngrok.Ngrok"
    exit 1
}

# ngrok 3.20+ required for current accounts
$verLine = (& ngrok version 2>&1 | Select-Object -First 1) -replace '[^0-9.]', ''
$verParts = $verLine.Split('.')
if ($verParts.Count -ge 2) {
    $major = [int]$verParts[0]
    $minor = [int]$verParts[1]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 20)) {
        Write-Host "Updating ngrok (3.20+ required)..."
        & ngrok update | Out-Null
    }
}

if (-not $script:HasToken) {
    Write-Host "[error] NGROK_AUTHTOKEN missing in .env"
    Write-Host ""
    Write-Host "1. Sign up (free): https://dashboard.ngrok.com/signup"
    Write-Host "2. Copy token:    https://dashboard.ngrok.com/get-started/your-authtoken"
    Write-Host "3. Add to .env:   NGROK_AUTHTOKEN=your_token_here"
    Write-Host "4. Run again:     .\start_ngrok.ps1"
    exit 1
}

$token = (python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('NGROK_AUTHTOKEN',''))").Trim()
& ngrok config add-authtoken $token | Out-Null

function Test-Api {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$($script:Port)/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch { return $false }
}

if (-not (Test-Api)) {
    Write-Host "Starting JalNetra API on port $($script:Port)..."
    Start-Process -FilePath "python" -ArgumentList "api_main.py" -WorkingDirectory $Root -WindowStyle Normal
    $deadline = (Get-Date).AddSeconds(45)
    while ((Get-Date) -lt $deadline) {
        if (Test-Api) { break }
        Start-Sleep -Seconds 2
    }
    if (-not (Test-Api)) {
        Write-Host "[error] API did not start on port $($script:Port). Run: python api_main.py"
        exit 1
    }
}

Write-Host "Starting ngrok tunnel -> http://127.0.0.1:$($script:Port)"
Start-Process -FilePath $ngrok.Source -ArgumentList "http $($script:Port)" -WorkingDirectory $Root

$deadline = (Get-Date).AddSeconds(20)
$publicUrl = $null
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 3
        $publicUrl = ($resp.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1).public_url
        if ($publicUrl) { break }
    } catch { }
    Start-Sleep -Seconds 1
}

Write-Host ""
if ($publicUrl) {
    Write-Host "=========================================="
    Write-Host "PUBLIC API URL:  $publicUrl"
    Write-Host "PUBLIC DOCS:     $publicUrl/docs"
    Write-Host "NGROK INSPECTOR: http://127.0.0.1:4040"
    Write-Host "=========================================="
} else {
    Write-Host "ngrok started - open http://127.0.0.1:4040 for your public URL"
}
