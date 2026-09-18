param([string]$Python = '', [int]$AppPort = 8502)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
if (-not $Python) {
    $esikVenvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $esikVenvPython) { $Python = $esikVenvPython }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
& $Python -c 'import streamlit, pandas, joblib, sklearn, xgboost'
if ($LASTEXITCODE -ne 0) { throw 'Python bağımlılıkları eksik. Önce kurulum.ps1 çalıştırın veya -Python ile hazır ortamınızı belirtin.' }
& $Python scripts/prepare_connections.py
if ($LASTEXITCODE -ne 0) { throw 'Yerel bağlantı ayarları hazırlanamadı.' }
$esikQueryProcess = $null
try {
    $esikServiceReady = $false
    try {
        $esikHealth = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/health' -TimeoutSec 2
        $esikServiceReady = $esikHealth.service -eq 'esik-company-query'
    } catch {}
    if ($esikServiceReady) {
        & $Python scripts/check_local_query.py
        if ($LASTEXITCODE -ne 0) { throw '8765 portunda başka proje kopyasına ait bir sorgu servisi var. Önce o servisi kapatıp bu klasörden yeniden başlatın.' }
    }
    if (-not $esikServiceReady) {
        $esikServerPath = Join-Path $PSScriptRoot 'esik_query_server.py'
        $esikQueryProcess = Start-Process -FilePath $Python -ArgumentList ('"' + $esikServerPath + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
        $esikQueryReady = $false
        for ($esikAttempt = 0; $esikAttempt -lt 15; $esikAttempt++) {
            if ($esikQueryProcess.HasExited) { throw 'Şirket sorgu servisi açılamadı. 8765 portunu ve yerel ayarları kontrol edin.' }
            & $Python scripts/check_local_query.py
            if ($LASTEXITCODE -eq 0) { $esikQueryReady = $true; break }
            Start-Sleep -Seconds 1
        }
        if (-not $esikQueryReady) { throw 'Şirket sorgu servisi zamanında hazır olmadı.' }
        Write-Host 'Yerel şirket sorgusu hazır. n8n kanal akışları bu servisi kullanır.'
    }
    & $Python -m streamlit run app.py --server.address 127.0.0.1 --server.port $AppPort
} finally {
    if ($null -ne $esikQueryProcess -and -not $esikQueryProcess.HasExited) {
        Stop-Process -Id $esikQueryProcess.Id
    }
}
