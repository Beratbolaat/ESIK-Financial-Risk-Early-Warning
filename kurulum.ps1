$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$esikPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $esikPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Sanal ortam oluşturulamadı.' }
}
& $esikPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Bağımlılıklar kurulamadı; hata mesajını kontrol edin.' }
Write-Host 'Kurulum tamamlandı. baslat.ps1 ile uygulamayı açabilirsiniz.'
