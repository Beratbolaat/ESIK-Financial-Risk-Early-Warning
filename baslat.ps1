$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
$esikPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $esikPython)) {
    throw 'Önce kurulum.ps1 ile yerel Python ortamını hazırlayın.'
}
& $esikPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
