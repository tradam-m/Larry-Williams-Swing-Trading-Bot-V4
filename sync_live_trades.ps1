param(
    [string]$Source = "C:\actions-runner\kraken_work\Larry-Williams-Swing-Trading-Bot-V4\Larry-Williams-Swing-Trading-Bot-V4\trades_v4.json",
    [string]$Destination = "C:\Trading\Larry-Williams-Swing-Trading-Bot-V4\trades_v4_live.json",
    [switch]$RunReport,
    [string]$ReportArgs = "--input trades_v4_live.json"
)

$ErrorActionPreference = "Stop"

function Fail($Message) {
    Write-Host "❌ $Message" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Source)) {
    Fail "File sorgente non trovato: $Source"
}

Copy-Item $Source $Destination -Force
Write-Host "💾 Copiato $Source -> $Destination"

if ($RunReport) {
    $python = "C:/Trading/Larry-Williams-Swing-Trading-Bot-V4/.venv/.venv/Scripts/activate/Scripts/python.exe"
    if (-not (Test-Path $python)) {
        Fail "Python non trovato in $python"
    }

    Write-Host "📊 Eseguo report: python report_v4.py $ReportArgs"
    & $python report_v4.py $ReportArgs.Split(' ')
}
