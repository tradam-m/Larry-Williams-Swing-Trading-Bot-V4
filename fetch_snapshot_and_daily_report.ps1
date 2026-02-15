param(
    [string]$Owner = "winningtrendingbots",
    [string]$Repo = "Larry-Williams-Swing-Trading-Bot-V4",
    [string]$Date = "",
    [switch]$NoCleanup
)

$ErrorActionPreference = "Stop"

function Fail($Message) {
    Write-Host "❌ $Message" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Fail "GitHub CLI non trovato. Installa gh e fai login con: gh auth login"
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "Python non trovato nel PATH. Attiva la venv e riprova."
}

$root = Get-Location
$snapshotsDir = Join-Path $root "snapshots"
$tempDir = Join-Path $root "artifact_latest"
$zipPath = Join-Path $root "artifact_latest.zip"

New-Item -ItemType Directory -Path $snapshotsDir -Force | Out-Null

$ts = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$snapshotPath = Join-Path $snapshotsDir "trades_v4_$ts.json"

Write-Host "🔎 Cerco ultimo artifact trading-logs-*..."
$artifactId = gh api "repos/$Owner/$Repo/actions/artifacts" --jq '.artifacts | map(select((.name|startswith("trading-logs-")) and (.expired==false))) | sort_by(.created_at) | last | .id'

if ([string]::IsNullOrWhiteSpace($artifactId) -or $artifactId -eq "null") {
    Fail "Nessun artifact valido trovato in GitHub Actions."
}

Write-Host "⬇️ Scarico artifact ID: $artifactId"
gh api "repos/$Owner/$Repo/actions/artifacts/$artifactId/zip" --output $zipPath

if (-not (Test-Path $zipPath)) {
    Fail "Download artifact fallito."
}

if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}

Expand-Archive $zipPath -DestinationPath $tempDir -Force

$sourceTrades = Join-Path $tempDir "trades_v4.json"
if (-not (Test-Path $sourceTrades)) {
    Fail "trades_v4.json non trovato nell'artifact scaricato."
}

Copy-Item $sourceTrades $snapshotPath -Force
Write-Host "💾 Snapshot creato: $snapshotPath"

if ([string]::IsNullOrWhiteSpace($Date)) {
    Write-Host "📊 Genero report giornaliero automatico (oggi)..."
    python report_v4.py --input $snapshotPath --daily --save
}
else {
    Write-Host "📊 Genero report giornaliero per data: $Date"
    python report_v4.py --input $snapshotPath --daily --date $Date --save
}

if (-not $NoCleanup) {
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }
    Write-Host "🧹 Pulizia completata"
}
else {
    Write-Host "ℹ️ Pulizia saltata (--NoCleanup)"
}

Write-Host "✅ Operazione completata"
