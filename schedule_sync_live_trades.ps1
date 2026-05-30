param(
    [string]$TaskName = "SyncLiveTrades",
    [int]$IntervalMinutes = 15,
    [switch]$WithReport,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$syncScript = Join-Path $scriptRoot "sync_live_trades.ps1"

function Fail($Message) {
    Write-Host "ERRORE: $Message" -ForegroundColor Red
    exit 1
}

if ($Remove) {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Task '$TaskName' rimosso."
    } else {
        Write-Host "Nessun task da rimuovere ('$TaskName' non esiste)."
    }
    return
}

if (-not (Test-Path $syncScript)) {
    Fail "sync_live_trades.ps1 non trovato in: $scriptRoot"
}

if ($IntervalMinutes -lt 5) {
    Fail "Intervallo minimo 5 minuti."
}

# Verifica gh autenticato
$ghStatus = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Yellow
    Write-Host "  gh CLI non autenticato!" -ForegroundColor Yellow
    Write-Host "  Esegui prima: gh auth login" -ForegroundColor Yellow
    Write-Host "  Poi rilancia questo script." -ForegroundColor Yellow
    Write-Host "========================================================" -ForegroundColor Yellow
    exit 1
}

$reportFlag = if ($WithReport) { " -RunReport" } else { "" }
$actionArgs = "-ExecutionPolicy Bypass -NonInteractive -File `"$syncScript`"$reportFlag"

$action    = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $actionArgs -WorkingDirectory $scriptRoot
$startTime = (Get-Date).AddMinutes(1)
$trigger   = New-ScheduledTaskTrigger -Once -At $startTime `
               -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
               -RepetitionDuration (New-TimeSpan -Days 3650)
$settings  = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
               -MultipleInstances IgnoreNew -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "Scarica trades_v4_live.json da GitHub Actions ogni $IntervalMinutes min" `
    -Force | Out-Null

Write-Host ""
Write-Host "Task '$TaskName' registrato: ogni $IntervalMinutes minuti (avvio alle $($startTime.ToShortTimeString()))" -ForegroundColor Green
Write-Host "Script: $syncScript"
Write-Host ""

# Esegui subito il primo sync
Write-Host "Eseguo primo sync adesso..."
& $syncScript $(if ($WithReport) { @("-RunReport") } else { @() })
