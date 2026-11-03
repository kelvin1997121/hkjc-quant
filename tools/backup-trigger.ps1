# Backup trigger for the cloud update, run by Windows Task Scheduler.
#
# GitHub's cron can be late or skip runs. Every 20 minutes during race hours this
# asks GitHub to run the "update" workflow -- but only if no update has started in
# the last 15 minutes, so once GitHub's own schedule works this does nothing.
# The analysis still runs in the cloud; this PC only sends the trigger.

$ErrorActionPreference = "Stop"
$repo = "kelvin1997121/hkjc-quant"
$gh = "C:\Program Files\GitHub CLI\gh.exe"
$env:Path += ";C:\Program Files\Git\cmd"
$log = Join-Path $PSScriptRoot "backup-trigger.log"   # gitignored

function Write-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg" | Add-Content -Path $log -Encoding utf8
    # keep the log small
    $lines = Get-Content $log -Encoding utf8
    if ($lines.Count -gt 1000) { $lines | Select-Object -Last 500 | Set-Content $log -Encoding utf8 }
}

try {
    # native commands don't throw in PowerShell 5.1, so check exit codes ourselves
    & $gh auth status *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "error: GitHub CLI is not logged in for this Windows user (run: gh auth login)"
        exit 1
    }
    $json = & $gh run list --repo $repo --workflow update.yml --limit 5 --json createdAt,status,event
    if ($LASTEXITCODE -ne 0) { Write-Log "error: could not list runs"; exit 1 }
    $runs = $json | ConvertFrom-Json
    $cutoff = [DateTime]::UtcNow.AddMinutes(-15)
    $busy = $runs | Where-Object { $_.status -in @("queued", "in_progress", "waiting", "pending") }
    $recent = $runs | Where-Object { [DateTimeOffset]::Parse($_.createdAt).UtcDateTime -gt $cutoff }
    if ($busy -or $recent) {
        $why = if ($busy) { "a run is in progress" } else { "a $($recent[0].event) run started recently" }
        Write-Log "skip: $why"
        exit 0
    }
    & $gh workflow run update.yml --repo $repo *> $null
    if ($LASTEXITCODE -ne 0) { Write-Log "error: could not trigger the workflow"; exit 1 }
    Write-Log "triggered update (no run in the last 15 min)"
} catch {
    Write-Log "error: $_"
    exit 1
}
