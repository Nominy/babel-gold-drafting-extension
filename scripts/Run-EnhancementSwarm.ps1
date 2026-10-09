[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Extension,
    [Parameter(Mandatory = $true)][string]$Grader,
    [Parameter(Mandatory = $true)][string]$Profile,
    [Parameter(Mandatory = $true)][string]$Coordinator,
    [string]$Browser = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Worker = Join-Path $PSScriptRoot "run-enhancement-swarm-worker.mjs"
$Node = (Get-Command node.exe).Source
$LogRoot = Join-Path (Split-Path -Parent $Profile) ".logs"
New-Item -ItemType Directory -Force $LogRoot | Out-Null
try {
    $lock = [System.IO.File]::Open((Join-Path $LogRoot "swarm.lock"), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
} catch { exit 0 }
$arguments = @("`"$Worker`"", "--extension", "`"$Extension`"", "--grader", "`"$Grader`"", "--profile", "`"$Profile`"", "--coordinator", "`"$Coordinator`"")
if ($Browser) { $arguments += @("--browser", "`"$Browser`"") }
$backoff = 5
try {
    while ($true) {
        $started = Get-Date
        try {
            $child = Start-Process -FilePath $Node -ArgumentList $arguments -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $LogRoot "worker.out.log") -RedirectStandardError (Join-Path $LogRoot "worker.err.log")
            $child.WaitForExit()
            $detail = "exit=$($child.ExitCode)"
        } catch { $detail = $_.Exception.Message }
        Add-Content -LiteralPath (Join-Path $LogRoot "supervisor.log") -Value ("{0:o} {1}" -f (Get-Date), $detail)
        if (((Get-Date) - $started).TotalSeconds -ge 60) { $backoff = 5 } else { $backoff = [Math]::Min(60, $backoff * 2) }
        Start-Sleep -Seconds $backoff
    }
} finally { $lock.Dispose() }
