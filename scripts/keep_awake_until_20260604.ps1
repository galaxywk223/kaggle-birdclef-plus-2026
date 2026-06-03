param(
    [datetimeoffset]$UntilUtc = ([datetimeoffset]"2026-06-04T00:10:00Z"),
    [int]$HeartbeatSeconds = 60
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "keep_awake_until_20260604.log"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class SleepUtil {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$ES_AWAYMODE_REQUIRED = [uint32]0x00000040

function Write-KeepAwakeLog {
    param([string]$Text)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"), $Text
    Add-Content -Path $LogPath -Value $line -Encoding utf8
    Write-Host $line
}

try {
    Write-KeepAwakeLog ("keep-awake started; until_utc={0}; heartbeat_seconds={1}" -f $UntilUtc.ToString("o"), $HeartbeatSeconds)

    while ([datetimeoffset]::UtcNow -lt $UntilUtc) {
        [void][SleepUtil]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED)
        Write-KeepAwakeLog ("keep-awake heartbeat; utc_now={0}" -f ([datetimeoffset]::UtcNow.ToString("o")))
        Start-Sleep -Seconds $HeartbeatSeconds
    }
}
finally {
    [void][SleepUtil]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-KeepAwakeLog "keep-awake stopped"
}
