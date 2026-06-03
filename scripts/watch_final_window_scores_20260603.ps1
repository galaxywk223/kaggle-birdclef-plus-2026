param(
    [string]$Competition = "birdclef-2026",
    [string]$KaggleExe = "kaggle",
    [datetimeoffset]$FastPollUtc = ([datetimeoffset]"2026-06-03T22:30:00Z"),
    [datetimeoffset]$DeadlineUtc = ([datetimeoffset]"2026-06-04T00:10:00Z"),
    [int]$SlowPollSeconds = 300,
    [int]$FastPollSeconds = 60
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogDir = Join-Path $ProjectRoot "logs"
$ArtifactDir = Join-Path $ProjectRoot "artifacts"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

$LogPath = Join-Path $LogDir ("final_window_score_monitor_20260603_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$SnapshotPath = Join-Path $ArtifactDir "final_window_submissions_latest_20260603.csv"
$StatePath = Join-Path $LogDir "final_window_score_monitor_state_20260603.json"

function Write-MonitorLog {
    param([string]$Text)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"), $Text
    Add-Content -Path $LogPath -Value $line -Encoding utf8
    Write-Host $line
}

function Convert-PublicScore {
    param([string]$Score)
    if ([string]::IsNullOrWhiteSpace($Score)) {
        return $null
    }
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $value = 0.0
    if ([double]::TryParse($Score, [System.Globalization.NumberStyles]::Float, $culture, [ref]$value)) {
        return $value
    }
    return $null
}

function Get-Submissions {
    $output = & $KaggleExe competitions submissions -c $Competition --csv 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        foreach ($line in $output) {
            Write-MonitorLog ("kaggle-error> {0}" -f $line)
        }
        return $null
    }

    $lines = @($output | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    if ($lines.Count -eq 0) {
        Write-MonitorLog "empty submissions output"
        return $null
    }

    $lines | Set-Content -Path $SnapshotPath -Encoding utf8
    try {
        return @($lines | ConvertFrom-Csv)
    }
    catch {
        Write-MonitorLog ("csv parse failed: {0}" -f $_.Exception.Message)
        return $null
    }
}

function Save-MonitorState {
    param(
        [string]$BestScore,
        [string]$BestDescription,
        [int]$FinalWindowRows,
        [int]$PendingRows
    )
    $payload = [ordered]@{
        updated_at_utc = ([datetimeoffset]::UtcNow.ToString("o"))
        best_public_score = $BestScore
        best_description = $BestDescription
        final_window_rows = $FinalWindowRows
        pending_rows = $PendingRows
    }
    $payload | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $StatePath -Encoding utf8
}

$MutexName = "Global\BirdCLEF2026FinalWindowScoreMonitor"
$Mutex = New-Object System.Threading.Mutex($false, $MutexName)
$HasMutex = $false

try {
    $HasMutex = $Mutex.WaitOne(0)
    if (-not $HasMutex) {
        Write-MonitorLog ("another score monitor already holds mutex {0}; exiting" -f $MutexName)
        exit 0
    }

    Write-MonitorLog ("score monitor started; fast_poll_utc={0}; deadline_utc={1}; slow_poll_seconds={2}; fast_poll_seconds={3}" -f $FastPollUtc.ToString("o"), $DeadlineUtc.ToString("o"), $SlowPollSeconds, $FastPollSeconds)

    while ([datetimeoffset]::UtcNow -lt $DeadlineUtc) {
        $rows = Get-Submissions
        if ($null -ne $rows) {
            $complete = @($rows | Where-Object { $_.status -match "COMPLETE" })
            $pending = @($rows | Where-Object { $_.status -match "PENDING|RUNNING|QUEUED" })
            $finalRows = @($rows | Where-Object { $_.description -like "BirdCLEF 2026 final window*" })

            $best = $null
            foreach ($row in $complete) {
                $score = Convert-PublicScore -Score $row.publicScore
                if ($null -eq $score) {
                    continue
                }
                if (($null -eq $best) -or ($score -gt $best.Score)) {
                    $best = [pscustomobject]@{
                        Score = $score
                        ScoreText = $row.publicScore
                        Description = $row.description
                        Date = $row.date
                    }
                }
            }

            $bestScoreText = ""
            $bestDescription = ""
            if ($null -ne $best) {
                $bestScoreText = $best.ScoreText
                $bestDescription = $best.Description
            }

            Save-MonitorState -BestScore $bestScoreText -BestDescription $bestDescription -FinalWindowRows $finalRows.Count -PendingRows $pending.Count
            Write-MonitorLog ("submissions: complete={0}; pending={1}; final_window_rows={2}; best_public={3}; best_desc={4}" -f $complete.Count, $pending.Count, $finalRows.Count, $bestScoreText, $bestDescription)
        }

        $sleepSeconds = $FastPollSeconds
        if ([datetimeoffset]::UtcNow -lt $FastPollUtc) {
            $sleepSeconds = $SlowPollSeconds
        }
        Start-Sleep -Seconds $sleepSeconds
    }

    Write-MonitorLog "score monitor stopped after deadline"
}
finally {
    if ($HasMutex) {
        [void]$Mutex.ReleaseMutex()
    }
    $Mutex.Dispose()
}
