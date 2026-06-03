param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [int]$PollSeconds = 600,
    [string]$SubmitAfterUtc = "2026-05-23T00:10:00Z",
    [int]$StopAfterHours = 16,
    [double]$BestToBeat = 0.949,
    [switch]$NoSubmit
)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Competition = "birdclef-2026"
$KaggleExe = "kaggle"
$ScoreLog = Join-Path $ProjectRoot "logs\score_push_2026-05-09.md"
$WatchLog = Join-Path $ProjectRoot ("logs\watch_birdclef_reset_queue_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$StateDir = Join-Path $ProjectRoot "logs\watch_state"
$ImprovedFlag = Join-Path $StateDir "birdclef_best_gt_0949.flag"

$Queue = @(
    [pscustomobject]@{
        Id = "cheny_exp071_perch_birdnet_unmapped"
        Kernel = "galaxy2025/birdclef-2026-cheny-exp071-perch-birdnet-unmapped"
        Version = "1"
        Message = "BirdCLEF 2026 Cheny Exp071 Perch BirdNET Unmapped run v1"
    },
    [pscustomobject]@{
        Id = "cheny_exp072_perch_blindspot"
        Kernel = "galaxy2025/birdclef-2026-cheny-exp072-perch-blindspot"
        Version = "1"
        Message = "BirdCLEF 2026 Cheny Exp072 Perch Blindspot run v1"
    },
    [pscustomobject]@{
        Id = "cheny_exp080_karnak_dual_arch"
        Kernel = "galaxy2025/birdclef-2026-cheny-exp080-karnak-dual-arch"
        Version = "1"
        Message = "BirdCLEF 2026 Cheny Exp080 Karnak Dual Arch run v1"
    }
)

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $WatchLog) | Out-Null

function Write-WatchLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"), $Message
    Add-Content -Path $WatchLog -Value $line -Encoding utf8
}

function Add-ScoreLog {
    param([string]$Markdown)
    Add-Content -Path $ScoreLog -Value ("`n" + $Markdown) -Encoding utf8
}

function Invoke-Logged {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    Write-WatchLog ("RUN: {0} {1}" -f $FilePath, ($Arguments -join " "))
    $output = & $FilePath @Arguments 2>&1
    $code = $LASTEXITCODE
    foreach ($line in $output) {
        Write-WatchLog ("OUT: {0}" -f $line)
    }
    Write-WatchLog ("EXIT: {0}" -f $code)
    return [pscustomobject]@{
        Code = $code
        Output = ($output -join "`n")
    }
}

function Get-SubmissionRows {
    $result = Invoke-Logged -FilePath $KaggleExe -Arguments @("competitions", "submissions", "-c", $Competition, "--csv")
    if ($result.Code -ne 0 -or [string]::IsNullOrWhiteSpace($result.Output)) {
        return @()
    }
    try {
        return @($result.Output | ConvertFrom-Csv)
    }
    catch {
        Write-WatchLog ("submission csv parse failed: {0}" -f $_.Exception.Message)
        return @()
    }
}

function Get-QueueItemFlagPath {
    param(
        [object]$Item,
        [string]$Suffix
    )
    return (Join-Path $StateDir ("{0}_{1}.flag" -f $Item.Id, $Suffix))
}

function Find-SubmissionForMessage {
    param(
        [array]$Rows,
        [string]$Message
    )
    $matches = @($Rows | Where-Object { $_.description -eq $Message })
    if ($matches.Count -eq 0) {
        return $null
    }
    return ($matches | Sort-Object date -Descending | Select-Object -First 1)
}

function Convert-PublicScore {
    param([string]$Score)
    if ([string]::IsNullOrWhiteSpace($Score)) {
        return $null
    }
    $value = 0.0
    if ([double]::TryParse(
            $Score,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$value
        )) {
        return $value
    }
    return $null
}

function Test-ImprovedSubmission {
    param(
        [object]$Item,
        [object]$Submission
    )
    if ($null -eq $Submission) {
        return $false
    }
    if (-not ($Submission.status -match "COMPLETE")) {
        return $false
    }
    $score = Convert-PublicScore -Score $Submission.publicScore
    if ($null -eq $score) {
        return $false
    }
    if ($score -gt $BestToBeat) {
        Add-ScoreLog ("## Watcher Stop: {0}`n`nThe queued submission `{1}` reached public score `{2}` at `{3}`, which is strictly greater than the configured target `{4}`. The reset watcher stopped before submitting additional queued kernels." -f $Item.Id, $Item.Message, $Submission.publicScore, $Submission.date, $BestToBeat.ToString([System.Globalization.CultureInfo]::InvariantCulture))
        New-Item -ItemType File -Force -Path $ImprovedFlag | Out-Null
        return $true
    }
    return $false
}

function Try-SubmitNext {
    if ($NoSubmit) {
        return
    }
    if (Test-Path $ImprovedFlag) {
        Write-WatchLog "improved-score flag exists; no further submissions will be attempted"
        return
    }
    $submitAfter = [datetimeoffset]::Parse($SubmitAfterUtc)
    if ([datetimeoffset]::UtcNow -lt $submitAfter) {
        Write-WatchLog ("submit not due until {0}" -f $SubmitAfterUtc)
        return
    }

    $rows = Get-SubmissionRows
    foreach ($item in $Queue) {
        $attemptFlag = Get-QueueItemFlagPath -Item $item -Suffix "submit_attempted"
        $scoredFlag = Get-QueueItemFlagPath -Item $item -Suffix "scored"
        $existing = Find-SubmissionForMessage -Rows $rows -Message $item.Message

        if ($null -ne $existing) {
            if (Test-ImprovedSubmission -Item $item -Submission $existing) {
                return
            }
            if (($existing.status -match "COMPLETE") -and -not [string]::IsNullOrWhiteSpace($existing.publicScore)) {
                if (!(Test-Path $scoredFlag)) {
                    Add-ScoreLog ("## Watcher Score Update: {0}`n`nThe queued submission reached `{1}` with public score `{2}` at `{3}`." -f $item.Id, $existing.status, $existing.publicScore, $existing.date)
                    New-Item -ItemType File -Force -Path $scoredFlag | Out-Null
                }
                continue
            }

            Write-WatchLog ("submission already exists and is not scored yet: {0} {1}" -f $item.Message, $existing.status)
            return
        }

        if (Test-Path $attemptFlag) {
            Write-WatchLog ("attempt flag exists but no submission row found for {0}; waiting before any retry" -f $item.Id)
            return
        }

        $result = Invoke-Logged -FilePath $KaggleExe -Arguments @(
            "competitions", "submit", $Competition,
            "-k", $item.Kernel,
            "-v", $item.Version,
            "-f", "submission.csv",
            "-m", $item.Message
        )
        if ($result.Code -eq 0) {
            New-Item -ItemType File -Force -Path $attemptFlag | Out-Null
            Add-ScoreLog ("## Watcher Submission Attempt: {0}`n`nThe queued kernel `{1}` version `{2}` was submitted after the configured UTC reset window. The submission table should be polled until a scored `COMPLETE` row appears." -f $item.Id, $item.Kernel, $item.Version)
        }
        else {
            Write-WatchLog ("submit failed for {0}; retry remains enabled" -f $item.Id)
        }
        return
    }
}

function Poll-QueuedScores {
    $rows = Get-SubmissionRows
    foreach ($item in $Queue) {
        $scoredFlag = Get-QueueItemFlagPath -Item $item -Suffix "scored"
        if (Test-Path $scoredFlag) {
            continue
        }
        $existing = Find-SubmissionForMessage -Rows $rows -Message $item.Message
        if ($null -eq $existing) {
            continue
        }
        if (Test-ImprovedSubmission -Item $item -Submission $existing) {
            return
        }
        if (($existing.status -match "COMPLETE") -and -not [string]::IsNullOrWhiteSpace($existing.publicScore)) {
            Add-ScoreLog ("## Watcher Score Update: {0}`n`nThe queued submission reached `{1}` with public score `{2}` at `{3}`." -f $item.Id, $existing.status, $existing.publicScore, $existing.date)
            New-Item -ItemType File -Force -Path $scoredFlag | Out-Null
        }
    }
}

Set-Location $ProjectRoot
Write-WatchLog "watcher started"
if ($NoSubmit) {
    Add-ScoreLog ("## Watcher Dry Run - Reset Queue`n`nA no-submit watcher dry run started at `{0}`. The dry run checks Kaggle submission parsing and exits without submitting queued kernels." -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"))
}
else {
    Add-ScoreLog ("## Watcher Started - Reset Queue`n`nA local watcher started at `{0}`. The watcher submits one queued BirdCLEF kernel at a time after `{1}`, polls Kaggle submission state, and records scored results before moving to the next queued kernel." -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"), $SubmitAfterUtc)
}

$deadline = (Get-Date).AddHours($StopAfterHours)
while ((Get-Date) -lt $deadline) {
    Poll-QueuedScores
    Try-SubmitNext

    if (Test-Path $ImprovedFlag) {
        Write-WatchLog "watcher stopped after a score above target"
        break
    }

    $doneCount = 0
    foreach ($item in $Queue) {
        if (Test-Path (Get-QueueItemFlagPath -Item $item -Suffix "scored")) {
            $doneCount += 1
        }
    }
    if (($doneCount -ge $Queue.Count) -or $NoSubmit) {
        Write-WatchLog "watcher completed all configured tasks"
        break
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-WatchLog "watcher stopped"
