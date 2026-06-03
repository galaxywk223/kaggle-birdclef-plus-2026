param(
    [string]$Competition = "birdclef-2026",
    [string]$KaggleExe = "kaggle",
    [string]$PythonExe = "python",
    [datetimeoffset]$StartUtc = ([datetimeoffset]::UtcNow),
    [datetimeoffset]$FastPollUtc = ([datetimeoffset]"2026-06-03T22:30:00Z"),
    [datetimeoffset]$DeadlineUtc = ([datetimeoffset]"2026-06-04T00:05:00Z"),
    [int]$SlowPollSeconds = 60,
    [int]$PollSeconds = 5,
    [int]$MaxAcceptedGroups = 5
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("final_window_submit_20260603_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$StatePath = Join-Path $LogDir "final_window_submit_state_20260603.json"

$Queue = @(
    @{
        Id = "exp142_p952_event_high"
        Group = "exp142_p952_event_high"
        Kernel = "galaxy2025/birdclef-2026-cheny-exp142-p952-event-high-safe"
        Version = 1
        Message = "BirdCLEF 2026 final window Exp142 P952 Event High Safe"
    },
    @{
        Id = "public_exp142_p952_event_high"
        Group = "exp142_p952_event_high"
        Kernel = "chenyfdws/bc26-exp142-hanijezo-p952-event-high-safe"
        Version = $null
        Message = "BirdCLEF 2026 final window Public Exp142 P952 Event High Safe"
    },
    @{
        Id = "exp141_p952_proxy"
        Group = "exp141_p952_proxy"
        Kernel = "galaxy2025/birdclef-2026-cheny-exp141-p952-proxy-safe"
        Version = 1
        Message = "BirdCLEF 2026 final window Exp141 P952 Proxy Safe"
    },
    @{
        Id = "public_exp141_p952_proxy"
        Group = "exp141_p952_proxy"
        Kernel = "chenyfdws/bc26-exp141-shahad-p952-proxy-safe"
        Version = $null
        Message = "BirdCLEF 2026 final window Public Exp141 P952 Proxy Safe"
    },
    @{
        Id = "lixinyin_exp142_p952_v51"
        Group = "lixinyin_exp142_p952_v51"
        Kernel = "galaxy2025/birdclef-2026-lixinyin-exp142-p952-v51"
        Version = 1
        Message = "BirdCLEF 2026 final window Lixinyin Exp142 P952 V51"
    },
    @{
        Id = "public_lixinyin_exp142_p952_v51"
        Group = "lixinyin_exp142_p952_v51"
        Kernel = "lixinyin/birdclef-2026-exp142-p952-v51"
        Version = $null
        Message = "BirdCLEF 2026 final window Public Lixinyin Exp142 P952 V51"
    },
    @{
        Id = "ulyanov_pantanal_repro_cpu"
        Group = "ulyanov_pantanal_repro_cpu"
        Kernel = "galaxy2025/birdclef-2026-ulyanov-pantanal-repro-cpu"
        Version = 1
        Message = "BirdCLEF 2026 final window Ulyanov Pantanal Repro CPU"
    },
    @{
        Id = "public_itshyao_s128_s124v2_g127_top2"
        Group = "itshyao_s128_s124v2_g127_top2"
        Kernel = "itshyao/birdclef-2026-s128-s124v2-g127-top2-rankblend"
        Version = $null
        Message = "BirdCLEF 2026 final window Public Itshyao S128 S124v2 G127 Top2 RankBlend"
    },
    @{
        Id = "public_itshyao_s123_s114_g124_f1_delta"
        Group = "itshyao_s123_s114_g124_f1_delta"
        Kernel = "itshyao/birdclef-2026-s123-s114-g124-f1-delta"
        Version = $null
        Message = "BirdCLEF 2026 final window Public Itshyao S123 S114 G124 F1 Delta"
    },
    @{
        Id = "public_itshyao_s122_s114_g123_f1_delta"
        Group = "itshyao_s122_s114_g123_f1_delta"
        Kernel = "itshyao/birdclef-2026-s122-s114-g123-f1-delta"
        Version = $null
        Message = "BirdCLEF 2026 final window Public Itshyao S122 S114 G123 F1 Delta"
    },
    @{
        Id = "public_itshyao_s121_s114_g116_f1_delta"
        Group = "itshyao_s121_s114_g116_f1_delta"
        Kernel = "itshyao/birdclef-2026-s121-s114-g116-f1-delta"
        Version = $null
        Message = "BirdCLEF 2026 final window Public Itshyao S121 S114 G116 F1 Delta"
    },
    @{
        Id = "public_jungchan_submission2_latest"
        Group = "jungchan_submission2_latest"
        Kernel = "jungchanryu/birdclef-submission2"
        Version = $null
        Message = "BirdCLEF 2026 final window Public Jungchan Submission2 Latest"
    },
    @{
        Id = "aotaku25_notebook02"
        Group = "aotaku25_notebook02"
        Kernel = "galaxy2025/birdclef-2026-aotaku25-notebook02"
        Version = 1
        Message = "BirdCLEF 2026 final window Aotaku25 Notebook02"
    },
    @{
        Id = "thomas_v2538d_smart_hybrid"
        Group = "thomas_v2538d_smart_hybrid"
        Kernel = "galaxy2025/birdclef-2026-thomas-v2538d-smart-hybrid"
        Version = 1
        Message = "BirdCLEF 2026 final window Thomas V2538d Smart Hybrid"
    }
)

$KernelReadyCache = @{}

function Write-SubmitLog {
    param([string]$Text)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"), $Text
    Add-Content -Path $LogPath -Value $line -Encoding utf8
    Write-Host $line
}

function ConvertTo-StateArray {
    param($Value)
    if ($null -eq $Value) {
        return @()
    }
    if ($Value -is [array]) {
        return @($Value | ForEach-Object { [string]$_ })
    }
    return @([string]$Value)
}

function Load-SubmitState {
    if (-not (Test-Path -LiteralPath $StatePath)) {
        return @{
            Submitted = @()
            SubmittedGroups = @()
        }
    }

    try {
        $raw = Get-Content -LiteralPath $StatePath -Raw -Encoding utf8
        if ([string]::IsNullOrWhiteSpace($raw)) {
            throw "state file is empty"
        }
        $state = $raw | ConvertFrom-Json
        return @{
            Submitted = @(ConvertTo-StateArray -Value $state.submitted)
            SubmittedGroups = @(ConvertTo-StateArray -Value $state.submitted_groups)
        }
    }
    catch {
        Write-SubmitLog ("state load failed; starting with empty state: {0}" -f $_.Exception.Message)
        return @{
            Submitted = @()
            SubmittedGroups = @()
        }
    }
}

function Save-SubmitState {
    param(
        [System.Collections.Generic.HashSet[string]]$Submitted,
        [System.Collections.Generic.HashSet[string]]$SubmittedGroups
    )

    $payload = [ordered]@{
        updated_at_utc = ([datetimeoffset]::UtcNow.ToString("o"))
        submitted = @($Submitted | Sort-Object)
        submitted_groups = @($SubmittedGroups | Sort-Object)
    }
    $payload | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $StatePath -Encoding utf8
}

function Invoke-Kaggle {
    param([string[]]$Arguments)
    $output = & $KaggleExe @Arguments 2>&1
    $code = $LASTEXITCODE
    foreach ($line in $output) {
        Write-SubmitLog ("kaggle> {0}" -f $line)
    }
    return @{
        Code = $code
        Output = ($output -join "`n")
    }
}

function Invoke-CodeSubmission {
    param([hashtable]$Item)
    $scriptPath = Join-Path $ProjectRoot "scripts\submit_code_with_body_20260603.py"
    $args = @(
        $scriptPath,
        "--competition", $Competition,
        "--file-name", "submission.csv",
        "--kernel", $Item.Kernel,
        "--message", $Item.Message
    )
    if ($null -ne $Item.Version) {
        $args += @("--version", ([string]$Item.Version))
    }
    $output = & $PythonExe @args 2>&1
    $code = $LASTEXITCODE
    foreach ($line in $output) {
        Write-SubmitLog ("submit-api> {0}" -f $line)
    }
    return @{
        Code = $code
        Output = ($output -join "`n")
    }
}

function Test-TransientSubmitError {
    param([string]$Output)
    if ([string]::IsNullOrWhiteSpace($Output)) {
        return $false
    }

    $transientPatterns = @(
        "requests\.exceptions\.SSLError",
        "requests\.exceptions\.ConnectionError",
        "requests\.exceptions\.ProxyError",
        "SSLError",
        "ConnectionError",
        "ProxyError",
        "Max retries exceeded",
        "UNEXPECTED_EOF_WHILE_READING",
        "EOF occurred",
        "connection reset",
        "temporarily unavailable",
        "RemoteDisconnected",
        "Read timed out",
        "ConnectTimeout",
        "RetryError"
    )

    foreach ($pattern in $transientPatterns) {
        if ($Output -match $pattern) {
            return $true
        }
    }
    return $false
}

function Get-KernelReady {
    param([string]$Kernel)
    if ($KernelReadyCache.ContainsKey($Kernel)) {
        return $true
    }

    $result = Invoke-Kaggle -Arguments @("kernels", "status", $Kernel)
    if ($result.Code -ne 0) {
        return $null
    }
    $ready = ($result.Output -match "COMPLETE")
    if ($ready) {
        $KernelReadyCache[$Kernel] = $true
    }
    return $ready
}

function Initialize-KernelReadyCache {
    Write-SubmitLog "warming COMPLETE kernel status cache"
    foreach ($item in $Queue) {
        if ($KernelReadyCache.ContainsKey($item.Kernel)) {
            continue
        }

        $ready = Get-KernelReady -Kernel $item.Kernel
        if ($ready -eq $true) {
            Write-SubmitLog ("cached COMPLETE status for {0}" -f $item.Id)
        }
        elseif ($ready -eq $false) {
            Write-SubmitLog ("preflight not COMPLETE for {0}" -f $item.Id)
        }
        else {
            Write-SubmitLog ("preflight status unavailable for {0}" -f $item.Id)
        }
    }
    Write-SubmitLog ("kernel status cache entries={0}" -f $KernelReadyCache.Count)
}

function Try-SubmitItem {
    param([hashtable]$Item)
    $ready = Get-KernelReady -Kernel $Item.Kernel
    if ($null -eq $ready) {
        Write-SubmitLog ("status unavailable for {0}; pausing this queue pass" -f $Item.Id)
        return $null
    }

    if (-not $ready) {
        Write-SubmitLog ("skip {0}: kernel is not COMPLETE" -f $Item.Id)
        return $false
    }

    Write-SubmitLog ("submit attempt {0}: {1} v{2}" -f $Item.Id, $Item.Kernel, $Item.Version)
    $result = Invoke-CodeSubmission -Item $Item

    if ($result.Code -eq 0) {
        Write-SubmitLog ("submit accepted {0}" -f $Item.Id)
        return $true
    }

    if ($result.Output -match "not accepting submissions|competition.*closed|deadline|Submissions have been disabled|has ended") {
        Write-SubmitLog ("competition closed after {0}" -f $Item.Id)
        return "closed"
    }

    if ($result.Output -match "daily Submission allowance|try again tomorrow") {
        Write-SubmitLog ("quota still locked after {0}" -f $Item.Id)
        return $null
    }

    if (Test-TransientSubmitError -Output $result.Output) {
        Write-SubmitLog ("transient submit error after {0}; pausing this queue pass" -f $Item.Id)
        return $null
    }

    Write-SubmitLog ("submit failed {0} code={1}" -f $Item.Id, $result.Code)
    return $false
}

$MutexName = "Global\BirdCLEF2026FinalWindowSubmit"
$Mutex = New-Object System.Threading.Mutex($false, $MutexName)
$HasMutex = $false

try {
    $HasMutex = $Mutex.WaitOne(0)
    if (-not $HasMutex) {
        Write-SubmitLog ("another final-window watcher already holds mutex {0}; exiting" -f $MutexName)
        exit 0
    }

    Write-SubmitLog ("final-window watcher started; start_utc={0}; fast_poll_utc={1}; deadline_utc={2}; slow_poll_seconds={3}; poll_seconds={4}; max_accepted_groups={5}" -f $StartUtc.ToString("o"), $FastPollUtc.ToString("o"), $DeadlineUtc.ToString("o"), $SlowPollSeconds, $PollSeconds, $MaxAcceptedGroups)
    Write-SubmitLog ("queue: {0}" -f (($Queue | ForEach-Object { $_.Id }) -join ", "))

    while ([datetimeoffset]::UtcNow -lt $StartUtc) {
        $remaining = $StartUtc - [datetimeoffset]::UtcNow
        Write-SubmitLog ("waiting for start window; remaining_seconds={0:N0}" -f $remaining.TotalSeconds)
        Start-Sleep -Seconds ([Math]::Min(300, [Math]::Max(1, [int][Math]::Floor($remaining.TotalSeconds))))
    }

    $state = Load-SubmitState
    $submitted = New-Object System.Collections.Generic.HashSet[string]
    foreach ($id in $state.Submitted) {
        if (-not [string]::IsNullOrWhiteSpace($id)) {
            [void]$submitted.Add($id)
        }
    }
    $submittedGroups = New-Object System.Collections.Generic.HashSet[string]
    foreach ($group in $state.SubmittedGroups) {
        if (-not [string]::IsNullOrWhiteSpace($group)) {
            [void]$submittedGroups.Add($group)
        }
    }
    Write-SubmitLog ("loaded state: submitted={0}; submitted_groups={1}" -f (($submitted | Sort-Object) -join ","), (($submittedGroups | Sort-Object) -join ","))
    Initialize-KernelReadyCache

    $competitionClosed = $false

    while ([datetimeoffset]::UtcNow -lt $DeadlineUtc) {
        foreach ($item in $Queue) {
            if ($submittedGroups.Count -ge $MaxAcceptedGroups) {
                Write-SubmitLog ("accepted group limit reached: {0}/{1}" -f $submittedGroups.Count, $MaxAcceptedGroups)
                break
            }
            if ($submitted.Contains($item.Id)) {
                continue
            }
            if ($submittedGroups.Contains($item.Group)) {
                Write-SubmitLog ("skip {0}: group already accepted ({1})" -f $item.Id, $item.Group)
                continue
            }

            $attempt = Try-SubmitItem -Item $item
            if ($attempt -eq "closed") {
                $competitionClosed = $true
                break
            }

            if ($attempt -eq $true) {
                [void]$submitted.Add($item.Id)
                [void]$submittedGroups.Add($item.Group)
                Save-SubmitState -Submitted $submitted -SubmittedGroups $submittedGroups
                continue
            }

            if ($attempt -eq $null) {
                break
            }
        }

        if ($competitionClosed) {
            break
        }

        if ($submittedGroups.Count -ge $MaxAcceptedGroups) {
            Write-SubmitLog ("max accepted groups reached; stopping submit loop")
            break
        }

        if ($submitted.Count -ge $Queue.Count) {
            Write-SubmitLog "all queued submissions accepted"
            break
        }

        $sleepSeconds = $PollSeconds
        if ([datetimeoffset]::UtcNow -lt $FastPollUtc) {
            $sleepSeconds = $SlowPollSeconds
        }
        Start-Sleep -Seconds $sleepSeconds
    }

    Save-SubmitState -Submitted $submitted -SubmittedGroups $submittedGroups
    Write-SubmitLog ("final-window watcher stopped; submitted={0}; submitted_groups={1}; deadline_reached={2}" -f (($submitted | Sort-Object) -join ","), (($submittedGroups | Sort-Object) -join ","), ([datetimeoffset]::UtcNow -ge $DeadlineUtc))
    Invoke-Kaggle -Arguments @("competitions", "submissions", "-c", $Competition, "--csv") | Out-Null
}
finally {
    if ($HasMutex) {
        [void]$Mutex.ReleaseMutex()
    }
    $Mutex.Dispose()
}
