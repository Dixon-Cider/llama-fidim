<#
.SYNOPSIS
  M10 - spec section 09 v1.1 acceptance run for the criteria that require
  loading models. Run this only when the GPUs are free.

.DESCRIPTION
  Executes the remaining acceptance criteria in one sequence and prints a
  pass/fail table with the evidence behind each verdict.

  SAFETY: this loads models into VRAM. A guard runs before every launch and
  aborts the whole run if projected commit would exceed -MaxCommitPercent
  (default 75, deliberately below the tool's own 90% block so the run stops
  well before the machine is at risk). Servers are always stopped on exit,
  including on Ctrl-C or an unexpected error.

  The 2026-08-01 freeze came from launching a second large server while the
  first was still loading. This script uses small-context profiles and waits
  for each server to report ready before considering the next.

  NOTE: ASCII only, deliberately. Windows PowerShell 5.1 reads .ps1 as ANSI
  unless the file has a BOM, so non-ASCII punctuation corrupts the parse.

.PARAMETER Phases
  Which phases to run. Default all. 1=two servers on two cards (incl.
  re-attach), 2=benchmark, 3=single-model split across both cards.

.PARAMETER MaxCommitPercent
  Abort threshold for projected commit. Default 75.

.EXAMPLE
  powershell -File scripts/m10-acceptance.ps1
  powershell -File scripts/m10-acceptance.ps1 -Phases 3
#>
[CmdletBinding()]
param(
  [int[]]$Phases = @(1, 2, 3),
  [int]$MaxCommitPercent = 75,
  [string]$Cli = "$PSScriptRoot\..\target\debug\llamactl.exe"
)

$ErrorActionPreference = 'Stop'

# Stable-key -> adapter LUID low dword, for attributing PDH counters to a
# physical card. LUIDs are volatile across driver updates (the 2026-08-25
# driver change moved bus03 0x1621C -> 0x16CEF and bus08 0x1B592 -> 0x1BAAD
# and silently failed every residency check), so derive them at start from
# PnP: DEVPKEY_Device_BusNumber -> DEVPKEY_Gpu_Luid, same as the tool does.
function Get-LuidByBus {
  $map = @{}
  foreach ($d in (Get-PnpDevice -Class Display -ErrorAction SilentlyContinue)) {
    $bus  = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_BusNumber' -ErrorAction SilentlyContinue).Data
    $luid = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName '{60B193CB-5276-4D0F-96FC-F173ABAD3EC6} 2' -ErrorAction SilentlyContinue).Data
    if ($null -eq $bus -or $null -eq $luid) { continue }
    $map[('bus{0:D2}' -f [int]$bus)] = [int64]$luid
  }
  foreach ($k in 'bus03', 'bus08') {
    if (-not $map.ContainsKey($k)) { throw "could not resolve adapter LUID for $k from PnP" }
  }
  return $map
}
$LuidByBus = Get-LuidByBus
Write-Host ("adapter LUIDs: bus03=0x{0:X} bus08=0x{1:X}" -f $LuidByBus['bus03'], $LuidByBus['bus08']) -ForegroundColor DarkGray

$script:Results = [System.Collections.ArrayList]::new()
$script:Started = [System.Collections.ArrayList]::new()

function Record($criterion, $verdict, $evidence) {
  [void]$script:Results.Add([pscustomobject]@{
    Criterion = $criterion; Verdict = $verdict; Evidence = $evidence
  })
  $colour = if ($verdict -eq 'PASS') { 'Green' } elseif ($verdict -eq 'FAIL') { 'Red' } else { 'Yellow' }
  Write-Host ("  [{0}] {1}" -f $verdict, $criterion) -ForegroundColor $colour
  if ($evidence) { Write-Host ("         {0}" -f $evidence) -ForegroundColor DarkGray }
}

function Get-Mem {
  $os = Get-CimInstance Win32_OperatingSystem
  $limit = $os.TotalVirtualMemorySize / 1MB
  $charge = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB
  [pscustomobject]@{
    RamFreeGiB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    CommitGiB  = [math]::Round($charge, 1)
    LimitGiB   = [math]::Round($limit, 1)
    Percent    = [math]::Round(100 * $charge / $limit, 1)
  }
}

# Abort the run rather than risk the machine.
function Assert-Headroom($label) {
  $m = Get-Mem
  Write-Host ("  memory @ {0}: commit {1} / {2} GiB ({3} pct), RAM free {4} GiB" -f `
    $label, $m.CommitGiB, $m.LimitGiB, $m.Percent, $m.RamFreeGiB) -ForegroundColor DarkGray
  if ($m.Percent -gt $MaxCommitPercent) {
    throw "ABORT: commit at $($m.Percent) pct exceeds the MaxCommitPercent $MaxCommitPercent guard. Nothing further will be launched."
  }
  return $m
}

# Dedicated + committed GPU bytes for a pid, keyed by adapter LUID.
function Get-GpuByLuid([int]$ProcId) {
  $out = @{}
  foreach ($counter in 'Dedicated Usage', 'Total Committed') {
    try {
      $samples = (Get-Counter "\GPU Process Memory(pid_$ProcId*)\$counter" -ErrorAction Stop).CounterSamples
    } catch { continue }
    foreach ($s in $samples) {
      if ($s.CookedValue -le 0) { continue }
      if ($s.InstanceName -notmatch '_luid_0x[0-9a-fA-F]+_0x([0-9a-fA-F]+)_') { continue }
      $luid = [Convert]::ToInt64($Matches[1], 16)
      if (-not $out.ContainsKey($luid)) { $out[$luid] = @{ Dedicated = 0; Committed = 0 } }
      if ($counter -eq 'Dedicated Usage') { $out[$luid].Dedicated += $s.CookedValue }
      else { $out[$luid].Committed += $s.CookedValue }
    }
  }
  return $out
}

function Get-Best($map, $luid) {
  if ($map.ContainsKey($luid)) {
    return [math]::Max($map[$luid].Dedicated, $map[$luid].Committed)
  }
  return 0
}

function Format-Card($map, $luid) {
  if ($map.ContainsKey($luid)) {
    return '{0:N1} GiB ded / {1:N1} GiB comm' -f ($map[$luid].Dedicated / 1GB), ($map[$luid].Committed / 1GB)
  }
  return 'nothing'
}

function Get-RunState($profileId) {
  $f = Get-ChildItem "$env:USERPROFILE\.llamactl\runs\*.json" -ErrorAction SilentlyContinue |
       Where-Object { $_.BaseName -like "$profileId-*" } | Select-Object -First 1
  if (-not $f) { return $null }
  return Get-Content $f.FullName -Raw | ConvertFrom-Json
}

# `& $Cli launch ... | Out-String` deadlocks: the detached llama-server
# inherits PowerShell's stdout pipe handle (Rust std does not restrict the
# inheritable-handle list), so the pipeline never sees EOF until the server
# exits. Start-Process -Wait is no better: in Windows PowerShell 5.1 it
# waits for the whole descendant tree, i.e. the server again. So we take
# the Process object and call WaitForExit(), which waits on the CLI's own
# process handle only; the redirect targets are plain files read back after.
function Invoke-CliToFile([string[]]$CliArgs) {
  $stdout = [System.IO.Path]::GetTempFileName()
  $stderr = [System.IO.Path]::GetTempFileName()
  try {
    $proc = Start-Process -FilePath $Cli -ArgumentList $CliArgs -PassThru `
      -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    # Touch the handle before the process exits, otherwise PS 5.1 leaves
    # ExitCode null afterwards.
    $null = $proc.Handle
    $proc.WaitForExit()
    $text = (Get-Content $stdout -Raw -ErrorAction SilentlyContinue) + "`n" +
            (Get-Content $stderr -Raw -ErrorAction SilentlyContinue)
    $code = $proc.ExitCode
    if ($null -eq $code) { $code = 0 }   # unknown: fall back to the ready match
    return @{ ExitCode = $code; Output = $text }
  } finally {
    Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
  }
}

function Start-Server($profileId) {
  Assert-Headroom "before $profileId" | Out-Null
  Write-Host "  launching $profileId ..." -ForegroundColor Cyan
  $r = Invoke-CliToFile @('launch', $profileId, '--ready-timeout', '600')
  $out = $r.Output
  if ($r.ExitCode -ne 0 -or $out -notmatch 'ready\.') {
    throw "launch of $profileId did not reach ready:`n$out"
  }
  [void]$script:Started.Add($profileId)
  return $out
}

function Stop-Server($profileId) {
  try { & $Cli stop $profileId 2>&1 | Out-Null } catch { }
  $script:Started.Remove($profileId)
}

function Stop-All {
  foreach ($p in @($script:Started)) {
    Write-Host "  stopping $p" -ForegroundColor DarkGray
    Stop-Server $p
  }
}

# ---------------------------------------------------------------- phases ----

function Invoke-Phase1 {
  Write-Host "`n=== PHASE 1: two servers, two cards, then re-attach ===" -ForegroundColor White

  $a = 'verify-12b'        # bus03
  $b = 'verify-26b-small'  # bus08
  Start-Server $a | Out-Null
  Start-Server $b | Out-Null

  $stateA = Get-RunState $a
  $stateB = Get-RunState $b
  if (-not $stateA -or -not $stateB) { Record 'Two servers launch' 'FAIL' 'run state missing'; return }

  $gpuA = Get-GpuByLuid $stateA.pid
  $gpuB = Get-GpuByLuid $stateB.pid
  $wantA = $LuidByBus['bus03']
  $wantB = $LuidByBus['bus08']

  $aOnRight = (Get-Best $gpuA $wantA) -gt 1GB
  $bOnRight = (Get-Best $gpuB $wantB) -gt 1GB
  $aStray = @($gpuA.Keys | Where-Object { $_ -ne $wantA -and (Get-Best $gpuA $_) -gt 0.5GB })
  $bStray = @($gpuB.Keys | Where-Object { $_ -ne $wantB -and (Get-Best $gpuB $_) -gt 0.5GB })

  $ev = "{0}(pid {1}) bus03: {2}; {3}(pid {4}) bus08: {5}" -f `
        $a, $stateA.pid, (Format-Card $gpuA $wantA), $b, $stateB.pid, (Format-Card $gpuB $wantB)

  if ($aOnRight -and $bOnRight -and $aStray.Count -eq 0 -and $bStray.Count -eq 0) {
    Record 'Two servers on two different GPUs, each on its intended card' 'PASS' $ev
  } else {
    Record 'Two servers on two different GPUs, each on its intended card' 'FAIL' `
      ("{0} (stray adapters: A={1} B={2})" -f $ev, $aStray.Count, $bStray.Count)
  }

  # iGPU must never be visible to a launched server.
  $vis = "$($stateA.visibility_env) / $($stateB.visibility_env)"
  if ($stateA.visibility_env -notmatch '\b1\b' -and $stateB.visibility_env -notmatch '\b1\b') {
    Record 'iGPU never visible to a launched server' 'PASS' "HIP_VISIBLE_DEVICES = $vis (index 1 = iGPU, absent from both)"
  } else {
    Record 'iGPU never visible to a launched server' 'FAIL' "HIP_VISIBLE_DEVICES = $vis"
  }

  # Re-attach: a fresh CLI process rebuilds live state from disk alone.
  # Judge only the two servers this phase launched: the listing may also carry
  # older crashed-run state that the tool keeps on purpose for diagnosis.
  $status = & $Cli status 2>&1 | Out-String
  $lineA = ($status -split "`r?`n" | Where-Object { $_ -match ('^\s*' + [regex]::Escape($a) + '\s') } | Select-Object -First 1)
  $lineB = ($status -split "`r?`n" | Where-Object { $_ -match ('^\s*' + [regex]::Escape($b) + '\s') } | Select-Object -First 1)
  if ($lineA -match '\bhealthy\b' -and $lineB -match '\bhealthy\b') {
    Record 'Closing and reopening the tool re-attaches to running servers' 'PASS' ("a new llamactl process listed both servers as healthy from run state alone:`n" + $lineA.Trim() + "`n" + $lineB.Trim())
  } else {
    Record 'Closing and reopening the tool re-attaches to running servers' 'FAIL' $status.Trim()
  }
}

function Invoke-Phase2 {
  Write-Host "`n=== PHASE 2: benchmark sweep ===" -ForegroundColor White
  $target = 'verify-26b-small'
  if ($script:Started -notcontains $target) { Start-Server $target | Out-Null }

  $out = & $Cli bench $target --tokens 128 2>&1 | Out-String
  Write-Host $out.Trim() -ForegroundColor DarkGray

  $serial = $null; $agg = $null; $per = $null
  if ($out -match 'serial:\s+([\d.]+) tok/s') { $serial = $Matches[1] }
  if ($out -match 'decode aggregate ([\d.]+) tok/s, per-stream ([\d.]+)') {
    $agg = $Matches[1]; $per = $Matches[2]
  }

  if ($serial -and $agg -and $per) {
    Record 'Benchmark produces aggregate and per-stream throughput' 'PASS' `
      "serial $serial tok/s, decode-aggregate $agg tok/s, per-stream $per tok/s"
  } else {
    Record 'Benchmark produces aggregate and per-stream throughput' 'FAIL' $out.Trim()
  }

  # Cold-cache handling: either it was marked cold and withheld, or it was a
  # warm run and got stored. Both are correct; silence would not be.
  $profPath = "$env:USERPROFILE\.llamactl\profiles\$target.json"
  if ($out -match 'baseline NOT saved: cold-cache') {
    Record 'Cold-cache first run marked and excluded from baselines' 'PASS' `
      'run flagged cold; sweep recorded to history but withheld from the profile baseline'
  } elseif ($out -match 'baseline saved to profile') {
    $prof = Get-Content $profPath -Raw | ConvertFrom-Json
    if ($prof.baseline.cold_cache -eq $false) {
      Record 'Cold-cache first run marked and excluded from baselines' 'PASS' `
        "warm run (model already loaded once); baseline stored with cold_cache=false"
    } else {
      Record 'Cold-cache first run marked and excluded from baselines' 'FAIL' 'a cold run was stored as a baseline'
    }
  } else {
    Record 'Cold-cache first run marked and excluded from baselines' 'FAIL' 'no cold/warm disposition reported'
  }

  # Baseline must carry the environment it was measured in.
  if (Test-Path $profPath) {
    $prof = Get-Content $profPath -Raw | ConvertFrom-Json
    if ($prof.baseline -and $prof.baseline.driver -and $prof.baseline.sdk -and $prof.baseline.profile_fingerprint) {
      Record 'Baseline stored against the exact profile revision' 'PASS' `
        "fingerprint $($prof.baseline.profile_fingerprint), driver $($prof.baseline.driver), sdk $($prof.baseline.sdk)"
    } else {
      Record 'Baseline stored against the exact profile revision' 'WARN' 'no warm baseline stored on this run'
    }
  }
}

function Invoke-Phase3 {
  Write-Host "`n=== PHASE 3: one model split across both cards ===" -ForegroundColor White
  Stop-All
  Start-Sleep -Seconds 5

  $p = 'qwen-split'
  Start-Server $p | Out-Null
  $state = Get-RunState $p
  if (-not $state) { Record 'Split model launches' 'FAIL' 'run state missing'; return }

  # A layer split is lazily resident: the non-main card can read near zero
  # dedicated until the first forward pass. Generate once to force it.
  try {
    $body = '{"prompt":"Count to five:","n_predict":24,"stream":false}'
    Invoke-RestMethod -Uri "http://127.0.0.1:$($state.port)/completion" -Method Post `
      -ContentType 'application/json' -Body $body -TimeoutSec 300 | Out-Null
  } catch {
    Write-Host "  (generation probe failed: $_)" -ForegroundColor Yellow
  }

  $gpu = Get-GpuByLuid $state.pid
  $a = $LuidByBus['bus03']; $b = $LuidByBus['bus08']
  $onA = (Get-Best $gpu $a) -gt 1GB
  $onB = (Get-Best $gpu $b) -gt 1GB
  $ev = "bus03: {0}; bus08: {1}; HIP_VISIBLE_DEVICES={2}" -f `
        (Format-Card $gpu $a), (Format-Card $gpu $b), $state.visibility_env

  if ($onA -and $onB) {
    Record 'One model split across both discrete GPUs, residency confirmed per card' 'PASS' $ev
  } else {
    Record 'One model split across both discrete GPUs, residency confirmed per card' 'FAIL' $ev
  }

  if ($state.visibility_env -eq '0,2') {
    Record 'Split pins visibility to exactly the two discrete cards' 'PASS' `
      "HIP_VISIBLE_DEVICES=$($state.visibility_env) - iGPU (index 1) excluded"
  } else {
    Record 'Split pins visibility to exactly the two discrete cards' 'FAIL' `
      "HIP_VISIBLE_DEVICES=$($state.visibility_env)"
  }
}

# ------------------------------------------------------------------ main ----

$startMem = $null
try {
  Write-Host "M10 acceptance run - spec section 09 v1.1" -ForegroundColor White
  Write-Host "guard: abort if projected commit exceeds $MaxCommitPercent pct" -ForegroundColor DarkGray
  if (-not (Test-Path $Cli)) { throw "llamactl not found at $Cli - run cargo build first" }
  $startMem = Assert-Headroom 'start'

  if ($Phases -contains 1) { Invoke-Phase1 }
  if ($Phases -contains 2) { Invoke-Phase2 }
  if ($Phases -contains 3) { Invoke-Phase3 }
}
catch {
  Write-Host "`nRUN ABORTED: $_" -ForegroundColor Red
  Record 'Run completed without aborting' 'FAIL' "$_"
}
finally {
  Write-Host "`n--- cleanup ---" -ForegroundColor White
  Stop-All
  Start-Sleep -Seconds 4
  $endMem = Get-Mem
  Write-Host ("  memory after cleanup: commit {0} / {1} GiB ({2} pct), RAM free {3} GiB" -f `
    $endMem.CommitGiB, $endMem.LimitGiB, $endMem.Percent, $endMem.RamFreeGiB) -ForegroundColor DarkGray
  if ($startMem -and ($endMem.CommitGiB - $startMem.CommitGiB) -gt 4) {
    Write-Host ("  NOTE: commit is {0:N1} GiB above the starting value - check for a stray llama-server" -f `
      ($endMem.CommitGiB - $startMem.CommitGiB)) -ForegroundColor Yellow
  }

  Write-Host "`n=== M10 RESULTS ===" -ForegroundColor White
  $script:Results | Format-Table -AutoSize -Wrap
  $fail = @($script:Results | Where-Object Verdict -eq 'FAIL').Count
  $pass = @($script:Results | Where-Object Verdict -eq 'PASS').Count
  Write-Host ("{0} passed, {1} failed" -f $pass, $fail) -ForegroundColor $(if ($fail) { 'Red' } else { 'Green' })
}
