<#
.SYNOPSIS
  Build llamactl and install it where a shortcut can point: the CLI and the
  GUI go to %LOCALAPPDATA%\Programs\llamactl, a Start Menu shortcut is
  created, and optionally the folder is added to the user PATH.

  Re-run after `git pull` / code changes to update the installed copies.
  Windows does not allow pinning to the taskbar from a script; open the
  Start Menu, right-click "llamactl", "Pin to taskbar".

.PARAMETER NoBuild
  Skip the cargo / tauri build and just copy what target\release holds.
.PARAMETER AddToPath
  Append the install folder to the user PATH so `llamactl` works in any shell.
.EXAMPLE
  powershell -File scripts\install.ps1
  powershell -File scripts\install.ps1 -AddToPath
#>
param(
  [switch]$NoBuild,
  [switch]$AddToPath
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $env:LOCALAPPDATA 'Programs\llamactl'
$release = Join-Path $repo 'target\release'

if (-not $NoBuild) {
  Write-Host "== building CLI" -ForegroundColor Cyan
  Push-Location $repo
  try {
    & cargo build --release -p llamactl-cli
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }
    Write-Host "== building GUI" -ForegroundColor Cyan
    Push-Location (Join-Path $repo 'ui')
    try {
      # A running GUI holds the exe open and the build cannot replace it.
      Get-Process llamactl-ui -ErrorAction SilentlyContinue | Stop-Process -Force
      & pnpm tauri build --no-bundle
      if ($LASTEXITCODE -ne 0) { throw "tauri build failed ($LASTEXITCODE)" }
    } finally { Pop-Location }
  } finally { Pop-Location }
}

foreach ($exe in 'llamactl.exe', 'llamactl-ui.exe') {
  if (-not (Test-Path (Join-Path $release $exe))) { throw "missing $release\$exe - build first" }
}

Write-Host "== installing to $dest" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $dest | Out-Null
# The installed GUI may be running too.
Get-Process llamactl-ui -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like "$dest*" } | Stop-Process -Force
# Keep-alive helpers hold llamactl.exe open; stop them, remember which runs
# had one, and restart them from the new binary afterwards. (Stop-Process,
# never Git-Bash taskkill: MSYS mangles /PID into a path.)
$helpers = Get-CimInstance Win32_Process -Filter "Name='llamactl.exe'" |
  Where-Object { $_.CommandLine -like '*keepalive*' }
$restart = @()
foreach ($h in $helpers) {
  if ($h.CommandLine -match '--port (\d+).*--interval (\d+).*--server-pid (\d+)') {
    $restart += @{ port = $Matches[1]; interval = $Matches[2]; serverPid = $Matches[3] }
  }
  Stop-Process -Id $h.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500
foreach ($exe in 'llamactl.exe', 'llamactl-ui.exe') {
  Copy-Item (Join-Path $release $exe) (Join-Path $dest $exe) -Force
}
foreach ($r in $restart) {
  $p = Start-Process -FilePath (Join-Path $dest 'llamactl.exe') -WindowStyle Hidden -PassThru `
    -ArgumentList @('keepalive', '--host', '127.0.0.1', '--port', $r.port, '--interval', $r.interval, '--server-pid', $r.serverPid)
  # Point the run state at the new helper pid so `stop` still kills it.
  Get-ChildItem "$env:USERPROFILE\.llamactl\runs\*-$($r.port).json" -ErrorAction SilentlyContinue | ForEach-Object {
    $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
    if ("$($j.pid)" -eq $r.serverPid) {
      $j | Add-Member -NotePropertyName keepalive_pid -NotePropertyValue $p.Id -Force
      # No BOM: llamactl reads these with serde_json.
      [IO.File]::WriteAllText($_.FullName, ($j | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))
    }
  }
  Write-Host ("== restarted keep-alive for port {0} (pid {1})" -f $r.port, $p.Id)
}

$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$lnk = Join-Path $startMenu 'llamactl.lnk'
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = Join-Path $dest 'llamactl-ui.exe'
$sc.WorkingDirectory = $dest
$sc.IconLocation = (Join-Path $dest 'llamactl-ui.exe') + ',0'
$sc.Description = 'llama.cpp launcher for the 2x R9700 box'
$sc.Save()
Write-Host "== Start Menu shortcut: $lnk" -ForegroundColor Cyan

if ($AddToPath) {
  $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (($userPath -split ';') -notcontains $dest) {
    [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $dest), 'User')
    Write-Host "== added $dest to the user PATH (new shells only)" -ForegroundColor Cyan
  } else {
    Write-Host "== $dest already on the user PATH"
  }
}

$ver = & (Join-Path $dest 'llamactl.exe') --version
Write-Host ""
Write-Host "installed: $ver"
Write-Host "  GUI : $dest\llamactl-ui.exe"
Write-Host "  CLI : $dest\llamactl.exe"
Write-Host "  to pin: Start Menu -> right-click 'llamactl' -> Pin to taskbar"
