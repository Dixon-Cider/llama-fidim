<#
.SYNOPSIS
  Build Llama FIDIM and install it for the current user, in the folder the
  setup.exe installer uses: the CLI, the GUI and the DiffusionGemma server
  (fidim-dg.exe) go to %LOCALAPPDATA%\Llama FIDIM, a Start Menu shortcut is
  created, and optionally the folder is added to the user PATH.

  Re-run after `git pull` / code changes to update the installed copies.
  An install in %LOCALAPPDATA%\Programs\LlamaFIDIM, where this script used
  to install, is retired: its files are removed (running helpers are moved
  aside and keep running) and, if that folder was on the user PATH, the new
  one takes its place there.
  Windows does not allow pinning to the taskbar from a script; open the
  Start Menu, right-click "Llama FIDIM", "Pin to taskbar".

.PARAMETER NoBuild
  Skip the cargo / tauri build and just copy what target\release holds.
.PARAMETER AddToPath
  Append the install folder to the user PATH so `fidim` works in any shell
  (the same as running `fidim path add` from the installed copy).
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
# The installer's folder (Tauri's default for a per-user install), so the
# two kinds of install update each other instead of living side by side.
$dest = Join-Path $env:LOCALAPPDATA 'Llama FIDIM'
# Where this script installed until the installer existed.
$oldDest = Join-Path $env:LOCALAPPDATA 'Programs\LlamaFIDIM'
$release = Join-Path $repo 'target\release'

# Delete $dir\$name. Windows refuses while it runs; then rename it to
# $name.old-<timestamp> instead, and the process keeps running from there
# (same pid, same image name, so `fidim stop` still finds it). First
# deletes the copies earlier installs renamed, unless they still run.
function Remove-OrMoveAside([string]$dir, [string]$name) {
  Get-ChildItem $dir -Filter "$name.old-*" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue
  $path = Join-Path $dir $name
  if (-not (Test-Path $path)) { return }
  try { Remove-Item $path -Force -ErrorAction Stop }
  catch {
    $aside = "$name.old-" + (Get-Date -Format yyyyMMddHHmmss)
    try {
      Rename-Item $path $aside -ErrorAction Stop
      Write-Host "   moved the running $name aside as $aside (it keeps running)"
    } catch {
      Write-Host "   could not remove $path or move it aside: $($_.Exception.Message)" -ForegroundColor Yellow
    }
  }
}

# What a build needs. Say so up front instead of failing halfway.
if (-not $NoBuild) {
  $missing = @()
  foreach ($tool in @(
      @{ cmd = 'cargo'; hint = 'Rust toolchain: https://rustup.rs (stable, MSVC)' },
      @{ cmd = 'pnpm';  hint = 'pnpm: npm install -g pnpm (Node 20+)' })) {
    if (-not (Get-Command $tool.cmd -ErrorAction SilentlyContinue)) { $missing += "  $($tool.cmd) - $($tool.hint)" }
  }
  if ($missing.Count) {
    Write-Host "Missing build tools:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host $_ }
    Write-Host "Also needed once: the Tauri CLI (pnpm install inside ui\ pulls it) and WebView2 (ships with Windows 11)."
    exit 1
  }
  if (-not (Test-Path (Join-Path $repo 'ui\node_modules'))) {
    Write-Host "== installing UI dependencies (pnpm install)" -ForegroundColor Cyan
    Push-Location (Join-Path $repo 'ui'); try { & pnpm install; if ($LASTEXITCODE -ne 0) { throw "pnpm install failed" } } finally { Pop-Location }
  }
}

if (-not $NoBuild) {
  # The binaries name the commit they were built from (fidim --version, the
  # sidebar). Tell the build which checkout and commit it is and whether the
  # tree has uncommitted changes, so an installed copy never passes for a
  # commit it does not match. Only for these builds: a caller's shell (run
  # with `& install.ps1`) must not keep the values.
  $saved = @{}
  foreach ($k in 'FIDIM_BUILD_ID', 'FIDIM_BUILD_MODIFIED') { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process') }
  if (Get-Command git -ErrorAction SilentlyContinue) {
    $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
      $head = & git -C $repo rev-parse HEAD 2>$null
      $gitOk = ($LASTEXITCODE -eq 0)
      if ($gitOk) { $changes = & git -C $repo status --porcelain --untracked-files=no 2>$null; $gitOk = ($LASTEXITCODE -eq 0) }
    } finally { $ErrorActionPreference = $eap }
    if ($gitOk) {
      $env:FIDIM_BUILD_ID = "$repo@$head"
      $env:FIDIM_BUILD_MODIFIED = if ($changes) { '1' } else { '0' }
      if ($changes) { Write-Host "== building uncommitted changes: the version will read <commit>-modified" -ForegroundColor Yellow }
    }
  }

  Write-Host "== building CLI" -ForegroundColor Cyan
  Push-Location $repo
  try {
    & cargo build --release -p fidim-cli
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }
    Write-Host "== building GUI" -ForegroundColor Cyan
    Push-Location (Join-Path $repo 'ui')
    try {
      # A running GUI holds the exe open and the build cannot replace it.
      Get-Process llama-fidim -ErrorAction SilentlyContinue | Stop-Process -Force
      & pnpm tauri build --no-bundle
      if ($LASTEXITCODE -ne 0) { throw "tauri build failed ($LASTEXITCODE)" }
    } finally { Pop-Location }
  } finally {
    Pop-Location
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
  }
}

# fidim-dg.exe is the DiffusionGemma server; without it every diffusion
# launch blocks at pre-flight, so -NoBuild must not install a set without it.
foreach ($exe in 'fidim.exe', 'fidim-dg.exe', 'llama-fidim.exe') {
  if (-not (Test-Path (Join-Path $release $exe))) { throw "missing $release\$exe - build first" }
}

# The tool was called llamactl until 2026-09-06: retire that install so two
# copies never fight over the same servers.
$legacy = Join-Path $env:LOCALAPPDATA 'Programs\llamactl'
if (Test-Path $legacy) {
  Write-Host "== retiring the old llamactl install at $legacy" -ForegroundColor Cyan
  Get-Process llamactl-ui -ErrorAction SilentlyContinue | Stop-Process -Force
  Get-CimInstance Win32_Process -Filter "Name='llamactl.exe'" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Milliseconds 500
  Remove-Item -Recurse -Force $legacy -ErrorAction SilentlyContinue
  $oldLnk = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\llamactl.lnk'
  if (Test-Path $oldLnk) { Remove-Item $oldLnk -Force }
  Write-Host "   (a taskbar pin to the old exe must be re-pinned by hand)"
}

Write-Host "== installing to $dest" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $dest | Out-Null
# The installed GUI may be running too, from here or from the old folder.
Get-Process llama-fidim -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like "$dest\*" -or $_.Path -like "$oldDest\*" } | Stop-Process -Force
# A running fidim-dg.exe is a live diffusion server holding a model in VRAM:
# never stop it. Windows lets a running image be renamed but not replaced, so
# move it aside; it keeps serving (same pid, same image name, so `fidim stop`
# still finds it) and the next install deletes the old copy once it exits.
# Done before the keep-alive helpers are stopped: if the rename fails, the
# script throws with every llama-server run still kept alive.
$dg = Join-Path $dest 'fidim-dg.exe'
Get-ChildItem $dest -Filter 'fidim-dg.exe.old-*' -ErrorAction SilentlyContinue |
  Remove-Item -Force -ErrorAction SilentlyContinue
if (Get-Process fidim-dg -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $dg }) {
  $aside = 'fidim-dg.exe.old-' + (Get-Date -Format yyyyMMddHHmmss)
  try { Rename-Item $dg $aside -ErrorAction Stop }
  catch {
    throw "a running diffusion server holds $dg and it could not be renamed aside: $($_.Exception.Message). Stop the diffusion profile (fidim stop <id>) and run the installer again."
  }
  Write-Host "== moved the running fidim-dg.exe aside as $aside (its server keeps running)"
}
# Keep-alive helpers hold fidim.exe open; stop them, remember which runs
# had one, and restart them afterwards, even when a copy fails. (Stop-Process,
# never Git-Bash taskkill: MSYS mangles /PID into a path.)
$helpers = Get-CimInstance Win32_Process -Filter "Name='fidim.exe'" |
  Where-Object { $_.CommandLine -like '*keepalive*' }
$restart = @()
foreach ($h in $helpers) {
  if ($h.CommandLine -match '--port (\d+).*--interval (\d+).*--server-pid (\d+)') {
    $restart += @{ port = $Matches[1]; interval = $Matches[2]; serverPid = $Matches[3] }
  }
  Stop-Process -Id $h.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500
try {
  foreach ($exe in 'fidim.exe', 'fidim-dg.exe', 'llama-fidim.exe') {
    try { Copy-Item (Join-Path $release $exe) (Join-Path $dest $exe) -Force -ErrorAction Stop }
    catch { throw "could not copy $exe into ${dest}: $($_.Exception.Message). A running copy may hold it open." }
  }
} finally {
  # From whichever fidim.exe is in place now (the new one, or the old one
  # when its copy failed).
  foreach ($r in $restart) {
    $p = Start-Process -FilePath (Join-Path $dest 'fidim.exe') -WindowStyle Hidden -PassThru `
      -ArgumentList @('keepalive', '--host', '127.0.0.1', '--port', $r.port, '--interval', $r.interval, '--server-pid', $r.serverPid)
    # Point the run state at the new helper pid so `stop` still kills it.
    Get-ChildItem "$env:USERPROFILE\.fidim\runs\*-$($r.port).json" -ErrorAction SilentlyContinue | ForEach-Object {
      $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
      if ("$($j.pid)" -eq $r.serverPid) {
        $j | Add-Member -NotePropertyName keepalive_pid -NotePropertyValue $p.Id -Force
        # No BOM: the tool reads these with serde_json.
        [IO.File]::WriteAllText($_.FullName, ($j | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))
      }
    }
    Write-Host ("== restarted keep-alive for port {0} (pid {1})" -f $r.port, $p.Id)
  }
}

# Retire the old folder now that the keep-alive helpers run from the new
# one: a copy still running there (a diffusion server, a terminal's
# `fidim live`) is moved aside as above, and the folder goes once nothing
# is left in it.
$retired = $false
if (Test-Path $oldDest) {
  Write-Host "== retiring the old install at $oldDest" -ForegroundColor Cyan
  foreach ($exe in 'fidim-dg.exe', 'fidim.exe', 'llama-fidim.exe') { Remove-OrMoveAside $oldDest $exe }
  if (-not (Get-ChildItem $oldDest -Force -ErrorAction SilentlyContinue)) {
    Remove-Item $oldDest -Force -ErrorAction SilentlyContinue
  } else {
    Write-Host "   $oldDest still holds files; the next install removes what is left of Llama FIDIM there"
  }
  $retired = $true
}
# A user PATH that named the old folder names the new one instead.
$fidim = Join-Path $dest 'fidim.exe'
$old = & $fidim --json path --dir $oldDest status | Out-String
if ($LASTEXITCODE -eq 0 -and $old.Trim() -and ($old | ConvertFrom-Json).status.user) {
  Write-Host "== the user PATH named $oldDest; moving it to $dest" -ForegroundColor Cyan
  & $fidim path --dir $oldDest remove
  if ($LASTEXITCODE -ne 0) { throw "fidim path remove failed ($LASTEXITCODE)" }
  & $fidim path add
  if ($LASTEXITCODE -ne 0) { throw "fidim path add failed ($LASTEXITCODE)" }
}

$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$lnk = Join-Path $startMenu 'Llama FIDIM.lnk'
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = Join-Path $dest 'llama-fidim.exe'
$sc.WorkingDirectory = $dest
$sc.IconLocation = (Join-Path $dest 'llama-fidim.exe') + ',0'
$sc.Description = 'Llama FIDIM - llama.cpp servers on AMD, your way'
$sc.Save()
Write-Host "== Start Menu shortcut: $lnk" -ForegroundColor Cyan
if ($retired) { Write-Host "   (a taskbar pin to the old folder must be re-pinned by hand)" }

if ($AddToPath) {
  # fidim edits the registry value itself: it keeps the value's type and
  # every other entry, and tells Explorer so new shells see the change.
  Write-Host "== user PATH" -ForegroundColor Cyan
  & $fidim path add
  if ($LASTEXITCODE -ne 0) { throw "fidim path add failed ($LASTEXITCODE)" }
}

$ver = & $fidim --version
Write-Host ""
Write-Host "installed: $ver"
Write-Host "  GUI : $dest\llama-fidim.exe"
Write-Host "  CLI : $dest\fidim.exe"
Write-Host "  DG  : $dest\fidim-dg.exe (the DiffusionGemma server the app starts)"
Write-Host "  to pin: Start Menu -> right-click 'Llama FIDIM' -> Pin to taskbar"
