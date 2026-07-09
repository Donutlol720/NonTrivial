$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
  $candidates = @()

  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { $candidates += $cmd.Source }

  $cmd = Get-Command py -ErrorAction SilentlyContinue
  if ($cmd) { $candidates += $cmd.Source }

  $cmd = Get-Command python3 -ErrorAction SilentlyContinue
  if ($cmd) { $candidates += $cmd.Source }

  $known = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "C:\Program Files\Python312\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python310\python.exe"
  )
  foreach ($p in $known) {
    if ($p -and (Test-Path $p)) { $candidates += $p }
  }

  foreach ($p in $candidates) {
    if (-not $p) { continue }
    if ($p -match "\\WindowsApps\\python(?:3)?\.exe$") { continue }
    if (Test-Path $p) { return $p }
  }

  return $null
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Resolve-PythonExe
if (-not $pythonExe) {
  Write-Host "No usable Python executable found."
  Write-Host "Install Python (or activate your conda env), then rerun:"
  Write-Host "  .\\run_pipeline.ps1"
  exit 1
}

Write-Host "Using Python:" $pythonExe
& $pythonExe (Join-Path $repoRoot "src\\run\\run_pipeline.py") @args
exit $LASTEXITCODE

