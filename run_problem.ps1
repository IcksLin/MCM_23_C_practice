param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigPath,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "E:\anaconda\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) { $PythonExe = "python" }
$ArgsList = @("-u", (Join-Path $ProjectRoot "scripts\run_with_config.py"), "--config", $ConfigPath)
if ($ValidateOnly) { $ArgsList += "--validate-only" }
& $PythonExe @ArgsList
exit $LASTEXITCODE
