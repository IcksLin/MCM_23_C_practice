param([Parameter(Mandatory=$true)][string]$ConfigPath,[switch]$ValidateOnly)
& (Join-Path $PSScriptRoot "..\run_problem.ps1") -ConfigPath $ConfigPath -ValidateOnly:$ValidateOnly
exit $LASTEXITCODE
