# Run from Windows PowerShell; moves the entire Linux disk, including Docker, to D:.
$ErrorActionPreference = 'Stop'
$wsl = "$env:WINDIR\System32\wsl.exe"
& $wsl --shutdown
if ($LASTEXITCODE -ne 0) { throw 'Could not stop WSL' }
Start-Sleep -Seconds 3
& $wsl --manage Ubuntu --move 'D:\WSL\Ubuntu'
if ($LASTEXITCODE -ne 0) { throw 'WSL move failed; inspect this log before proceeding' }
& $wsl --distribution Ubuntu --user semyon --exec /bin/true
if ($LASTEXITCODE -ne 0) { throw 'Could not restart Ubuntu' }
Write-Output 'Ubuntu moved to D:\WSL\Ubuntu and restarted.'
