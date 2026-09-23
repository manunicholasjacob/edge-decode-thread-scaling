if (Test-Path C:\llmpc\spin.pids) {
  foreach ($id in (Get-Content C:\llmpc\spin.pids) -split ',') {
    if ($id) { try { Stop-Process -Id ([int]$id) -Force -ErrorAction Stop } catch {} }
  }
  Remove-Item C:\llmpc\spin.pids -ErrorAction SilentlyContinue
}
Write-Output ("python remaining: " + @(Get-Process python -ErrorAction SilentlyContinue).Count)
