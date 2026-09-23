# Start one self-limiting spinner pinned to each free logical processor.
# Writes the PIDs so the caller can kill them even if this script dies.
param([int]$Seconds = 600)
$masks = @(0x10000, 0x20000, 0x40000, 0x80000)   # logical processors 16-19
$ids = @()
foreach ($m in $masks) {
  $p = Start-Process python -ArgumentList 'C:\llmpc\spin.py', "$Seconds" -PassThru -WindowStyle Hidden
  Start-Sleep -Milliseconds 400
  if (-not $p.HasExited) {
    try { $p.ProcessorAffinity = [IntPtr]$m } catch { Write-Output "affinity failed for $($p.Id)" }
    $ids += $p.Id
  }
}
$ids -join ',' | Out-File -FilePath C:\llmpc\spin.pids -Encoding ascii
Write-Output "started $($ids.Count) spinners: $($ids -join ',')"
