# Hold a system wake lock for the duration of a benchmark campaign.
#
# WHY THIS EXISTS. On 22 Sep the 7B sweep lost seven hours inside five cells:
# normal cells take 72 to 85 seconds, those five took 10 to 155 minutes, and two
# of them recorded decode samples of 0.24 and 0.04 tok/s against a ~10 tok/s
# median. The machine was entering modern standby between invocations once the
# interactive session went idle. A wake lock owned by the session dies with the
# session; a benchmark that runs for hours unattended has to hold its own.
param([int]$Seconds = 14400)
$sig = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$k = Add-Type -MemberDefinition $sig -Name Power -Namespace Win32 -PassThru
# ES_CONTINUOUS 0x80000000 | ES_SYSTEM_REQUIRED 0x00000001
$r = $k::SetThreadExecutionState([uint32]"0x80000001")
if ($r -eq 0) { Write-Output "FAILED to set execution state"; exit 1 }
Write-Output "wake lock held for $Seconds s (pid $PID)"
Start-Sleep -Seconds $Seconds
$k::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
Write-Output "wake lock released"
