"""Exact previous notify_one regression; timeout/crash/sanitizer is never red."""
import subprocess,sys
r=subprocess.run([sys.argv[1]],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=12)
expected='ASSERT: pending submission must wake worker while deadline waiter exists\n'
if r.returncode!=70 or r.stdout or r.stderr!=expected:
 raise SystemExit('unexpected control outcome: '+repr((r.returncode,r.stdout,r.stderr)))
print('PASS precise notify_one failure control (status70, exact assertion, no sanitizer output)')
