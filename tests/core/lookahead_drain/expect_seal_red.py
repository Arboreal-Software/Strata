"""Only removing the submit seal must hit the idle-guard assertion after cleanup."""
import subprocess,sys
r=subprocess.run([sys.argv[1]],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=12)
expected='ASSERT: idle retained guard must reject submission\n'
if r.returncode!=70 or r.stdout or r.stderr!=expected:
 raise SystemExit('unexpected seal control outcome: '+repr((r.returncode,r.stdout,r.stderr)))
print('PASS precise missing-submit-seal control (status70, exact assertion, no sanitizer output)')
