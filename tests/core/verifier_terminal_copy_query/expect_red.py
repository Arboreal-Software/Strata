"""Only assertion 86 with exact diagnostic qualifies as the old defect."""
import subprocess,sys
try:
    result=subprocess.run([sys.argv[1],'--copy-error'],capture_output=True,text=True,timeout=20)
except subprocess.TimeoutExpired:
    raise SystemExit('timeout is not expected red')
expected='FAIL: copy query error must not report pipeline completion\n'
if result.returncode!=86 or result.stdout or result.stderr!=expected:
    raise SystemExit('unexpected old result: '+repr((result.returncode,result.stdout,result.stderr)))
print('old copy-query false completion reproduced precisely')
