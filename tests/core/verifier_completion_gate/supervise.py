# Exact process has threads only, no forked descendants; timeout kills/reaps all.
import subprocess,sys
try:
 result=subprocess.run([sys.argv[1]],capture_output=True,text=True,timeout=20)
except subprocess.TimeoutExpired:
 raise SystemExit('gate fixture timeout; exact process killed/reaped')
assert result.returncode==0,(result.returncode,result.stdout,result.stderr)
assert not result.stdout and not result.stderr,(result.stdout,result.stderr)
print('completion gate fixture PASS')
