# Test workers are threads in this exact process, never forked descendants.
# subprocess.run kills and reaps the entire process on timeout, so a blocked
# actual HostGraphLaunch join cannot orphan a worker after the supervisor exits.
import subprocess,sys
try:
 p=subprocess.run([sys.argv[1]],capture_output=True,text=True,timeout=20)
except subprocess.TimeoutExpired:
 raise SystemExit('fixed fixture deadline exceeded; process killed/reaped')
print(p.stdout,end='')
print(p.stderr,end='',file=sys.stderr)
assert p.returncode==0,(p.returncode,p.stdout,p.stderr)
assert p.stdout=='PASS: actual HostGraphLaunch and chain completion extraction\n',p.stdout
