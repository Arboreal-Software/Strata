import subprocess,sys
p=subprocess.run([sys.argv[1]],capture_output=True,text=True,timeout=20)
assert p.returncode==86, (p.returncode,p.stdout,p.stderr)
assert p.stderr=='ASSERT: late asynchronous chain launch error must refuse completion\n',p.stderr
print('PASS: exact old unchecked chain-launch completion assertion')
