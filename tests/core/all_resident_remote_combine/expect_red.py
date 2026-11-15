import subprocess,sys
p=subprocess.run([sys.argv[1]],capture_output=True,text=True,timeout=15)
assert p.returncode==86,(p.returncode,p.stdout,p.stderr)
assert p.stderr=='ASSERT: all-resident stage must not add stale remote sum\n',p.stderr
print('PASS: exact stale remote sum expected red')
