import subprocess,sys
for args in ([],['--chain-first']):
    try:
        r=subprocess.run([sys.argv[1]]+args,capture_output=True,text=True,timeout=20)
    except subprocess.TimeoutExpired:
        raise SystemExit('timeout is not expected red')
    s=r.stdout+r.stderr
    print(s,end='')
    if r.returncode!=1 or s.splitlines()!=['FAIL: serial copy error must not report completion']:
        raise SystemExit('old source did not reach exact completion assertion')
