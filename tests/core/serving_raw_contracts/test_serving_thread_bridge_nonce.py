#!/usr/bin/env python3
"""Run the actual OS nonce producer in two fresh CPU-only subprocesses."""
import json
from pathlib import Path
import subprocess
import sys
binary=Path(sys.argv[1]).resolve()
values=[]
for _ in range(2):
    result=subprocess.run([str(binary),'--os-epoch'],capture_output=True,text=True,timeout=10)
    if result.returncode!=0:raise RuntimeError('actual OS epoch producer failed: '+result.stderr)
    text=result.stdout.strip()
    if not text.isascii() or not text.isdecimal() or not 0<int(text)<1<<64:
        raise RuntimeError('invalid actual OS epoch output')
    values.append(int(text))
if values[0]==values[1]:raise RuntimeError('actual fresh-process scheduler nonces collided')
print(json.dumps(dict(actual_processes=2,producer='Linux getrandom GRND_NONBLOCK',epochs=values,
                     distinct_observed=True,global_uniqueness_proven=False,native_gpu_qualified=False)))
