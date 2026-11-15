from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[3]
d=Path(__file__).resolve().parent
s=(root/'src/core/verify.cpp').read_text()
a=s.index('    if (chain) {',s.index('// (#646 staged the next stage'))
b=s.index('    progress_at("verify window: waiting for the GPU to finish',a)
assert (d/'fixed.inc').read_text()==s[a:b], 'actual complete chain branch changed'
for name,digest in json.loads((d/'pins.json').read_text()).items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('actual chain branch and HostGraphLaunch source freshness PASS')
