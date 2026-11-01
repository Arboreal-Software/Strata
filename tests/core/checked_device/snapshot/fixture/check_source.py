import pathlib,json,hashlib
r=pathlib.Path(__file__).resolve().parent.parent
for f,h in json.loads((r/'source-pins.json').read_text()).items():
 assert hashlib.sha256((r/'source'/f).read_bytes()).hexdigest()==h,f
s=(r/'source/src/core/verify.cpp').read_text()
a=s.index('bool Verifier::release_gpu_waits(');b=s.index('void Verifier::trace_ev(',a)
assert (r/'fixture/release.inc').read_text()==s[a:b]
print('current integrated checked-device/release extraction freshness PASS')
