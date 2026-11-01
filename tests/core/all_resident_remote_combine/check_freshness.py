from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[3];d=Path(__file__).resolve().parent
for name,digest in json.loads((d/'pins.json').read_text()).items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
s=(r/'src/core/verify.cpp').read_text()
a=s.index('        // The zero-doorbell graph never resets/populates remote sums.')
b=s.index('        stamp(l, 24, grp);',a)
assert (d/'fixed.inc').read_text()==s[a:b]
s=(r/'src/core/remote_expert_opt.cu').read_text();actual=''
for start,end in [('void RemoteExpertOpt::begin(', 'void RemoteExpertOpt::prepare('),('void RemoteExpertOpt::accumulate(', 'void RemoteExpertOpt::copy_rows('),('void RemoteExpertOpt::combine(', 'bool RemoteExpertOpt::owns(')]:
 actual+=s[s.index(start):s.index(end,s.index(start))]
assert (d/'methods.inc').read_text()==actual
h=(r/'include/strata/core/verify.hpp').read_text();a=h.index('    bool ar_on() const');b=h.index('\n',a)
assert (d/'ar_on.inc').read_text()==h[a:b]+'\n'
print('actual remote producer/combine and callsite freshness PASS')
