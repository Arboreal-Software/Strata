from pathlib import Path
import hashlib,json,sys
d=Path(__file__).resolve().parent;root=d.parents[2]
for n,h in json.loads((d/'pins.json').read_text()).items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
s=(root/'src/core/verify.cpp').read_text();a=s.index('bool Verifier::prepare_serial_completion_gate(');b=s.index('bool Verifier::set_strict_retirement(',a)
assert (d/'factory.inc').read_text()==s[a:b]
h=(root/'include/strata/core/verify.hpp').read_text();a=h.index('    void mark_completion_unhealthy()');b=h.index('    bool strict_retirement_',a)
assert (d/'hooks.inc').read_text()==h[a:b]
assert 'bool Verifier::release_gpu_waits(int timeout_ms) {\n    mark_completion_unhealthy();' in s
assert 'terminal_copy_query_failed_.store(true' not in s
assert s.count('latch_copy_query_failure();')==2
assert 'Verifier::~Verifier() {\n    mark_completion_unhealthy();' in s

assert "p->retirement_init_started_" in (d/"factory.inc").read_text()
print("actual gate/factory/monotonic hooks freshness PASS")
