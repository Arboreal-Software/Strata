import pathlib,json,hashlib
r=pathlib.Path(__file__).resolve().parent.parent
for tree,pins in [('before','base-pins.json'),('source','source-pins.json')]:
 for f,h in json.loads((r/pins).read_text()).items():
  assert hashlib.sha256((r/tree/f).read_bytes()).hexdigest()==h,(tree,f)
s=(r/'before/src/core/verify.cpp').read_text()
s=s.replace('stats_on(int dev)', 'stats_on(int dev, bool checked = false)').replace('const OnDevice on(dev);','const OnDevice on(dev, checked);').replace('route_resident_cfg().stats_on(device_)','route_resident_cfg().stats_on(device_, strict_retirement_)').replace('const OnDevice on_device(device_);','const OnDevice on_device(device_, strict_retirement_);').replace('const OnDevice back(device_);','const OnDevice back(device_, strict_retirement_);').replace('const OnDevice on_next(next_->device_);','const OnDevice on_next(next_->device_, strict_retirement_ || next_->strict_retirement_);')
a=s.index('bool Verifier::run(int T,');b=s.index('    using namespace strata::kernels;',a)
s=s[:b]+'''    // A new prelaunch thread uses the child's own policy. Refuse a mixed
    // chain before capture/launch rather than silently downgrading its scopes.
    if (strict_retirement_ && next_ != nullptr && !next_->strict_retirement_)
        retirement_failed("Verifier unchecked child device scope");
'''+s[b:]
a=s.index('bool Verifier::release_gpu_waits(');b=s.index('    const OnDevice on_device(device_, strict_retirement_);',a)
s=s[:b]+'''    // An unused strict verifier may still participate in no-work cleanup.
    // Never treat a partially initialized/stream-owning object as this case.
    if (strict_retirement_ && !retirement_init_started_ && device_ < 0 && !cs_ && !copy_) {
        trace_ev("RELEASE-DRAINED", -1, -1, 0);
        return true;
    }
'''+s[b:]
assert s==(r/'source/src/core/verify.cpp').read_text()
# Full equality above preserves the uncommitted chain launch repair and all
# unrelated source, not merely a hand-picked token from that repair.
assert (r/'before/include/strata/core/verify.hpp').read_bytes()==(r/'source/include/strata/core/verify.hpp').read_bytes()

s=(r/'source/src/core/verify.cpp').read_text()
a=s.index('bool Verifier::release_gpu_waits(');b=s.index('void Verifier::trace_ev(',a)
assert (r/'fixture/release.inc').read_text()==s[a:b]
