from pathlib import Path
import hashlib,json
d=Path(__file__).resolve().parent;r=d.parents[2]
for n,h in json.loads((d/'pins.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
s=(r/'src/core/remote_expert_opt.cu').read_text();a=s.index('bool RemoteExpertOpt::set_strict_retirement(');b=s.index('size_t RemoteExpertOpt::metadata_bytes()',a)
assert (d/'lifecycle.inc').read_text()==s[a:b]
for n in ['remote_expert_opt.hpp','strict_retirement.hpp']:assert (d/n).read_bytes()==(r/'include/strata/core'/n).read_bytes()
s=(r/'sycl/include/strata/core/remote_experts.hpp').read_text();a=s.index('    bool set_strict_retirement(');b=s.index('\n    /// Initialise',a)
assert (d/'helper-setter.inc').read_text()==s[a:b]
