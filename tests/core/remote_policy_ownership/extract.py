#!/usr/bin/env python3
"""Generate a tiny CPU fixture from the CURRENT complete production methods.
Not a compiler or test runner. No device/model input is read.
"""
from pathlib import Path
import argparse
import hashlib
import json
p=argparse.ArgumentParser()
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
base=Path(__file__).resolve().parent
hp=a.source_root/'include/strata/core/remote_experts.hpp'
cp=a.source_root/'src/core/remote_experts.cpp'
h=hp.read_text(); c=cp.read_text()
start=h.index('class RemoteExperts {')
end=h.index('\n};\n\n/// STRATA_PREFILL_HELPER_SOURCE',start)+4
header=h[start:end]
start=c.index('constexpr int64_t H =')
end=c.index('} // namespace',start)
helpers=c[start:end]
start=c.index('// BEGIN owned remote routing admission')
end=c.index('// END owned remote routing admission',start)+len('// END owned remote routing admission')
methods=c[start:end]
def region(first,last):
    lo=c.index(first);return c[lo:c.index(last,lo)]
methods+='\nRemoteExperts::~RemoteExperts() { close(); }\n'
methods+=region('void RemoteExperts::close()', 'bool RemoteExperts::open(')
methods+=region('bool RemoteExperts::begin_impl(', 'bool RemotePrefillSource::copy_blob(')
text=(base/'prefix.inc').read_text()
text+='\nnamespace strata::core {\n'+header+'\nnamespace {\n'+helpers+'\n}\n'+methods+'\n}\n'
text+=(base/'cases.inc').read_text()
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(text)
a.output.with_suffix('.sources.json').write_text(json.dumps({str(f.relative_to(a.source_root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (hp,cp)},indent=2)+'\n')
