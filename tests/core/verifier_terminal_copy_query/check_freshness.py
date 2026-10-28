"""Exact extraction freshness; entry adapters prove first guards, not full large methods."""
from pathlib import Path
import json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def extract(text,signature):
    assert text.count(signature)==1, signature
    start=text.index(signature); cursor=text.index('{',start)+1; depth=1
    while depth:
        depth += (text[cursor]=='{')-(text[cursor]=='}'); cursor+=1
    return text[start:cursor]+'\n'
verify=(ROOT/'src/core/verify.cpp').read_text()
drain=(ROOT/'src/core/verify_drain.cpp').read_text()
header=(ROOT/'include/strata/core/verify.hpp').read_text()
actual=''.join(extract(verify,'bool Verifier::'+n+'(') for n in ('done','pl_finish','prestage','run_slots','commit_slots'))
actual+=extract(drain,'bool Verifier::wait_commit(')+extract(drain,'Verifier::SnapshotDrain Verifier::poll_snapshot_drain(')
assert actual==(HERE/'fixed-methods.inc').read_text(), 'stale complete methods'
a=header.index('    std::atomic<bool> terminal_copy_query_failed_')
b=header.index('    bool capture_commit(',a)
assert header[a:b]==(HERE/'latch.inc').read_text(), 'stale latch/helper'
start=actual.index('        const cudaError_t copy_status = cudaStreamQuery(copy_);')
end=actual.index('        copy_used_ = false;',start)
before=actual[:start]+'        if (cudaStreamQuery(copy_) == cudaErrorNotReady) return false;\n'+actual[end:]
assert before==(HERE/'before-methods.inc').read_text(), 'unexpected old mutation'
entries=[]; parts=[]
for m in re.finditer(r'^(bool|int) Verifier::(\w+)\([^;]*?\)\s*(?:const\s*)?\{\n([^\n]+)',verify,re.M):
    ret,name,line=m.groups()
    if 'refuse_copy_query_failure' not in line and 'terminal_copy_query_failed_.load' not in line: continue
    entries.append([ret,name,line])
    parts.append(f'    {ret} entry_{name}(std::string& err) {{\n{line}\n        ++effects; return '+('true' if ret=='bool' else '1')+';\n    }\n')
parts.append('    bool check_entry_guards(std::string& err) {\n        return '+'\n            && '.join(f'entry_{name}(err) == '+('false' if ret=='bool' else '-1') for ret,name,line in entries)+';\n    }\n')
assert entries==json.loads((HERE/'entry-guards.json').read_text()), 'entry guard membership/position changed'
assert ''.join(parts)==(HERE/'entry-guards.inc').read_text(), 'stale first-guard adapters'
for mode in ('before','fixed'):
    assert (HERE/(mode+'.cpp')).read_text()=='#include "prefix.inc"\n#include "'+mode+'-methods.inc"\n#include "cases.inc"\n'
print('complete methods, latch and actual first-guard adapters current')
