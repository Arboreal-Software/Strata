"""Prepared freshness checker; source only until separately authorized execution."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
root=here.parents[2]
def method(text,signature):
    begin=text.index(signature);cursor=text.index('{',begin)+1;depth=1
    while depth:
        depth+=(text[cursor]=='{')-(text[cursor]=='}');cursor+=1
    return text[begin:cursor]+'\n'
vmm=(root/'src/core/vmm.cpp').read_text()
batch=(root/'src/core/vmm_managed.cpp').read_text()
pool=(root/'src/core/owned_qsa_elastic.cpp').read_text()
header=(root/'include/strata/core/owned_qsa_elastic.hpp').read_text()
a=batch.index('namespace {');b=batch.index('bool DetachedVmmChunk::allocate',a)
expected={
 'batch-methods.inc':batch[a:b]+method(batch,'bool PreparedVmmAllocationBatch::host_bound(')+method(batch,'bool PreparedVmmAllocationBatch::prepare('),
 'range-methods.inc':method(vmm,'bool VmmRange::managed_host_bound(')+method(vmm,'void VmmRange::release_managed_metadata()')+method(vmm,'bool VmmRange::prepare_managed_metadata('),
 'owned-cleanup.inc':method(pool,'bool OwnedQsaElasticPool::retire_checked()')+method(pool,'OwnedQsaElasticPool::~OwnedQsaElasticPool()'),
 'host-release.inc':method(vmm,'void VmmRange::release() { if(retirement_generation_)'),
 'chunk-methods.inc':method(vmm,'DetachedVmmChunk::DetachedVmmChunk(DetachedVmmChunk&&')+method(vmm,'DetachedVmmChunk::~DetachedVmmChunk()'),
}
a=header.index('class OwnedQsaElasticPool final:');b=header.index('\n};',a)+3
expected['owned-class.inc']=header[a:b]+'\n'
a=pool.index('struct OwnedQsaElasticPool::Storage');b=pool.index('const QsaElasticPlan& OwnedQsaElasticPool::plan()',a)
expected['owned-methods.inc']=pool[a:b]
for name,text in expected.items():
    if (here/name).read_text()!=text:raise SystemExit('stale actual extract: '+name)
for name,term in [('range-methods','!lease->canonical_owner(lease)||'),('batch-methods','!physical->canonical_owner(physical)||!metadata->canonical_owner(metadata)||'),('owned-methods','!l->canonical_owner(l)||')]:
    fixed=expected[name+'.inc']
    if fixed.count(term)!=1:raise SystemExit('canonical counterfactual no longer singular: '+name)
    if (here/(name+'-no-canonical.inc')).read_text()!=fixed.replace(term,'',1):raise SystemExit('unexpected counterfactual: '+name)
for name,digest in json.loads((here/'source-pins.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise SystemExit('changed source pin: '+name)
print('actual canonical host admission extracts and single-guard controls current')
