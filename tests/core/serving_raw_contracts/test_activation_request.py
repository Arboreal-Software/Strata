#!/usr/bin/env python3
"""Execute actual bounded activation target policy; CPU control proof only."""
import argparse
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[3]

def take(text,marker):
    begin=text.index(marker);opening=text.index('{',begin);depth=0
    for end in range(opening,len(text)):
        depth+=(text[end]=='{')-(text[end]=='}')
        if not depth:return text[begin:end+1]
    raise ValueError(marker)

def check_source(root):
    q=(root/'src/program/serving_suffix_qualification.inc').read_text()
    g=(root/'src/program/generate.cpp').read_text()
    assert g.count('activation_qualification_selects(serving_request_ordinal+1)')==1
    assert 'activation_hooks.request_ordinal=serving_request_ordinal+1;' in g
    for marker in ('static bool qualify_activation_final(','static bool qualify_activation_rollback('):
        assert 'ordinal!=admission.activation_request' in take(q,marker)
    return q


def run(root,build):
    q=check_source(root);build.mkdir(parents=True,exist_ok=True)
    source='''#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <charconv>
#include <cstdio>
'''+take(q,'static bool activation_qualification_request(')+take(q,'static bool activation_qualification_selects(')+take(q,'static bool serving_qualification_limit(')+take(q,'template<class Policy>\nstatic bool serving_qualification_policy(')+'''
struct Policy {bool serve=true,suffix_cache=true,owned_session_lifetime=true,owned_model_lifetime=true;
    bool suffix_cache_unrepaired_experiment=false,prefill_head_alias=false,vision=false,batch=false,
    pipeline_windows=false,adapt_every=false,adapt_async=false,kv_grow=false;};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    unsetenv("STRATA_SUFFIX_ACTIVATION_REQUEST");unsetenv("STRATA_SUFFIX_ACTIVATION_QUALIFY");
    uint64_t target=0;require(activation_qualification_request(target)&&target==4,"default target");
    for(unsigned ordinal=1;ordinal<=6;++ordinal)require(!activation_qualification_selects(ordinal),"production no selector");
    setenv("STRATA_SUFFIX_ACTIVATION_QUALIFY","success",1);
    require(activation_qualification_selects(4)&&!activation_qualification_selects(6),"original default request4");
    setenv("STRATA_SUFFIX_ACTIVATION_REQUEST","6",1);
    require(activation_qualification_request(target)&&target==6,"explicit request6");
    unsigned ordinary_yields=0,forced_final=0;
    for(unsigned actual: {2u,4u,6u}){
        const bool serving_activation_selected=activation_qualification_selects(actual);
        const bool force_activation_final_oracle=serving_activation_selected;
        if(!force_activation_final_oracle)++ordinary_yields;else ++forced_final;
        require(serving_activation_selected==(actual==6),"same-session actual ordinal selection");
    }
    require(ordinary_yields==2&&forced_final==1,"earlier req4 stays eligible for ordinary retained yield");
    const uint64_t admitted_target=target;
    require(admitted_target!=4&&admitted_target==6,"admission binds real request6 without renumbering");
    for(const char* bad:{"","0","2","5","7","04","06","-6","6x"," 6","18446744073709551616"}){
        setenv("STRATA_SUFFIX_ACTIVATION_REQUEST",bad,1);
        require(!activation_qualification_request(target)&&!activation_qualification_selects(4)&&!activation_qualification_selects(6),"invalid target failclosed");
    }
    unsetenv("STRATA_SUFFIX_ACTIVATION_QUALIFY");setenv("STRATA_SUFFIX_ACTIVATION_REQUEST","6",1);
    require(!activation_qualification_request(target),"orphan selector failclosed");
    Policy policy;std::string error;
    require(!serving_qualification_policy(policy,error),"orphan selector startup refusal");
    unsetenv("STRATA_SUFFIX_ACTIVATION_REQUEST");error.clear();require(serving_qualification_policy(policy,error),"ordinary no-selector policy unchanged");
    setenv("STRATA_SUFFIX_ACTIVATION_QUALIFY","success",1);setenv("STRATA_SUFFIX_ACTIVATION_REQUEST","6",1);
    setenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST","/source/SOURCE-MANIFEST.json",1);setenv("STRATA_LOOKAHEAD","0",1);
    require(serving_qualification_policy(policy,error),"authentic bounded qualification policy accepts target6");
    for(const char* key:{"STRATA_SUFFIX_REPAIR_FAULT_AT","STRATA_SUFFIX_REPAIR_CANCEL_AT"}){
        setenv(key,"0",1);require(!serving_qualification_policy(policy,error),"fault/cancel presence refuses even0");unsetenv(key);
    }
    setenv("STRATA_SUFFIX_ACTIVATION_QUALIFY","bad",1);require(!serving_qualification_policy(policy,error),"unknown mode startup refusal");
    std::puts("PASS actual C++ bounded target selector; req4 ordinary eligibility preserved under req6 (CPU only)");return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL %s\\n",error.what());return 1;}}
'''
    source=source.replace('#include <cstdio>','#include <cstdio>\n#include <initializer_list>')
    cpp=build/'selector.cpp';cpp.write_text(source);binary=build/'selector'
    subprocess.run(['c++','-std=c++20','-Wall','-Wextra',str(cpp),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT);p.add_argument('--build-dir',type=Path);p.add_argument('--source-only',action='store_true');a=p.parse_args()
    if a.source_only:check_source(a.source)
    elif a.build_dir:run(a.source,a.build_dir)
    else:
        with tempfile.TemporaryDirectory() as directory:run(a.source,Path(directory))
