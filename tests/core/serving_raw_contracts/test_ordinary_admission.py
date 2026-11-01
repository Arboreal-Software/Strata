#!/usr/bin/env python3
"""Compile the actual serving admission expressions with CPU-only session fixtures.

This tests admission/provenance control flow, not device completion. Negative controls
remove each safety condition from the extracted production expressions.
"""
from pathlib import Path
import subprocess
import re
import tempfile

ROOT = Path(__file__).resolve().parents[3]
PREFIX = r'''
#include "strata/core/serving_raw_metadata.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
using namespace strata::core;
namespace strata::program {
struct ServingRawSession {
    using Phase = serving_raw::Phase;
    StateCut current{};
    uint64_t capacity=1024;
    uint64_t max_tokens() const { return capacity; }
    StateCut cut() const { return current; }
    bool healthy() const { return true; }
    bool request_open() const { return false; }
    bool private_window_open() const { return false; }
    Phase phase() const { return Phase::drained; }
};
}
int main(int argc, char** argv) {
    if (argc != 2) return 90;
    std::string name=argv[1];
    struct Options { bool pipeline_windows=false,adapt_every=false,adapt_async=false,prefill_head_alias=false; } o;
    bool pipe=false; int max_new=4,req_temperature=0,req_penalty_last_n=0;
    std::vector<int> req_bias,req_imgs,live_imgs;
    std::vector<int32_t> ids{1,2,3}, live{1,2,3};
    int slot_source=-1; bool borrow=false,incoming=false,from_live=true;
    bool owned_batch_allocation=false; int admit_slot=-1;
    int64_t resume=3;
    serving_raw::TrackedState serving_tracked_state;
    serving_raw::TrackedStateFacts facts{1,live,4,true,0,{}};
    if (serving_raw::track_state_certificate(facts,1024,serving_tracked_state) != serving_raw::TrackedStateRefusal::none) return 91;
    strata::program::ServingRawSession session{serving_tracked_state.cut};
    auto* serving_raw_session=&session;
    struct Owners { std::vector<int> parked_identities() const {return {1};} } owners;
    auto* serving_repair_owners=&owners;
    bool serving_raw_fence_used_this_request=false,serving_raw_fence_running=false;
    void* serving_suffix_active=nullptr; uint64_t serving_raw_cut=1;
    bool expected=true;
    std::vector<int> consumed{1}; bool serving_read_gave_way=false,nf_hit=false;
    bool reset_expected=false;
    if(name=="reset_empty"){consumed.clear();reset_expected=true;}
    if(name=="reset_read"){serving_read_gave_way=true;reset_expected=true;}
    if(name=="reset_nonfinite"){nf_hit=true;reset_expected=true;}
    if(name=="absent"){serving_raw_session=nullptr; expected=false;}
    else if(name=="zero"){resume=0; from_live=false; serving_tracked_state.clear();}
    else if(name=="uncertified"){serving_tracked_state.clear(); expected=false;}
    else if(name=="slot"){slot_source=0; expected=false;}
    else if(name=="borrow"){borrow=true; expected=false;}
    else if(name=="incoming"){incoming=true; expected=false;}
    else if(name=="checkpoint"){from_live=false; expected=false;}
    else if(name=="different_live"){live[0]=9; expected=false;}
    else if(name=="different_cut"){session.current.id++; expected=false;}
    else if(name=="different_resume"){resume=2; expected=false;}
    else if(name=="inexact"){
        facts.state_exact=false; facts.approximation={100,window_history_digest(live)};
        if(serving_raw::track_state_certificate(facts,1024,serving_tracked_state)!=serving_raw::TrackedStateRefusal::none)return 92;
        session.current=serving_tracked_state.cut;
    }
    else if(name!="live" && name!="reset_empty" && name!="reset_read" && name!="reset_nonfinite")return 93;
'''
SUFFIX = r'''
    if(should_reset!=reset_expected)return 23;
    if(ordinary_fence_admissible!=expected)return 21;
    if(ordinary_fence_admissible && resume>0 && !ordinary_inherits_certificate)return 22;
    return 0;
}
'''


def extract(source):
    begin=source.index('            const bool ordinary_inherits_certificate =')
    end=source.index('            if (ordinary_fence_admissible) {',begin)
    reset=re.search(r'else if \((serving_read_gave_way \|\| nf_hit.*?)\) \{', source)
    if reset is None: raise RuntimeError('ordinary reset disposition missing')
    return source[begin:end]+'\nconst bool should_reset = '+reset.group(1)+';\n'


def run(directory):
    directory=Path(directory)
    body=extract((ROOT/'src/program/generate.cpp').read_text())
    variants=[('production',body,None),
              ('no_null_guard',body.replace('                serving_raw_session != nullptr &&\n                !o.pipeline_windows', '                !o.pipeline_windows'), 'absent'),
              ('no_empty_reset',body.replace(' || consumed.empty()', ''), 'reset_empty'),
              ('no_eligibility',body.replace(' && ordinary_start_certifiable',''), 'uncertified'),
              ('no_restore_guard',body.replace('slot_source < 0 && !borrow && !incoming && from_live &&',''), 'slot'),
              ('no_history_binding',body.replace('std::equal(live.begin(), live.end(), serving_tracked_state.accepted.begin())','true'), 'different_live')]
    cases=('absent','zero','uncertified','slot','borrow','incoming','checkpoint',
           'different_live','different_cut','different_resume','inexact','live',
           'reset_empty','reset_read','reset_nonfinite')
    for label,block,negative_case in variants:
        if negative_case and block == body:
            raise RuntimeError(f'{label}: negative control did not change production expressions')
        cpp=directory/(label+'.cpp'); binary=directory/label
        cpp.write_text(PREFIX+block+SUFFIX)
        subprocess.run(['c++','-std=c++20','-O0','-I'+str(ROOT/'include'),str(cpp),
                        str(ROOT/'src/core/window_hash.cpp'),'-o',str(binary)],check=True)
        for case in ((negative_case,) if negative_case else cases):
            result=subprocess.run([str(binary),case],capture_output=True,timeout=10)
            if (result.returncode==0) == bool(negative_case):
                raise RuntimeError(f'{label}/{case}: unexpected exit {result.returncode}')
    print('PASS actual ordinary admission: null owner, uncertified inheritance, all restore sources, cut/history binding, reset dispositions; negative controls fail')


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='strata-ordinary-admission-') as directory:
        run(directory)
