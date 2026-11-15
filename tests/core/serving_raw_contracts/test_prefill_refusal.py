#!/usr/bin/env python3
"""Extract real paid-init policy and refusal continuation; CPU-only recording RAW adapter."""
import argparse
from pathlib import Path
import subprocess
from test_serving_thread_bridge_source import body


def actual(source, marker):
    at = source.index(marker)
    block = body(source, marker)
    return source[at:source.index(block, at)] + block


def generate(root, mutation=None):
    cpp = (root / 'src/prefill/prefill.cpp').read_text()
    driver = (root / 'src/program/generate.cpp').read_text()
    lazy = actual(cpp, 'inline double cpu_share_explicit()') + '\n' + actual(cpp, 'inline double cpu_share_env()')
    observer = actual(cpp, 'core::PrefillCpuShareObservation Prefill::cpu_share_policy() const noexcept')
    paid = actual(cpp, 'if(workspace_credit_&&cpu_share_env()!=0.0)')
    fallback = actual(driver, 'auto sfx_resume_exact =')
    tail = actual(driver, 'if(!serving_raw_session||(!serving_raw_fence_running&&!serving_ordinary_fence)')
    result_at = driver.index('std::fprintf(stderr, "strata serve: suffix cache (repaired): refused before the "')
    result_end = driver.index('sfx_release_donor();', result_at) + len('sfx_release_donor();')
    legacy_result = driver[result_at:result_end]
    cpu_at = driver.index('!cpu.explicit_resolved||cpu.mode!=', driver.index('auto sfx_prefix_policy ='))
    cpu_end = driver.index(')', cpu_at)
    cpu_reject = driver[cpu_at:cpu_end]
    if mutation == 'cold-unresolved':
        paid = ''
    if mutation == 'close-fallback':
        fallback = fallback[:fallback.index('{')] + '{(void)main;(void)prefix_cut;sfx_release_donor();}'
    return r'''
#include "strata/core/serving_raw_contracts.hpp"
#include "strata/core/window_state.hpp"
#include "strata/core/runtime_policy_observation.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstring>
using namespace strata;
static void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
namespace strata::core {
[[noreturn]] void retirement_failed(const char* m){throw std::runtime_error(m);}
struct SessionState{};struct ExpertSource{};
}
namespace strata::prefill {
bool g_share_default=false,g_share_explicit_observed=false,g_share_max_observed=false,g_share_max_presence_observed=false;
bool g_share_max_explicit=false;
double g_share_explicit_value=0;int64_t g_share_max_value=0,g_stream_min_share=0;
constexpr int64_t kStreamAllMinDefault=1024;
''' + lazy + r'''
struct Prefill {
 struct Impl {const core::ExpertSource* src=nullptr;void* pp=nullptr;bool cpu_dead=false;
 double cpu_share_now=.5;bool cpu_gate=false;int cpu_nratio=0;bool cpu_pend=false;int pend_wall=0;};
 std::unique_ptr<Impl> impl_=std::make_unique<Impl>();void* cpu_pool_=nullptr;
 std::shared_ptr<const void> workspace_credit_;
 core::PrefillCpuShareObservation cpu_share_policy()const noexcept;
 bool init_paid(std::string& err){
''' + paid + r'''
  return true;
 }
};
''' + observer + r'''
}
namespace strata::program {
struct ServingRawSession {
 using Phase=core::serving_raw::Phase;
 core::serving_raw::Lifecycle lifecycle;core::StateCut state{7,2692,2692,2692,0,{}};
 bool healthy_=true,fail_begin=false;int begins=0,closes=0;
 ServingRawSession(){
 // A genuinely initialized MAIN transaction is idle; Lifecycle defaults absent.
 lifecycle.set_observed_phase(Phase::idle);
 require(lifecycle.apply(core::serving_raw::Lifecycle::Event::open_request),"initial open");
 require(lifecycle.apply(core::serving_raw::Lifecycle::Event::begin_work),"initial begin");
 require(lifecycle.apply(core::serving_raw::Lifecycle::Event::drain_work),"actual completed prefix");}
 bool healthy()const{return healthy_;}bool request_open()const{return lifecycle.request_open();}
 Phase phase()const{return lifecycle.phase();}const core::StateCut& cut()const{return state;}
 bool begin_work(std::string&){if(fail_begin)return false;if(!lifecycle.apply(core::serving_raw::Lifecycle::Event::begin_work))return false;++begins;return true;}
 bool close_request(std::string&){if(!lifecycle.apply(core::serving_raw::Lifecycle::Event::close_request))return false;++closes;return true;}
};
}
static void run_fallback(const std::string& mode){
 bool serving_threads=mode!="legacy";
 auto serving_raw_session=std::make_shared<program::ServingRawSession>();
 auto original=serving_raw_session;auto prefix_cut=original->cut();
 bool serving_raw_fence_running=false,serving_ordinary_fence=false,serving_start_certified_exact=false;
 bool serving_suffix_active=false;uint64_t serving_raw_cut=prefix_cut.id;
 core::BundleHandle serving_donor;core::StateCut serving_donor_cut{};
 auto sfx_release_donor=[&](){std::string e;require(serving_raw_session->close_request(e),"legacy release closes only drained fence");};
''' + fallback + r''';
 if(mode=="result"){
 int64_t sfx_repair_at=2692,serving_suffix_boundary=2692;
 struct {std::string detail="genuine refusal";} result;
''' + legacy_result + r'''
 require(serving_threads&&!original->request_open()&&original->closes==1&&original->begins==0&&
 sfx_repair_at==-1&&serving_suffix_boundary==-1,"legacy result remains release even with scheduler enabled");return;
 }
 if(mode=="health")original->healthy_=false;
 if(mode=="phase")original->lifecycle.set_observed_phase(program::ServingRawSession::Phase::capturing);
 if(mode=="ordinary")serving_ordinary_fence=true;
 if(mode=="suffix")serving_suffix_active=true;
 if(mode=="running")serving_raw_fence_running=true;
 if(mode=="begin-failure")original->fail_begin=true;
 if(mode=="history")original->state.history[0]=1;
 if(mode=="watermark")++original->state.output_watermark;
 if(mode=="accepted"){--prefix_cut.accepted;original->state=prefix_cut;}
 if(mode=="exact"){--prefix_cut.exact_frontier;original->state=prefix_cut;}
 if(mode=="swapped")serving_raw_session=std::make_shared<program::ServingRawSession>();
 if(mode=="closed"){std::string e;require(original->close_request(e),"close negative");}
 if(mode=="cut")++prefix_cut.id;
 const bool negative=mode=="swapped"||mode=="closed"||mode=="cut"||mode=="health"||mode=="phase"||mode=="ordinary"||mode=="suffix"||mode=="running"||mode=="begin-failure"||mode=="history"||mode=="watermark"||mode=="accepted"||mode=="exact";
 bool caught=false;try{sfx_resume_exact(original,prefix_cut);}catch(const std::runtime_error&){caught=true;}
 if(negative){require(caught&&original->begins==0,"malformed refusal cannot start work");return;}
 require(!caught,"valid fallback must resume");
 if(mode=="legacy"){require(!original->request_open()&&original->closes==1&&original->begins==0,"legacy unchanged");return;}
 auto actual_tail=[&](std::string& e)->bool{
''' + tail + r'''
 return true;};
 std::string e;require(actual_tail(e),"actual managed tail requires same-request running fence");
 require(original.get()==serving_raw_session.get()&&original->begins==1&&original->closes==0&&
 serving_ordinary_fence&&serving_start_certified_exact&&!serving_raw_fence_running,
 "fallback keeps admitted request, begins real work and enables ordinary final certification");
 require(original->lifecycle.apply(core::serving_raw::Lifecycle::Event::drain_work),"ordinary completion drains actual running lifecycle");
 require(original->close_request(e)&&original->closes==1&&!original->request_open()&&
 original->phase()==program::ServingRawSession::Phase::drained,"ordinary finalization closes exactly once");
 require(!original->close_request(e)&&original->closes==1,"duplicate close refused");
}
int main(int argc,char** argv){try{
 require(argc==2,"mode required");std::string mode=argv[1];
 if(mode.starts_with("cpu-")){
  setenv("STRATA_PREFILL_CPU_SHARE",mode=="cpu-auto"?"auto":"0",1);
  prefill::Prefill p;require(!prefill::g_share_explicit_observed,"cold lazy policy");
  auto cpu=p.cpu_share_policy();require(!cpu.explicit_resolved&&cpu.mode==core::NumericPolicyMode::unresolved&&!cpu.pool_present,
   "actual cold observer does not force policy read");
  const bool old_reject=(''' + cpu_reject + r''');require(old_reject,"actual strict prefix guard refuses cold unresolved policy");
  if(mode!="cpu-default-off")p.workspace_credit_=std::make_shared<int>(1);
  std::string e;const bool initialized=p.init_paid(e);cpu=p.cpu_share_policy();
  if(mode=="cpu-default-off")require(initialized&&!cpu.explicit_resolved,"unpaid default path never consumes policy");
  else if(mode=="cpu-auto")require(!initialized&&cpu.explicit_resolved&&cpu.mode==core::NumericPolicyMode::adaptive,"paid nonzero policy refused before allocation");
  else {require(initialized&&cpu.explicit_resolved&&cpu.mode==core::NumericPolicyMode::disabled&&!cpu.pool_present,"real paid init resolves disabled policy without pool");
   require(!(''' + cpu_reject + r'''),"actual unchanged strict CPU guard accepts resolved disabled policy");}
 }else run_fallback(mode);
 std::puts("PASS extracted prefill policy/refusal contract");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
'''


def run(binary):
    for mode in ['cpu-zero', 'cpu-auto', 'cpu-default-off', 'seal', 'capture', 'base', 'result', 'swapped', 'closed', 'cut', 'health', 'phase', 'ordinary', 'suffix', 'running', 'begin-failure', 'history', 'watermark', 'accepted', 'exact', 'legacy']:
        subprocess.run([str(binary), mode], check=True, timeout=10)
    print('PASS21 extracted cold-policy/refusal cases (recording RAW with actual Lifecycle)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--generate', type=Path)
    parser.add_argument('--mutation', choices=['cold-unresolved', 'close-fallback'])
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    if args.generate:
        args.generate.parent.mkdir(parents=True, exist_ok=True)
        args.generate.write_text(generate(args.source_root, args.mutation))
    elif args.binary:
        run(args.binary.resolve(strict=True))
    else:
        parser.error('--generate or --binary required')
