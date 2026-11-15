#!/usr/bin/env python3
"""Compile actual active repair decision, trusted age projection and selection lambda.

CPU control flow only. Real ancestry/projection/eligibility/selector execute over
host controller fixtures; no device completion or native repair is inferred.
"""
from pathlib import Path
import subprocess
import argparse
import tempfile

ROOT=Path(__file__).resolve().parents[3]
PREFIX=r'''
#include "serving_repair_scheduler.hpp"
#include "serving_repair_age.hpp"
#include "strata/core/window_state.hpp"
#include <cstdio>
#include <memory>
#include <string>
using namespace strata::program;
namespace core=strata::core;
namespace strata::program {
struct FixtureData { core::RepairSpatialCoverage coverage; FixtureData(const core::RepairSpatialCoverage& c):coverage(c){} };
struct FixturePublished { core::SnapshotDescriptor metadata; uint64_t serial=3; std::shared_ptr<FixtureData> data; };
struct FixturePrivate { core::StateCut cut_{}; };
struct FixtureController { bool busy_=false; std::shared_ptr<FixturePublished> published_; std::shared_ptr<FixturePrivate> private_; uint64_t private_forwarded() const {return private_->cut_.forwarded;} };
class ServingSuffixRepairOperation { public:
struct Outcome { std::shared_ptr<FixtureController> controller; bool retired=false; core::SnapshotDescriptor published; };
PROJECT
};
}
int main(int argc,char** argv){
    if(argc!=2)return 90;std::string name=argv[1];
    struct Runtime {uint64_t remote_policy_incarnation=7;} runtime;auto* owned_runtime=&runtime;
    ServingRepairSchedulerConfig serving_repair_schedule;
    uint64_t serving_repair_random_state=3,serving_repair_consecutive_escalated=0;
    int owner=1;auto* serving_repair_owners=&owner;
    struct Options{int max_context=1024;}o;
    bool piped=false;auto pipe_inflight=[](){return false;};auto batch_on=[](){return false;};
    struct Signal{bool stop_requested=false,input_error=false,end_of_input=false;uint64_t queued_input_lines=0;} queue;
    auto serving_idle_queue_signal=[&](){return queue;};
    int produced_n=1,max_new=2;
    ServingSuffixRepairOperation::Outcome operation;
    core::RepairCausalRoot root;root.identity[0]=1;root.original_history[0]=2;
    root.engine_generation=1;root.window_generation=1;root.runtime_incarnation=7;
    root.original_cut=1;root.original_begin=4;root.original_count=12;root.first_unresolved_unix_ms=100;
    root.monotonic_birth={7,0};
    if(name=="unknown")root.monotonic_birth={};
    core::RepairAncestry ancestry;
    if(core::make_repair_ancestry({&root,1},{64,core::RepairAncestry::retained_bytes()},ancestry)!=core::RepairAncestryStatus::ok)return 91;
    core::RepairSpatialCoverage coverage;
    if(core::make_repair_spatial_coverage(ancestry,4,16,coverage)!=core::RepairAncestryStatus::ok)return 92;
    operation.controller=std::make_shared<FixtureController>();
    operation.controller->published_=std::make_shared<FixturePublished>();
    operation.controller->published_->metadata.cut.forwarded=16;
    operation.controller->published_->metadata.cut.id=11;
    operation.controller->published_->data=std::make_shared<FixtureData>(coverage);
    operation.controller->private_=std::make_shared<FixturePrivate>();
    operation.controller->private_->cut_.forwarded=8;
    operation.published=operation.controller->published_->metadata;
    bool expected=false;
    if(name=="legacy"){serving_repair_owners=nullptr;expected=true;}
    else if(name=="idle"){produced_n=max_new;expected=true;}
    else if(name=="queued"){produced_n=max_new;queue.queued_input_lines=1;}
    else if(name=="terminal"){produced_n=max_new;queue.stop_requested=true;}
    else if(name=="aged"||name=="unknown"){
        serving_repair_schedule.enabled=true;serving_repair_schedule.policy.live_repair_age_escalation_ns=1;
        serving_repair_schedule.live_weight=100;serving_repair_schedule.foreground_weight=1;
        serving_repair_schedule.max_consecutive=1;
        expected=name=="aged";
    }else if(name!="young")return 93;
SELECTOR
DECISION
    if(repair_admitted!=expected)return 21;
    return 0;
}
'''


def extract(source):
    a=source.index('    static ServingRepairAgeObservation repair_age(')
    b=source.index('    // Each call performs at most four exact tokens.',a)
    project=source[a:b]
    a=source.index('        auto serving_select_repair =')
    # Stop at this helper's own return/closure, not at the later parked tick:
    # owned-slot maintenance is a separate intervening lambda.
    end='            return selected;\n        };'
    b=source.index(end,a)+len(end)
    selector=source[a:b]
    a=source.index('                    bool repair_admitted=true;')
    b=source.index('                    if(repair_admitted &&',a)
    decision=source[a:b]
    # The queued request reaches a decision, rather than bypassing it before the tick.
    if 'while (serving_repair_idle_tick()) {}' not in source:
        raise RuntimeError('outer boundary no longer evaluates queued demand')
    # Each parked selection refreshes actual owner ancestry before selecting.
    tick=source[source.index('        auto serving_repair_idle_tick ='):source.index('        for (;;) {',source.index('        auto serving_repair_idle_tick ='))]
    if not tick.index('::repair_age(payload,now)')<tick.index('serving_select_repair(boundary,candidates,"parked")')<tick.index('advance_parked_owner('):
        raise RuntimeError('parked age/selection/execution order changed')
    return project,selector,decision


def run(directory):
    project,selector,decision=extract((ROOT/'src/program/generate.cpp').read_text())
    variants=[('production',decision,None),
              ('no_active_gate',decision.replace('!repair_selected.foreground','true'),'young')]
    for label,block,negative in variants:
        if negative and block==decision:raise RuntimeError('negative control did not change production')
        cpp=Path(directory)/(label+'.cpp');binary=Path(directory)/label
        cpp.write_text(PREFIX.replace('PROJECT',project).replace('SELECTOR',selector).replace('DECISION',block))
        subprocess.run(['c++','-std=c++20','-O0','-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/program'),
                        str(cpp),str(ROOT/'src/core/window_repair_ancestry.cpp'),'-o',str(binary)],check=True)
        for case in ((negative,) if negative else ('legacy','young','unknown','queued','idle','terminal','aged')):
            result=subprocess.run([str(binary),case],capture_output=True,timeout=10)
            if (result.returncode==0)==bool(negative):
                raise RuntimeError(f'{label}/{case} unexpected exit {result.returncode}: {result.stderr.decode()}')
    print('PASS actual serving age/selection/active gate seams; young/unknown/queued/terminal/aged/legacy cases; negative control fails')



# The owned-slot dispatch seam reuses the same real projection/selector inputs.
# Queue state and the final repair_unit call are recording boundaries only.
def take_dispatch_block(source,marker,semi=False):
    if source.count(marker)!=1:
        raise RuntimeError('dispatch extraction marker is not unique: '+marker)
    start=source.index(marker);opening=source.index('{',start);depth=0
    for at in range(opening,len(source)):
        if source[at]=='{':depth+=1
        elif source[at]=='}':
            depth-=1
            if not depth:return source[start:at+1]+(';' if semi else '')+'\n'
    raise RuntimeError('unclosed dispatch extraction: '+marker)


def run_dispatch(directory):
    source=(ROOT/'src/program/generate.cpp').read_text()
    project,selector,decision=extract(source)
    signal=take_dispatch_block(source,'        auto serving_idle_queue_signal =',True)
    signal_type=take_dispatch_block((ROOT/'src/program/serving_repair_owners.hpp').read_text(),
                                    'struct ServingIdleQueueSignal {',True)
    anchor='            const auto selected=serving_select_repair(boundary,std::span<const strata::program::ServingRepairSelectionCandidate>(candidates.data(),std::min(bs.size(),candidates.size())),"owned-slots");\n'
    if source.count(anchor)!=1:raise RuntimeError('unmanaged selection anchor not unique')
    start=source.index(anchor)+len(anchor)
    end=source.index('        };\n        // One selection and at most ONE bounded private unit.',start)
    tail=source[start:end]
    # Remove only the actual new guard for a negative control, not a mirror of it.
    guard_start='            const auto before=serving_idle_queue_signal();\n'
    guard_end='            auto& lane=bs[selected.candidate];\n'
    if tail.count(guard_start)!=1 or tail.count(guard_end)!=1:
        raise RuntimeError('owned-slot final guard boundary not unique')
    a=tail.index(guard_start);b=tail.index(guard_end,a)
    without_guard=tail[:a]+tail[b:]
    if without_guard==tail:raise RuntimeError('guard-removal negative did not change production')
    shared=PREFIX[:PREFIX.index('int main(')]
    recording=r"""
static inline unsigned dispatch_calls=0;
static bool repair_unit(Outcome&,bool,std::string&,uint64_t) {++dispatch_calls;return true;}
"""
    shared=shared.replace('PROJECT',project+recording)
    setup=PREFIX[PREFIX.index('    ServingSuffixRepairOperation::Outcome operation;'):PREFIX.index('    bool expected=false;')]
    template=(Path(__file__).resolve().parent/'dispatch_arrival.cpp.in').read_text()
    cases=('idle_no_arrival','idle_queue_arrival','aged_queue_arrival','fresh_stop','fresh_error','fresh_eof')
    for label,actual_tail,negative in [('owned_dispatch',tail,False),('missing_final_guard',without_guard,True)]:
        code=template
        replacements={'SHARED_PREFIX':shared,'QUEUE_SIGNAL_TYPE':signal_type,'ACTUAL_QUEUE_SIGNAL':signal,
                      'ANCESTRY_SETUP':setup,'ACTUAL_UNMANAGED_TAIL':actual_tail}
        for marker,actual in replacements.items():
            if code.count(marker)!=1:raise RuntimeError('dispatch template marker not unique: '+marker)
            code=code.replace(marker,actual)
        cpp=Path(directory)/(label+'.cpp');binary=Path(directory)/label;cpp.write_text(code)
        subprocess.run(['c++','-std=c++20','-O0','-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/program'),
                        str(cpp),str(ROOT/'src/core/window_repair_ancestry.cpp'),'-o',str(binary)],check=True)
        for case in (('idle_queue_arrival','fresh_stop') if negative else cases):
            result=subprocess.run([str(binary),case],capture_output=True,text=True,timeout=10)
            if negative:
                if result.returncode!=21 or 'calls=1 expected=0 advanced=1' not in result.stdout:
                    raise RuntimeError(f'{label}/{case}: zero-dispatch assertion did not fail: {result.returncode}: {result.stdout} {result.stderr}')
            elif result.returncode!=0:
                raise RuntimeError(f'{label}/{case}: unexpected exit {result.returncode}: {result.stdout} {result.stderr}')
            print(f'PASS {label}/{case}: '+result.stdout.strip())
    print('PASS actual owned-slot selector/queue-signal/dispatch tail; six boundary cases; guard removal fails queue and STOP assertions')


# Clarified managed coverage: actual outer q=r loop, joint selection, cohort and
# burst. CPU172 native/time boundaries remain explicit in the single template.
def run_prefill_burst(directory):
    source=(ROOT/'src/program/generate.cpp').read_text()
    def span(first,last,start=0):
        a=source.index(first,start);b=source.index(last,a)
        return source[a:b]
    block=take_dispatch_block
    cohort=block(source,'auto serving_foreground_only_cohort=',True)
    batch_on=block(source,'auto batch_on =',True)
    ack=block(source,'auto serving_ack_foreground_lane=',True)
    batch_prefix=span('auto batch_step =','            if (S == 0) return true;')+'            if (S == 0) return true;\n'
    start=source.index('const int64_t quantum=planned?')
    entry=span('if(!serving_raw_session||','                            if(!serving_process_controls())',start)
    joint=span('                            if(!serving_process_controls())','                            auto report_turn=',start)
    choose=block(source,'auto choose_and_service=',True)
    after_peer='                            if(other_serviced)continue;\n'
    if source.count(after_peer)!=1:raise RuntimeError('actual prefill peer continuation is not unique')
    advance=span('                            serving_repair_consecutive_escalated=0;\n                            at=end;',
                 '                        }\n                        return true;',start)
    eos=span(r'                    std::printf("BT %d %d\n", b, (int) y);','                }\n                if (batch_mtp && sl.active)',source.index('auto batch_step ='))
    pieces='                int64_t piece = piece_now();\n                if (piece <= 0) return run(a0, b0);\n'
    if source.count(pieces)!=1:raise RuntimeError('actual positive-piece entry is not unique')
    pieces+=block(source,'for (int64_t q = a0; q < b0;)')
    guard=span('#if defined(STRATA_SERVING_SUFFIX_REPAIR_AVAILABLE)\n                    // The next managed piece',
               '                    if (decode_share <= 0.0 || !batch_on()) continue;')
    if pieces.count(guard)!=1 or 'if(serving_threads)continue;' not in guard:
        raise RuntimeError('actual managed interchunk guard missing/ambiguous')
    world=(ROOT/'tests/core/serving_raw_contracts/test_serving_thread_bridge.cpp').read_text().split('int main(',1)[0]
    verify=(ROOT/'include/strata/kernels/verify_kernels.hpp').read_text()
    bound=next(line for line in verify.splitlines() if 'constexpr int kVerifyMaxT =' in line)
    eos_default=next(line for line in source.splitlines() if 'std::vector<int64_t> eos_ids = ' in line)
    decode=span('static const double decode_share =','            bool batch_fatal = false;')
    template=(Path(__file__).resolve().parent/'prefill_burst.cpp.in').read_text()
    fragments={'ACTUAL_WORLD':world,'ACTUAL_VERIFY_BOUND':bound,'ACTUAL_EOS_DEFAULT':eos_default,
               'ACTUAL_BATCH_ON':batch_on,'ACTUAL_ACK':ack,'ACTUAL_COHORT':cohort,
               'ACTUAL_BATCH_PREFIX':batch_prefix,'ACTUAL_EOS':eos,'ACTUAL_JOINT':joint,
               'ACTUAL_ENTRY':entry,'ACTUAL_CHOOSE':choose,'ACTUAL_AFTER_PEER':after_peer,
               'ACTUAL_ADVANCE':advance,'ACTUAL_DECODE_SHARE':decode}
    cases=('priority','equal','stop','cancel','eof','error','legacy','peer-only')
    for label,outer,negative in [('managed_coverage',pieces,False),('missing_managed_coverage',pieces.replace(guard,'',1),True)]:
        code=template
        for marker,value in {**fragments,'ACTUAL_PIECES':outer}.items():
            if code.count(marker)!=1:raise RuntimeError('prefill template marker missing/ambiguous: '+marker)
            code=code.replace(marker,value)
        cpp=Path(directory)/(label+'.cpp');binary=Path(directory)/label;cpp.write_text(code)
        subprocess.run(['c++','-std=c++20','-O0','-DSTRATA_SERVING_SUFFIX_REPAIR_AVAILABLE=1','-pthread',
                        '-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/program'),str(cpp),
                        *[str(ROOT/'src/core'/name) for name in ('thread_priority.cpp','thread_dependencies.cpp','window_budget.cpp')],
                        '-o',str(binary)],check=True)
        for case in (('priority',) if negative else cases):
            result=subprocess.run([str(binary),case],capture_output=True,text=True,timeout=10)
            if negative:
                if result.returncode!=21 or 'COVERAGE_ASSERT lower_peer_calls=3 expected=0 current_prefill_ready=1' not in result.stdout:
                    raise RuntimeError(f'{label}/{case}: same coverage assertion did not fail21: {result.returncode}: {result.stdout} {result.stderr}')
                print('NEGATIVE_CONTROL_DETECTED actual guard removal / high-priority coverage exit21')
            elif result.returncode:
                raise RuntimeError(f'{label}/{case}: unexpected exit {result.returncode}: {result.stdout} {result.stderr}')
            else:print(f'PASS {label}/{case}: '+result.stdout.strip())


def execute(directory):
    run(directory)
    run_dispatch(directory)
    run_prefill_burst(directory)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--build-dir',type=Path)
    args=parser.parse_args()
    if args.build_dir:
        args.build_dir.mkdir(parents=True,exist_ok=True)
        execute(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-scheduler-seams-') as directory:execute(directory)
