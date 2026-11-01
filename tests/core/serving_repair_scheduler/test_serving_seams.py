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

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--build-dir',type=Path)
    args=parser.parse_args()
    if args.build_dir:
        args.build_dir.mkdir(parents=True,exist_ok=True)
        run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-scheduler-seams-') as directory:run(directory)
