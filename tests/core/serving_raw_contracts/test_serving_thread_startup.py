#!/usr/bin/env python3
"""Generate an actual-source startup/input-envelope CPU regression. No model/device calls."""
import argparse,importlib.util
from pathlib import Path

def generate(root):
    root=Path(root);source=(root/'src/program/generate.cpp').read_text()
    path=root/'tests/core/serving_raw_contracts/test_serving_thread_bridge_source.py'
    spec=importlib.util.spec_from_file_location('actual_managed_startup_extract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    marker='if(!serving_threads_config.valid || (serving_threads_config.enabled&&'
    start=source.index(marker);opening=source.index('{',start)
    guard=source[start:opening]+module.body(source,marker)
    start=source.index('        size_t input_max_line_bytes=0, input_max_queue_lines=0;')
    end=source.index('        const auto in_shared = std::make_shared<StdinShared>();',start)
    envelope=source[start:end]
    held=source[source.index('in_shared->reservation=std::move(input_reservation);',end):]
    held=held[:held.index(';')+1]
    return r'''// Generated from the ACTUAL serving guard and paid reader block.
// Hardware/model loaders intentionally absent; real WindowBudget/ReservationLease.
#define STRATA_WINDOW_STATE_AVAILABLE 1
#define STRATA_SERVING_SUFFIX_REPAIR_AVAILABLE 1
#include "serving_thread_bridge.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
namespace core=strata::core;
namespace program=strata::program;
struct Options {bool verified_inputs_active=false,suffix_cache=true;int64_t max_context=512;};
struct Verified {std::unique_ptr<core::WindowBudget> budget;};
struct ReaderOwner {std::shared_ptr<const core::ReservationLease> reservation;};
struct Output {size_t line=0,queue=0;std::shared_ptr<ReaderOwner> reader;};
static void require(bool ok,const char* text){if(!ok){std::fprintf(stderr,"FAIL %s\n",text);std::exit(1);}}
int actual_startup(Options o,bool owned_batch_allocation,const program::ServingThreadConfig& serving_threads_config,
        const program::ServingRepairSchedulerConfig& serving_repair_schedule,core::WindowBudget* model_budget){
''' + guard + r'''
    return 0;
}
int actual_envelope(Options o,bool serving_threads,Verified& verified_inputs,
        std::unique_ptr<core::WindowBudget>& model_budget,Output& output){
''' + envelope + r'''
    const auto in_shared=std::make_shared<ReaderOwner>();
''' + held + r'''
    output.line=input_max_line_bytes;output.queue=input_max_queue_lines;output.reader=in_shared;
    return 0;
}
static std::unique_ptr<core::WindowBudget> budget(uint64_t host,bool dual=true){
    return std::make_unique<core::WindowBudget>(core::WindowResources{dual?std::vector<uint64_t>{1,1}:std::vector<uint64_t>{0},host},
        core::WindowResources{dual?std::vector<uint64_t>{0,0}:std::vector<uint64_t>{0}});
}
static uint64_t reserved(const std::unique_ptr<core::WindowBudget>& b){return b->snapshot().reserved.host_bytes;}
int main(){
    // Explicit existing qualification values. They are test inputs, not defaults.
    auto policy=program::parse_serving_thread_config(program::kServingThreadPolicy,
        "-100,100,100,60000,16,8,64,128,256,1024,16,8388608");
    const auto repair=program::parse_serving_repair_scheduler_config({"1000000000000000000","1","1","2","0"});
    require(policy.enabled&&policy.valid&&repair.enabled&&repair.valid,"actual policy parsers");
    Options o;auto model=budget(64<<20);Verified verified;
    require(actual_startup(o,true,policy,repair,model.get())==0,"existing owned2 qualification profile without private snapshot loader must start");
    require(!o.verified_inputs_active&&!verified.budget,"no implicit private loader activation");
    require(actual_startup(o,false,policy,repair,model.get())==2,"owned2 remains mandatory");
    require(actual_startup(o,true,policy,repair,nullptr)==2,"actual model ledger remains mandatory");
    auto missing_suffix=o;missing_suffix.suffix_cache=false;
    require(actual_startup(missing_suffix,true,policy,repair,model.get())==2,"RAW suffix ownership mode remains mandatory");
    auto bad=policy;bad.valid=false;
    require(actual_startup(o,true,bad,repair,model.get())==2,"invalid explicit policy refuses");
    auto absent_repair=repair;absent_repair.enabled=false;
    require(actual_startup(o,true,policy,absent_repair,model.get())==2,"complete repair policy remains mandatory");
    auto off=program::parse_serving_thread_config(nullptr,nullptr);
    require(actual_startup(missing_suffix,false,off,absent_repair,nullptr)==0,"default-off old profile untouched");
    constexpr uint64_t line=512*12+65536,charge=65536+4*(8+4)*line;
    Output managed;
    require(actual_envelope(o,true,verified,model,managed)==0&&managed.line==line&&managed.queue==8,"managed raw tokens bounded without private loader");
    require(managed.reader&&managed.reader->reservation&&managed.reader->reservation->canonical_owner(managed.reader->reservation)&&
        managed.reader->reservation->resources().host_bytes==charge&&reserved(model)==charge,"actual model ledger pays whole reader envelope");
    require(managed.reader->reservation->resources().device_bytes==std::vector<uint64_t>({0,0}),"actual model ledger dimensions retained");
    auto late_reader=managed.reader;managed.reader.reset();
    require(reserved(model)==charge,"detached reader retains reservation after serving scope drops it");
    late_reader.reset();require(reserved(model)==0,"reservation releases only after final reader owner");
    Output legacy;
    require(actual_envelope(o,false,verified,model,legacy)==0&&legacy.line==0&&legacy.queue==0&&
        !legacy.reader->reservation&&reserved(model)==0,"default-off input behavior unchanged");
    auto private_o=o;private_o.verified_inputs_active=true;
    Output missing_private;
    require(actual_envelope(private_o,true,verified,model,missing_private)==1&&reserved(model)==0,
        "selected private loader never silently falls back to model ledger");
    verified.budget=budget(64<<20,false);Output private_reader;
    require(actual_envelope(private_o,true,verified,model,private_reader)==0&&reserved(verified.budget)==charge&&reserved(model)==0,
        "verified loader mode retains its original separate ledger");
    require(private_reader.reader->reservation->resources().device_bytes==std::vector<uint64_t>({0}),"private input ledger dimensions unchanged");
    private_reader.reader.reset();require(reserved(verified.budget)==0,"private reader lifetime releases normally");
    auto small=budget(1<<20);Output refused;
    require(actual_envelope(o,true,verified,small,refused)==1&&reserved(small)==0,"real input quota refusal has no leaked charge");
    std::unique_ptr<core::WindowBudget> missing;
    require(actual_envelope(o,true,verified,missing,refused)==1,"managed missing model ledger refuses before reader");
    auto invalid=o;invalid.max_context=0;
    require(actual_envelope(invalid,true,verified,model,refused)==1,"zero context bounded reader refuses");
    invalid.max_context=INT64_MAX;
    require(actual_envelope(invalid,true,verified,model,refused)==1,"input arithmetic overflow refuses before reserve");
    require(reserved(model)==0,"all negative paths release/no mutation");
    std::puts("actual managed startup/input-envelope regression PASS");
}
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--generate',type=Path,required=True)
    args=parser.parse_args();code=generate(args.source_root)
    args.generate.parent.mkdir(parents=True,exist_ok=True);args.generate.write_text(code)
    print('Generated actual source guard/envelope regression; no compiler/model/device action')
