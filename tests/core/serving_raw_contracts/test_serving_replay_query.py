#!/usr/bin/env python3
"""Actual original/fixed serving C1 arm + verifier poll + RAW poison CPU seams.

The CUDA functions are recording APIs, not hardware. The original method snapshot
is byte-pinned to the failed134748 source. No supplied observe() call substitutes
for executing the actual controller query and actual Verifier poll methods.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ORIGINAL_CONTROLLER = 'd802a8816e25581d1f908eafc5ff612592ccd7eb5977d0addaf03f436accf71d'
ORIGINAL_METHOD = '5ff9e8f18704818ffae8ee5fd3c09e7e6658946ab25d24d10e88124c9c02b083'


def take(source, marker, semi=False):
    begin = source.index(marker)
    opening = source.index('{', begin)
    depth = 0
    for at in range(opening, len(source)):
        if source[at] == '{':
            depth += 1
        elif source[at] == '}':
            depth -= 1
            if not depth:
                return source[begin:at+1] + (';' if semi else '') + '\n'
    raise RuntimeError('unclosed actual method ' + marker)


CPP = r'''
#include "serving_suffix_fault_qualification.hpp"
#include "strata/core/resident_deferred_fault.hpp"
#include "strata/core/resident_fault_precondition.hpp"
#include "strata/core/window_device.hpp"
#include "strata/core/window_state.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
namespace seam {
bool post_trap=false;int device=0;int trace_fd=-1,sentinel_fd=-1;unsigned traced=0;
cudaError_t query_status=cudaErrorIllegalInstruction;bool nonsticky_control=false,before_get_error=false,prerequisite_error=false;
void (*after_launch)()=nullptr;
void trace(char c){if(trace_fd>=0&&traced++<64&&::write(trace_fd,&c,1)!=1)std::_Exit(94);}
}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device){seam::trace(seam::post_trap?'G':'g');
    if(seam::before_get_error)return cudaErrorInvalidValue;
    if(seam::post_trap&&!seam::nonsticky_control)return cudaErrorIllegalInstruction;
    *device=seam::device;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaSetDevice(int device){seam::trace(seam::post_trap?'S':'s');seam::device=device;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaPeekAtLastError(){seam::trace(seam::post_trap?'P':'p');return seam::prerequisite_error?cudaErrorInvalidValue:cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaGetLastError(){seam::trace('C');return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize(){seam::trace(seam::post_trap?'Y':'y');return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t,unsigned* flags){seam::trace('f');*flags=1;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream){
    seam::trace(stream==reinterpret_cast<cudaStream_t>(uintptr_t(0x2000))?'Q':'O');
    if(!seam::post_trap)return cudaSuccess;
    const auto status=seam::query_status;
    if(seam::nonsticky_control)seam::query_status=cudaSuccess;
    return status;}
extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t){seam::trace('E');return cudaErrorIllegalInstruction;}
extern "C" const char* CUDARTAPI cudaGetErrorString(cudaError_t){seam::trace('R');return "recording CUDA error";}
void require(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL %s\n",why);std::exit(90);}}
struct Sentinel {~Sentinel(){if(seam::sentinel_fd>=0&&::write(seam::sentinel_fd,"D",1)!=1)std::_Exit(92);}};
namespace strata::core {
struct Verifier {
    enum class SnapshotDrain {pending,drained,unhealthy,unsupported};
    bool batch_copy_failed_=false,commit_pending_=false,copy_used_=false;
    std::atomic<bool> released_{false};std::string snapshot_failure_,commit_failure_;
    int device_=0;cudaStream_t cs_=nullptr,sh_cs_=nullptr,copy_=nullptr,side_=nullptr,side2_=nullptr,df_side_[2]{};
    cudaEvent_t commit_done_=nullptr;Verifier* next_=nullptr;
    std::shared_ptr<Sentinel> owner=std::make_shared<Sentinel>();
    Verifier(int device,cudaStream_t stream):device_(device),cs_(stream){}
    cudaStream_t stream()const{return cs_;}
    bool refuse_copy_query_failure(std::string&){return false;}
    bool handoff_failed(std::string&){return false;}
    SnapshotDrain poll_snapshot_drain(std::string&);
};
ACTUAL_POLL
namespace resident_deferred_fault {
cudaError_t launch_private_stage_trap(cudaStream_t stream)noexcept {
    require(stream==reinterpret_cast<cudaStream_t>(uintptr_t(0x2000)),"recording launch actual bound stage1 stream");
    seam::trace('L');seam::post_trap=true;if(seam::after_launch)seam::after_launch();return cudaSuccess;
}
}
}
namespace strata::program {
namespace core=::strata::core;
using core::StateCut;
using core::Verifier;
RAW_RESULT
class SuffixRawTransaction {public:
    enum class Phase {idle,running,drained,capturing,restoring,poisoned};
    struct Stage {int device=0;void* session=nullptr;};
    struct Allocation {bool good=true;std::array<Stage,2> entries{};std::shared_ptr<Sentinel> owner=std::make_shared<Sentinel>();
        bool healthy()const{return good;}void mark_unhealthy(){good=false;}const auto& stages()const{return entries;}} allocation_;
    struct Storage {std::shared_ptr<Sentinel> owner=std::make_shared<Sentinel>();};std::shared_ptr<Storage> pending_;
    std::shared_ptr<const void> runtime_=std::make_shared<Sentinel>();std::thread::id controller_=std::this_thread::get_id();
    Phase phase_=Phase::drained;core::StateCut cut_{};uint64_t max_tokens_=512;std::vector<core::Verifier*> verifiers_;
    static bool present(const core::StateDigest& digest){return std::any_of(digest.begin(),digest.end(),[](auto value){return value!=0;});}
    static SuffixRawResult invalid(const char* why){return {SuffixRawStatus::invalid,why,{}};}
RAW_METHODS
};
namespace resident_repair_detail {
constexpr size_t maximum_stages=8;enum class RuntimePhase {ready,drained};
struct Runtime {
    struct Service {const void* active_=nullptr;bool unhealthy_=false;std::thread::id controller_=std::this_thread::get_id();};
    struct Completion {bool committed()const{return true;}};
    struct Dispatch {unsigned callback_depth_=0;std::unique_ptr<Completion> last_completion_=std::make_unique<Completion>();
        uint64_t last_completed_operation_=1;};
    struct Stage {void* session=nullptr;std::shared_ptr<const void> session_owner;std::unique_ptr<core::Verifier> verifier;
        cudaStream_t stream=nullptr;struct {int device=0;} service;};
    std::shared_ptr<Service> service_=std::make_shared<Service>();std::shared_ptr<Dispatch> dispatch_=std::make_shared<Dispatch>();
    std::array<Stage,2> stages_{};size_t stage_count_=2;RuntimePhase phase_=RuntimePhase::drained;
    SuffixRawTransaction::Allocation allocation_;std::thread::id controller_=std::this_thread::get_id();uint64_t operation_=1;
};
class ResidentRepairController {public:
    bool busy_=true,cancel_requested_=false,serving_fault_fired_=false,finished_=false;
    core::resident_serving_fault::Binding serving_fault_{};
    std::shared_ptr<Runtime> runtime_=std::make_shared<Runtime>();SuffixRawTransaction live_;
    std::unique_ptr<SuffixRawTransaction> private_=std::make_unique<SuffixRawTransaction>();
    std::vector<int32_t> private_history_;size_t receipt_count_=1;
    struct Published {uint64_t serial=2;};std::shared_ptr<Published> published_=std::make_shared<Published>();
    std::unique_ptr<core::Verifier> replacement_;
    inline static ResidentRepairController* hook_=nullptr;
    [[noreturn]] static void fatal(const char* why){core::retirement_failed(why);}
ACTUAL_PAUSE
ACTUAL_COMPLETE
ACTUAL_ARM
    static void run_case(const core::resident_serving_fault::Binding& binding,const char* mode){
        ResidentRepairController controller;controller.serving_fault_=binding;
        for(size_t i=0;i<2;++i){auto& stage=controller.runtime_->stages_[i];
            auto session=std::make_shared<int>(int(i));stage.session_owner=session;stage.session=session.get();
            stage.service.device=int(i);stage.stream=reinterpret_cast<cudaStream_t>(uintptr_t((i+1)*0x1000));
            stage.verifier=std::make_unique<core::Verifier>(int(i),stage.stream);
            controller.private_->allocation_.entries[i]={int(i),stage.session};
            controller.runtime_->allocation_.entries[i]={int(i),stage.session};controller.live_.allocation_.entries[i]={int(i),stage.session};
            controller.private_->verifiers_.push_back(stage.verifier.get());}
        controller.runtime_->stages_[0].verifier->next_=controller.runtime_->stages_[1].verifier.get();
        controller.private_->phase_=SuffixRawTransaction::Phase::running;controller.private_->cut_={8,40,40,40,1,{}};
        controller.private_->cut_.history[0]=1;controller.private_history_.resize(41,1);
        core::resident_serving_fault::Snapshot snapshot;snapshot.publication=2;snapshot.live_cut=69;snapshot.target_cut=70;
        snapshot.accepted=293;snapshot.forwarded=292;snapshot.exact=40;snapshot.watermark=1;snapshot.private_cut=8;snapshot.repaired=40;
        snapshot.known=59;snapshot.generated=1;snapshot.foreground=60;snapshot.attachments=1;
        snapshot.receipt_operation=60;snapshot.receipt_publication=2;snapshot.receipt_cut=69;snapshot.receipt_token=11855;snapshot.history[0]=1;
        if(!std::strcmp(mode,"719"))seam::query_status=cudaErrorLaunchFailure;
        if(!std::strcmp(mode,"unsupported"))seam::query_status=cudaErrorInvalidResourceHandle;
        if(!std::strcmp(mode,"success")||!std::strcmp(mode,"event_only"))seam::query_status=cudaSuccess;
        if(!std::strcmp(mode,"event_only")){controller.runtime_->stages_[1].verifier->commit_pending_=true;
            controller.runtime_->stages_[1].verifier->commit_done_=reinterpret_cast<cudaEvent_t>(uintptr_t(0x4000));}
        if(!std::strcmp(mode,"not_ready"))seam::query_status=cudaErrorNotReady;
        if(!std::strcmp(mode,"before_get"))seam::before_get_error=true;
        if(!std::strcmp(mode,"prerequisite"))seam::prerequisite_error=true;
        if(!std::strcmp(mode,"normal_after_query"))seam::nonsticky_control=true;
        // Replacement is admitted/owned BEFORE launch; the CPU hook swaps only
        // already-owned objects, preserving the same sticky-error lease rule.
        controller.replacement_=std::make_unique<core::Verifier>(1,controller.runtime_->stages_[1].stream);hook_=&controller;
        if(!std::strcmp(mode,"wrong_verifier"))seam::after_launch=[](){hook_->runtime_->stages_[1].verifier.swap(hook_->replacement_);};
        if(!std::strcmp(mode,"wrong_stream"))seam::after_launch=[](){hook_->runtime_->stages_[1].stream=reinterpret_cast<cudaStream_t>(uintptr_t(0x3000));};
        Pause pause(controller);core::resident_deferred_fault::Scope scope;
        controller.arm_serving_replay_fault(scope,pause,snapshot);
        core::StateCut cut{70,41,41,41,1,{}};cut.history[0]=1;
        controller.complete(*controller.private_,cut);
ACTUAL_UNEXPECTED_COMPLETION
    }
};
}
}
int main(int argc,char** argv){
    require(argc==2,"one recording CPU case");
    int trace=::open("/output/C1-API.trace",O_WRONLY|O_CREAT|O_EXCL,0600);
    int sentinel=::open("/output/C1-OWNER.trace",O_WRONLY|O_CREAT|O_EXCL,0600);
    require(trace>=0&&sentinel>=0,"fresh bounded API/owner trace files");seam::trace_fd=trace;seam::sentinel_fd=sentinel;
    strata::program::ServingSuffixFaultQualification manager;std::string error;
    auto credit=std::shared_ptr<const void>(new Sentinel,[](const void* p){delete static_cast<const Sentinel*>(p);});
    std::array<char,65> identity{};std::fill_n(identity.data(),64,'a');
    require(manager.configure("2:private_replay",true,error)&&manager.prepare(2,credit,identity,identity,error),"actual manager FD/admission/binding");
    strata::program::resident_repair_detail::ResidentRepairController::run_case(manager.binding(),argv[1]);
    std::_Exit(91);
}
'''


def run(build):
    root = Path(__file__).resolve().parents[3]
    names = ['src/program/resident_repair_controller.inc', 'src/core/verify_drain.cpp',
             'src/program/suffix_raw_runtime.cpp', 'src/program/suffix_raw_runtime.hpp',
             'tests/core/serving_raw_contracts/serving_replay_query_original.inc',
             'src/program/serving_suffix_fault_qualification.hpp', 'include/strata/core/resident_deferred_fault.hpp']
    pins = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in names}
    controller, drain, raw, raw_h, original = [(root / f).read_text() for f in names[:5]]
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_METHOD:
        raise RuntimeError('original failed134748 method snapshot changed')
    replacements = {
        'ACTUAL_POLL': take(drain, 'Verifier::SnapshotDrain Verifier::poll_snapshot_drain('),
        'RAW_RESULT': take(raw_h, 'enum class SuffixRawStatus', True) + take(raw_h, 'struct SuffixRawResult', True),
        'RAW_METHODS': ''.join(take(raw, marker).replace('SuffixRawTransaction::', '', 1) for marker in
                               ['SuffixRawResult SuffixRawTransaction::poison(', 'SuffixRawResult SuffixRawTransaction::poll_all(',
                                'SuffixRawResult SuffixRawTransaction::complete_driver_work(']),
        'ACTUAL_PAUSE': take(controller, '    struct Pause', True),
        'ACTUAL_COMPLETE': take(controller, '    void complete('),
        'ACTUAL_UNEXPECTED_COMPLETION': 'fatal("serving replay fault unexpectedly certified a new cut");',
    }
    unexpected = 'fatal("serving replay fault unexpectedly certified a new cut");'
    if controller.count(unexpected) != 1:
        raise RuntimeError('actual missed-poison failstop changed')
    replacements['ACTUAL_UNEXPECTED_COMPLETION'] = unexpected
    fixed = take(controller, '    void arm_serving_replay_fault(')
    output = Path('/output')
    summaries = []
    for variant, method in [('original', original), ('fixed', fixed)]:
        code = CPP.replace('ACTUAL_ARM', method)
        for key, value in replacements.items():
            code = code.replace(key, value)
        path = build / (variant + '-serving-replay-query.cpp')
        binary = build / (variant + '-serving-replay-query')
        path.write_text(code)
        subprocess.run(['c++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Wconversion', '-Werror',
                        '-DSTRATA_WINDOW_STATE_AVAILABLE=1', '-DSTRATA_RESIDENT_FAULT_QUALIFICATION=1',
                        '-I' + str(root / 'include'), '-I' + str(root / 'src/program'),
                        '-isystem', '/usr/local/cuda/targets/x86_64-linux/include', str(path), '-o', str(binary)], check=True)
        cases = ['715'] if variant == 'original' else ['715', '719', 'wrong_verifier', 'wrong_stream', 'unsupported',
                                                        'success', 'not_ready', 'event_only', 'before_get', 'prerequisite', 'normal_after_query']
        for case in cases:
            for name in ['RESIDENT-FAULT.jsonl', 'C1-API.trace', 'C1-OWNER.trace']:
                (output / name).unlink(missing_ok=True)
            result = subprocess.run([str(binary), case], capture_output=True, text=True, timeout=10)
            records = [json.loads(line) for line in (output / 'RESIDENT-FAULT.jsonl').read_text().splitlines()]
            trace = (output / 'C1-API.trace').read_text()
            owner = (output / 'C1-OWNER.trace').read_bytes()
            query = [r for r in records if r['phase'] == 'cudaStreamQuery']
            failure = [r for r in records if r['phase'] == 'raw_failure']
            if result.returncode != 1 or owner or 'RETURN' in result.stdout:
                raise RuntimeError('actual noreturn/owner retention failed: ' + repr((variant, case, result, trace, owner)))
            if variant == 'original':
                expected = 'serving replay fault query attribution unavailable'
                if query or failure or 'L' not in trace or trace.split('L', 1)[1] != 'G':
                    raise RuntimeError('original actual GetDevice-before-query negative did not reproduce')
            elif case in ['715', '719']:
                expected = 'resident actual all-stage completion refused'
                if len(query) != 1 or query[0]['status'] != int(case) or len(failure) != 1:
                    raise RuntimeError('actual bound query/original RAW poison missing')
                actual = failure[0]
                if any(actual.get(f) != 1 for f in ['raw_poisoned', 'allocation_unhealthy', 'cut_unchanged', 'failure_owner_retained']):
                    raise RuntimeError('original RAW did not retain actual failure/cut/allocation obligations')
                if trace.split('L', 1)[1] != 'QSG' or 'C' in trace:
                    raise RuntimeError('bound query was not first post-launch CUDA API before original poll')
            else:
                expected = {'wrong_verifier': 'bound query attribution unavailable', 'wrong_stream': 'bound query attribution unavailable',
                            'unsupported': 'bound stream returned unsupported status', 'success': 'query deadline or poll bound',
                            'not_ready': 'query deadline or poll bound', 'event_only': 'query deadline or poll bound',
                            'before_get': 'device prerequisite refused', 'prerequisite': 'resident fault prerequisite refused',
                            'normal_after_query': 'unexpectedly certified a new cut'}[case]
                if failure or (case in ['wrong_verifier', 'wrong_stream', 'success', 'not_ready', 'event_only', 'before_get', 'prerequisite'] and query):
                    raise RuntimeError('negative CPU sequence cannot masquerade as attributed fault/RAW poison')
                if case in ['before_get', 'prerequisite'] and ('L' in trace or 'Q' in trace):
                    raise RuntimeError('failed precondition reached actual launch/query')
                if case == 'event_only' and 'E' in trace:
                    raise RuntimeError('event-only status entered bound stream evidence')
            if expected not in result.stderr:
                raise RuntimeError('actual source noreturn origin differs: ' + repr((variant, case, result.stderr)))
            summaries.append({'variant': variant, 'case': case, 'exit': result.returncode, 'trace': trace,
                              'phases': [r['phase'] for r in records], 'origin': expected})
            print('PASS actual C1 sequence', variant, case, 'trace=' + trace, flush=True)
    if any(hashlib.sha256((root / f).read_bytes()).hexdigest() != value for f, value in pins.items()):
        raise RuntimeError('C1 method inputs changed during fixture')
    (build / 'result.json').write_text(json.dumps({'cases': summaries, 'source_pins': pins, 'original_controller': ORIGINAL_CONTROLLER,
                                                'original_method': ORIGINAL_METHOD, 'native_proof': False}, indent=2) + '\n')
    print('source_pins=' + repr(pins), flush=True)
    print('original_controller=' + ORIGINAL_CONTROLLER + ' original_method=' + ORIGINAL_METHOD +
          ' fixed_method=' + hashlib.sha256(fixed.encode()).hexdigest(), flush=True)
    print('PASS actual original/fixed C1 arm, Verifier poll, RAW poison and noreturn CPU seams; no hardware fault proof', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    if args.build_dir:
        args.build_dir.mkdir(parents=True, exist_ok=True)
        run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-serving-replay-query-') as path:
            run(Path(path))


if __name__ == '__main__':
    main()
