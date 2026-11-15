#!/usr/bin/env python3
"""Actual serving fault helpers and finish/RAW/retire/close propagation at CPU seams.

CUDA calls, native allocations and dispatch are recording seams. Native error
codes supplied by those seams are NOT hardware evidence. Actual source methods,
FD writer/ownership, lifecycle, poison transitions and noreturn exits execute.
Requires a fresh writable /output (the bounded CPU qualification container).
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

PRELUDE = r'''
#include "serving_suffix_fault_qualification.hpp"
#include "strata/core/resident_copy_fault.hpp"
#include "strata/core/resident_deferred_fault.hpp"
#include "strata/core/serving_raw_contracts.hpp"
#include "strata/core/window_state.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
namespace seam {int calls=0,kind=-1,destruct_fd=-1;cudaError_t result=cudaErrorInvalidMemcpyDirection;}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device){*device=1;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaSetDevice(int){return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaPeekAtLastError(){return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize(){return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t,unsigned* flags){*flags=0;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void*,const void*,size_t,cudaMemcpyKind kind,cudaStream_t){++seam::calls;seam::kind=int(kind);return seam::result;}
void require(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL %s\n",why);std::exit(90);}}
struct Sentinel {~Sentinel(){if(seam::destruct_fd>=0&&::write(seam::destruct_fd,"D",1)!=1)std::_Exit(92);}};
namespace strata::core {
struct Verifier {
    enum class SnapshotDrain {drained,pending,unhealthy,unsupported};SnapshotDrain drain=SnapshotDrain::drained;
    SnapshotDrain poll_snapshot_drain(std::string&){return drain;}
};
}
namespace strata::program {
namespace core=::strata::core;
using namespace core;
RAW_RESULT
class SuffixRawTransaction {
public:
    enum class Phase {idle,running,drained,capturing,restoring,poisoned};
    struct Stage {int device=0;};
    struct Allocation {
        bool good=true;std::vector<Stage> entries{{0},{1}};std::shared_ptr<Sentinel> owner=std::make_shared<Sentinel>();
        bool healthy()const{return good;}void mark_unhealthy(){good=false;}
        const auto& stages()const{return entries;}
    } allocation_;
    struct Storage {std::shared_ptr<Sentinel> owner=std::make_shared<Sentinel>();};
    std::shared_ptr<Storage> pending_;
    std::shared_ptr<const void> runtime_=std::make_shared<Sentinel>();
    std::thread::id controller_=std::this_thread::get_id();bool initialized_=true;
    Phase phase_=Phase::drained;core::StateCut cut_{};uint64_t max_tokens_=64;
    std::vector<core::Verifier*> verifiers_;
    static bool present(const core::StateDigest& h){return std::any_of(h.begin(),h.end(),[](uint8_t b){return b!=0;});}
    static SuffixRawResult invalid(const char* why){return {SuffixRawStatus::invalid,why,{}};}
RAW_METHODS
    static cudaError_t invoke_copy(core::ResidentCopyFault& fault){
        core::ResidentCopyFaultScope scope(&fault);std::array<unsigned char,16> dst{},src{};
        return core::ResidentCopyFault::copy(dst.data(),src.data(),dst.size(),cudaMemcpyDefault,reinterpret_cast<cudaStream_t>(uintptr_t(1)));
    }
    struct Source {const core::SnapshotDescriptor* value;const auto* get()const{return value;}};
    SuffixRawResult validate_fault(core::ResidentCopyFault* fault,const Source& source){
        std::array<int,2> config{};
ACTUAL_RESTORE_GUARD
        return {SuffixRawStatus::ok,{},{}};
    }
};
namespace resident_repair_detail {
constexpr size_t maximum_stages=8;
struct Dispatch {bool permit=true;unsigned calls=0;bool retire_initialized(std::string&){++calls;return permit;}};
struct Runtime {std::shared_ptr<Dispatch> dispatch_=std::make_shared<Dispatch>();};
class ResidentRepairController {
public:
    std::shared_ptr<Runtime> runtime_=std::make_shared<Runtime>();
    SuffixRawTransaction live_;std::unique_ptr<SuffixRawTransaction> private_=std::make_unique<SuffixRawTransaction>();
    bool busy_=false,retired_=false,cancelled_=false;
    struct Receipt {uint64_t operation=7,publication=2,cut=19;int32_t token=1;};
    struct Published {core::SnapshotDescriptor metadata;uint64_t serial=2;};
    std::shared_ptr<Published> published_=std::make_shared<Published>();
    std::array<Receipt,64> receipts_{};size_t receipt_count_=1;
    uint64_t next_cut_=30;bool finish_permit=true;
    [[noreturn]] static void fatal(const char* why){core::retirement_failed(why);}
    uint64_t private_forwarded()const{return private_->cut_.forwarded;}
    bool finish_request_current(uint64_t,std::string&){return finish_permit;}
ACTUAL_COMPLETE
ACTUAL_RETIRE
    static void copy_case(const core::resident_serving_fault::Binding& binding,core::resident_serving_fault::Snapshot snapshot,bool standalone=false,bool wrong=false){
        core::ResidentCopyFault fault=standalone?core::ResidentCopyFault(binding.fd(),2,20,19,132,36,wrong?60:4,1,1):core::ResidentCopyFault(binding,snapshot,1);
        fault.stage0_completed_=true; // recording CPU copy-stage seam, NOT native stage0 evidence
        seam::calls=0;seam::kind=-1;
        require(SuffixRawTransaction::invoke_copy(fault)==seam::result&&seam::calls==1&&seam::kind==5&&fault.fired_,"actual copy hook invokes API kind5 once and returns unchanged seam status");
    }
    static void raw_purpose_case(const core::resident_serving_fault::Binding& binding,core::resident_serving_fault::Snapshot snapshot,bool corrupt){
        core::ResidentCopyFault fault(binding,snapshot,1);SuffixRawTransaction tx;
        tx.cut_={snapshot.live_cut,snapshot.accepted,snapshot.forwarded,snapshot.exact,snapshot.watermark,snapshot.history};
        core::SnapshotDescriptor source;source.cut=tx.cut_;source.cut.id=snapshot.target_cut;
        source.cut.exact_frontier=snapshot.repaired;
        if(corrupt)++source.cut.output_watermark;
        const auto result=tx.validate_fault(&fault,{&source});
        require(bool(result)!=corrupt,"actual RAW serving-purpose guard binds source/destination cuts");
    }
    static void replay_completion_case(const core::resident_serving_fault::Binding& binding,core::resident_serving_fault::Snapshot snapshot){
        ResidentRepairController controller;core::Verifier first,second;second.drain=core::Verifier::SnapshotDrain::unhealthy;
        controller.private_->verifiers_={&first,&second};controller.private_->phase_=SuffixRawTransaction::Phase::running;
        controller.private_->cut_={snapshot.private_cut,snapshot.repaired,snapshot.repaired,snapshot.repaired,snapshot.watermark,snapshot.history};
        core::resident_deferred_fault::Scope scope;
        const auto stream=reinterpret_cast<cudaStream_t>(uintptr_t(1));scope.arm_serving(binding,snapshot,&second,stream);
        core::resident_deferred_fault::observe(&second,core::resident_deferred_fault::QueryApi::event,nullptr,cudaErrorIllegalInstruction);
        require(!scope.observed_,"event error cannot masquerade as actual stream-query observation");
        core::resident_deferred_fault::observe(&first,core::resident_deferred_fault::QueryApi::stream,stream,cudaErrorIllegalInstruction);
        require(!scope.observed_,"wrong verifier cannot observe selected fault");
        core::resident_deferred_fault::observe(&second,core::resident_deferred_fault::QueryApi::stream,stream,cudaErrorIllegalInstruction);
        require(scope.observed_&&scope.observed_status_==cudaErrorIllegalInstruction,"actual observer binds query API/verifier/stream/status CPU seam");
        core::StateCut next{snapshot.target_cut,41,41,41,1,{}};next.history[0]=1;
        controller.complete(*controller.private_,next);require(false,"unhealthy actual RAW completion must noreturn");
    }
};
} // resident_repair_detail
using Phase=core::serving_raw::Phase;
inline Phase raw_phase(SuffixRawTransaction::Phase phase){
    using P=SuffixRawTransaction::Phase;
    if(phase==P::running)return Phase::running;
    if(phase==P::poisoned)return Phase::poisoned;
    return phase==P::drained?Phase::drained:Phase::idle;
}
class ServingRawSession {
public:
    core::serving_raw::Lifecycle lifecycle_;std::shared_ptr<SuffixRawTransaction> tx_=std::make_shared<SuffixRawTransaction>();bool capture_permit=true;
    ServingRawSession(){lifecycle_.set_observed_phase(Phase::drained);(void)lifecycle_.apply(core::serving_raw::Lifecycle::Event::open_request);}
    SuffixRawTransaction::Phase phase()const{return tx_->phase_;}
    struct CaptureResult {bool ok;std::string detail;explicit operator bool()const{return ok;}};
    CaptureResult capture(const core::SnapshotDescriptor&,bool& destination){destination=capture_permit;return {capture_permit,"capture refused CPU seam"};}
ACTUAL_CLOSE
ACTUAL_ABANDON
};
struct ServingSuffixRepairOperation {
    using Controller=resident_repair_detail::ResidentRepairController;
    struct Outcome {std::shared_ptr<Controller> controller=std::make_shared<Controller>();std::vector<int32_t> request_tokens{3,2,0};
        uint64_t generated=1,next_cut=0,exact_frontier=0,request_receipt_base=0;core::SnapshotDescriptor published;bool retired=false,certified_exact=false,repaired_final=false;};
ACTUAL_EXACT_READY
ACTUAL_FINISH
ACTUAL_FAILSTOP
};
} // namespace strata::program
using Manager=strata::program::ServingSuffixFaultQualification;
namespace faults=strata::core::resident_serving_fault;
using Controller=strata::program::resident_repair_detail::ResidentRepairController;
using Operation=strata::program::ServingSuffixRepairOperation;
using Session=strata::program::ServingRawSession;
std::array<char,65> identity(){std::array<char,65> h{};std::fill_n(h.data(),64,'a');return h;}
void fresh(){(void)::unlink("/output/RESIDENT-FAULT.jsonl");}
faults::Snapshot snapshot(faults::Site site){faults::Snapshot s;
    s.publication=site==faults::Site::final_exact_copy?3:2;s.live_cut=19;s.accepted=293;s.forwarded=292;s.exact=40;
    s.watermark=1;s.private_cut=18;s.repaired=site==faults::Site::final_exact_copy?292:(site==faults::Site::private_replay?40:44);s.target_cut=30;
    s.known=59;s.generated=1;s.foreground=60;s.attachments=site==faults::Site::final_exact_copy?2u:1u;
    s.receipt_operation=60;s.receipt_publication=2;s.receipt_cut=19;s.receipt_token=1;s.history[0]=7;return s;
}
template<class F> void child_failure(F run,const char* expected){
    int pipefd[2],errors[2];require(pipe(pipefd)==0&&pipe(errors)==0,"sentinel/error pipe");pid_t pid=fork();require(pid>=0,"fork");
    if(!pid){close(pipefd[0]);close(errors[0]);require(dup2(errors[1],STDERR_FILENO)>=0,"child stderr binding");close(errors[1]);
        seam::destruct_fd=pipefd[1];run();if(::write(pipefd[1],"RETURN",6)!=6)std::_Exit(93);std::_Exit(91);}
    close(pipefd[1]);close(errors[1]);int status=0;require(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==1,"exact actual noreturn exit1");
    char bytes[4096];require(::read(pipefd[0],bytes,sizeof bytes)==0,"noreturn boundary cannot unwind/free live sentinel owners or return");close(pipefd[0]);
    const auto n=::read(errors[0],bytes,sizeof bytes);close(errors[0]);require(n>0&&std::string(bytes,size_t(n)).find(expected)!=std::string::npos,"exact source failure origin required, not arbitrary child exit1");
}
void init(Operation::Outcome& out){
    auto& c=*out.controller;auto& m=c.published_->metadata;m.cut={19,4,3,3,1,{}};m.cut.history[0]=1;
    m.accepted_tokens={3,2,0,1};m.pending_tokens={1};c.private_->cut_.forwarded=3;
}
int main(){
    std::string error;
    for(const char* value:{"","0:private_replay","02:private_replay","2:unknown","2:private_replay:extra","65537:final_exact_copy"}){
        Manager invalid;require(!invalid.configure(value,true,error),"strict invalid serving fault selector");
    }
    {Manager invalid;require(!invalid.configure("2:private_replay",false,error),"unsupported/unqualified/mutually-exclusive policy refuses");}
    for(const auto& pair:std::array<std::pair<const char*,faults::Site>,3>{{{"2:private_replay",faults::Site::private_replay},{"2:second_partial_copy",faults::Site::second_partial_copy},{"2:final_exact_copy",faults::Site::final_exact_copy}}}){
        fresh();bool released=false,closed=false;int fd=-1;
        {
            Manager manager;require(manager.configure(pair.first,true,error)&&manager.target(2)&&!manager.target(1),"actual selector binds ordinal/site");
            auto credit=std::shared_ptr<const void>(new int(7),[&](const void* p){released=true;closed=fd>=0&&fcntl(fd,F_GETFD)<0&&errno==EBADF;delete static_cast<const int*>(p);});
            std::weak_ptr<const void> weak=credit;
            require(manager.prepare(2,credit,identity(),identity(),error),"real fresh durable FD preparation");fd=manager.binding().fd();
            require(credit.use_count()==2&&manager.binding().valid(),"FD owner aliases original reservation owner");credit.reset();
            require(!weak.expired()&&!released,"original credit stays alive through FD/borrowers");
            const auto s=snapshot(pair.second);
            if(pair.second==faults::Site::private_replay){
                child_failure([&]{Controller::replay_completion_case(manager.binding(),s);},"resident actual all-stage completion refused");
                std::ifstream input("/output/RESIDENT-FAULT.jsonl");std::string records((std::istreambuf_iterator<char>(input)),{});
                require(records.find("\"raw_poisoned\":1,\"allocation_unhealthy\":1,\"cut_unchanged\":1,\"failure_owner_retained\":1")!=std::string::npos,
                    "actual RAW completion poisons allocation, preserves old cut and retains failure owner");
            }else {
                require(faults::copy_scope_valid(manager.binding(),s),"dynamic actual known59/generated1/FG60 serving purpose accepted");
                auto invalid=s;invalid.foreground=4;require(!faults::copy_scope_valid(manager.binding(),invalid),"cannot substitute standalone FG4 for actual serving history");
                Controller::raw_purpose_case(manager.binding(),s,false);Controller::raw_purpose_case(manager.binding(),s,true);
                Controller::copy_case(manager.binding(),s);
                if(pair.second==faults::Site::second_partial_copy){
                    Controller::copy_case(manager.binding(),s,true);
                    child_failure([&]{Controller::copy_case(manager.binding(),s,true,true);},"partial-copy fault reached invalid operation scope");
                }
            }
        }
        require(released&&closed,"real File closes before final original credit release");
    }
    fresh();
    { // Refused pre-mutation setup retains original credit until the refusing owner is destroyed; never closes a foreign FD.
        int existing=open("/output/RESIDENT-FAULT.jsonl",O_WRONLY|O_CREAT|O_EXCL,0600);require(existing>=0,"existing receipt control");
        bool released=false;{Manager manager;require(manager.configure("2:private_replay",true,error),"refusal selector");
            auto credit=std::shared_ptr<const void>(new int(7),[&](const void* p){released=true;delete static_cast<const int*>(p);});
            require(!manager.prepare(2,credit,identity(),identity(),error),"existing receipt refuses before mutation");credit.reset();require(!released,"refusing owner still retains admitted metadata credit");}
        require(released&&fcntl(existing,F_GETFD)>=0,"refusal releases credit without closing existing receipt owner");close(existing);
    }
    fresh();
    for(int mode=0;mode<3;++mode){
        Operation::Outcome out;init(out);Session session;
        if(mode==0)session.capture_permit=false;
        if(mode==1)out.controller->runtime_->dispatch_->permit=false;
        if(mode==2)out.controller->finish_permit=false;
        auto private_owner=std::weak_ptr<Sentinel>(out.controller->private_->allocation_.owner);
        require(!Operation::finish(session,out,error)&&out.controller&&out.controller->private_&&!out.retired&&!private_owner.expired(),
                "actual finish capture/retire/publication refusal keeps controller/private owner and cannot report success");
    }
    child_failure([]{Operation::Outcome out;init(out);Session session;out.controller->runtime_->dispatch_->permit=false;
ACTUAL_MAIN_FINISH_FAILURE
    },"FAILSTOP final exact publication/retirement");
    child_failure([]{Session owner;owner.tx_->phase_=strata::program::SuffixRawTransaction::Phase::running;
        Session* serving_raw_session=&owner;std::string close_error;
ACTUAL_MAIN_CLOSE_FAILURE
    },"FAILSTOP closing retained request");
    // The newly introduced retained-yield close is its own failure path with its own exact origin: a yield that
    // cannot close its request must failstop through the yield message, must not unwind the open work fence as an
    // ordinary local, and must not be mistaken for the retained-request close above.
    child_failure([]{Session owner;owner.tx_->phase_=strata::program::SuffixRawTransaction::Phase::running;
        Session* serving_raw_session=&owner;std::string close_error;
ACTUAL_MAIN_YIELD_CLOSE_FAILURE
    },"FAILSTOP closing retained-yield request");
    std::puts("PASS actual serving fault ownership/purpose/RAW-poison/finish/retire/close CPU seams; no native fault proof");return 0;
}
'''


def take(source,marker,semi=False):
    start=source.index(marker);opening=source.index('{',start);depth=0
    for at in range(opening,len(source)):
        if source[at]=='{':depth+=1
        elif source[at]=='}':
            depth-=1
            if not depth:return source[start:at+1]+(';' if semi else '')+'\n'
    raise RuntimeError(marker)


def run(build):
    root=Path(__file__).resolve().parents[3]
    files=['src/program/generate.cpp','src/program/resident_repair_controller.inc','src/program/suffix_raw_runtime.cpp',
           'src/program/suffix_raw_runtime.hpp','src/program/serving_suffix_fault_qualification.hpp',
           'include/strata/core/resident_serving_fault.hpp','include/strata/core/resident_copy_fault.hpp',
           'include/strata/core/resident_deferred_fault.hpp','include/strata/core/resident_fault_receipt.hpp']
    pins={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
    generate=(root/files[0]).read_text();controller=(root/files[1]).read_text();raw=(root/files[2]).read_text();raw_h=(root/files[3]).read_text()
    result=take(raw_h,'enum class SuffixRawStatus',True)+take(raw_h,'struct SuffixRawResult',True)
    methods=''
    for marker in ['SuffixRawResult SuffixRawTransaction::poison(', 'SuffixRawResult SuffixRawTransaction::poll_all(', 'SuffixRawResult SuffixRawTransaction::complete_driver_work(']:
        methods+=take(raw,marker).replace('SuffixRawTransaction::','',1)
    guard=take(raw,'        if(fault){')
    close=take(raw,'bool ServingRawSession::close_request(').replace('ServingRawSession::','',1)
    abandon=take(raw,'void ServingRawSession::abandon(').replace('ServingRawSession::','',1)
    replacements={'RAW_RESULT':result,'RAW_METHODS':methods,'ACTUAL_RESTORE_GUARD':guard,
                  'ACTUAL_COMPLETE':take(controller,'    void complete('),'ACTUAL_RETIRE':take(controller,'    bool retire('),
                  'ACTUAL_CLOSE':close,'ACTUAL_ABANDON':abandon,
                  'ACTUAL_EXACT_READY':take(generate,'    static bool exact_ready('),
                  'ACTUAL_FINISH':take(generate,'    static bool finish(ServingRawSession&'),
                  'ACTUAL_FAILSTOP':take(generate,'    [[noreturn]] static void failstop(ServingRawSession&')}
    finish_failure=take(generate,'                if(!strata::program::ServingSuffixRepairOperation::finish(*serving_raw_session,operation,err))')
    replacements['ACTUAL_MAIN_FINISH_FAILURE']='Session* serving_raw_session=&session;auto& operation=out;std::string err;\n'+finish_failure
    # Message-anchored extraction with uniqueness assertions. The bare `close_request(close_error)` form is no
    # longer unique: the retained-yield request close added at generate.cpp:20356 and the retained-request close
    # this gate has always meant at :20482 both match it, and substring extraction silently took the FIRST one,
    # so the child reported the yield origin and the parent saw an arbitrary exit 90. Anchor each extraction on
    # the failstop MESSAGE that identifies the intended site and refuse ambiguity up front rather than relying
    # on indentation or positional order. Expected origins are unchanged.
    close_marker=('                        if(!serving_raw_session->close_request(close_error)){\n'
                  '                            std::fprintf(stderr,"strata serve: suffix cache (repaired): '
                  'FAILSTOP closing retained request: %s\\n",close_error.c_str());')
    yield_close_marker=('                if(!serving_raw_session->close_request(close_error)){\n'
                        '                        std::fprintf(stderr,"strata serve: suffix cache (repaired): '
                        'FAILSTOP closing retained-yield request: %s\\n",close_error.c_str());')
    for name,marker in (('retained-request close',close_marker),('retained-yield close',yield_close_marker)):
        if generate.count(marker)!=1:
            raise SystemExit('%s extraction marker is not unique: %d matches'
                             % (name,generate.count(marker)))
    replacements['ACTUAL_MAIN_CLOSE_FAILURE']=take(generate,close_marker)
    replacements['ACTUAL_MAIN_YIELD_CLOSE_FAILURE']=take(generate,yield_close_marker)
    code=PRELUDE
    for key,value in replacements.items():code=code.replace(key,value)
    path=build/'actual-serving-fault.cpp';binary=build/'actual-serving-fault';path.write_text(code)
    subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-Wconversion','-Werror',
                    '-DSTRATA_WINDOW_STATE_AVAILABLE=1','-DSTRATA_RESIDENT_FAULT_QUALIFICATION=1',
                    '-I'+str(root/'include'),'-I'+str(root/'src/program'),'-isystem','/usr/local/cuda/targets/x86_64-linux/include',str(path),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=30)
    marker='    strata::program::ServingSuffixFaultQualification serving_fault_qualification;'
    start=generate.rfind('#if defined(STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE)',0,generate.index(marker))
    end=generate.index('    // Validate the dedicated profile before model startup',start)
    gate=generate[start:end]
    off=build/'actual-fault-off.cpp';off_binary=build/'actual-fault-off'
    off.write_text('#include "serving_suffix_fault_qualification.hpp"\n#include <cstdlib>\n#include <cstdio>\nint main(int argc,char** argv){\nif(argc>1)setenv("STRATA_SUFFIX_REPAIR_FAULT_AT",argv[1],1);else unsetenv("STRATA_SUFFIX_REPAIR_FAULT_AT");\n'+gate+'return 0;\n}\n')
    subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-Wconversion','-Werror',
                    '-I'+str(root/'include'),'-I'+str(root/'src/program'),str(off),'-o',str(off_binary)],check=True)
    subprocess.run([str(off_binary)],check=True,timeout=10)
    for value in ['', '2:private_replay','2:second_partial_copy','2:final_exact_copy']:
        result=subprocess.run([str(off_binary),value],capture_output=True,text=True,timeout=10)
        if result.returncode!=2 or 'serving fault selector requires supported fault-ON qualification build' not in result.stderr:
            raise RuntimeError('actual fault-OFF premodel presence gate failed: '+repr(result))
    qualification=(root/'src/program/serving_suffix_qualification.inc').read_text()
    enabled=take(qualification,'static bool serving_qualification_enabled()')
    exact=take(generate,'    static bool exact_first()')
    prelude=r'''#include "serving_suffix_fault_qualification.hpp"
#include <cstdlib>
#include <cstdio>
namespace strata::program {
struct SuffixExperimentDriver { ENABLED };
struct ServingSuffixRepairOperation { EXACT };
}
int main(int argc,char** argv){
    struct Options {
        bool serve=true,suffix_cache=true,suffix_cache_unrepaired_experiment=false,owned_session_lifetime=true,owned_model_lifetime=true;
        int spec=0,lookup_chain=0,suffix_draft=0,batch=0,pipeline_windows=0,adapt_every=0,adapt_async=0;
        bool vision=false,prefill_head_alias=false,suffix_repair_experiment=false;
        bool suffix_repair_fault_second_arena=false,suffix_repair_fault_second_copy=false,suffix_repair_fault_deferred_trap=false;
        std::string layer_split="24",split_device="1",mtp,draft,draft_model,spec_oracle,spec_follow,suffix_repair_fault_receipt;
    } o;
    setenv("STRATA_SUFFIX_REPAIR_FAULT_AT","2:private_replay",1);setenv("STRATA_SUFFIX_REPAIR_QUALIFY","1",1);
    const std::string choice=argc>1?argv[1]:"valid";
    if(choice=="split0")o.layer_split="0";
    if(choice=="split_auto")o.layer_split="auto";
    if(choice=="split_malformed")o.layer_split="24,";
    if(choice=="device0")o.split_device="0";
    if(choice=="device_malformed")o.split_device="1,2";
    if(choice=="spec")o.spec=2;
    if(choice=="cancel")setenv("STRATA_SUFFIX_REPAIR_CANCEL_AT","2:import",1);
    if(choice=="standalone")o.suffix_repair_fault_second_copy=true;
    if(choice=="absent"){unsetenv("STRATA_SUFFIX_REPAIR_FAULT_AT");o.layer_split="unsupported";}
GATE
    return 0;
}
'''
    premodel=build/'actual-fault-premodel.cpp';premodel_binary=build/'actual-fault-premodel'
    premodel.write_text(prelude.replace('ENABLED',enabled).replace('EXACT',exact).replace('GATE',gate))
    subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-Wconversion','-Werror',
                    '-DSTRATA_WINDOW_STATE_AVAILABLE=1','-DSTRATA_RESIDENT_FAULT_QUALIFICATION=1',
                    '-I'+str(root/'include'),'-I'+str(root/'src/program'),'-isystem','/usr/local/cuda/targets/x86_64-linux/include',str(premodel),'-o',str(premodel_binary)],check=True)
    for choice in ['valid','absent','split0','split_auto','split_malformed','device0','device_malformed','spec','cancel','standalone']:
        result=subprocess.run([str(premodel_binary),choice],capture_output=True,text=True,timeout=10)
        expected=0 if choice in ['valid','absent'] else 2
        if result.returncode!=expected:raise RuntimeError('actual premodel split/device/policy gate failed '+choice+': '+repr(result))
        print('PASS actual premodel control '+choice+' exit'+str(result.returncode),flush=True)
    if any(hashlib.sha256((root/f).read_bytes()).hexdigest()!=h for f,h in pins.items()):raise RuntimeError('source mutated during actual fixture')
    print('source_pins='+repr(pins),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-dir',type=Path);args=parser.parse_args()
    if args.build_dir:args.build_dir.mkdir(parents=True,exist_ok=True);run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-serving-fault-') as directory:run(Path(directory))


if __name__=='__main__':main()
