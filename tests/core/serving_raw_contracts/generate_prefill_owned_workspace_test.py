#!/usr/bin/env python3
"""Extract actual producer/driver/binding code; backend byte counter is a CPU fixture."""
import argparse
from pathlib import Path
import re
from test_serving_thread_bridge_source import body

# CUDA is the recording boundary. Ring and strict-retirement method bodies below
# are actual headers, including their error/uncertainty decisions. This queue is
# deliberately deferred until a checked sync; it is not native DMA evidence.
CUDA = r'''
#include <cstring>
#include <algorithm>
#include <sys/wait.h>
#include <unistd.h>
using cudaError_t=int;using cudaStream_t=void*;
struct RecordedEvent{int device;};using cudaEvent_t=RecordedEvent*;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorUnknown=2;
constexpr unsigned cudaHostAllocPortable=1,cudaEventDisableTiming=2;
enum cudaMemcpyKind{cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost};
namespace recording {
int device=0,allocs=0,frees=0,events=0,copies=0,record_error=0,sync_error=0;
int fail_alloc=-1,missing_event=-1,free_receipt=-1;
std::weak_ptr<const void> payer;bool require_paid=false;
struct Copy{void* dst;const void* src;size_t bytes;};Copy queue[64];size_t queued=0;
void flush(){for(size_t i=0;i<queued;++i)std::memcpy(queue[i].dst,queue[i].src,queue[i].bytes);queued=0;}
void paid(){if(require_paid)require(!payer.expired(),"actual workspace credit survives last ring free");}
}
static cudaError_t cudaGetDevice(int* d){*d=recording::device;return cudaSuccess;}
static cudaError_t cudaSetDevice(int d){recording::device=d;return cudaSuccess;}
static cudaError_t cudaGetLastError(){return cudaSuccess;}
static const char* cudaGetErrorString(cudaError_t){return "recording CUDA error";}
static cudaError_t cudaHostAlloc(void** p,size_t n,unsigned){
 recording::paid();if(recording::allocs==recording::fail_alloc){*p=nullptr;return cudaErrorUnknown;}
 *p=std::malloc(n);if(!*p)return cudaErrorUnknown;++recording::allocs;return cudaSuccess;
}
static cudaError_t cudaFreeHost(void* p){
 recording::paid();require(recording::queued==0,"no slot free before checked completion");
 if(recording::free_receipt>=0){char c='F';(void)::write(recording::free_receipt,&c,1);}
 ++recording::frees;std::free(p);return cudaSuccess;
}
static cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e,unsigned){
 if(recording::events==recording::missing_event){*e=nullptr;return cudaSuccess;}
 *e=new RecordedEvent{recording::device};++recording::events;return cudaSuccess;
}
static cudaError_t cudaEventDestroy(cudaEvent_t e){require(e&&e->device==recording::device,"event free on its device");delete e;return cudaSuccess;}
static cudaError_t cudaStreamSynchronize(cudaStream_t){if(recording::sync_error)return recording::sync_error;recording::flush();return cudaSuccess;}
static cudaError_t cudaDeviceSynchronize(){return cudaStreamSynchronize(nullptr);}
static cudaError_t cudaEventSynchronize(cudaEvent_t e){if(!e||e->device!=recording::device)return cudaErrorInvalidValue;return cudaStreamSynchronize(nullptr);}
static cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t){if(!e||e->device!=recording::device)return cudaErrorInvalidValue;return recording::record_error;}
static cudaError_t cudaMemcpyAsync(void* d,const void* s,size_t n,cudaMemcpyKind,cudaStream_t){
 require(recording::queued<64,"bounded recording queue");++recording::copies;recording::queue[recording::queued++]={d,s,n};return cudaSuccess;
}
'''

RING_CHECKS = r'''
static void ring_contracts(){
 using core::PinnedRing;
 {
  PinnedRing legacy;char src[4]={'t','e','x','t'},dst[4]={};
  require(legacy.h2d(dst,src,4,nullptr)==cudaSuccess,"legacy positive upload");
  legacy.retire_strict();require(std::memcmp(src,dst,4)==0,"legacy upload completes");
 }
 auto ledger=budget(512ULL<<20);core::ModelGeometry g;core::SessionState ss;core::ExpertSource src;std::string err;
 auto admit=[&](prefill::Prefill& p){require(actual_driver(p,g,ss,&src,16,true,0,true,ledger,err),"real workspace admission before preparation");
  recording::payer=p.workspace_credit_;recording::require_paid=true;};
 {
  prefill::Prefill p;admit(p);int a=recording::allocs,e=recording::events;
  require(p.init_binding(g,ss,&src,16,reinterpret_cast<void*>(1),false,err),"paid four-slot init positive");
  require(recording::allocs==a+4&&recording::events==e+4,"all four actual slots/events prepared");
  auto& r=p.impl_->up_ring;char input[8]={'b','o','u','n','d','e','d','!'},device[8]={},output[8]={};
  for(int i=0;i<9;++i)require(r.h2d(device,input,8,nullptr)==cudaSuccess,"paid slot rotation");
  require(r.d2h_sync(output,device,8,nullptr)==cudaSuccess&&std::memcmp(input,output,8)==0,"paid download checked completion");
  require(r.reserve(65536)&&r.h2d(nullptr,nullptr,0,nullptr)==cudaSuccess,"bounded reserve and zero no-op");
  require(recording::allocs==a+4&&recording::events==e+4,"no pinned allocations/events after freeze");
 }
 require(ledger->snapshot().reserved.host_bytes==0,"Prefill ring/Impl freed before reservation drop");
 for(int failure=0;failure<7;++failure){
  prefill::Prefill p;admit(p);require(p.init_binding(g,ss,&src,16,reinterpret_cast<void*>(1),false,err),"fresh paid failure fixture");
  auto& r=p.impl_->up_ring;char src_bytes[8]={},dst[8]={};int copies=recording::copies;
  if(failure==0)require(!r.reserve(SIZE_MAX),"oversize refuses without growth");
  if(failure==1){recording::device=1;require(!r.acquire(8),"wrong-device refusal");recording::device=0;}
  if(failure==2){void* stage=r.acquire(8);require(stage&&r.submit(dst,src_bytes,8,nullptr)!=cudaSuccess,"forged acquired pointer refuses");}
  if(failure==3){require(r.acquire(8)&&!r.acquire(8),"double-acquire refusal");}
  if(failure==4){::setenv("STRATA_TEST_PAGEABLE","1",1);require(!r.reserve(1),"fresh forced-pageable refusal");::unsetenv("STRATA_TEST_PAGEABLE");}
  if(failure==5){void* stage=r.acquire(8);require(stage&&r.submit(dst,stage,7,nullptr)!=cudaSuccess,"exact acquired extent required");}
  if(failure==6)require(r.h2d(dst,src_bytes,8,reinterpret_cast<void*>(1))!=cudaSuccess,"foreign stream rejected before submission");
  auto first=r.owned_error();require(first!=cudaSuccess&&r.h2d(nullptr,nullptr,0,nullptr)==first,"first error survives zero-byte operation");
  require(!r.acquire(1)&&recording::copies==copies,"failed owned acquire never direct-copies");
 }
 for(int partial=0;partial<3;++partial){
  prefill::Prefill p;admit(p);int a=recording::allocs;
  if(partial==0)recording::missing_event=recording::events+2;
  if(partial==1)recording::fail_alloc=recording::allocs+2;
  if(partial==2)::setenv("STRATA_TEST_PAGEABLE","1",1);
  require(!p.init_binding(g,ss,&src,16,reinterpret_cast<void*>(1),false,err),"partial preparation refuses ready");
  require(!p.impl_->up_ring.owned_ready()&&p.workspace_credit_&&!recording::payer.expired(),"paid partial-init still retains real reservation");
  if(partial==2)require(recording::allocs==a,"forced-pageable init performs no slot allocation");
  recording::missing_event=recording::fail_alloc=-1;::unsetenv("STRATA_TEST_PAGEABLE");
 }
 {
  prefill::Prefill p;admit(p);require(p.init_binding(g,ss,&src,16,reinterpret_cast<void*>(1),false,err),"uncertain paid init");
  char from[8]={},to[8]={};recording::record_error=cudaErrorUnknown;
  require(p.impl_->up_ring.h2d(to,from,8,nullptr)==cudaErrorUnknown&&recording::queued==1,"event-record failure keeps submitted DMA uncertain");
  require(p.workspace_credit_&&!recording::payer.expired(),"uncertain submission retains credit");
  require(!p.impl_->up_ring.reserve(SIZE_MAX)&&p.impl_->up_ring.owned_error()==cudaErrorUnknown,"later refusal preserves original event-record error");
  recording::record_error=0;require(p.impl_->up_ring.drain_owned(),"actual checked drain resolves uncertainty");
  require(p.impl_->up_ring.owned_error()==cudaErrorUnknown,"successful cleanup cannot erase original operation failure");
 }
 // Both standalone retirement and the extracted Prefill ring-release phase must
 // fail-stop BEFORE any slot free when completion cannot be established.
 for(int prefill_release=0;prefill_release<2;++prefill_release){
  int receipts[2];require(::pipe(receipts)==0,"bounded cleanup receipt pipe");pid_t child=::fork();require(child>=0,"fork fail-stop check");
  if(child==0){
   ::close(receipts[0]);recording::free_receipt=receipts[1];
   prefill::Prefill p;admit(p);require(p.init_binding(g,ss,&src,16,reinterpret_cast<void*>(1),false,err),"fail-stop preparation");
   char from[8]={},to[8]={};recording::record_error=cudaErrorUnknown;
   require(p.impl_->up_ring.h2d(to,from,8,nullptr)==cudaErrorUnknown,"uncertain submission before fail-stop");
   recording::sync_error=cudaErrorUnknown;
   if(prefill_release)p.release();else p.impl_->up_ring.release();
   std::_Exit(7);
  }
  ::close(receipts[1]);int status=0;require(::waitpid(child,&status,0)==child,"join fail-stop child");char receipt=0;
  require(::read(receipts[0],&receipt,1)==0,"failed drain frees no slot");::close(receipts[0]);
  require(WIFEXITED(status)&&WEXITSTATUS(status)==1,"strict retirement fails closed");
 }
 recording::require_paid=false;recording::payer.reset();
 require(ledger->snapshot().reserved.host_bytes==0,"all recording ring groups release real workspace");
}
'''


def generate(root):
    cpp = (root / 'src/prefill/prefill.cpp').read_text()
    driver = (root / 'src/program/generate.cpp').read_text()
    header = (root / 'include/strata/prefill/prefill.hpp').read_text()
    def actual(source, marker):
        at = source.index(marker)
        block = body(source, marker)
        return source[at:source.index(block, at)] + block
    producer = actual(cpp, 'WorkspaceAdmission Prefill::admit_owned_workspace(')
    bank_target = actual(cpp, 'bool Prefill::owned_position_bank_target(')
    guard = actual(cpp, 'if(workspace_credit_&&')
    prepare = actual(cpp, 'static bool prepare_paid_uploads(')
    init = body(cpp, 'bool Prefill::init(')
    device_guard = actual(init, 'if(workspace_credit_){')
    release = body(cpp, 'void Prefill::release()')
    # The executable release seam uses the three actual final-phase statements;
    # real worker join/whole-Impl teardown ordering also has source wiring guards.
    retire = re.search(r'impl_->up_ring\.retire_strict\(\);', release)[0]
    delete_impl = re.search(r'if \(strict_retirement_\) impl_\.reset\(\);', release)[0]
    drop_credit = release[release.rindex('workspace_credit_.reset();'):].splitlines()[0]
    strict = (root / 'include/strata/core/strict_retirement.hpp').read_text()
    ring = (root / 'include/strata/core/pinned_ring.hpp').read_text()
    strict = strict.replace('#pragma once', '').replace('#include <cuda_runtime.h>', '')
    ring = ring.replace('#pragma once', '').replace('#include <cuda_runtime.h>', '').replace(
        '#include "strata/core/strict_retirement.hpp"', strict)
    device_scope = (root / 'include/strata/core/on_device.hpp').read_text().replace(
        '#pragma once', '').replace('#include <cuda_runtime.h>', '').replace(
        '#include "strata/core/strict_retirement.hpp"', '')
    driver_body = body(driver, 'auto admit_prefill_workspace =')
    constants = re.search(r'constexpr int64_t N =[^;]+;', cpp)[0]
    threads = re.search(r'inline constexpr int kMaxStagerThreads =[^;]+;', header)[0]
    return r'''
#include "strata/prefill/owned_workspace.hpp"
#include "strata/core/window_budget.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
static void require(bool b,const char* why){if(!b){std::fprintf(stderr,"FAIL: %s\n",why);std::abort();}}
''' + CUDA + '\n' + ring + '\n' + device_scope + r'''
namespace strata::program {class OwnedPositionBankBinding;}
namespace strata::core {
struct ModelGeometry{int64_t n_embd=2560,hc=4,hc_lr=320,n_expert=16,n_layers=2;};
struct SessionState{int k=10;};
struct ExpertSource{bool transient(int64_t,int64_t)const{return false;}};
}
namespace strata::kernels{constexpr int kStepCount=32,PLE_N_HEADS=4;}
namespace strata::prefill {
using cudaEvent_t=void*;
''' + constants + '\n' + threads + r'''
''' + prepare + r'''
static int64_t MAXBLOB(){return 4096;} // explicit backend byte-count fixture, no GPU allocation claim
struct Stager{struct Job{const uint8_t* src;size_t bytes;core::ExpertSource* from=nullptr;int32_t l=0,e=0;};};
struct Prefill {
    // Other Prefill backends remain a representation boundary. The real ring is
    // a member so the actual producer prices its metadata through sizeof(Impl).
    struct Impl{const core::ModelGeometry* g=nullptr;const core::SessionState* ss=nullptr;bool pp=false,help_pp=false;int device=-1;core::PinnedRing up_ring;};
    using WorkspaceAdmitter=std::function<WorkspaceAdmission(const OwnedWorkspaceBytes&,std::shared_ptr<const void>&)>;
    std::shared_ptr<const program::OwnedPositionBankBinding> position_banks_;
    bool owned_position_bank_target(const core::ModelGeometry*,const core::SessionState*,int)const noexcept;
    bool strict_retirement_=true;void* cpu_pool_=nullptr;Prefill* helper_=nullptr;void* embd_rows=nullptr;
    std::function<void()> on_routes,on_taps;
    std::shared_ptr<const void> workspace_credit_;
    const core::ModelGeometry* workspace_geometry_=nullptr;const core::SessionState* workspace_session_=nullptr;
    const core::ExpertSource* workspace_source_=nullptr;int64_t workspace_chunk_=0;
    bool workspace_borrowed_=false,workspace_successor_=false;
    std::unique_ptr<Impl> impl_=std::make_unique<Impl>();Prefill* next_=nullptr;
    static uint64_t bytes_needed(const core::ModelGeometry&,const core::SessionState&,int64_t chunk,bool){return (16ULL<<20)+uint64_t(chunk)*4096;}
    static uint64_t bytes_needed_owned(const core::ModelGeometry& g,const core::SessionState& s,int64_t c,bool src){return bytes_needed(g,s,c,src);}
    WorkspaceAdmission admit_owned_workspace(const core::ModelGeometry&,const core::SessionState&,core::ExpertSource*,int64_t,bool,const WorkspaceAdmitter&,std::string&);
    void release(){if(!impl_)return;
''' + retire + '\n' + delete_impl + '\n' + drop_credit + r'''
    }
    ~Prefill(){release();}
    bool init_binding(const core::ModelGeometry& g,core::SessionState& ss,core::ExpertSource* src,int64_t chunk,void* borrow,bool borrow_spill,std::string& err){
''' + guard + r'''
''' + device_guard + r'''
        // Recording backend init representation; the target method itself is actual.
        impl_->g=&g;impl_->ss=&ss;
        return prepare_paid_uploads(impl_->up_ring,uint64_t(chunk),uint64_t(g.n_expert),nullptr,err);
    }
};
''' + producer + '\n' + bank_target + r'''
}
using namespace strata;
static prefill::WorkspaceAdmission last_admission=prefill::WorkspaceAdmission::invalid;
static bool actual_driver(prefill::Prefill& consumer,const core::ModelGeometry& g,const core::SessionState& session,
                          core::ExpertSource* source,int64_t chunk,bool borrowed,int device,bool enabled,
                          std::unique_ptr<core::WindowBudget>& model_budget,std::string& err){
    const bool owned_batch_allocation=enabled;struct Options{int64_t prefill_chunk;};Options o{chunk};
    // Native allocation inspection is recording-only here; actual production
    // session/device membership and health guards themselves are extracted.
    struct Stage{const core::SessionState* session;int device;};
    struct Sessions{Stage stage;bool healthy()const{return true;}std::vector<Stage> stages()const{return {stage};}};
    struct Runtime{Sessions sessions;};Runtime runtime{{{&session,device}}};auto* owned_runtime=&runtime;
    cudaSetDevice(device);
    auto invoke=[&](prefill::Prefill& consumer,const core::SessionState& session,core::ExpertSource* source,bool borrowed,int device)->prefill::WorkspaceAdmission
#define STRATA_WINDOW_STATE_AVAILABLE 1
''' + driver_body + r''';
    last_admission=invoke(consumer,session,source,borrowed,device);
    return last_admission==prefill::WorkspaceAdmission::ready;
}
static auto budget(uint64_t host,uint64_t device=512ULL<<20){return std::make_unique<core::WindowBudget>(core::WindowResources{{device,device},host,host},core::WindowResources{{0,0}});}
''' + RING_CHECKS + r'''
int main(){
    core::ModelGeometry g;core::SessionState a,b;core::ExpertSource source;std::string err;
    auto ledger=budget(512ULL<<20);
    std::weak_ptr<const void> late;
    {
        prefill::Prefill primary,secondary;primary.next_=&secondary;
        require(actual_driver(secondary,g,b,&source,1024,true,1,true,ledger,err),"actual secondary producer/ledger admission");
        require(actual_driver(primary,g,a,&source,1024,true,0,true,ledger,err),"actual successor producer/ledger admission");
        const auto paid=ledger->snapshot().reserved;
        require(paid.device_bytes==std::vector<uint64_t>({2ULL<<20,2ULL<<20})&&paid.host_bytes>0&&paid.pinned_bytes==paid.host_bytes,
            "actual callback pays standalone token pages and host/pinned workspace; arena already paid");
        late=primary.workspace_credit_;require(!late.expired(),"actual producer retains genuine reservation");
        require(primary.init_binding(g,a,&source,1024,reinterpret_cast<void*>(1),false,err),"actual init accepts identical owners/shape");
        require(primary.owned_position_bank_target(&g,&a,0),"actual bank target accepts real issued workspace and matching backend inventory");
        core::SessionState foreign_bank_session;
        require(!primary.owned_position_bank_target(&g,&foreign_bank_session,0)&&!primary.owned_position_bank_target(&g,&a,1),
            "actual bank target rejects foreign session/device without minting authority");
        require(!primary.init_binding(g,a,&source,512,reinterpret_cast<void*>(1),false,err),"actual init rejects unquoted chunk");
        core::SessionState other;
        require(!primary.init_binding(g,other,&source,1024,reinterpret_cast<void*>(1),false,err),"actual init rejects replacement session");
        require(!primary.init_binding(g,a,&source,1024,nullptr,false,err),"actual init rejects new own allocation under borrowed quote");
        primary.helper_=&secondary;
        require(!primary.init_binding(g,a,&source,1024,reinterpret_cast<void*>(1),false,err),"actual init rejects uncovered helper after quote");
        primary.helper_=nullptr;
        require(!actual_driver(primary,g,a,&source,1024,true,0,true,ledger,err),"actual producer refuses replacement before checked reset");
    }
    require(late.expired()&&ledger->snapshot().reserved.host_bytes==0,"actual producer owner retains then releases admission");
    {
        auto short_host=budget(1);prefill::Prefill p;
        require(!actual_driver(p,g,a,&source,1024,false,0,true,short_host,err)&&short_host->snapshot().reserved.host_bytes==0&&!p.workspace_credit_,
            "actual ledger refusal happens before init without partial charge");
        require(last_admission==prefill::WorkspaceAdmission::budget_refused,"actual quota refusal is eligible for existing checked smaller-chunk retry");
    }
    {
        prefill::Prefill p;auto current=ledger->snapshot().reserved;
        require(actual_driver(p,g,a,&source,1024,false,0,false,ledger,err)&&!p.workspace_credit_&&ledger->snapshot().reserved==current,
            "actual default-off driver path unchanged");
        require(!actual_driver(p,g,a,&source,0,false,0,true,ledger,err),"actual zero shape refused");
        require(last_admission==prefill::WorkspaceAdmission::invalid,"invalid shape is never mislabeled capacity shrink");
        p.cpu_pool_=reinterpret_cast<void*>(1);
        require(!actual_driver(p,g,a,&source,1024,false,0,true,ledger,err),"actual uncovered CPU consumer refused");
    }
    ring_contracts();
    std::puts("PASS extracted paid Prefill producer, actual ledger callback, real frozen ring and init owner binding");
}
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--generate', type=Path, required=True)
    args = parser.parse_args()
    args.generate.parent.mkdir(parents=True, exist_ok=True)
    args.generate.write_text(generate(args.source_root))
