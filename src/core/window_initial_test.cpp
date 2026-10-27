#include "strata/core/window_restore.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/core/window_selection.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "window_native_test_fixture.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static int device=0,zeros=0,binds=0,drains=0,fail_zero=0,fail_bind=0,fail_drain=0;
static bool active=false,throw_zero=false,fail_restore_device=false,fail_return_only=false;
static cudaError_t last_error=cudaSuccess;
static std::map<void*,int> allocations;
extern "C" cudaError_t cudaGetDevice(int* d){*d=device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){if(active&&fail_restore_device&&device==1&&d==0)return cudaErrorUnknown;device=d;return cudaSuccess;}
extern "C" cudaError_t cudaMalloc(void** p,size_t n){*p=std::malloc(n);allocations[*p]=device;return *p?cudaSuccess:cudaErrorMemoryAllocation;}
extern "C" cudaError_t cudaFree(void* p){check(allocations.at(p)==device,"free owning device");allocations.erase(p);std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){return active&&++drains==fail_drain?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t cudaGetLastError(){auto error=last_error;last_error=cudaSuccess;return error;}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected initial error";}
namespace strata::kernels { std::size_t native_q8_1_bytes(int,int){return 4096;} }
namespace strata::core {
bool session_private_plan(const ModelGeometry&,int64_t,int64_t,int64_t,int64_t,SessionPrivatePlan& p){p={1024,4096,512};return true;}
uint64_t ple_run_scratch_bytes(){return 512;}
uint64_t session_init_private(const ModelGeometry& g,int64_t,int64_t,void* arena,SessionState& s,SessionInitUploads&,int64_t lo,int64_t hi){
    s.layer_lo=lo;s.layer_hi=hi;s.qsa_ord0=lo/4;s.qsa_alloc=1;
    s.qsa_states=new QsaState[size_t(g.n_qsa_layers())]();s.qsa_states[s.qsa_ord0].kv_int8=true;
    s.ple_hist=static_cast<float*>(arena);return 1024;
}
bool qsa_state_release_host(QsaState&,std::string&){return true;}
void session_release(SessionState&){}
void session_zero(SessionState& s,const ModelGeometry&,const float* initial,void* stream){
    check(initial==nullptr&&stream==nullptr,"empty initial zero uses null residual and default stream");
    ++zeros;s.ple_prev[0]=s.ple_prev[1]=-1;s.ple_token=-1;
    if(zeros==fail_zero){if(throw_zero)throw std::runtime_error("zero exception");last_error=cudaErrorUnknown;}
}
bool session_zero_checked(SessionState& s,const ModelGeometry& g,const float* initial,void* stream,std::string& error){
    session_zero(s,g,initial,stream);
    if(fail_return_only){error="direct reset submission failure with clean last-error";return false;}
    if(last_error!=cudaSuccess){error="reset submission failure";return false;}return true;
}

bool conversation_snapshot_validate(const SavedConversation&,const SessionState&,const ModelGeometry&,const QsaState*,std::string&){check(false,"fresh init must not validate invented snapshot");return false;}
ConversationRestore conversation_snapshot_restore(const SavedConversation&,SessionState&,const ModelGeometry&,const QsaState*,std::string&){check(false,"fresh init must not restore invented snapshot");return ConversationRestore::transfer_failed;}
}
struct Runtime : NativeRuntimeAttachment {};
template<class Fn>void isolated(Fn fn){const auto p=fork();check(p>=0,"fork");if(!p){fn();std::exit(0);}int status=0;check(waitpid(p,&status,0)==p&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"isolated initial case");}
int main(){
    const auto g=window_test::geometry();const auto key=window_test::metadata().key;
    WindowResources capacity{{100000,100000},32<<20,10000,0,0,2},zero{{0,0},0,0,0,0,0};
    std::vector<NativeSessionRequest> requests{{0,0,4,true},{1,4,8,false}};
#if defined(STRATA_GPU_METAL)
    {
        WindowBudget budget(capacity,zero);NativeSessionAllocation allocation;PreparedNativeWindow ready;
        check(prepare_owned_native_initial(key,g,allocation,{},budget,1024,1<<20,true,ready).status==NativeRestoreStatus::invalid&&
              zeros==0&&allocations.empty(),"Metal refusal precedes all device/session work");
        std::puts("window_initial Metal refusal seam passed");return 0;
    }
#endif
    auto bind=[](SessionState& s,std::string& error){check(s.ple_token==-1,"zero before binder");if(++binds==fail_bind){error="binding failure";return false;}return true;};
    std::vector<NativeSessionBinder> binders{bind,bind};
    {
        WindowBudget budget(capacity,zero);NativeSessionAllocation allocation;
        check(bool(allocate_native_sessions(g,16,1,requests,budget,allocation)),"factory allocation");
        auto* first=allocation.stages()[0].session;
        PreparedNativeWindow ready;active=true;
        auto prepare=[&](const auto& b){return prepare_owned_native_initial(key,g,allocation,b,budget,1024,1<<20,true,ready);};
        auto bad=binders;bad[1]={};check(prepare(bad).status==NativeRestoreStatus::invalid&&zeros==0&&!ready,"all binders validated before zero");
        const auto hi=first->layer_hi;first->layer_hi=3;
        check(prepare(binders).status==NativeRestoreStatus::invalid&&zeros==0,"broken topology before zero");first->layer_hi=hi;
        auto tiny=capacity;tiny.host_bytes=1;tiny.pinned_bytes=0;WindowBudget small(tiny,zero);
        check(prepare_owned_native_initial(key,g,allocation,binders,small,1024,1<<20,true,ready).status==NativeRestoreStatus::budget_refused&&zeros==0,"metadata refused before device work");
        auto runtime=std::make_shared<Runtime>();std::weak_ptr<Runtime> held=runtime;
        check(bool(prepare_owned_native_initial(key,g,allocation,binders,budget,1024,1<<20,true,ready,runtime)),"fresh initial prepares");
        check(!ready.descriptor()&&zeros==2&&binds==2&&drains==2&&device==0,"no snapshot; all stages zero/bind/drain and restore device");
        check(prepare(binders).status==NativeRestoreStatus::invalid&&zeros==2&&ready&&!ready.descriptor(),"alias claim refuses and preserves output");
        {
            WindowSelection selected(ready);const auto token=selected.token();
            StateCut initial_cut;initial_cut.id=1;initial_cut.history=window_history_digest({});
            check(token.key==key&&token.cut==initial_cut,"initial exact key and first empty-history cut");
            SelectedWindowWork access;check(selected.admit(token,access),"initial selection admits");
            check(access.borrow([&](const auto& views,const auto& attachment){check(views.size()==2&&views[0].session==first&&views[1].session==allocation.stages()[1].session&&attachment==runtime,"exact factory sessions/runtime");}),"borrow initial");
            check(selected.quiesce(token),"quiesce initial");auto work=access.lease();auto cut=token.cut;++cut.id;
            check(work.finish(work.operation(),key,{cut,cut},true,true),"initial work finishes");
            WindowRead read;const auto settled=selected.token();check(selected.acquire_read(settled,read),"initial cut read");
            check(!selected.replace(settled,read,ready),"descriptorless replacement refused");
            runtime.reset();ready={};check(!held.expired(),"selection retains runtime attachment");
        }
        check(held.expired(),"runtime releases with selected owner");active=false;allocation={};
        check(allocations.empty()&&budget.snapshot().reserved==zero,"fresh owner and reservations released");
    }
    for(int mode=0;mode<7;++mode)isolated([&]{
        active=false;zeros=binds=drains=0;WindowBudget budget(capacity,zero);NativeSessionAllocation allocation;
        check(bool(allocate_native_sessions(g,16,1,requests,budget,allocation)),"failure factory fixture");
        auto alias=allocation;auto owner=std::weak_ptr<const void>(allocation.stages()[0].owner);
        auto runtime=std::make_shared<Runtime>();std::weak_ptr<Runtime> runtime_owner=runtime;
        if(mode==0)fail_zero=2;
        if(mode==1)fail_bind=2;
        if(mode==2)fail_drain=2;
        if(mode==3)last_error=cudaErrorUnknown;
        if(mode==4){fail_zero=1;throw_zero=true;}
        if(mode==5)fail_restore_device=true;
        if(mode==6)fail_return_only=true;
        active=true;PreparedNativeWindow ready;
        auto failure=prepare_owned_native_initial(key,g,allocation,binders,budget,1024,1<<20,true,ready,runtime);
        check(failure.status==NativeRestoreStatus::unhealthy&&failure.failure&&!ready&&!allocation.healthy()&&!alias.healthy(),"zero/bind/drain/error failure poisons aliases and refuses output");
        check(mode!=3||zeros==0,"preexisting CUDA error not erased as healthy");
        runtime.reset();allocation={};alias={};check(!owner.expired()&&!runtime_owner.expired()&&budget.snapshot().reserved.shadows==1,"pending failure retains all owners and charge");
        active=false;fail_restore_device=false;failure.failure->retire_after_teardown();
        check(runtime_owner.expired()&&owner.expired()&&!allocations.empty()&&native_session_quarantine_count()==1&&budget.snapshot().reserved.shadows==1,"confirmed teardown releases runtime but poisoned factory retains storage lease");
    });
    isolated([&]{
        active=false;zeros=binds=drains=0;WindowBudget budget(capacity,zero);
        NativeSessionAllocation first,second;
        check(bool(allocate_native_sessions(g,16,1,requests,budget,first)),"existing initial fixture");
        check(bool(allocate_native_sessions(g,16,1,requests,budget,second)),"replacement failure fixture");
        PreparedNativeWindow ready;active=true;
        check(bool(prepare_owned_native_initial(key,g,first,binders,budget,1024,1<<20,true,ready)),"existing output prepares");
        WindowSelection selected(ready);const auto token=selected.token();
        auto bad=binders;bad[1]={};const auto before=zeros;
        check(prepare_owned_native_initial(key,g,second,bad,budget,1024,1<<20,true,ready).status==NativeRestoreStatus::invalid&&zeros==before,
              "bad binder preserves existing prepared output before writes");
        auto launch_error=binders;launch_error[1]=[](SessionState&,std::string&){last_error=cudaErrorUnknown;return true;};
        auto failed=prepare_owned_native_initial(key,g,second,launch_error,budget,1024,1<<20,true,ready);
        check(failed.status==NativeRestoreStatus::unhealthy&&failed.failure&&ready&&!ready.descriptor(),"binder launch error retained despite success return");
        bool same=false;try{WindowSelection duplicate(ready);}catch(const std::invalid_argument&){same=true;}
        check(same&&selected.token()==token,"failed preparation did not replace existing claimed prepared output");
        failed.failure->retire_after_teardown();active=false;
    });
    std::puts("window_initial_test passed");
}
