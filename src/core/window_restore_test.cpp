#include "strata/core/window_restore.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "window_native_test_fixture.hpp"
#include "strata/core/window_selection.hpp"
#include <thread>
#include <atomic>
#include <cstdio>
#include <cstdlib>

using namespace strata::core;
using namespace strata::core::window_test;
static int current_device=0,validations=0,writes=0,bindings=0,fail_validate=0,fail_write=0,fail_binding=0;
static int get_calls=0,set_calls=0,fail_get_at=0,fail_set_at=0;
static cudaError_t sync_status=cudaSuccess;
extern "C" cudaError_t cudaGetDevice(int* d) { if(++get_calls==fail_get_at)return cudaErrorUnknown;*d=current_device;return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int d) { if(++set_calls==fail_set_at)return cudaErrorUnknown;current_device=d;return cudaSuccess; }
extern "C" cudaError_t cudaDeviceSynchronize() { return sync_status; }
extern "C" const char* cudaGetErrorString(cudaError_t) { return "injected sync failure"; }
extern "C" cudaError_t cudaGetLastError(){return cudaSuccess;}
namespace strata::core {
void session_zero(SessionState&,const ModelGeometry&,const float*,void*){}
bool session_zero_checked(SessionState&,const ModelGeometry&,const float*,void*,std::string&){return true;}
bool conversation_snapshot_validate(const SavedConversation&,const SessionState&,const ModelGeometry&,const QsaState*,std::string& error) {
    if(++validations==fail_validate) { error="injected final-stage validation failure";return false; }return true;
}
ConversationRestore conversation_snapshot_restore(const SavedConversation&,SessionState& session,const ModelGeometry&,const QsaState*,std::string& error) {
    ++writes;session.ple_prev[0]=999;
    if(writes==fail_write) { error="injected partial restore failure";return ConversationRestore::transfer_failed; }
    return ConversationRestore::restored;
}
}
static void check(bool ok,const char* why) { if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);} }
struct Runtime final : NativeRuntimeAttachment { explicit Runtime(int n):id(n){} int id; };
struct Destination {
    SessionState sessions[2];QsaState qsa[2]{};
    Destination() {
        for(int i=0;i<2;++i) {
            qsa[i].kv_int8=true;
            auto& s=sessions[i];s.layer_lo=i*4;s.layer_hi=i*4+4;s.qsa_states=qsa;s.qsa_alloc=1;s.qsa_ord0=i;
            if(!i) s.ple_hist=reinterpret_cast<float*>(uintptr_t(1));
        }
    }
};
int main() {
    auto meta=metadata();auto g=geometry();
    auto native=saved(0,4,1,true);native.stage_images.push_back(saved(4,8,1,false));
    std::vector<NativeWindowStage> config{{0,4,{1},true},{4,8,{1},false}};
    WindowStateLayout layout;
    check(bool(native_window_layout(g,config,meta.key,meta.cut,1024,1<<20,layout)),"fixture layout");
    SnapshotDescriptor descriptor;
    check(bool(describe_native_window(native,g,layout,meta,{{{7,8},9},{{-1,-1},-1}},descriptor)),"fixture native descriptor");
    BundleHandle source;check(bool(BundleHandle::freeze(descriptor,meta.key,layout,source)),"fixture immutable source");
    WindowBudget budget(WindowResources{{100,100},32<<20,0,0,0,2},WindowResources{{0,0},0,0,0,0,0});
    auto dst=std::make_shared<Destination>();
    auto reservation=budget.reserve(WindowResources{{10,10},1024,0,0,0,1},ReservationPurpose::private_work).lease;
    auto stages=[&] {
        std::vector<NativePrivateStage> result;
        for(int i=0;i<2;++i) result.push_back({i,&dst->sessions[i],dst,[](std::string& e){
            if(++bindings==fail_binding) {e="injected private binding failure";return false;}return true;}});
        return result;
    };
    auto reset=[] { current_device=get_calls=set_calls=fail_get_at=fail_set_at=0;validations=writes=bindings=fail_validate=fail_write=fail_binding=0;sync_status=cudaSuccess; };
    PreparedNativeWindow ready;
    auto runtime=std::make_shared<Runtime>(1);
    auto prepare=[&] { return prepare_native_restore(source,meta.key,g,stages(),reservation,budget,1024,1<<20,true,ready,runtime); };
    reset();fail_validate=2;
    check(prepare().status==NativeRestoreStatus::invalid && writes==0 && bindings==0 && !ready,
          "last-stage validation failure precedes every write");
    reset();fail_get_at=1;
    auto device_failed=prepare();
    check(device_failed.status==NativeRestoreStatus::unhealthy&&device_failed.failure&&writes==0,
          "first device query error cannot be erased by a later successful query");
    device_failed.failure->retire_after_teardown();
    reset();fail_set_at=1;
    device_failed=prepare();
    check(device_failed.status==NativeRestoreStatus::unhealthy&&device_failed.failure&&writes==1,
          "second-stage device selection failure prevents its writes");
    device_failed.failure->retire_after_teardown();
    reset();fail_set_at=2;
    device_failed=prepare();
    check(device_failed.status==NativeRestoreStatus::unhealthy&&device_failed.failure&&writes==2&&!ready,
          "failed caller-device restoration cannot publish a prepared gang");
    device_failed.failure->retire_after_teardown();
    reset();check(bool(prepare()),"complete private restore prepares");
    check(validations==2 && writes==2 && bindings==2 && ready.descriptor()->cut==meta.cut,"all stage state and bindings prepared");
    check(dst->sessions[0].ple_prev[0]==7 && dst->sessions[0].ple_prev[1]==8 && dst->sessions[0].ple_token==9,
          "actual PLE token state restored after checkpoint helper");
    check(current_device==0,"owning-device selection restored");
    reset();check(prepare().status==NativeRestoreStatus::invalid && writes==0,"prepared destination cannot be overwritten by another transaction");
    const auto original_destination=dst;
    dst=std::make_shared<Destination>();
    reservation=budget.reserve(WindowResources{{10,10},1024,0,0,0,1},ReservationPurpose::private_work).lease;
    const auto* old=ready.descriptor();const uint64_t held=budget.snapshot().reserved.host_bytes;
    reset();fail_write=2;
    auto failed=prepare();
    check(failed.status==NativeRestoreStatus::unhealthy && failed.failure && writes==2 && bindings==1,
          "late partial transfer failure retains transaction");
    check(ready.descriptor()==old && budget.snapshot().reserved.host_bytes>held,"failed prepare cannot replace existing prepared result");
    check(original_destination->sessions[0].ple_prev[0]==7,"existing prepared destination remains untouched");
    failed.failure->retire_after_teardown();
    check(budget.snapshot().reserved.host_bytes==held,"teardown retires failed staging only");
    reset();fail_binding=1;failed=prepare();
    check(failed.status==NativeRestoreStatus::unhealthy && failed.failure && writes==1,"binding failure prevents ready state");
    failed.failure->retire_after_teardown();
    reset();sync_status=cudaErrorUnknown;failed=prepare();
    check(failed.status==NativeRestoreStatus::unhealthy && failed.failure,"post-binding device failure prevents ready state");
    failed.failure->retire_after_teardown();
    auto wrong=meta.key;++wrong.selection_epoch;reset();
    check(prepare_native_restore(source,wrong,g,stages(),reservation,budget,1024,1<<20,true,ready).status==NativeRestoreStatus::invalid && writes==0,
          "stale identity rejected before device writes");
    ready={};check(budget.snapshot().reserved.host_bytes==1024,"prepared handle owns staging accounting until release");
    reset();check(bool(prepare()),"initial selection fixture prepares");
    {
        WindowSelection selected(ready);
        bool duplicate=false;
        try { WindowSelection again(ready); } catch(const std::invalid_argument&) { duplicate=true; }
        check(duplicate,"one prepared gang cannot back two execution selections");
        auto initial=selected.token();SelectedWindowWork access;
        check(selected.admit(initial,access),"selected writer and runtime admitted atomically");
        auto work=access.lease();
        const auto old_session=&dst->sessions[0];
        check(access.borrow([&](const auto& views,const auto& attachment){
            check(views.size()==2&&views[0].session==old_session&&views[1].device==1,"admitted exact stages");
            check(std::dynamic_pointer_cast<Runtime>(attachment)->id==1,"admitted runtime attachment");
            check(selected.token()==initial,"borrow callback runs outside selection lock");
        }),"active work borrow");
        check(!selected.admit(initial,access)&&access.lease().operation()==work.operation(),"busy admission leaves output unchanged");
        check(selected.quiesce(initial),"selected quiescence closes admission");
        check(access.borrow([](const auto&,const auto&){}),"admitted operation retains borrow during quiescing");
        auto completed=initial.cut;++completed.id;
        check(work.finish(work.operation(),initial.key,{completed,completed},true,true),"selected operation completes common cut");
        bool called=false;
        check(!access.borrow([&](const auto&,const auto&){called=true;})&&!called,"completed work cannot borrow");
        auto settled=selected.token();WindowRead read;
        SelectedWindowRead read_access;
        check(selected.acquire_read(settled,read_access),"selected read with exact stages");
        check(read_access.borrow([&](const auto& views,const auto& attachment){
            check(views[0].session==old_session&&std::dynamic_pointer_cast<const Runtime>(attachment)->id==1,"read exact old runtime");
            check(selected.token()==settled,"read callback outside lock");
        }),"quiesced read borrow");
        check(!selected.acquire_read(initial,read_access)&&read_access.token()==settled,"stale read leaves output unchanged");
        check(!selected.acquire_read(initial,read),"old cut token cannot acquire settled snapshot");
        check(selected.acquire_read(settled,read),"settled selected cut held");
        auto old_owner=std::weak_ptr<Destination>(dst);
        auto old_runtime=std::weak_ptr<Runtime>(runtime);
        runtime=std::make_shared<Runtime>(2);
        dst=std::make_shared<Destination>();
        reservation=budget.reserve(WindowResources{{10,10},1024,0,0,0,1},ReservationPurpose::private_work).lease;
        reset();check(bool(prepare()),"replacement gang prepared independently");
        check(!selected.replace(initial,read,ready),"stale selection cut cannot commit");
        auto foreign_lease=budget.reserve(WindowResources{{0,0},1,0,0,0,0},ReservationPurpose::foreground).lease;
        WindowExecution foreign(settled.key,settled.cut,2,std::make_shared<int>(1),foreign_lease);
        check(foreign.begin_quiesce(settled.key),"foreign gate quiesces");
        WindowRead foreign_read;check(foreign.acquire_read(settled.key,foreign_read),"foreign read fixture");
        check(!selected.replace(settled,foreign_read,ready),"equal metadata cannot substitute a foreign execution read hold");
        auto stale_append=settled;++stale_append.key.append_epoch;
        check(!selected.replace(stale_append,read,ready),"stale append identity cannot commit");
        std::atomic<int> committed{0};
        std::thread a([&]{if(selected.replace(settled,read,ready))++committed;});
        std::thread b([&]{if(selected.replace(settled,read,ready))++committed;});
        a.join();b.join();
        check(committed==1,"concurrent replacement has one atomic winner");
        check(!selected.admit(settled,work) && !selected.resume(settled),"old selection cannot mutate or resume retired gang");
        check(!read.usable() && !old_owner.expired(),"old readers retain retired storage without execution authority");
        check(!read_access.borrow([](const auto&,const auto&){}),"retired read denies access while retaining owner");
        check(!old_runtime.expired(),"old work/read retain old runtime attachment");
        check(!selected.admit(settled,access)&&access.token()==initial,"stale writer output unchanged");
        read={};work={};access={};read_access={};check(old_runtime.expired(),"old attachment released with final access");
        check(old_owner.expired(),"retired gang released after final operation/read hold");
        auto current=selected.token();check(current.publication==settled.publication+1,"publication serial fences same-content replacement");
        check(selected.admit(current,access),"replacement gang admits runtime work");
        work=access.lease();
        check(access.borrow([&](const auto& views,const auto& attachment){
            check(views[0].session==&dst->sessions[0]&&views[0].session!=old_session,"replacement selects new session");
            check(std::dynamic_pointer_cast<Runtime>(attachment)->id==2,"replacement selects new attachment");
        }),"new runtime borrowed");
        selected.quiesce(current);auto cut=current.cut;++cut.id;
        check(work.finish(work.operation(),current.key,{cut,cut},true,true),"replacement work drains");
        work={};access={};
        auto done=selected.token();
        check(selected.resume(done),"resume for unhealthy access fixture");
        check(selected.admit(done,access),"unhealthy access fixture admitted");
        work=access.lease();
        check(work.mark_unhealthy(work.operation()),"mark selected operation unhealthy");
        check(!access.borrow([](const auto&,const auto&){}),"unhealthy work cannot borrow");
        check(work.retire_after_teardown(work.operation()),"test confirms teardown of mock devices");
        check(!access.borrow([](const auto&,const auto&){}),"retired work cannot borrow");
        work={};access={};
    }
    ready={};reset();check(bool(prepare()),"pending-owner fixture prepares");
    auto pending_session=std::weak_ptr<Destination>(dst);
    auto pending_runtime=std::weak_ptr<Runtime>(runtime);
    WindowWork pending;
    {
        WindowSelection selected(ready);SelectedWindowWork access;
        check(selected.admit(selected.token(),access),"pending-owner fixture admits");
        pending=access.lease();
    }
    ready={};dst.reset();runtime.reset();
    check(!pending_session.expired()&&!pending_runtime.expired(),"pending execution retains runtime after access and selection disappear");
    check(pending.mark_unhealthy(pending.operation()),"pending-owner fixture marks failed work");
    check(pending.retire_after_teardown(pending.operation()),"pending-owner fixture confirms mock teardown");
    pending={};
    check(pending_session.expired()&&pending_runtime.expired(),"runtime and sessions release only after pending completion handle retires");
    std::puts("window_restore_test passed");
}
