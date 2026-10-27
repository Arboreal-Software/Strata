#include "strata/core/window_restore.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/core/window_selection.hpp"
#include "strata/core/window_manager.hpp"
#include <thread>
#include <condition_variable>
#include <limits>
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
static int device=0,zeros=0,drains=0,fail_zero=0,fail_drain=0;
static bool active=false,throw_zero=false,fail_restore_device=false;
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
    s.ple_hist=lo==0?static_cast<float*>(arena):nullptr;return 1024;
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
    if(last_error!=cudaSuccess){error="reset submission failure";return false;}return true;
}

bool conversation_snapshot_validate(const SavedConversation&,const SessionState&,const ModelGeometry&,const QsaState*,std::string&){return true;}
ConversationRestore conversation_snapshot_restore(const SavedConversation&,SessionState&,const ModelGeometry&,const QsaState*,std::string&){return ConversationRestore::restored;}
}
using S=WindowManagerStatus;
struct Barrier {
    std::mutex mutex;std::condition_variable cv;bool arrived=false,released=false;
    void stop(){std::unique_lock lock(mutex);arrived=true;cv.notify_all();cv.wait(lock,[&]{return released;});}
    void wait(){std::unique_lock lock(mutex);cv.wait(lock,[&]{return arrived;});}
    void release(){std::lock_guard lock(mutex);released=true;cv.notify_all();}
};
static thread_local Barrier* replacement_barrier=nullptr;
static thread_local Barrier* publication_barrier=nullptr;
// Only the private test source copy calls these at the two unlocked busy boundaries.
namespace strata::core {
void manager_test_replace_busy(){if(replacement_barrier)replacement_barrier->stop();}
void manager_test_publish_busy(){if(publication_barrier)publication_barrier->stop();}
}
struct Runtime : NativeRuntimeAttachment {
    std::function<void()> on_destroy;
    ~Runtime(){if(on_destroy)on_destroy();}
};
struct Fixture {
    NativeSessionAllocation allocation;PreparedNativeWindow prepared;
    std::shared_ptr<Runtime> runtime;
};
static WindowKey key(uint64_t generation){auto k=window_test::metadata().key;k.window_generation=generation;k.window_id="window-"+std::to_string(generation);k.session_id="user-"+std::to_string(generation);return k;}
static Fixture fresh(WindowBudget& budget,const WindowKey& k){
    Fixture f;const auto g=window_test::geometry();
    check(bool(allocate_native_sessions(g,16,1,{{0,0,4,true},{1,4,8,false}},budget,f.allocation)),"allocate initial fixture");
    f.runtime=std::make_shared<Runtime>();
    check(bool(prepare_owned_native_initial(k,g,f.allocation,{[](auto&,auto&){return true;},[](auto&,auto&){return true;}},
        budget,1024,1<<20,true,f.prepared,f.runtime)),"prepare real fresh window");
    return f;
}
static WindowManagerLimits limits(size_t entries=8){
    auto k=key(1);return {k.engine_generation,77,entries,128,0,0,{{k.model_profile,k.topology,1},{k.model_profile,k.topology,1}}};
}
static ManagedWindowToken reg(WindowManager& m,Fixture& f,const WindowKey& k,uint64_t runtime_class=1){
    WindowRegistrationTicket ticket;ManagedWindowToken out;
    check(m.reserve_registration(k,runtime_class,ticket)==S::ok,"reserve entry");
    check(m.publish_registration(ticket,f.prepared,out)==S::ok,"publish fresh");return out;
}
static WindowLaneToken lane(WindowManager& m,size_t i=0){WindowLaneToken out;check(m.lane_snapshot(i,out)==S::ok,"lane snapshot");return out;}
static ManagedWindowToken token(WindowManager& m,const ManagedWindowToken& old){ManagedWindowSnapshot out;check(m.entry_snapshot(old.registration,out)==S::ok,"entry snapshot");return out.token;}
static void finish(ManagedWindowWork& work,const ManagedWindowToken& t){auto lease=work.lease();auto cut=t.selection.cut;++cut.id;check(lease.finish(lease.operation(),t.selection.key,{cut,cut},true,true),"common cut finish");}
int main(){
    alarm(45);
    WindowBudget budget(WindowResources{{1000000,1000000},256<<20,10000,0,0,100},WindowResources{{0,0},0,0,0,0,0});
    {
        WindowManager m(limits(),budget);
        auto a=fresh(budget,key(1)),b=fresh(budget,key(2));
        auto ta=reg(m,a,key(1)),tb=reg(m,b,key(2));
        check(!a.prepared.descriptor()&&ta.selection.cut.id==1&&ta.selection.cut.history==window_history_digest({}),"fresh initial metadata no snapshot");
        WindowLaneToken bound;auto empty=lane(m);
        check(m.switch_active(empty,ta,bound)==S::ok,"bind A");auto old_a_lane=bound;
        ManagedWindowWork wa;check(m.admit(bound,wa)==S::ok,"A admission");
        check(wa.token()==ta,"admitted full registration and selection token");
        check(wa.selected().borrow([&](const auto& views,const auto& runtime){check(views[0].session==a.allocation.stages()[0].session&&runtime==a.runtime,"exact A runtime and sessions");}),"A borrow");
        check(m.request_park(bound)==S::ok,"close A admission");
        auto closed=lane(m);WindowLaneToken untouched=empty;
        check(m.switch_active(closed,tb,untouched)==S::pending&&untouched==empty,"wait for actual work drain no partial route");
        check(m.admit(old_a_lane,wa)==S::stale,"old routing fenced");
        auto lease=wa.lease();auto cut=ta.selection.cut;++cut.id;
        check(!lease.finish(lease.operation(),ta.selection.key,{cut,cut},true,false),"undrained finish refused");
        finish(wa,ta);ta=token(m,ta);closed=lane(m);
        SelectedWindowRead read_a;check(m.acquire_read(ta,read_a)==S::ok,"capture hold A");
        check(m.switch_active(closed,tb,bound)==S::ok,"A to B");
        check(read_a.borrow([&](const auto& views,const auto& runtime){check(views[0].session==a.allocation.stages()[0].session&&runtime==a.runtime,"parked exact read owner");}),"A read while B selected");
        ManagedWindowWork wb;check(m.admit(bound,wb)==S::ok,"B admission");finish(wb,tb);tb=token(m,tb);
        check(m.request_park(lane(m))==S::ok,"park B");closed=lane(m);
        check(m.switch_active(closed,ta,untouched)==S::read_held&&lane(m)==closed,"target read holds prevent resume without route change");
        read_a={};check(m.switch_active(closed,ta,bound)==S::ok,"B to A");
        check(bound.window->selection.cut.id==2&&token(m,tb).selection.cut.id==2,"independent cuts retained");
        check(m.admit(old_a_lane,wa)==S::stale,"ABA old lane rejected");
        // Independent lanes can execute distinct windows at the same time.
        auto empty1=lane(m,1);WindowLaneToken bound1;
        check(m.switch_active(empty1,tb,bound1)==S::ok,"bind B independent lane");
        ManagedWindowWork two_a,two_b;
        std::thread first([&]{check(m.admit(bound,two_a)==S::ok,"concurrent A admit");});
        std::thread second([&]{check(m.admit(bound1,two_b)==S::ok,"concurrent B admit");});first.join();second.join();
        check(m.request_cancel(ta)==S::ok,"cancel A exact entry");
        check(!two_a.selected().borrow([](const auto&,const auto&){}),"cancelled work cannot newly borrow");
        check(m.retire(ta)==S::bound,"bound cancelled not retired");
        check(m.unbind(lane(m),untouched)==S::pending,"cancel waits drain");
        auto cancelled=two_a.lease();check(cancelled.finish(cancelled.operation(),ta.selection.key,{},true,true),"cancel drains original work");
        check(m.unbind(lane(m),untouched)==S::ok,"cancelled unbind");
        check(m.retire(ta)==S::ok&&m.request_cancel(ta)==S::stale,"retire fences stale entry");
        finish(two_b,tb);tb=token(m,tb);
        // Same-lane resume has no self-held witness deadlock.
        check(m.request_park(lane(m,1))==S::ok,"park B again");
        check(m.switch_active(lane(m,1),tb,bound1)==S::ok,"same lane resume");
        check(m.request_cancel(tb)==S::ok,"cancel idle B");check(m.unbind(lane(m,1),untouched)==S::ok,"unbind B");
        check(m.retire(tb)==S::ok,"retire B");
        WindowRegistrationTicket stale;check(m.reserve_registration(key(1),1,stale)==S::stale,"retired key cannot ordinary-register");
    }
    {
        WindowManager m(limits(2),budget);WindowRegistrationTicket a,b,c;
        check(m.reserve_registration(key(1),1,a)==S::ok,"ticket A");
        check(m.reserve_registration(key(2),1,b)==S::ok,"ticket B");
        check(m.reserve_registration(key(3),1,c)==S::capacity,"bounded entries");
        a={};check(m.reserve_registration(key(3),1,c)==S::ok,"ticket destructor frees pending slot");
        check(m.reserve_registration(key(1),1,a)==S::stale,"cancelled generation not reused");
        auto bad=key(4);bad.thread_id.assign(129,'x');check(m.reserve_registration(bad,1,a)==S::invalid,"bounded identity strings");
        bad=key(4);++bad.engine_generation;check(m.reserve_registration(bad,1,a)==S::invalid,"wrong engine");
        check(m.cancel_registration(b)==S::ok,"explicit pending cancel");
        auto fixture=fresh(budget,key(2));ManagedWindowToken out;
        check(m.publish_registration(b,fixture.prepared,out)==S::stale,"delayed prepared result refused");
    }
    {
        auto l=limits();l.last_serial=std::numeric_limits<uint64_t>::max();WindowManager m(l,budget);WindowRegistrationTicket ticket;
        check(m.reserve_registration(key(1),1,ticket)==S::exhausted,"serial exhaustion refuses");
        l=limits();l.last_window_generation=std::numeric_limits<uint64_t>::max();WindowManager old(l,budget);
        check(old.reserve_registration(key(1),1,ticket)==S::exhausted,"creation exhaustion cannot wrap");
    }
    {
        WindowManager m(limits(),budget);auto a=fresh(budget,key(1));auto ta=reg(m,a,key(1),2);WindowLaneToken output=lane(m);
        check(m.switch_active(output,ta,output)==S::incompatible,"runtime placement refused");
        auto b=fresh(budget,key(2));auto tb=reg(m,b,key(2));
        WindowLaneToken initial=lane(m),other=lane(m,1),r1,r2;S s1=S::invalid,s2=S::invalid;
        std::thread first([&]{s1=m.switch_active(initial,tb,r1);});std::thread second([&]{s2=m.switch_active(other,tb,r2);});first.join();second.join();
        check((s1==S::ok&&s2==S::bound)||(s2==S::ok&&s1==S::bound),"one target cannot bind twice");
    }
    {
        WindowManager m(limits(),budget);auto a=fresh(budget,key(1));auto ta=reg(m,a,key(1));WindowLaneToken active_lane;
        check(m.switch_active(lane(m),ta,active_lane)==S::ok,"unhealthy bind");ManagedWindowWork work;
        check(m.admit(active_lane,work)==S::ok,"unhealthy admit");auto lease=work.lease();check(lease.mark_unhealthy(lease.operation()),"mark unhealthy");
        // Close still-bound admission by explicit park may refuse unhealthy; unbind must remain pending until teardown.
        check(m.request_cancel(ta)==S::unhealthy,"uncertain cancel refused");
        check(lease.retire_after_teardown(lease.operation()),"confirmed teardown");
        check(m.request_cancel(ta)==S::ok,"terminal cancellation closes routing");
        WindowLaneToken empty;check(m.unbind(lane(m),empty)==S::ok&&m.retire(ta)==S::ok,"retire only after teardown");
    }
    {
        // Real owner destructor calls back into manager, so any held registry mutex hangs under alarm.
        WindowManager m(limits(),budget);auto a=fresh(budget,key(1));auto ta=reg(m,a,key(1));
        int calls=0;a.runtime->on_destroy=[&]{WindowLaneToken out;check(m.lane_snapshot(0,out)==S::ok,"destructor reentry");++calls;};
        a={};check(m.request_cancel(ta)==S::ok&&m.retire(ta)==S::ok&&calls==1,"retirement outside lock");
    }
    {
        // Actual execution completion between old-token capture and quiesce/cancel.
        auto resource=budget.reserve(WindowResources{{0,0},1,0,0,0,0},ReservationPurpose::foreground).lease;
        StateCut cut;cut.id=1;cut.history=window_history_digest({});WindowExecution execution(key(1),cut,1,std::make_shared<int>(1),resource);
        WindowWork work;check(execution.admit(key(1),work,&cut),"fence execution admit");auto next=cut;++next.id;
        check(work.finish(work.operation(),key(1),{next},true,true),"finish before old-cut control");
        check(!execution.cancel(key(1),&cut)&&!execution.begin_quiesce(key(1),&cut),"stale cut atomically refused despite same key");
        check(execution.view().phase==WindowExecutionPhase::running,"stale control leaves running");
        check(execution.begin_quiesce(key(1),&next)&&execution.cancel(key(1),&next),"current cut accepted");
    }
#if defined(STRATA_WINDOW_MANAGER_TEST_SEAM)
    {
        WindowManager m(limits(1),budget);auto f=fresh(budget,key(1));WindowRegistrationTicket ticket;
        check(m.reserve_registration(key(1),1,ticket)==S::ok,"busy publication ticket");
        Barrier gate;S published=S::invalid;ManagedWindowToken out;
        std::thread publisher([&]{publication_barrier=&gate;published=m.publish_registration(ticket,f.prepared,out);});
        gate.wait();ManagedWindowSnapshot snap;check(m.entry_snapshot(ticket.id(),snap)==S::busy,"pending publication busy");
        // Immutable-id cancellation avoids sharing mutable ticket state across callers.
        check(m.cancel_registration(ticket.id())==S::ok,"cancel busy publication");
        WindowRegistrationTicket other;check(m.reserve_registration(key(2),1,other)==S::capacity,"cancelled busy publication retains bounded slot");
        gate.release();publisher.join();
        check(m.reserve_registration(key(2),1,other)==S::ok,"slot returns after cancelled publisher retires");
        check(published==S::cancelled,"late publication refused after cancellation");
    }
#endif
    {
        WindowManager m(limits(),budget);auto original=fresh(budget,key(1));auto ta=reg(m,original,key(1));
        SelectedWindowRead read;check(m.acquire_read(ta,read)==S::ok,"parked replacement read");
        auto saved=window_test::saved(0,4,1,true);saved.stage_images.push_back(window_test::saved(4,8,1,false));
        auto metadata=window_test::metadata();metadata.key=key(1);++metadata.key.revision;++metadata.key.selection_epoch;
        const auto g=window_test::geometry();WindowStateLayout layout;SnapshotDescriptor descriptor;
        check(bool(native_window_layout(g,{{0,4,{1},true},{4,8,{1},false}},metadata.key,metadata.cut,1024,1<<20,layout)),"replacement layout");
        check(bool(describe_native_window(saved,g,layout,metadata,{{{7,8},9},{{-1,-1},-1}},descriptor)),"replacement descriptor");
        BundleHandle bundle;check(bool(BundleHandle::freeze(descriptor,metadata.key,layout,bundle)),"replacement bundle");
        NativeSessionAllocation allocation;
        check(bool(allocate_native_sessions(g,16,1,{{0,0,4,true},{1,4,8,false}},budget,allocation)),"replacement allocation");
        PreparedNativeWindow replacement;auto runtime=std::make_shared<Runtime>();
        check(bool(prepare_owned_native_restore(bundle,metadata.key,g,allocation,{[](auto&,auto&){return true;},[](auto&,auto&){return true;}},budget,1024,1<<20,true,replacement,runtime)),"real private replacement");
        auto output=ta;check(m.replace_parked(ta,read.lease(),original.prepared,output)==S::invalid&&output==ta,"fresh replacement refused unchanged");
        S replaced=S::invalid;
        WindowLaneToken next;
#if defined(STRATA_WINDOW_MANAGER_TEST_SEAM)
        Barrier gate;
        std::thread worker([&]{replacement_barrier=&gate;replaced=m.replace_parked(ta,read.lease(),replacement,output);});
        gate.wait();ManagedWindowSnapshot snapshot;
        check(m.entry_snapshot(ta.registration,snapshot)==S::busy,"restore busy observable");
        check(m.switch_active(lane(m),ta,next)==S::busy&&m.request_cancel(ta)==S::busy&&m.retire(ta)==S::busy,"busy replacement prevents activation cancellation retirement");
        gate.release();worker.join();
#else
        replaced=m.replace_parked(ta,read.lease(),replacement,output);
#endif
        check(replaced==S::ok&&output.selection.key==metadata.key,"replacement publishes exact new token");
        check(!read.borrow([](const auto&,const auto&){}),"retired old read no longer usable");
        auto weak=std::weak_ptr<Runtime>(original.runtime);int retired=0;
        original.runtime->on_destroy=[&]{auto now=token(m,output);check(now==output,"old owner reentry after replacement");++retired;};
        original={};check(!weak.expired(),"old read retains prior runtime");
        read={};check(weak.expired()&&retired==1,"old read release outside registry lock");
        check(m.switch_active(lane(m),output,next)==S::ok,"resume restored entry");
        ManagedWindowWork work;check(m.admit(next,work)==S::ok,"restored admission");
        check(work.selected().borrow([&](const auto& views,const auto& attached){check(views[0].session==allocation.stages()[0].session&&attached==runtime,"restored exact runtime session");}),"restored borrow");
        finish(work,output);
    }
    {
        WindowManager m(limits(),budget);auto ka=key(1),kb=key(2);kb.thread_id="different-thread";
        auto a=fresh(budget,ka),b=fresh(budget,kb);auto ta=reg(m,a,ka),tb=reg(m,b,kb);
        WindowLaneToken la,lb;check(m.switch_active(lane(m),ta,la)==S::ok&&m.switch_active(lane(m,1),tb,lb)==S::ok,"two users on different threads");
        ManagedWindowWork wa,wb;check(m.admit(la,wa)==S::ok&&m.admit(lb,wb)==S::ok,"independent thread admissions");finish(wa,ta);finish(wb,tb);
    }
    {
        WindowManager m(limits(),budget);auto a=fresh(budget,key(1));auto ta=reg(m,a,key(1));WindowLaneToken bound;
        check(m.switch_active(lane(m),ta,bound)==S::ok,"output owner bind");ManagedWindowWork output;
        check(m.admit(bound,output)==S::ok,"output owner admit");finish(output,ta);ta=token(m,ta);
        check(m.request_cancel(ta)==S::ok,"output owner cancel");WindowLaneToken empty;
        check(m.unbind(lane(m),empty)==S::ok&&m.retire(ta)==S::ok,"output still retains retired entry");
        int destroyed=0;a.runtime->on_destroy=[&]{auto snapshot=lane(m);check(snapshot.window.has_value(),"output owner destructor reentry");++destroyed;};a={};
        check(destroyed==0,"old output retains runtime");auto b=fresh(budget,key(2));auto tb=reg(m,b,key(2));
        check(m.switch_active(empty,tb,bound)==S::ok&&m.admit(bound,output)==S::ok&&destroyed==1,"replacing prior work output retires outside lock");finish(output,tb);
    }
    std::puts("window_manager_test passed");
}
