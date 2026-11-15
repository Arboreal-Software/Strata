#include "window_session_elastic_test_mocks.hpp"
#include "strata/core/window_restore.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/core/window_selection.hpp"
#include "strata/core/window_manager.hpp"
#include "strata/core/window_admission.hpp"
#include <thread>
#include <type_traits>
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
static_assert(!std::is_copy_constructible_v<WindowAdmission>);
static_assert(!std::is_copy_assignable_v<WindowAdmission>);
static_assert(!std::is_move_constructible_v<WindowAdmission>);
static_assert(!std::is_move_assignable_v<WindowAdmission>);
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
static void finish(const ManagedWindowWork& work,const ManagedWindowToken& t){auto lease=work.lease();auto cut=t.selection.cut;++cut.id;check(lease.finish(lease.operation(),t.selection.key,{cut,cut},true,true),"common cut finish");}

static thread_local int allocation_countdown=-1;
void* operator new(std::size_t n){
    if(allocation_countdown==0){allocation_countdown=-1;throw std::bad_alloc();}
    if(allocation_countdown>0)--allocation_countdown;
    if(auto* p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n){return ::operator new(n);}
static void* watched_record_metadata=nullptr;
static WindowBudget* watched_record_budget=nullptr;
static uint64_t watched_record_charge=0;
static bool saw_record_metadata_retirement=false;
static void before_delete(void* p) noexcept {
    if(p&&p==watched_record_metadata){
        watched_record_metadata=nullptr; // budget snapshot may allocate/free its own vectors
        check(watched_record_budget->snapshot().reserved.host_bytes>=watched_record_charge,
              "record metadata remains charged through its destructor");
        saw_record_metadata_retirement=true;
    }
}
__attribute__((noinline)) void operator delete(void* p) noexcept{before_delete(p);std::free(p);}
__attribute__((noinline)) void operator delete(void* p,std::size_t) noexcept{before_delete(p);std::free(p);}
__attribute__((noinline)) void operator delete[](void* p) noexcept{before_delete(p);std::free(p);}
__attribute__((noinline)) void operator delete[](void* p,std::size_t) noexcept{before_delete(p);std::free(p);}
using A=AdmissionStatus;
static AdmissionLimits admission_limits(){AdmissionLimits l{8,4,1<<20,65536,0,{}};l.max_units.fill(16);return l;}
static WindowKey other_key(){auto k=key(2);k.thread_id="other";return k;}
struct World {
    WindowBudget budget{WindowResources{{1000000,1000000},256<<20,10000,0,0,100},WindowResources{{0,0},0,0,0,0,0}};
    ThreadPriorities priorities{1,{-100,100,100,1000,8,8,128}};
    ThreadDependencies dependencies{1,{16,32},priorities};
    WindowManager manager{limits(),budget};
    WindowAdmission admission{admission_limits(),priorities,dependencies,manager,budget};
    Fixture a=fresh(budget,key(1)),b=fresh(budget,other_key());
    ManagedWindowToken ta=reg(manager,a,key(1)),tb=reg(manager,b,other_key());
    ThreadKey thread=priorities.create_thread("t",0,0).snapshot.key;
    ThreadKey other=priorities.create_thread("other",0,0).snapshot.key;
    OperationKey op,op2;
    World(){
        op=dependencies.register_operation(thread,1,DependencyWorkClass::foreground,dependencies.snapshot().revision,0).key;
        op2=dependencies.register_operation(other,2,DependencyWorkClass::foreground,dependencies.snapshot().revision,0).key;
        WindowLaneToken out;check(manager.switch_active(lane(manager),ta,out)==S::ok,"world bind");
    }
    AdmissionCandidate candidate(OperationKey operation,ManagedWindowToken window,size_t lane_id=0){
        return {operation,window,lane(manager,lane_id),1,1,AdmissionUnit::decode,true,true,false,false,std::make_shared<int>(7)};
    }
    void drain(WindowAdmissionHandle& h){auto* work=h.work();check(work,"ready work");finish(*work,work->token());check(admission.release(h)==A::released,"release exact complete lease");}
};
int main(){
    strata::core::window_test::check_fixed_kv_elastic_mock_contract();
 alarm(60);
 check(!WindowWork{}.resolved(),"empty work unresolved");
 {
    World w;
    auto idle=w.priorities.create_thread("idle-without-operation",3,0);
    check(bool(idle),"idle thread registered without dependency operation");
    auto actor=w.priorities.register_activity(idle.snapshot.key,"user",1,0);
    check(bool(actor),"idle actor registered");
    ActivityKey identity{idle.snapshot.key,"user",actor.snapshot.activity.front().incarnation};
    check(bool(w.priorities.set_activity(identity,1,9,25,0)),"idle actor boost accepted");
    auto timer=w.admission.refresh(0);
    check(timer.status==A::no_work&&timer.next_expiry_ms&&*timer.next_expiry_ms==25,
          "admission timer includes actors absent from operation graph");
    timer=w.admission.refresh(25);
    check(timer.status==A::no_work&&!timer.next_expiry_ms,"idle deadline serviced without new request");
    auto expired=w.priorities.snapshot(idle.snapshot.key,25);
    check(bool(expired)&&expired.snapshot.base==3&&expired.snapshot.local_effective==3&&
          !expired.snapshot.activity.front().active,"timer expires idle boost without changing base");
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    check(w.admission.admit(0,{&c,1},false,h).status==A::ready,"actual admission");
    check(h.work()->token()==w.ta&&h.cut()==1,"exact manager token and admission cut");
    check(h.priority()->thread.applied_revision==h.priority()->thread.accepted_revision,"applied at admission");
    check(w.admission.release(h)==A::unresolved,"cannot release in-flight exact lease");
    w.drain(h);check(!h.work(),"retired handle cannot execute");
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);auto parked=w.candidate(w.op2,w.tb);
    // Both same thread/equal priority: earlier minted operation deterministically wins.
    AdmissionCandidate cs[]{parked,c};WindowAdmissionHandle h;
    check(w.admission.admit(0,cs,false,h).candidate==1,"deterministic tie");w.drain(h);
    auto r=w.dependencies.finish(w.op,DependencyState::complete,w.dependencies.snapshot().revision);check(bool(r),"complete actual drained milestone");
    check(w.admission.admit(0,{&parked,1},false,h).status==A::routing_needed,"parked best routes before dispatch");
 }
 {
    World w;auto r=w.priorities.register_activity(w.thread,"user-a",1,0);check(bool(r),"actor a");
    ActivityKey a{w.thread,"user-a",r.snapshot.activity.front().incarnation};
    r=w.priorities.set_activity(a,1,50,10,0);check(bool(r),"boost a");
    r=w.priorities.register_activity(w.thread,"user-b",r.snapshot.accepted_revision,0);check(bool(r),"actor b");
    ActivityKey b{w.thread,"user-b",r.snapshot.activity.back().incarnation};
    check(bool(w.priorities.set_activity(b,1,30,20,0)),"boost b");
    check(bool(w.priorities.set_base(w.thread,1,-5,0)),"base decrease retained");
    auto decision=w.admission.refresh(0);check(decision.next_expiry_ms==10,"timer armed without request");
    check(w.priorities.snapshot(w.thread,0).snapshot.local_effective==45,"boost max not sum");
    check(w.admission.refresh(10).next_expiry_ms==20,"timer expires first user without new input");
    check(w.priorities.snapshot(w.thread,10).snapshot.local_effective==25,"second actor remains");
    check(!w.priorities.set_activity(a,1,50,10,10),"stale expired actor cannot refresh");
    w.admission.refresh(20);check(w.priorities.snapshot(w.thread,20).snapshot.local_effective==-5,"expiry restores edited base");
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);c.resource_ready=false;
    auto maintenance=w.dependencies.register_operation(w.other,3,DependencyWorkClass::saved_maintenance,w.dependencies.snapshot().revision,0).key;
    auto saved=w.candidate(maintenance,w.tb);saved.idle_authorized=true;saved.foreground_return_protected=true;
    AdmissionCandidate cs[]{c,saved};WindowAdmissionHandle h;
    check(w.admission.admit(0,cs,false,h).status==A::no_work,"resource-blocked foreground still vetoes maintenance");
    check(w.admission.admit(0,{&saved,1},true,h).status==A::no_work,"external demand vetoes idle routing");
    check(w.admission.admit(0,{&saved,1},false,h).status==A::routing_needed,"idle maintenance routes only after gates");
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    check(w.admission.admit(0,{&c,1},false,h).status==A::ready,"cancel admission");auto lease=h.recovery_lease();
    check(w.manager.request_cancel(w.ta)==S::ok,"exact cancel");check(w.admission.release(h)==A::unresolved,"cancel is not drain");
    check(lease.finish(lease.operation(),w.ta.selection.key,{},true,true),"actual terminal drain");
    WindowLaneToken empty;check(w.manager.unbind(lane(w.manager),empty)==S::ok,"unbind cancelled");
    check(w.manager.retire(w.ta)==S::ok,"manager snapshot now absent");
    check(w.admission.release(h)==A::released,"retained original state proves resolved despite absent manager");
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    check(w.admission.admit(0,{&c,1},false,h).status==A::ready,"unhealthy admission");auto lease=h.recovery_lease();
    check(lease.mark_unhealthy(lease.operation()),"mark unhealthy");check(w.admission.release(h)==A::unresolved,"unhealthy is retained");
    check(lease.retire_after_teardown(lease.operation()),"test confirms teardown");check(w.admission.release(h)==A::released,"only confirmed teardown resolves");
 }
 {
    const auto child=fork();check(child>=0,"quarantine child");
    if(child==0){
    World w;auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    check(w.admission.admit(0,{&c,1},false,h).status==A::ready,"newer op test");auto old=h.recovery_lease();finish(*h.work(),w.ta);
    auto updated=token(w.manager,w.ta);auto next=lane(w.manager);ManagedWindowWork later;
    check(w.manager.admit(next,later)==S::ok,"advance without retiring old admission violates reclamation order");
    check(!old.resolved()&&w.admission.release(h)==A::unresolved,"newer operation is not old completion proof");
    finish(later,updated);
    // Deliberate quarantine is isolated from parent leak checking.
    _exit(0);
    }
    int status=0;check(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"conservative newer-operation quarantine");
 }
 {
    World w;
    check(bool(w.priorities.set_base(w.other,1,20,0)),"other user priority");
    auto c=w.candidate(w.op,w.ta),other=w.candidate(w.op2,w.tb);WindowAdmissionHandle h;
    AdmissionCandidate cs[]{c,other};
    check(w.admission.admit(0,cs,false,h).status==A::routing_needed,"higher priority parked user is not bypassed");
    auto mid=w.dependencies.register_operation(w.thread,9,DependencyWorkClass::foreground,w.dependencies.snapshot().revision,0).key;
    check(bool(w.dependencies.update_dependencies({{w.op2,mid,true},{mid,w.op,true}},w.dependencies.snapshot().revision)),"transitive foreground waiter edges");
    check(w.admission.admit(0,cs,false,h).status==A::ready&&h.work()->token()==w.ta,"inherited foreground prerequisite admitted");
    check(h.priority()->effective_priority==20&&h.priority()->priority_source_thread==w.other,"inheritance source visible");
    check(w.priorities.snapshot(w.other,0).snapshot.applied_cut==h.cut(),"inherited source acknowledged at actual admission");w.drain(h);
    check(bool(w.dependencies.finish(w.op2,DependencyState::cancelled,w.dependencies.snapshot().revision)),"waiter cancel");
    w.admission.refresh(0);auto graph=w.dependencies.snapshot();
    check(graph.operations.front().effective_priority==0,"waiter cancellation removes urgency");
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    c.amount=17;check(w.admission.admit(0,{&c,1},false,h).status==A::invalid,"unit bound");c.amount=1;
    ++c.window.selection.publication;check(w.admission.admit(0,{&c,1},false,h).status==A::stale,"stale exact selection");
    c=w.candidate(w.op,w.ta);check(w.admission.admit(0,{&c,1},false,h).status==A::ready,"owner retention test");
    WindowAdmissionHandle recovered;h={};check(w.admission.outstanding(0,recovered)&&recovered.work(),"dropped handle remains recoverable");
    w.drain(recovered);
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    check(w.admission.admit(0,{&c,1},false,h).status==A::ready,"replace old work admitted");finish(*h.work(),w.ta);
    auto ta=token(w.manager,w.ta);check(w.manager.request_park(lane(w.manager))==S::ok,"park completed old work");
    WindowLaneToken empty;check(w.manager.unbind(lane(w.manager),empty)==S::ok,"unbind before replacement");
    SelectedWindowRead read;check(w.manager.acquire_read(ta,read)==S::ok,"old replacement read");
    auto saved=window_test::saved(0,4,1,true);saved.stage_images.push_back(window_test::saved(4,8,1,false));
    auto metadata=window_test::metadata();metadata.key=ta.selection.key;++metadata.key.revision;++metadata.key.selection_epoch;
    const auto g=window_test::geometry();WindowStateLayout layout;SnapshotDescriptor descriptor;
    check(bool(native_window_layout(g,{{0,4,{1},true},{4,8,{1},false}},metadata.key,metadata.cut,1024,1<<20,layout)),"replace layout");
    check(bool(describe_native_window(saved,g,layout,metadata,{{{7,8},9},{{-1,-1},-1}},descriptor)),"replace descriptor");
    BundleHandle bundle;check(bool(BundleHandle::freeze(descriptor,metadata.key,layout,bundle)),"replace bundle");
    NativeSessionAllocation allocation;check(bool(allocate_native_sessions(g,16,1,{{0,0,4,true},{1,4,8,false}},w.budget,allocation)),"replace allocation");
    PreparedNativeWindow replacement;
    check(bool(prepare_owned_native_restore(bundle,metadata.key,g,allocation,{[](auto&,auto&){return true;},[](auto&,auto&){return true;}},w.budget,1024,1<<20,true,replacement,std::make_shared<Runtime>())),"replace preparation");
    ManagedWindowToken replaced;
    check(w.manager.replace_parked(ta,read.lease(),replacement,replaced)==S::ok,"replace actual storage");
    check(w.admission.release(h)==A::released,"old exact lease resolves independently of replacement token");
 }
 {
    World w;WindowLaneToken bound;
    check(w.manager.switch_active(lane(w.manager,1),w.tb,bound)==S::ok,"second independent lane");
    auto c=w.candidate(w.op,w.ta),d=w.candidate(w.op2,w.tb,1);WindowAdmissionHandle a,b;
    check(w.admission.admit(0,{&c,1},false,a).status==A::ready,"first independent unit");
    check(w.admission.admit(0,{&d,1},false,b).status==A::ready,"second independent unit");
    check(a.work()->token()!=b.work()->token()&&!a.recovery_lease().resolved()&&!b.recovery_lease().resolved(),"two exact simultaneous leases");
    w.drain(a);w.drain(b);
 }
 {
    World w;auto l=admission_limits();l.last_admission_cut=UINT64_MAX;
    WindowAdmission exhausted(l,w.priorities,w.dependencies,w.manager,w.budget);auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    check(exhausted.admit(0,{&c,1},false,h).status==A::exhausted,"admission serial never wraps");
 }
 {
    World w;auto l=admission_limits();l.max_outstanding=1;
    WindowAdmission bounded(l,w.priorities,w.dependencies,w.manager,w.budget);auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    check(bounded.admit(0,{&c,1},false,h).status==A::ready,"single-slot admission");
    check(bounded.admit(0,{&c,1},false,h).status==A::capacity,"unresolved capacity bounded");
    finish(*h.work(),w.ta);check(bounded.release(h)==A::released,"bounded slot reclaimed");
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);auto l=admission_limits();
    WindowBudget small(WindowResources{{0,0},l.workspace_bytes+1024,0,0,0,0},WindowResources{{0,0},0,0,0,0,0});
    WindowAdmission bounded(l,w.priorities,w.dependencies,w.manager,small);WindowAdmissionHandle h;
    check(bounded.admit(0,{&c,1},false,h).status==A::budget,"metadata budget before manager admission");
    ManagedWindowSnapshot view;check(w.manager.entry_snapshot(w.ta.registration,view)==S::ok&&!view.execution.in_flight,"budget refusal has no work");
 }
 {
    World w;check(bool(w.priorities.set_base(w.other,1,100,0)),"repair high base");
    auto live=w.dependencies.register_operation(w.other,5,DependencyWorkClass::live_repair,w.dependencies.snapshot().revision,0).key;
    auto c=w.candidate(w.op,w.ta),repair=w.candidate(live,w.tb);AdmissionCandidate cs[]{repair,c};WindowAdmissionHandle h;
    check(w.admission.admit(0,cs,false,h).status==A::ready&&h.work()->token()==w.ta,"hard foreground class outranks boosted repair");w.drain(h);
 }
 {
    World w;auto c=w.candidate(w.op,w.ta);bool retired=false;
    auto owner=std::make_shared<Runtime>();owner->on_destroy=[&]{check(w.admission.refresh(0).status==A::no_work,"authorization destructor reactor reentry");retired=true;};
    c.authorization=owner;owner.reset();WindowAdmissionHandle h;
    check(w.admission.admit(0,{&c,1},false,h).status==A::ready,"reentry admission");c.authorization.reset();
    w.drain(h);check(retired,"authorization retired after registry mutation");
 }
 {
    World w;auto r=w.priorities.register_activity(w.thread,"max-deadline",1,UINT64_MAX-10);check(bool(r),"max deadline actor");
    ActivityKey actor{w.thread,"max-deadline",r.snapshot.activity.front().incarnation};
    check(bool(w.priorities.set_activity(actor,1,10,10,UINT64_MAX-10)),"maximum expiry valid");
    auto timer=w.admission.refresh(UINT64_MAX-10);
    check(timer.next_expiry_ms&&*timer.next_expiry_ms==UINT64_MAX,"maximum deadline distinct from no timer");
    check(!w.admission.refresh(UINT64_MAX).next_expiry_ms,"maximum deadline expires without input");
 }
 {
    World w;const std::string actor(96,'a');
    check(bool(w.priorities.register_activity(w.thread,actor,1,0)),"long retained metadata actor");
    auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
    check(w.admission.admit(0,{&c,1},false,h).status==A::ready,"metadata lifetime admission");
    check(!h.priority()->thread.activity.empty(),"receipt owns actor metadata");
    const auto* metadata=h.priority()->thread.activity.front().lease_id.data();
    w.drain(h); // released receipt remains retained by the external handle
    watched_record_metadata=const_cast<char*>(metadata);watched_record_budget=&w.budget;
    watched_record_charge=w.budget.snapshot().reserved.host_bytes;
    saw_record_metadata_retirement=false;
    h={};
    check(saw_record_metadata_retirement,"actual receipt string destructor observed");
    check(w.budget.snapshot().reserved.host_bytes+admission_limits().record_bytes==watched_record_charge,
          "record charge released after final retained metadata");
    watched_record_budget=nullptr;
 }
 // Every recovery/refusal branch has its own bounded fault search and count.
 // Inherited cases must observe the first registry ack committed and the second
 // still unapplied; merely reaching acknowledgement_pending is insufficient.
 enum class AckCase { direct_recover,direct_expiry,direct_edit,partial_recover,
                      target_expiry,target_edit,source_expiry,source_edit,count };
 constexpr size_t case_count=static_cast<size_t>(AckCase::count);
 int reached[case_count]={},partial_count=0,recovered_count=0,blocked_count=0,refused_count=0;
 for(size_t scenario=0;scenario<case_count;++scenario){
   const auto mode=static_cast<AckCase>(scenario);
   const bool inherited=scenario>=static_cast<size_t>(AckCase::partial_recover);
   const bool target_expires=mode==AckCase::direct_expiry||mode==AckCase::target_expiry;
   const bool source_expires=mode==AckCase::source_expiry;
   const bool target_edits=mode==AckCase::direct_edit||mode==AckCase::target_edit;
   const bool source_edits=mode==AckCase::source_edit;
   const bool recover=mode==AckCase::direct_recover||mode==AckCase::partial_recover;
   for(int fault=0;fault<256&&!reached[scenario];++fault){
     World w;
     auto add_actor=[&](const ThreadKey& thread,const char* name,int boost,uint64_t ttl){
       auto registered=w.priorities.register_activity(thread,name,1,0);
       check(bool(registered),"decision actor registration");
       ActivityKey actor{thread,name,registered.snapshot.activity.front().incarnation};
       check(bool(w.priorities.set_activity(actor,1,boost,ttl,0)),"decision actor activity");
     };
     add_actor(w.thread,"target-actor-with-allocating-id",5,target_expires?10:100);
     if(inherited){
       add_actor(w.other,"source-actor-with-allocating-id",20,source_expires?10:100);
       check(bool(w.priorities.set_base(w.other,1,40,0)),"source higher base");
       check(bool(w.dependencies.update_dependencies({{w.op2,w.op,true}},w.dependencies.snapshot().revision)),
             "actual waiting consumer raises prerequisite priority");
     }
     auto c=w.candidate(w.op,w.ta);WindowAdmissionHandle h;
     allocation_countdown=fault;auto result=w.admission.admit(0,{&c,1},false,h);allocation_countdown=-1;
     if(result.status!=A::acknowledgement_pending){
       if(result.status==A::ready)w.drain(h);
       else {++refused_count;check(!h.work(),"pre-admission failure has no executable receipt");}
       continue;
     }
     check(!h.work()&&bool(h.recovery_lease()),"pending ack retains exact non-executable work");
     const auto target_before=w.priorities.snapshot(w.thread,0).snapshot;
     const auto source_before=w.priorities.snapshot(inherited?w.other:w.thread,0).snapshot;
     const bool partial=inherited&&target_before.applied_revision==target_before.accepted_revision&&
         target_before.applied_cut==h.cut()&&source_before.applied_revision<source_before.accepted_revision;
     if(inherited&&!partial){
       check(w.admission.recover_acknowledgement(h)==A::ready,"nonpartial fault recovers before search continues");
       w.drain(h);continue;
     }
     ++reached[scenario];if(partial)++partial_count;
     const auto cut=h.cut();const auto op=h.recovery_lease().operation();const auto token=*h.token();
     const auto target_receipt=h.priority()->thread;const auto source_receipt=*h.inherited_source();
     if(inherited){
       check(h.priority()->priority_source==w.op2&&source_receipt.key==w.other,
             "record contains genuine inherited source identity");
       check(target_receipt.applied_revision<target_before.applied_revision,
             "partial registry ack is not prematurely published into receipt");
     }
     auto orphan=w.priorities.create_thread("idle-no-operation",3,0);check(bool(orphan),"orphan thread");
     auto actor=w.priorities.register_activity(orphan.snapshot.key,"idle-user",1,0);check(bool(actor),"orphan actor");
     ActivityKey orphan_key{orphan.snapshot.key,"idle-user",actor.snapshot.activity.front().incarnation};
     check(bool(w.priorities.set_activity(orphan_key,1,30,1,0)),"orphan boost");
     check(bool(w.priorities.set_base(orphan.snapshot.key,1,-7,0)),"orphan independent base edit");
     const uint64_t expected_deadline=target_expires||source_expires?10:100;
     allocation_countdown=0;auto timer=w.admission.refresh(1);allocation_countdown=-1;
     check(timer.status==A::acknowledgement_pending&&timer.next_expiry_ms==expected_deadline,
           "pending ack timer maintenance is allocation-free");
     check(w.priorities.next_activity_expiry_ms()==expected_deadline,
           "registry itself expired orphan before any snapshot can advance its time");
     check(w.priorities.snapshot(orphan.snapshot.key,0).error==PriorityError::invalid_time,
           "registry clock already advanced");
     auto expired=w.priorities.snapshot(orphan.snapshot.key,1).snapshot;
     check(expired.local_effective==-7&&!expired.activity.front().active,"orphan expiry restores edited base");
     check(w.admission.refresh(0).status==A::invalid,"pending does not hide backwards time");
     if(target_expires||source_expires){
       allocation_countdown=0;auto expiry=w.admission.refresh(10);allocation_countdown=-1;
       check(expiry.status==A::acknowledgement_pending,"expiry-only remains pending until reconciliation");
       // No base mutation in expiry-only cases: a base edit must not mask stale expiry ack.
     }else if(target_edits||source_edits){
       const auto& changed=source_edits?w.other:w.thread;
       const uint64_t base_version=source_edits?2:1;
       check(bool(w.priorities.set_base(changed,base_version,7,1)),"edit-only changes one decision source");
       check(w.priorities.next_activity_expiry_ms()==100,"edit-only occurs before either activity expires");
     }
     const auto status=w.admission.recover_acknowledgement(h);
     check(h.cut()==cut&&h.recovery_lease().operation()==op&&*h.token()==token,
           "reconciliation preserves original cut operation and complete token");
     check(h.priority()->thread.accepted_revision==target_receipt.accepted_revision&&
           h.inherited_source()->accepted_revision==source_receipt.accepted_revision,
           "reconciliation never substitutes new accepted decision revisions");
     if(recover){
       ++recovered_count;check(status==A::ready&&h.work(),"unrelated expiry recovers original exact unit");
       check(h.priority()->thread.applied_cut==cut&&h.inherited_source()->applied_cut==cut,
             "successful retry publishes original applied cut");w.drain(h);
     }else{
       ++blocked_count;check(status==A::acknowledgement_blocked&&!h.work(),"changed decision source blocks execution");
       check(h.priority()->thread.applied_revision==target_receipt.applied_revision&&
             h.priority()->thread.applied_cut==target_receipt.applied_cut&&
             h.inherited_source()->applied_revision==source_receipt.applied_revision&&
             h.inherited_source()->applied_cut==source_receipt.applied_cut,
             "blocked receipt retains unapplied publication fields");
       const uint64_t now=target_expires||source_expires?10:1;
       if(partial){
         auto first=w.priorities.snapshot(w.thread,now).snapshot;
         auto second=w.priorities.snapshot(w.other,now).snapshot;
         check(first.applied_revision==target_before.applied_revision&&first.applied_cut==target_before.applied_cut,
               "first committed registry ack survives blocked second ack without rollback");
         check(second.applied_revision==source_before.applied_revision&&second.applied_cut==source_before.applied_cut,
               "failed inherited ack never advances source applied state");
       }
       check(w.admission.release(h)==A::unresolved,"blocked lease cannot release before drain");
       check(w.admission.refresh(now+1).status==A::acknowledgement_pending,"blocked state still permits timer maintenance");
       check(w.manager.request_cancel(w.ta)==S::ok,"cancel exact blocked window");
       auto lease=h.recovery_lease();check(lease.finish(op,w.ta.selection.key,{},true,true),"actual retained lease drains cancellation");
       check(w.admission.release(h)==A::released,"release only after exact cancellation drain");
     }
   }
   if(reached[scenario]!=1)std::fprintf(stderr,"unreached acknowledgement scenario %zu\n",scenario);
   check(reached[scenario]==1,"each named acknowledgement scenario must reach its intended fault phase");
 }
 check(recovered_count==2&&blocked_count==6&&partial_count==5&&refused_count>0,
       "independent recovery blocked partial and preadmission counts all reached");
 std::printf("window admission passed: recoveries %d, blocked %d, partial inherited %d, allocation refusals %d\n",
             recovered_count,blocked_count,partial_count,refused_count);
}
