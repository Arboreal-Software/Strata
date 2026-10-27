#include "window_session_elastic_test_mocks.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/core/window_restore.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "window_native_test_fixture.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <functional>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
static int current_device=0,mallocs=0,frees=0,inits=0,fail_malloc=0,fail_init=0;
static bool fail_free=false;
static int device_calls=0,host_frees=0,fail_drain_device=-1,fail_set_target=-1;
static bool check_retirement_order=false;
static std::set<int> drained_devices;
static void check_gang_drained();
static std::function<void()> during_drain,during_free;
static int restore_writes=0;
static std::map<void*,int> device_allocations;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
template<class Fn>static void isolated(Fn fn){
    const auto pid=fork();check(pid>=0,"fork failure fixture");
    if(pid==0){fn();std::exit(0);}
    int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"isolated failure case");
}
extern "C" cudaError_t cudaGetDevice(int* p){++device_calls;*p=current_device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){++device_calls;if(d==fail_set_target)return cudaErrorUnknown;current_device=d;return cudaSuccess;}
extern "C" cudaError_t cudaMalloc(void** p,size_t n){
    ++device_calls;
    if(++mallocs==fail_malloc)return cudaErrorMemoryAllocation;
    *p=std::malloc(n);device_allocations[*p]=current_device;return *p?cudaSuccess:cudaErrorMemoryAllocation;
}
extern "C" cudaError_t cudaFree(void* p){
    check_gang_drained();
    if(during_free)during_free();
    if(fail_free)return cudaErrorUnknown;
    check(device_allocations.at(p)==current_device,"release selects owning device");
    device_allocations.erase(p);std::free(p);++frees;return cudaSuccess;
}
static void check_gang_drained(){
    if(check_retirement_order)check(drained_devices==std::set<int>{0,1},"all gang devices drain before any pinned/device free");
}
extern "C" cudaError_t cudaDeviceSynchronize(){
    ++device_calls;if(during_drain)during_drain();
    if(current_device==fail_drain_device)return cudaErrorUnknown;
    if(check_retirement_order)drained_devices.insert(current_device);
    return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected allocator CUDA error";}
namespace strata::kernels { std::size_t native_q8_1_bytes(int,int){return 4096;} }
extern "C" cudaError_t cudaGetLastError(){return cudaSuccess;}
namespace strata::core {
void session_zero(SessionState&,const ModelGeometry&,const float*,void*){}
bool session_zero_checked(SessionState&,const ModelGeometry&,const float*,void*,std::string&){return true;}
bool session_private_plan(const ModelGeometry&,int64_t,int64_t,int64_t,int64_t,SessionPrivatePlan& p){p={1024,4096,512};return true;}
uint64_t ple_run_scratch_bytes(){return 512;}
uint64_t session_init_private(const ModelGeometry& g,int64_t,int64_t,void* arena,SessionState& s,
                              SessionInitUploads& uploads,int64_t lo,int64_t hi){
    ++inits;s.layer_lo=lo;s.layer_hi=hi;s.qsa_ord0=lo/4;s.qsa_alloc=1;
    s.qsa_states=new QsaState[size_t(g.n_qsa_layers())]();
    s.qsa_states[s.qsa_ord0].kv_int8=true;
    s.qsa_states[s.qsa_ord0].host_step=static_cast<int32_t*>(std::malloc(32));
    s.ple_hist=static_cast<float*>(arena);
    uploads.qsa.resize(size_t(g.n_qsa_layers()));uploads.qsa[size_t(s.qsa_ord0)].rope_cos.resize(8);
    if(inits==fail_init){uploads.error="injected partial init";return 0;}return 1024;
}
bool qsa_state_release_host(QsaState& q,std::string&){check_gang_drained();++host_frees;std::free(q.host_step);q.host_step=nullptr;return true;}
void session_release(SessionState&){}
bool conversation_snapshot_validate(const SavedConversation&,const SessionState&,const ModelGeometry&,const QsaState*,std::string&){return true;}
ConversationRestore conversation_snapshot_restore(const SavedConversation&,SessionState&,const ModelGeometry&,const QsaState*,std::string&){++restore_writes;return ConversationRestore::restored;}
}
int main(int argc,char** argv){
    strata::core::window_test::check_fixed_kv_elastic_mock_contract();
    ModelGeometry g;g.n_layers=8;g.qsa_interval=4;
    WindowResources capacity{{100000,100000},100000,10000,0,0,2},zero{{0,0},0,0,0,0,0};
    const std::vector<NativeSessionRequest> requests{{0,0,4,true},{1,4,8,false}};
    if(argc>1&&std::string(argv[1])=="--retirement-drain"){
        // Actual allocator Storage/owner/budget, mocked CUDA and session buffers.
        // Producer closure is caller-established; no callback submits new work.
        {
            WindowBudget budget(capacity,zero);NativeSessionAllocation out;
            check(bool(allocate_native_sessions(g,16,1,requests,budget,out)),"retirement drain fixture");
            current_device=7;drained_devices.clear();check_retirement_order=true;
            const auto before_host=host_frees;
            out={};check_retirement_order=false;
            check(drained_devices==std::set<int>{0,1}&&host_frees==before_host+2&&current_device==7,
                  "all stages drain before host release and original device restored");
            check(device_allocations.empty()&&budget.snapshot().reserved==zero,"normal retirement releases actual complete owner and quota");
        }
        for(int scenario=0;scenario<5;++scenario)isolated([&]{
            WindowBudget budget(capacity,zero);NativeSessionAllocation out;
            check(bool(allocate_native_sessions(g,16,1,requests,budget,out)),"failed drain fixture");
            auto* a=out.stages()[0].session;auto* b=out.stages()[1].session;
            std::weak_ptr<const ReservationLease> lease=out.reservation();
            const auto held=budget.snapshot().reserved;const auto old_frees=frees,old_host_frees=host_frees;
            current_device=7;drained_devices.clear();check_retirement_order=true;
            if(scenario<2)fail_drain_device=scenario;
            else fail_set_target=scenario==4?7:scenario-2;
            out={};
            check(!lease.expired()&&a->qsa_states[a->qsa_ord0].host_step&&b->qsa_states[b->qsa_ord0].host_step,
                  "failed drain/select/restore retains actual session owner and pinned targets");
            check(frees==old_frees&&host_frees==old_host_frees&&budget.snapshot().reserved==held&&native_session_quarantine_count()==1,
                  "no borrowed free or quota release before full gang drain and restoration");
            if(scenario!=4)check(current_device==7,"failed drain/select restores original device");
            const auto calls=device_calls;out={};check(device_calls==calls,"uncertain retired owner is not retried");
        });
        {
            auto small=capacity;small.device_bytes[1]=1;WindowBudget budget(small,zero);NativeSessionAllocation out;
            const auto before=device_calls;
            check(allocate_native_sessions(g,16,1,requests,budget,out).status==NativeAllocationStatus::budget_refused,
                  "host-only gang admission refuses");
            check(device_calls==before&&out.stages().empty()&&budget.snapshot().reserved==zero,
                  "host-only unwind performs no device work");
        }
        std::puts("window_allocator retirement drain tests passed");return 0;
    }
    if(argc>1&&std::string(argv[1])=="--transfer"){
        auto fresh_pair=[] {
            std::vector<std::unique_ptr<SessionState>> sessions;
            sessions.push_back(std::make_unique<SessionState>());
            sessions.push_back(std::make_unique<SessionState>());
            return sessions;
        };
        WindowBudget budget(capacity,zero);NativeSessionAllocation out;
        check(bool(allocate_native_sessions(g,16,1,requests,budget,out)),"existing output fixture");
        const auto original=out.stages()[0].session;
        auto fresh=fresh_pair();auto* first=fresh[0].get();auto* second=fresh[1].get();
        int marker=0;
        const std::vector<std::function<void(SessionState&)>> bind_resource{
            [&](SessionState& s){s.qsa_states=reinterpret_cast<QsaState*>(&marker);},
            [&](SessionState& s){s.qsa_state_arena=&marker;},
            [&](SessionState& s){s.gdn.state=reinterpret_cast<float*>(&marker);},
            [&](SessionState& s){s.block.gr.xn=reinterpret_cast<float*>(&marker);},
            [&](SessionState& s){s.ple.table=reinterpret_cast<strata::kernels::PleTable*>(&marker);},
            [&](SessionState& s){s.ple.read_context=std::shared_ptr<strata::kernels::PleReadContext>(
                reinterpret_cast<strata::kernels::PleReadContext*>(&marker),[](auto*){});},
            [&](SessionState& s){s.ple.scratch=reinterpret_cast<float*>(&marker);},
            [&](SessionState& s){s.ple.emb_dev=reinterpret_cast<float*>(&marker);},
            [&](SessionState& s){s.ple.emb_host=reinterpret_cast<float*>(&marker);},
            [&](SessionState& s){s.ple.w.key_native_q8_1=&marker;},
            [&](SessionState& s){s.ple.w.key_native_data=&marker;}
        };
        const auto before_mallocs=mallocs;
        for(const auto& bind:bind_resource){
            bind(*fresh[1]);
            check(allocate_native_sessions(g,16,1,requests,fresh,budget,out).status==NativeAllocationStatus::invalid,
                  "bound resource rejected before ownership transfer");
            check(fresh[0].get()==first&&fresh[1].get()==second&&out.stages()[0].session==original&&mallocs==before_mallocs,
                  "rejection preserves all caller owners and previous output without CUDA");
            *fresh[1]=SessionState{};
        }
        auto small=capacity;small.device_bytes[1]=1;WindowBudget constrained(small,zero);
        check(allocate_native_sessions(g,16,1,requests,fresh,constrained,out).status==NativeAllocationStatus::budget_refused&&
              fresh[0].get()==first&&fresh[1].get()==second&&out.stages()[0].session==original,
              "budget refusal preserves incoming pointers and output");
        auto absent=std::move(fresh[1]);
        check(allocate_native_sessions(g,16,1,requests,fresh,budget,out).status==NativeAllocationStatus::invalid&&fresh[0].get()==first,
              "null owner rejected before any transfer");fresh[1]=std::move(absent);
        check(bool(allocate_native_sessions(g,16,1,requests,fresh,budget,out))&&fresh.size()==2&&!fresh[0]&&!fresh[1],
              "success empties input elements without resizing caller vector");
        check(out.stages()[0].session==first&&out.stages()[1].session==second&&out.healthy(),
              "factory preserves both preexisting SessionState addresses");
        auto held=out.stages()[1].owner;out={};check(second->qsa_states!=nullptr,"borrowed owner retains transferred session");
        held.reset();check(device_allocations.empty()&&budget.snapshot().reserved==zero,"transferred sessions release with factory owner");
        isolated([&]{
            NativeSessionAllocation prior;
            check(bool(allocate_native_sessions(g,16,1,requests,budget,prior)),"failure prior output fixture");
            const auto* old=prior.stages()[0].session;
            auto doomed=fresh_pair();auto* a=doomed[0].get();auto* b=doomed[1].get();
            fail_init=inits+2;const auto old_frees=frees;
            check(allocate_native_sessions(g,16,1,requests,doomed,budget,prior).status==NativeAllocationStatus::unhealthy,
                  "partial initialization fails after transfer");
            check(!doomed[0]&&!doomed[1]&&prior.stages()[0].session==old&&a->qsa_states&&b->qsa_states&&frees==old_frees,
                  "partial failure keeps transferred objects alive and old output unchanged");
            check(native_session_quarantine_count()==1&&budget.snapshot().reserved.shadows==2,
                  "partial transfer failure retains new gang plus prior output reservations");
        });
        isolated([&]{
            NativeSessionAllocation failure;auto doomed=fresh_pair();auto* a=doomed[0].get();auto* b=doomed[1].get();
            fail_malloc=mallocs+1;
            check(allocate_native_sessions(g,16,1,requests,doomed,budget,failure).status==NativeAllocationStatus::unhealthy,
                  "first device allocation failure follows complete ownership transfer");
            check(!doomed[0]&&!doomed[1]&&a->qsa_states==nullptr&&b->qsa_states==nullptr&&failure.stages().empty()&&
                  native_session_quarantine_count()==1&&budget.snapshot().reserved.shadows==1,
                  "even uninitialized transferred stages and lease remain factory quarantined");
        });
        std::puts("window_allocator session-transfer tests passed");return 0;
    }
    if(argc>1){
        WindowBudget budget(capacity,zero);NativeSessionAllocation stable,next;
        check(bool(allocate_native_sessions(g,16,1,requests,budget,stable)),"health query fixture");
        int drain_queries=0,free_queries=0;
        during_drain=[&]{
            std::thread query([&]{
                check(stable.healthy(),"other allocation health readable during device drain");
                check(native_session_quarantine_count()==0,"quarantine readable during device drain");
                ++drain_queries;
            });query.join();
        };
        check(bool(allocate_native_sessions(g,16,1,requests,budget,next)),"drain callback query completes");
        during_drain={};
        during_free=[&]{
            std::thread query([&]{
                check(stable.healthy(),"other allocation health readable during device free");
                check(native_session_quarantine_count()==0,"quarantine readable during device free");
                ++free_queries;
            });query.join();
        };
        next={};during_free={};
        check(drain_queries==2&&free_queries==4,"queries exercised actual per-device drain/free seams");
        stable={};check(device_allocations.empty()&&budget.snapshot().reserved==zero,"query test releases all storage");
        isolated([&]{
            NativeSessionAllocation existing,candidate;
            check(bool(allocate_native_sessions(g,16,1,requests,budget,existing)),"poison callback fixture");
            during_drain=[&]{
                std::thread poison([&]{existing.mark_unhealthy();
                    check(!existing.healthy(),"poison is visible while drain is active");
                    (void)native_session_quarantine_count();});poison.join();
            };
            const auto before_frees=frees;
            check(allocate_native_sessions(g,16,1,requests,budget,candidate).status==NativeAllocationStatus::unhealthy&&candidate.stages().empty(),
                  "concurrent global poison prevents final candidate publication");
            during_drain={};existing={};
            check(native_session_quarantine_count()==2&&frees==before_frees&&budget.snapshot().reserved.shadows==2,
                  "poisoned original and uncertain candidate retain all storage and both leases");
        });
        std::puts("window_allocator lock-scope tests passed");return 0;
    }
    {
        WindowBudget budget(capacity,zero);NativeSessionAllocation out;
        check(bool(allocate_native_sessions(g,16,1,requests,budget,out)),"fresh gang allocated");
        check(inits==2&&mallocs==4&&out.stages().size()==2,"all native and PLE allocations owned");
        auto* a=out.stages()[0].session;auto* b=out.stages()[1].session;
        check(a!=b&&a->ple_hist!=b->ple_hist&&a->qsa_states!=b->qsa_states,"private stage storage does not alias");
        check(a->ple.token==&a->ple_token&&a->ple.prev==a->ple_prev&&a->ple.hist==a->ple_hist&&
              a->ple.emb_dev&&a->ple.emb_host&&a->ple.scratch&&!a->ple.table,"PLE mutable buffers are fresh and reader stays unbound");
        auto held=out.stages()[1].owner;out={};
        check(frees==0&&budget.snapshot().reserved.shadows==1,"borrowed stage owner retains entire gang and charge");
        held.reset();check(frees==4&&device_allocations.empty()&&budget.snapshot().reserved==zero,"last owner releases complete gang on correct devices");
    }
    {
        auto small=capacity;small.device_bytes[1]=1023;WindowBudget budget(small,zero);NativeSessionAllocation out;
        const auto before=mallocs;
        check(allocate_native_sessions(g,16,1,requests,budget,out).status==NativeAllocationStatus::budget_refused&&mallocs==before,
              "second-device shortage refuses before first allocation");
        check(budget.snapshot().reserved==zero,"failed gang admission leaves no charges");
    }
    {
        using namespace strata::core::window_test;
        auto model=geometry();auto meta=metadata();
        auto image=saved(0,4,1,true);image.stage_images.push_back(saved(4,8,1,true));
        WindowStateLayout layout;
        check(bool(native_window_layout(model,{{0,4,{1},true},{4,8,{1},true}},meta.key,meta.cut,1024,1<<20,layout)),"owned restore source layout");
        SnapshotDescriptor descriptor;
        check(bool(describe_native_window(image,model,layout,meta,{{{7,8},9},{{-1,-1},-1}},descriptor)),"owned restore source descriptor");
        BundleHandle source;check(bool(BundleHandle::freeze(descriptor,meta.key,layout,source)),"owned restore immutable source");
        auto larger=capacity;larger.host_bytes=32<<20;WindowBudget budget(larger,zero);NativeSessionAllocation allocation;
        check(bool(allocate_native_sessions(model,16,1,requests,budget,allocation)),"restore destination comes from actual allocator owner");
        PreparedNativeWindow ready;int binds=0;
        std::vector<NativeSessionBinder> binders;
        for(const auto& stage:allocation.stages())binders.push_back([expected=stage.session,&binds](SessionState& actual,std::string&){
            check(&actual==expected,"private binder receives correct allocated session");++binds;return true;});
        auto prepare=[&]{return prepare_owned_native_restore(source,meta.key,model,allocation,binders,budget,1024,1<<20,true,ready);};
        auto missing=binders;missing[1]={};
        check(prepare_owned_native_restore(source,meta.key,model,allocation,missing,budget,1024,1<<20,true,ready).status==NativeRestoreStatus::invalid&&restore_writes==0,
              "missing final binder refused before first write");
        check(bool(prepare())&&restore_writes==2&&binds==2,"factory-owned gang restored and privately bound");
        check(prepare().status==NativeRestoreStatus::invalid&&restore_writes==2,"same owned gang cannot back overlapping restore transactions");
        const auto prior_frees=frees;allocation={};
        check(frees==prior_frees&&budget.snapshot().reserved.shadows==1,"prepared handle retains factory buffers and reservations");
        ready={};check(device_allocations.empty()&&budget.snapshot().reserved==zero,"retiring prepared state releases actual allocation owner");
        isolated([&]{
            NativeSessionAllocation doomed;
            check(bool(allocate_native_sessions(model,16,1,requests,budget,doomed)),"failure invalidation fixture allocated");
            auto alias=doomed;PreparedNativeWindow failed_output;
            std::vector<NativeSessionBinder> bad(2,[](SessionState&,std::string& error){error="injected binding failure";return false;});
            auto failure=prepare_owned_native_restore(source,meta.key,model,doomed,bad,budget,1024,1<<20,true,failed_output);
            check(failure.status==NativeRestoreStatus::unhealthy&&failure.failure&&!doomed.healthy()&&!alias.healthy(),
                  "failed owned restore invalidates every allocation handle alias");
            const auto prior_writes=restore_writes;
            check(prepare_owned_native_restore(source,meta.key,model,alias,bad,budget,1024,1<<20,true,failed_output).status==NativeRestoreStatus::unhealthy&&restore_writes==prior_writes,
                  "unhealthy destination cannot be reused before or after teardown retirement");
            failure.failure->retire_after_teardown();doomed={};alias={};
            check(native_session_quarantine_count()==1&&budget.snapshot().reserved.shadows==1,
                  "teardown retirement cannot turn stale device pointers into healthy allocations");
        });
    }
    {
        WindowBudget budget(capacity,zero);NativeSessionAllocation out;
        auto native_requests=requests;native_requests[0].ple_native_key=true;
        const auto before=mallocs;
        auto tight=capacity;tight.device_bytes[0]=1024+strata::kernels::NG_N_EMBD*sizeof(float)+512;
        WindowBudget constrained(tight,zero);
        check(allocate_native_sessions(g,16,1,native_requests,constrained,out).status==NativeAllocationStatus::budget_refused&&mallocs==before,
              "native-key-only capacity shortage refuses whole gang before CUDA allocation");
        check(bool(allocate_native_sessions(g,16,1,native_requests,budget,out)),"native PLE scratch gang allocated");
        const auto& ple=out.stages()[0].session->ple;
        check(mallocs==before+5&&ple.w.key_native_q8_1&&ple.w.key_native_q8_1!=ple.scratch&&
              ple.w.key_native_q8_1!=ple.emb_dev,"native key scratch is private and disjoint");
        out={};check(device_allocations.empty()&&budget.snapshot().reserved==zero,"native scratch releases with gang");
        native_requests[1].ple_native_key=true;
        check(allocate_native_sessions(g,16,1,native_requests,budget,out).status==NativeAllocationStatus::invalid,
              "native key scratch requires PLE stage buffers");
    }
    isolated([&]{
        WindowBudget budget(capacity,zero);NativeSessionAllocation out;
        const auto before=native_session_quarantine_count();fail_init=inits+2;
        check(allocate_native_sessions(g,16,1,requests,budget,out).status==NativeAllocationStatus::unhealthy&&out.stages().empty(),
              "partial second-stage initialization never publishes output");
        check(native_session_quarantine_count()==before+1&&budget.snapshot().reserved.shadows==1,
              "uncertain initialization retains allocation and budget until worker restart");fail_init=0;
        check(allocate_native_sessions(g,16,1,requests,budget,out).status==NativeAllocationStatus::unhealthy,
              "quarantined allocator refuses further private work");
    });
    isolated([&]{
        WindowBudget budget(capacity,zero);NativeSessionAllocation out;
        check(bool(allocate_native_sessions(g,16,1,requests,budget,out)),"release-failure fixture allocated");
        const auto before=native_session_quarantine_count();fail_free=true;out={};fail_free=false;
        check(native_session_quarantine_count()==before+1&&budget.snapshot().reserved.shadows==1,
              "failed device free cannot erase accounting");
    });
    std::puts("window_allocator_test passed");
}
