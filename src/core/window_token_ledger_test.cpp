#include "strata/core/window_restore.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/core/window_selection.hpp"
#include "strata/core/window_manager.hpp"
#include "strata/core/window_token_ledger.hpp"
#include "strata/core/window_capture.hpp"
#include "strata/core/verify.hpp"
#include <new>
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
bool session_zero_checked(SessionState& session,const ModelGeometry& geometry,const float* initial,void* stream,std::string& error){
    try{session_zero(session,geometry,initial,stream);}catch(const std::exception& e){error=e.what();return false;}
    if(last_error!=cudaSuccess){error="injected checked reset failure";return false;}return true;
}
bool conversation_snapshot_validate(const SavedConversation&,const SessionState&,const ModelGeometry&,const QsaState*,std::string&){return true;}
ConversationRestore conversation_snapshot_restore(const SavedConversation&,SessionState&,const ModelGeometry&,const QsaState*,std::string&){return ConversationRestore::restored;}
}
using S=WindowManagerStatus;
struct Runtime : NativeRuntimeAttachment {
    std::unique_ptr<WindowTokenLedger> ledger;
    std::unique_ptr<std::array<Verifier,2>> capture_verifiers;
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
using TS=WindowTokenStatus;
static thread_local int fail_new=0;
void* operator new(std::size_t n){if(fail_new && --fail_new==0)throw std::bad_alloc();if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
__attribute__((noinline)) void operator delete(void* p) noexcept {std::free(p);}
__attribute__((noinline)) void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void* operator new[](std::size_t n){return ::operator new(n);}
__attribute__((noinline)) void operator delete[](void* p) noexcept {::operator delete(p);}
__attribute__((noinline)) void operator delete[](void* p,std::size_t) noexcept {::operator delete(p);}
static thread_local std::function<void()> after_finish;
namespace strata::core {void ledger_test_after_finish(){auto fn=std::move(after_finish);after_finish={};if(fn)fn();}}
static std::map<const Verifier*,Verifier::SnapshotDrain> capture_drains;
static WindowManager* capture_manager=nullptr;
static ManagedWindowToken capture_token;
static int capture_copies=0,capture_fail_at=0;
namespace strata::core {
Verifier::~Verifier()=default;
Verifier::SnapshotDrain Verifier::poll_snapshot_drain(std::string&){return capture_drains.at(this);}
struct VerifierDrainTestPeer {static void bind(Verifier& v,SessionState& s,int device){v.ss_=&s;v.device_=device;}};
bool conversation_snapshot_save_private(SavedConversation& out,const ConversationView& view,
    const SessionState& session,const ModelGeometry&,std::string& error){
    ++capture_copies;WindowLaneToken attempt;
    check(capture_manager && capture_manager->switch_active(lane(*capture_manager),capture_token,attempt)==S::read_held,"exported read blocks resume during native copy");
    auto directory=std::move(out.stage_images);out=window_test::saved(int(session.layer_lo),int(session.layer_hi),1,session.ple_hist!=nullptr);
    out.stage_images=std::move(directory);out.live.ids=view.ids;out.cvec=view.cvec;
    if(capture_copies==capture_fail_at){error="injected exported capture failure";return false;}return true;
}
}
struct Harness {
    WindowManager manager;
    Fixture fixture;
    ManagedWindowToken entry;
    std::unique_ptr<WindowTokenLedger> ledger;
    Harness(WindowBudget& budget,uint64_t max=64,uint64_t serial=0,uint64_t incarnation=77,WindowKey k=key(1))
        :manager([&]{auto l=limits();l.manager_incarnation=incarnation;return l;}(),budget),fixture(fresh(budget,k)),entry(reg(manager,fixture,k)) {
        ledger=std::make_unique<WindowTokenLedger>(entry.selection.key,entry.selection.cut,WindowTokenLedgerLimits{max,2,serial},budget);
        WindowLaneToken active_lane;check(manager.switch_active(lane(manager),entry,active_lane)==S::ok,"ledger initial bind");
    }
    ManagedWindowWork admit(){ManagedWindowWork work;check(manager.admit(lane(manager),work)==S::ok,"ledger actual admit");return work;}
    ManagedWindowToken refresh(){entry=token(manager,entry);return entry;}
    WindowTokenCandidateView proposal(const WindowTokenCandidate& c){WindowTokenCandidateView out;check(ledger->inspect(c,out)==TS::ok,"candidate inspect");return out;}
    WindowTokenReceipt commit(const WindowTokenCandidate& c){auto v=proposal(c);WindowTokenReceipt out;check(ledger->finish(c,manager,{v.proposed_cut,v.proposed_cut},true,true,out)==TS::ok,"exact ledger commit");refresh();return out;}
    void cancel(const WindowTokenCandidate& c,ManagedWindowWork& work){check(manager.request_cancel(entry)==S::ok,"terminal cancel");auto w=work.lease();check(w.finish(w.operation(),entry.selection.key,{},true,true),"terminal drain base key");check(ledger->reconcile_terminal(c,manager)==TS::cancelled,"terminal reconcile");}
};
struct ImportSource {BundleHandle bundle;WindowStateLayout layout;};
static ImportSource import_source(const WindowKey& old) {
    const auto g=window_test::geometry();auto meta=window_test::metadata();meta.key=old;
    ++meta.key.revision;++meta.key.selection_epoch;++meta.key.append_epoch;
    auto native=window_test::saved(0,4,1,true);native.stage_images.push_back(window_test::saved(4,8,1,false));
    ImportSource out;SnapshotDescriptor descriptor;
    check(bool(native_window_layout(g,{{0,4,{1},true},{4,8,{1},false}},meta.key,meta.cut,1024,1<<20,out.layout)),"import source layout");
    check(bool(describe_native_window(native,g,out.layout,meta,{{{7,8},9},{{-1,-1},-1}},descriptor)),"import source descriptor");
    check(bool(BundleHandle::freeze(descriptor,meta.key,out.layout,out.bundle)),"import source immutable freeze");return out;
}
static Fixture import_destination(WindowBudget& budget,const ImportSource& source,std::unique_ptr<WindowTokenLedger> ledger={}) {
    Fixture f;const auto g=window_test::geometry();
    check(bool(allocate_native_sessions(g,16,1,{{0,0,4,true},{1,4,8,false}},budget,f.allocation)),"import private factory");
    f.runtime=std::make_shared<Runtime>();f.runtime->ledger=std::move(ledger);
    check(bool(prepare_owned_native_restore(source.bundle,source.bundle.get()->key,g,f.allocation,
        {[](auto&,auto&){return true;},[](auto&,auto&){return true;}},budget,1024,1<<20,true,f.prepared,f.runtime)),"import actual prepared restore");return f;
}
static void park_import(Harness& h) {
    check(h.manager.request_park(lane(h.manager))==S::ok,"import closes current admission");WindowLaneToken empty;
    check(h.manager.unbind(lane(h.manager),empty)==S::ok,"import source unbound");
}
int main(){
    alarm(60);
    WindowBudget budget(WindowResources{{1000000,1000000},256<<20,10000,0,0,100},WindowResources{{0,0},0,0,0,0,0});
    {
        Harness h(budget);WindowTokenCandidate candidate;const int32_t* address=nullptr;
        auto work=h.admit();const std::vector<int32_t> prompt{1,2,3};
        check(h.ledger->stage_accept(h.entry,work,prompt,candidate)==TS::ok,"stage entire accepted prompt");
        check(h.ledger->view().cut.accepted==0,"staged tokens not yet published");
        h.ledger->borrow_published([&](auto ids,size_t pending){check(ids.empty()&&pending==0,"published empty before finish");address=ids.data();check(h.ledger->view().cut.id==1,"callback reenters without mutex");});
        auto proposed=h.proposal(candidate);WindowTokenReceipt receipt;
        check(h.ledger->finish(candidate,h.manager,{proposed.proposed_cut,proposed.proposed_cut},true,false,receipt)==TS::unresolved,"undrained work not published");
        check(h.ledger->view().cut.accepted==0&&h.ledger->view().pending_serial!=0,"undrained candidate retained");
        receipt=h.commit(candidate);check(receipt.cut.accepted==3&&receipt.cut.forwarded==0&&!receipt.output,"prompt acceptance metadata unit");
        auto stale=candidate;work=h.admit();check(h.ledger->stage_forward(h.entry,work,2,candidate)==TS::ok,"stage prompt chunk");h.commit(candidate);
        check(h.ledger->finish(stale,h.manager,{},true,true,receipt)==TS::stale,"old candidate cannot finish new work");
        work=h.admit();check(h.ledger->stage_decode(h.entry,work,candidate)==TS::ok,"bind decode before sampling");
        check(h.ledger->finish(candidate,h.manager,{},true,true,receipt)==TS::awaiting_sample,"no output or finish before sample");
        check(h.ledger->record_sample(candidate,99)==TS::ok&&h.ledger->record_sample(candidate,98)==TS::invalid,"one sampled output identity");
        receipt=h.commit(candidate);check(receipt.output==99&&receipt.cut.accepted==4&&receipt.cut.forwarded==3&&receipt.cut.output_watermark==1,"EOS output remains pending");
        std::vector<int32_t> expected{1,2,3,99};
        h.ledger->borrow_published([&](auto ids,size_t pending){check(ids.data()==address&&std::equal(ids.begin(),ids.end(),expected.begin())&&pending==3,"no prefix relocation pending EOS");});
        check(receipt.cut.history==window_history_digest(expected),"canonical accepted history");
        work=h.admit();const std::vector<int32_t> suffix{4,5};check(h.ledger->stage_accept(h.entry,work,suffix,candidate)==TS::ok,"continue after EOS pending");receipt=h.commit(candidate);
        expected.insert(expected.end(),suffix.begin(),suffix.end());check(h.ledger->matches_history_prefix(expected),"continuation full prefix");expected[3]=98;check(!h.ledger->matches_history_prefix(expected),"edited prefix rejected");
        check(receipt.cut.forwarded==3&&receipt.cut.accepted==6&&receipt.key.append_epoch==3,"append once per accepted transaction");
        work=h.admit();check(h.ledger->stage_decode(h.entry,work,candidate)==TS::invalid,"continuation pending EOS plus suffix must forward before sample");
        check(h.ledger->stage_forward(h.entry,work,3,candidate)==TS::ok,"forward old EOS plus new prompt tail");
        try {h.ledger->borrow_candidate(candidate,[&](auto,size_t){check(h.ledger->finish(candidate,h.manager,{},true,true,receipt)==TS::busy,"borrow blocks mutation");throw std::runtime_error("unwind");});}catch(const std::runtime_error&){}
        h.commit(candidate);check(!h.ledger->view().phase,"borrow unwind and completed candidate clear pending");
    }
    {
        Harness h(budget);auto work=h.admit();WindowTokenCandidate c;const std::vector<int32_t> prompt{1,2};
        check(h.ledger->stage_accept(h.entry,work,prompt,c)==TS::ok,"cancel append fixture");h.cancel(c,work);
        check(h.ledger->view().cut.accepted==0&&!h.ledger->view().phase,"cancel truncates unpublished suffix");
    }
    {
        Harness h(budget);auto work=h.admit();WindowTokenCandidate c;const std::vector<int32_t> prompt{1,2};
        check(h.ledger->stage_accept(h.entry,work,prompt,c)==TS::ok,"cancellation bool fixture");h.commit(c);
        work=h.admit();check(h.ledger->stage_forward(h.entry,work,1,c)==TS::ok,"forward cancelled fixture");auto v=h.proposal(c);
        check(h.manager.request_cancel(h.entry)==S::ok,"cancel before ledger finish");WindowTokenReceipt output;output.output=123;
        check(h.ledger->finish(c,h.manager,{v.proposed_cut,v.proposed_cut},true,true,output)==TS::unresolved&&output.output==123,"finish true on cancellation is not publication");
        check(h.ledger->view().cut.forwarded==0&&h.ledger->reconcile_terminal(c,h.manager)==TS::cancelled,"observe aborted original cut");
    }
    {
        Harness h(budget);auto work=h.admit();WindowTokenCandidate c;const std::vector<int32_t> prompt{1};
        check(h.ledger->stage_accept(h.entry,work,prompt,c)==TS::ok,"failed work fixture");
        auto lease=work.lease();check(lease.mark_unhealthy(lease.operation()),"mark failed device work");
        auto v=h.proposal(c);WindowTokenReceipt output;
        check(h.ledger->finish(c,h.manager,{v.proposed_cut,v.proposed_cut},true,true,output)==TS::unresolved,"unhealthy retained no publication");
        c={};check(h.ledger->pending_candidate(c)==TS::ok,"lost caller candidate recoverable");
        check(lease.retire_after_teardown(lease.operation()),"confirmed actual teardown seam");
        check(h.ledger->reconcile_terminal(c,h.manager)==TS::cancelled,"teardown releases candidate");
    }
    {
        Harness a(budget),b(budget,64,0,78);auto wa=a.admit(),wb=b.admit();WindowTokenCandidate c;const std::vector<int32_t> p{1};
        check(wa.selected().token()==wb.selected().token()&&wa.lease().operation()==wb.lease().operation()&&wa.token()!=wb.token(),"same selection/op distinct manager incarnation");
        check(a.ledger->stage_accept(a.entry,wb,p,c)==TS::stale,"foreign actual work registration rejected");
        check(a.ledger->stage_accept(a.entry,wa,p,c)==TS::ok,"actual work identity");WindowTokenCandidateView view;
        check(b.ledger->inspect(c,view)==TS::stale,"candidate belongs to exact ledger");a.cancel(c,wa);
        check(b.ledger->stage_accept(b.entry,wb,p,c)==TS::ok,"other ledger binds own work");b.cancel(c,wb);
    }
    {
        auto k=key(1);k.window_id.assign(40,'w');Harness h(budget,4,0,77,k);auto work=h.admit();WindowTokenCandidate c;
        const auto before=budget.snapshot().reserved;const std::vector<int32_t> p{1,2};
        fail_new=1;check(h.ledger->stage_accept(h.entry,work,p,c)==TS::allocation_failed&&!c,"actual candidate allocation failure");
        check(budget.snapshot().reserved==before&&!h.ledger->view().phase&&h.ledger->view().cut.accepted==0,"allocation failure preserves ledger charge");
        check(h.ledger->stage_accept(h.entry,work,p,c)==TS::ok,"retry before device work");auto v=h.proposal(c);WindowTokenReceipt output;output.output=42;
        const std::vector<StateCut> completed{v.proposed_cut,v.proposed_cut};
        fail_new=1;check(h.ledger->finish(c,h.manager,completed,true,true,output)==TS::allocation_failed&&output.output==42,"receipt allocation before finish leaves output");
        h.commit(c);work=h.admit();check(h.ledger->stage_decode(h.entry,work,c)==TS::invalid,"cannot sample before all but last pending prompt input forwarded");
        check(h.ledger->stage_forward(h.entry,work,1,c)==TS::ok,"forward prompt before sample");h.commit(c);
        work=h.admit();check(h.ledger->stage_decode(h.entry,work,c)==TS::ok,"sample allocation fixture");
        fail_new=1;check(h.ledger->record_sample(c,7)==TS::allocation_failed,"canonical hasher allocation failure after sample");
        check(h.ledger->view().cut.accepted==2&&h.ledger->view().phase==WindowTokenPhase::unresolved,"sample candidate retained unpublished");
        check(h.ledger->record_sample(c,7)==TS::ok,"same sample hash retry");h.commit(c);
        work=h.admit();const std::vector<int32_t> too_many{8,9};WindowTokenCandidate preserved=c;
        check(h.ledger->stage_accept(h.entry,work,too_many,c)==TS::limit,"token bound refusal");
        WindowTokenCandidateView old;check(h.ledger->inspect(c,old)==TS::ok&&old.phase==WindowTokenPhase::committed,"refusal preserves prior output candidate");
        check(h.ledger->stage_forward(h.entry,work,1,c)==TS::ok,"bound refusal does not strand work");h.commit(c);
    }
    {
        Harness h(budget,64,std::numeric_limits<uint64_t>::max());auto work=h.admit();WindowTokenCandidate c;const std::vector<int32_t> p{1};
        check(h.ledger->stage_accept(h.entry,work,p,c)==TS::exhausted,"serial exhaustion no wrap");
        check(h.manager.request_cancel(h.entry)==S::ok,"exhaustion cancels admitted work");auto w=work.lease();check(w.finish(w.operation(),h.entry.selection.key,{},true,true),"exhaustion drain");
        auto k=key(1);k.append_epoch=std::numeric_limits<uint64_t>::max();Harness e(budget,64,0,79,k);auto ew=e.admit();
        check(e.ledger->stage_accept(e.entry,ew,p,c)==TS::exhausted,"append exhaustion");check(e.manager.request_cancel(e.entry)==S::ok,"append cancel");auto lease=ew.lease();check(lease.finish(lease.operation(),e.entry.selection.key,{},true,true),"append drain");
    }
    {
        StateCut initial;initial.id=1;initial.history=window_history_digest({});const auto before=budget.snapshot().reserved;
        bool succeeded=false;int failures=0;
        for(int point=1;point<=64&&!succeeded;++point){
            fail_new=point;
            try{WindowTokenLedger ledger(key(1),initial,{64,2,0},budget);succeeded=true;}
            catch(const std::bad_alloc&){++failures;}
            catch(const std::runtime_error&){++failures;}
            fail_new=0;check(budget.snapshot().reserved==before,"constructor allocation failure/success returns exact charge");
        }
        check(succeeded&&failures>1,"all constructor allocation points tested until first success");
    }
    {
        WindowBudget tiny(WindowResources{{0,0},1,0,0,0,0},WindowResources{{0,0},0,0,0,0,0});
        StateCut initial;initial.id=1;initial.history=window_history_digest({});bool refused=false;
        try{WindowTokenLedger ledger(key(1),initial,{64,2,0},tiny);}catch(const std::runtime_error&){refused=true;}
        check(refused&&tiny.snapshot().reserved.host_bytes==0,"budget refusal no leaked charge");
    }
#if defined(STRATA_WINDOW_LEDGER_TEST_SEAM)
    {
        auto k=key(1);k.window_id.assign(40,'q');Harness h(budget,64,0,77,k);auto work=h.admit();WindowTokenCandidate c;const std::vector<int32_t> p{1};
        check(h.ledger->stage_accept(h.entry,work,p,c)==TS::ok,"post-finish allocation fixture");h.commit(c);
        work=h.admit();check(h.ledger->stage_decode(h.entry,work,c)==TS::ok&&h.ledger->record_sample(c,9)==TS::ok,"stage real output before observation failure");
        auto v=h.proposal(c);const std::vector<StateCut> cuts{v.proposed_cut,v.proposed_cut};WindowTokenReceipt output;output.output=54;
        after_finish=[] {fail_new=1;};bool failed=false;
        try{h.ledger->finish(c,h.manager,cuts,true,true,output);}catch(const std::bad_alloc&){failed=true;}
        check(failed&&output.output==54&&h.ledger->view().phase==WindowTokenPhase::unresolved,"post-finish snapshot allocation retains candidate without output");
        ManagedWindowSnapshot actual;check(h.manager.entry_snapshot(h.entry.registration,actual)==S::ok&&actual.token.selection.cut==v.proposed_cut&&!actual.execution.in_flight,"device execution cut already published");
        check(h.ledger->view().cut.accepted==1&&h.ledger->view().cut.output_watermark==0,"host ledger still old while observation uncertain");
        output=h.commit(c);check(output.output==9&&h.ledger->view().cut.accepted==2&&output.cut.output_watermark==1,"exact observation retry publishes output once without repeated finish");
        output.output=54;check(h.ledger->finish(c,h.manager,cuts,true,true,output)==TS::stale&&output.output==54,"replayed candidate cannot emit duplicate output");
    }
    {
        // This deliberately violates reactor serialization and retains uncertain
        // ownership by contract. Child exits without claiming teardown/cleanup.
        auto pid=fork();check(pid>=0,"fork uncertain publication");if(!pid){
            Harness h(budget);auto work=h.admit();WindowTokenCandidate c;const std::vector<int32_t> p{1};
            check(h.ledger->stage_accept(h.entry,work,p,c)==TS::ok,"uncertain candidate");auto v=h.proposal(c);WindowTokenReceipt out;out.output=55;
            after_finish=[&]{auto advanced=h.refresh();auto newer=h.admit();auto lease=newer.lease();auto cut=advanced.selection.cut;++cut.id;check(lease.finish(lease.operation(),advanced.selection.key,{cut,cut},true,true),"misbehaving reactor advanced before observation");};
            check(h.ledger->finish(c,h.manager,{v.proposed_cut,v.proposed_cut},true,true,out)==TS::unresolved&&out.output==55,"post-finish advance refuses output");
            check(h.ledger->view().cut.accepted==0&&h.ledger->view().pending_serial!=0,"published device state not rolled back or falsely host-published");
            check(budget.snapshot().reserved.host_bytes>0,"uncertain reservation retained");_exit(0);
        }int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"uncertain child result");
    }
#endif
    {
        Harness h(budget);park_import(h);auto source=import_source(h.entry.selection.key);
        std::unique_ptr<WindowTokenLedger> imported;WindowTokenImport ticket;
        check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,source.layout,{64,2,0},budget,imported,ticket)==TS::ok,"prepare validated history import");
        check(!imported->view().active&&!h.ledger->view().active&&ticket,"both ledgers gated during import");
        check(imported->borrow_published([](auto,size_t){})==TS::inactive,"private imported history not executable before activation");
        auto* ledger=imported.get();auto destination=import_destination(budget,source,std::move(imported));auto clone=import_destination(budget,source);
        ManagedWindowToken selected;check(h.manager.replace_parked(h.entry,ticket.old_read(),destination.prepared,selected)==S::ok,"actual manager publishes prepared restore");
        check(h.manager.matches_prepared(selected,destination.prepared)&&!h.manager.matches_prepared(selected,clone.prepared),"same metadata is not prepared storage identity");
        check(ledger->activate_import(ticket,h.manager,selected,clone.prepared)==TS::stale&&ticket&&!ledger->view().active,"wrong private storage cannot activate");
        check(h.ledger->abort_import(ticket,h.manager)==TS::stale&&!h.ledger->view().active,"cannot abort into old ledger after replacement");
        fail_new=1;check(ledger->activate_import(ticket,h.manager,selected,destination.prepared)==TS::allocation_failed&&ticket&&!ledger->view().active,"activation observation allocation retains inactive ticket");
        check(h.manager.matches_prepared(selected,destination.prepared),"activation failure never rolls back actual selection");
        check(ledger->activate_import(ticket,h.manager,selected,destination.prepared)==TS::ok&&!ticket&&ledger->view().active&&!h.ledger->view().active,"exact activation retry retires old ledger");
        check(ledger->view().key==source.bundle.get()->key&&ledger->view().cut==source.bundle.get()->cut,"all imported epochs/frontiers/output exact");
        check(ledger->activate_import(ticket,h.manager,selected,destination.prepared)==TS::stale,"consumed ticket replay rejected");
        ledger->borrow_published([&](auto tokens,size_t pending){check(tokens.size()==10&&pending==9&&tokens.back()==10,"canonical accepted pending imported");});
        WindowLaneToken bound;check(h.manager.switch_active(lane(h.manager),selected,bound)==S::ok,"resume imported runtime");ManagedWindowWork work;
        check(h.manager.admit(bound,work)==S::ok,"admit real imported selection");WindowTokenCandidate candidate;
        check(ledger->stage_decode(selected,work,candidate)==TS::ok&&ledger->record_sample(candidate,99)==TS::ok,"continue imported final pending input");
        WindowTokenCandidateView next;check(ledger->inspect(candidate,next)==TS::ok,"import decode proposal");WindowTokenReceipt output;
        check(ledger->finish(candidate,h.manager,{next.proposed_cut,next.proposed_cut},true,true,output)==TS::ok&&output.output==99&&output.cut.accepted==11&&output.cut.forwarded==10&&output.cut.output_watermark==8&&output.key.append_epoch==2,"no output/append reset on imported EOS");
        check(h.manager.request_park(lane(h.manager))==S::ok,"pause imported output before regression attempts");WindowLaneToken parked;
        check(h.manager.unbind(lane(h.manager),parked)==S::ok,"unbind imported output");auto current=token(h.manager,selected);
        for(int mode=0;mode<3;++mode){
            auto regressed=*source.bundle.get();regressed.key=current.selection.key;++regressed.key.revision;++regressed.key.selection_epoch;++regressed.key.append_epoch;
            regressed.cut.id=current.selection.cut.id+1;regressed.cut.output_watermark=current.selection.cut.output_watermark+1;
            if(mode==0)regressed.cut.output_watermark=current.selection.cut.output_watermark-1;
            if(mode==1)regressed.key.append_epoch=current.selection.key.append_epoch-1;
            if(mode==2)regressed.cut.id=current.selection.cut.id-1;
            for(auto& device:regressed.devices)device.cut=regressed.cut;
            BundleHandle bundle;check(bool(BundleHandle::freeze(regressed,regressed.key,source.layout,bundle)),"valid immutable regression source");
            std::unique_ptr<WindowTokenLedger> refused;WindowTokenImport refused_ticket;
            check(ledger->prepare_import(h.manager,current,bundle,regressed.key,source.layout,{64,2,0},budget,refused,refused_ticket)==TS::invalid&&!refused&&!refused_ticket&&ledger->view().active,"output append and cut regressions refused without relabeling");
        }
        std::weak_ptr<Runtime> held=destination.runtime;destination.runtime.reset();destination.prepared={};
        auto latest=token(h.manager,selected);check(h.manager.request_cancel(latest)==S::ok,"retire imported owner");WindowLaneToken empty;
        check(h.manager.unbind(lane(h.manager),empty)==S::ok&&h.manager.retire(latest)==S::ok,"remove imported registry owner");
        work={};check(held.expired(),"no prepared-runtime-ledger ownership cycle");
    }
    {
        Harness h(budget);park_import(h);auto source=import_source(h.entry.selection.key);std::unique_ptr<WindowTokenLedger> imported;WindowTokenImport ticket;
        auto wrong=source.bundle.get()->key;++wrong.provenance_epoch;
        check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,wrong,source.layout,{64,2,0},budget,imported,ticket)==TS::invalid&&!imported&&!ticket,"wrong expected provenance refused");
        auto layout=source.layout;layout.model_profile[0]^=1;
        check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,layout,{64,2,0},budget,imported,ticket)==TS::invalid,"wrong model layout refused");
        WindowBudget tiny(WindowResources{{0,0},1,0,0,0,0},WindowResources{{0,0},0,0,0,0,0});const auto old=h.ledger->view();
        check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,source.layout,{64,2,0},tiny,imported,ticket)==TS::budget_refused,"import private budget refusal");
        check(h.ledger->view().active&&h.ledger->view().key==old.key&&h.ledger->view().cut==old.cut&&!h.ledger->view().import_serial&&!imported&&!ticket,"failed private allocation restores old state unchanged");
        const auto before=budget.snapshot().reserved;bool succeeded=false;int failures=0;
        for(int point=1;point<=256&&!succeeded;++point){
            fail_new=point;auto result=h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,source.layout,{64,2,0},budget,imported,ticket);fail_new=0;
            if(result==TS::ok){succeeded=true;check(h.ledger->abort_import(ticket,h.manager)==TS::ok,"safe exact pre-selection abort");imported.reset();}
            else{check(result==TS::allocation_failed||result==TS::budget_refused,"injected import allocation status");++failures;check(!imported&&!ticket,"failed import outputs unchanged");}
            check(h.ledger->view().active&&!h.ledger->view().import_serial&&budget.snapshot().reserved==before,"every failed import returns exact private charge and read hold");
        }
        check(succeeded&&failures>1,"import allocation sweep until first success");
        check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,source.layout,{64,2,0},budget,imported,ticket)==TS::ok,"ticket abandonment fixture");
        ticket={};check(!h.ledger->view().active&&!imported->view().active,"abandoned published ticket never reactivates writes");
        auto* preserved=imported.get();check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,source.layout,{64,2,0},budget,imported,ticket)==TS::inactive&&!h.ledger->view().active&&imported.get()==preserved&&!ticket,"failed retry cannot resurrect abandoned inactive ledger");
    }
    {
        Harness h(budget);auto source=import_source(h.entry.selection.key);auto work=h.admit();WindowTokenCandidate candidate;const std::vector<int32_t> p{1};
        check(h.ledger->stage_accept(h.entry,work,p,candidate)==TS::ok,"pending import conflict fixture");
        std::unique_ptr<WindowTokenLedger> imported;WindowTokenImport ticket;
        check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,source.layout,{64,2,0},budget,imported,ticket)==TS::busy&&!imported&&!ticket,"cannot import current pending work");h.cancel(candidate,work);
    }
    {
        Harness h(budget);park_import(h);auto source=import_source(h.entry.selection.key);std::unique_ptr<WindowTokenLedger> imported;WindowTokenImport ticket;
        check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,source.layout,{64,2,0},budget,imported,ticket)==TS::ok,"post-selection abandonment fixture");
        auto* ledger=imported.get();auto destination=import_destination(budget,source,std::move(imported));ManagedWindowToken selected;
        check(h.manager.replace_parked(h.entry,ticket.old_read(),destination.prepared,selected)==S::ok,"replacement before abandonment");
        ticket={};check(!ledger->view().active&&!h.ledger->view().active&&h.manager.matches_prepared(selected,destination.prepared),"abandonment preserves selected owner and both write fences");
        WindowLaneToken bound;check(h.manager.switch_active(lane(h.manager),selected,bound)==S::ok,"deliberately bypass driver activation check");ManagedWindowWork work;
        check(h.manager.admit(bound,work)==S::ok,"admitted route still cannot bypass ledger gate");WindowTokenCandidate c;
        check(ledger->stage_decode(selected,work,c)==TS::inactive&&h.ledger->stage_decode(selected,work,c)==TS::inactive,"neither inactive ledger may stage GPU unit");
        check(h.manager.request_cancel(selected)==S::ok,"cancel bypassed admission");auto lease=work.lease();check(lease.finish(lease.operation(),selected.selection.key,{},true,true),"drain bypassed admission");
    }
    {
        Harness h(budget);WindowTokenCandidate candidate;auto work=h.admit();const std::vector<int32_t> prompt{1,2,3,4,5,6,7,8,9};
        check(h.ledger->stage_accept(h.entry,work,prompt,candidate)==TS::ok,"capture accepted prompt");WindowTokenCapture exported;
        check(h.ledger->export_capture(h.manager,h.entry,true,budget,exported)==TS::busy&&!exported,"no pending transaction export");h.commit(candidate);
        check(h.ledger->export_capture(h.manager,h.entry,true,budget,exported)==TS::stale,"running selection not exported");
        work=h.admit();check(h.ledger->stage_forward(h.entry,work,8,candidate)==TS::ok,"capture forward prompt");h.commit(candidate);
        work=h.admit();check(h.ledger->stage_decode(h.entry,work,candidate)==TS::ok&&h.ledger->record_sample(candidate,10)==TS::ok,"capture sampled pending EOS");h.commit(candidate);park_import(h);
        auto foreign=h.entry;++foreign.registration.manager;
        check(h.ledger->export_capture(h.manager,foreign,true,budget,exported)==TS::stale&&!exported,"foreign registration export refused");
        const auto before=budget.snapshot().reserved.host_bytes;
        check(h.ledger->export_capture(h.manager,h.entry,true,budget,exported)==TS::ok&&exported.usable(),"exact ledger export");
        const auto& meta=exported.metadata();check(meta.key==h.entry.selection.key&&meta.cut==h.entry.selection.cut&&meta.pending_tokens==std::vector<int32_t>{10}&&meta.cut.output_watermark==1&&meta.devices.empty(),"pending EOS and output preserved without tensor fabrication");
        check(meta.cut.history==window_history_digest(meta.accepted_tokens)&&budget.snapshot().reserved.host_bytes>before,"canonical history copy charged");
        auto retained=exported;exported={};WindowLaneToken next;
        check(h.manager.switch_active(lane(h.manager),h.entry,next)==S::read_held,"copied capture handle retains source read");exported=retained;
        WindowBudget tiny(WindowResources{{0,0},1,0,0,0,0},WindowResources{{0,0},0,0,0,0,0});auto* previous=&exported.metadata();
        check(h.ledger->export_capture(h.manager,h.entry,true,tiny,exported)==TS::budget_refused&&&exported.metadata()==previous,"budget failure preserves prior capture handle");
        const auto reserved=budget.snapshot().reserved;bool success=false;int failures=0;
        for(int point=1;point<=128&&!success;++point){
            fail_new=point;auto status=h.ledger->export_capture(h.manager,h.entry,true,budget,exported);fail_new=0;
            if(status==TS::ok){success=true;exported={};}else{check(status==TS::allocation_failed||status==TS::budget_refused,"export allocation refusal");++failures;check(&exported.metadata()==previous,"allocation failure prior output unchanged");}
            check(budget.snapshot().reserved==reserved,"export allocation failure releases temporary charge/read");
        }
        check(success&&failures>1,"all export allocations tested to first success");exported=retained;
        h.fixture.runtime->capture_verifiers=std::make_unique<std::array<Verifier,2>>();
        for(size_t i=0;i<2;++i){auto& verifier=(*h.fixture.runtime->capture_verifiers)[i];VerifierDrainTestPeer::bind(verifier,*h.fixture.allocation.stages()[i].session,int(i));capture_drains[&verifier]=Verifier::SnapshotDrain::drained;}
        capture_manager=&h.manager;capture_token=h.entry;capture_copies=0;capture_fail_at=0;BundleHandle bundle;NativeCaptureResult captured;
        auto capture=[&]{check(exported.selected_read().borrow([&](const auto& views,const auto& runtime){
            auto owner=std::dynamic_pointer_cast<const Runtime>(runtime);check(owner&&owner->capture_verifiers,"exact runtime verifier owners");
            std::vector<NativeCaptureStage> stages;for(size_t i=0;i<views.size();++i)stages.push_back({views[i].device,views[i].session,&(*owner->capture_verifiers)[i]});
            captured=capture_native_window(exported.selected_read().lease(),window_test::geometry(),stages,exported.metadata(),1024,1<<24,budget,bundle);
        }),"native capture borrows exact selected sessions/runtime");};
        capture();check(bool(captured)&&capture_copies==2&&bundle.get()->pending_tokens==meta.pending_tokens&&bundle.get()->cut==meta.cut&&bundle.get()->accepted_tokens==meta.accepted_tokens,"actual native capture consumes exported history");
        for(const auto& stage:bundle.get()->devices)check(stage.cut==meta.cut,"all captured stage cuts exact");
        capture_drains[&(*h.fixture.runtime->capture_verifiers)[1]]=Verifier::SnapshotDrain::pending;capture_copies=0;capture();check(captured.status==NativeCaptureStatus::pending&&capture_copies==0,"pending native drain leaves export held");
        capture_drains[&(*h.fixture.runtime->capture_verifiers)[1]]=Verifier::SnapshotDrain::drained;
        exported={};retained={};check(h.manager.switch_active(lane(h.manager),h.entry,next)==S::ok,"resume after final export copy released");
        capture_manager=nullptr;
    }
    {
        Harness old(budget),next(budget,64,0,78);park_import(old);park_import(next);WindowTokenCapture out;
        check(old.ledger->export_capture(old.manager,old.entry,true,budget,out)==TS::ok,"old output read owner");
        int retired=0;old.fixture.runtime->on_destroy=[&]{check(next.ledger->view().active,"output owner reenters exporting ledger");++retired;};
        check(old.manager.request_cancel(old.entry)==S::ok&&old.manager.retire(old.entry)==S::ok,"retire read-held old manager entry");old.fixture={};
        check(!out.usable()&&retired==0,"invalidated read still retains runtime");
        check(next.ledger->export_capture(next.manager,next.entry,true,budget,out)==TS::ok&&retired==1,"prior output destruction occurs outside ledger mutex");
    }
    {
        Harness h(budget);park_import(h);auto source=import_source(h.entry.selection.key);auto approximate=*source.bundle.get();
        approximate.cut.exact_frontier=8;ApproximationRegion region;region.begin=8;region.end=9;region.first_unresolved_unix_ms=1;region.ancestor[0]=1;approximate.approximation={region};
        for(auto& device:approximate.devices)device.cut=approximate.cut;
        check(bool(BundleHandle::freeze(approximate,approximate.key,source.layout,source.bundle)),"immutable approximate source with ancestry");
        std::unique_ptr<WindowTokenLedger> imported;WindowTokenImport ticket;
        check(h.ledger->prepare_import(h.manager,h.entry,source.bundle,source.bundle.get()->key,source.layout,{64,2,0},budget,imported,ticket)==TS::ok,"approximate history import");
        auto* ledger=imported.get();auto destination=import_destination(budget,source,std::move(imported));ManagedWindowToken selected;
        check(h.manager.replace_parked(h.entry,ticket.old_read(),destination.prepared,selected)==S::ok&&ledger->activate_import(ticket,h.manager,selected,destination.prepared)==TS::ok,"select exact approximate provenance");
        WindowTokenCapture out;check(ledger->export_capture(h.manager,selected,true,budget,out)==TS::invalid&&!out,"missing approximation ancestry refused");
        check(ledger->export_capture(h.manager,selected,false,budget,out,source.bundle)==TS::invalid,"wrong steering provenance refused");
        check(ledger->export_capture(h.manager,selected,true,budget,out,source.bundle)==TS::ok&&out.metadata().cut.exact_frontier==8&&out.metadata().approximation.size()==1&&out.metadata().approximation[0].ancestor==region.ancestor,"export preserves exact frontier and immutable ancestry");
    }
    std::puts("window_token_ledger_test passed");
}
