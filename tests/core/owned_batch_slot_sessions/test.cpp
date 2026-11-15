// Actual two-slot production owner + native retirement methods + real budget.
// Allocation/CUDA boundaries record host lifetime/control only, never native fit/drain proof.
#include "owned_batch_slot_sessions.hpp"
#include "strata/core/native_batch_slot_binding.hpp"
#include "strata/core/native_batch_group_completion.hpp"
#include "strata/core/native_batch_setup_admission.hpp"
#include "strata/core/stream_watch.hpp"
#include <map>
#include <future>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <bit>
#include <cmath>
#include <cerrno>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "serving_thread_bridge.hpp"
#include "strata/core/window_device.hpp"
#include "strata/core/pinned_ring.hpp"
#include "strata/prefill/owned_workspace.hpp"
#include "strata/core/serving_raw_metadata.hpp"
#include <span>
#include <thread>
#include "strata/core/window_state.hpp"
#include "strata/core/serving_raw_contracts.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <new>
#include <stdexcept>
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/mrope.hpp"
#include "../core/window_hash_vendor/sha256.h"

// Recording CUDA storage ONLY for the new bank factory. Production owner,
// allocator, admission and binding methods below are extracted unchanged.
namespace bank_recording {
inline void* pointers[24]{};inline size_t bytes[24]{};inline int devices[24]{};
inline int allocations=0,frees=0,fail_at=-1,enclosing_frees=0;
inline bool require_enclosing_payment=false,copy_fail=false;
inline int free_receipt=-1;
inline strata::core::WindowBudget* ledger=nullptr;
inline uint64_t required_bank_device=0;
inline void paid_bank(){
    if(!ledger)std::abort();
    const auto live=ledger->snapshot().reserved;
    if(!live.host_bytes||live.device_bytes.size()!=2||
       live.device_bytes[0]<required_bank_device||live.device_bytes[1]<required_bank_device)std::abort();
}
}
inline int cudaMalloc(void** out,size_t n){
    if(bank_recording::allocations==bank_recording::fail_at){*out=nullptr;return 1;}
    if(bank_recording::allocations>=24)std::abort();
    auto* p=std::malloc(n);if(!p)return 1;
    const int i=bank_recording::allocations++;bank_recording::pointers[i]=p;
    bank_recording::bytes[i]=n;bank_recording::devices[i]=seam::device;
    seam::native_used[seam::device]+=n;*out=p;return cudaSuccess;
}
inline int cudaMemcpy(void* dst,const void* src,size_t n,int){if(bank_recording::copy_fail)return 1;std::memcpy(dst,src,n);return cudaSuccess;}
inline int recorded_bank_free(void* p){
    bank_recording::paid_bank();
    for(int i=0;i<bank_recording::allocations;++i)if(bank_recording::pointers[i]==p){
        if(bank_recording::devices[i]!=seam::device)std::abort();
        seam::native_used[seam::device]-=bank_recording::bytes[i];bank_recording::pointers[i]=nullptr;
        if(bank_recording::free_receipt>=0){const char f='F';(void)::write(bank_recording::free_receipt,&f,1);}
        ++bank_recording::frees;std::free(p);return cudaSuccess;
    }
    std::abort();
}
// Rename just the recording CUDA boundary; actual owner source stays intact.
#define cudaFree recorded_bank_free
#include "owned_vision_position_banks.hpp"
#undef cudaFree
// Actual private data-owner declarations only, without pulling in a different
// native SessionState than this existing recording boundary. No image issuer or
// payload is fabricated by these legacy text/bank groups.
#include "owned_vision_request.hpp"
#include "owned_admission_cancel.hpp"
#include "strata/core/conversation_cache.hpp"
ACTUAL_IMAGE_PLE_DECL;
namespace strata::program {
class ServingThreadBridge;
ACTUAL_IMAGE_OPERATION_DECL;
ACTUAL_IMAGE_DIRECTORY_DECL;
ACTUAL_IMAGE_WITNESS_DECL;
}


ACTUAL_IMAGE_SIZE_HELPERS
namespace strata::core {
// Bounded backing representation, after genuine native ledger admission.
void SessionState::prepare_recording(const ModelGeometry& geometry){
    if(!seam::image_mode||max_cells<=0||max_cells>128||qsa_alloc!=1||qsa_ord0<0||qsa_ord0>1)std::abort();
    ConversationStateSizes sizes{};std::string error;
    if(!conversation_session_sizes(geometry,*this,sizes,error))std::abort();
    recorded_gdn.assign(sizes.gdn,std::byte{});gdn_state=recorded_gdn.data();
    recorded_ple.assign(sizes.ple,std::byte{});ple_hist=recorded_ple.data();
    auto& q=qsa_states[qsa_ord0];q.kv_int8=true;q.max_cells=max_cells;q.n_pages=max_cells/4;q.n_slots=q.n_pages;
    q.idx_pooled_rows=max_cells/4+1;
    std::array<uint64_t,5> bytes{};
    if(!conversation_kv_part_sizes(q,geometry,max_cells,true,bytes,error))std::abort();
    for(size_t i=0;i<5;++i){if(bytes[i]>(1U<<20))std::abort();q.backing[i].assign(size_t(bytes[i]),std::byte{});}
    q.backing[5].assign(sizes.tail,std::byte{});q.backing[6].assign(sizes.dead,std::byte{});q.backing[7].assign(sizes.block_pos,std::byte{});
    q.k_pool=q.k_q=q.backing[0].data();q.v_pool=q.v_q=q.backing[1].data();
    q.k_scale=q.backing[2].data();q.v_scale=q.backing[3].data();q.idx_pooled=q.backing[4].data();
    q.idx_tail=q.backing[5].data();q.idx_dead=q.backing[6].data();q.idx_block_pos=q.backing[7].data();
}
struct ExpertSource {bool transient(int64_t,int64_t)const{return false;}};
namespace {
std::recursive_mutex operation_mutex;
std::mutex allocator_mutex;
bool allocator_failed=false;
struct Stage {
    int device=-1;
    std::unique_ptr<SessionState> session;
    std::vector<std::shared_ptr<QsaElasticView>> elastic_pools;
    void* ple_native_key=nullptr;void* ple_scratch=nullptr;void* ple_embedding=nullptr;void* arena=nullptr;
};
struct Storage {
    std::shared_ptr<const ReservationLease> reservation;
    std::vector<Stage> stages;
    bool terminal=false,uncertain=false,retired=false,poisoned=false;
};
std::vector<std::shared_ptr<Storage>> quarantine;
}
// Actual production convenience allocator creates one fresh session object per stage.
ACTUAL_CONVENIENCE_ALLOCATOR
// Recording physical boundary, using the REAL allocation type and REAL WindowBudget lease.
NativeAllocationResult allocate_native_sessions(const ModelGeometry& geometry,int64_t cells,int64_t k,
    const std::vector<NativeSessionRequest>& requests,std::vector<std::unique_ptr<SessionState>>& fresh,
    WindowBudget& budget,NativeSessionAllocation& out){
    ++seam::allocation_calls;
    if(seam::allocation_calls==seam::allocation_fail)return {NativeAllocationStatus::budget_refused,"recording gang refusal"};
    if(fresh.size()!=requests.size()||requests.empty()||requests.back().layer_end!=geometry.n_layers)
        return {NativeAllocationStatus::invalid,"recording topology refusal"};
    WindowResources resources{std::vector<uint64_t>(budget.snapshot().capacity.device_bytes.size()),128,0,0,0,1};
    for(const auto& request:requests)resources.device_bytes.at(size_t(request.device))+=100;
    auto reserved=budget.reserve(resources,ReservationPurpose::private_work);
    if(!reserved)return {NativeAllocationStatus::budget_refused,"actual budget refused"};
    auto storage=std::make_shared<Storage>();storage->reservation=reserved.lease;
    storage->stages.resize(requests.size());
    for(size_t i=0;i<requests.size();++i){auto& stage=storage->stages[i];stage.device=requests[i].device;
        stage.session=std::move(fresh[i]);stage.session->device=stage.device;
        stage.session->layer_lo=requests[i].layer_begin;stage.session->layer_hi=requests[i].layer_end;
        stage.session->max_cells=cells;stage.session->k=k;stage.session->qsa_states=new QsaState[2];
        if(seam::image_mode){stage.session->qsa_ord0=int64_t(i);stage.session->qsa_alloc=1;stage.session->gdn_ord0=int64_t(i);stage.session->gdn_alloc=1;
            stage.session->prepare_recording(geometry);}}
    for(size_t i=0;i<requests.size();++i){
        if(seam::allocation_calls==seam::partial_fail&&i==1){
            storage->uncertain=true;allocator_failed=true;quarantine.push_back(storage);
            return {NativeAllocationStatus::unhealthy,"recording second-stage physical refusal"};
        }
        storage->stages[i].arena=reinterpret_cast<void*>(uintptr_t(i+1));
    }
    NativeSessionAllocation allocation;
    allocation.reservation_=reserved.lease;allocation.storage_=storage;
    for(auto& stage:storage->stages)allocation.stages_.push_back({stage.device,stage.session.get(),storage});
    allocation.healthy_=[storage]{return !storage->terminal&&!storage->uncertain&&!storage->poisoned&&!allocator_failed;};
    allocation.poison_=[storage]{storage->poisoned=true;storage->uncertain=true;};
    out=std::move(allocation);return {NativeAllocationStatus::allocated,{}};
}
// These terminal methods are extracted unchanged from src/core/window_allocator.cpp.
ACTUAL_PREPARE_RETIREMENT
ACTUAL_DRAIN_RETIREMENT
// Actual verifier inspection bodies run over recording private-field inventory.
// No batch graph/submission/completion is represented by this host fixture.
struct OnDevice {int previous=0;OnDevice(int device,bool){cudaGetDevice(&previous);cudaSetDevice(device);}~OnDevice(){cudaSetDevice(previous);}};
bool batch_self_commits(const int*,int){return false;}
using PoolMultiFn=void(*)();
class Verifier {public:
    static void* operator new(size_t n){return ::operator new(n);}
    static void* operator new(size_t,void* p)noexcept{return p;}
    static void operator delete(void*,void*)noexcept{}
    static void operator delete(void* p)noexcept{
        if(bank_recording::require_enclosing_payment){
            bank_recording::paid_bank();
            ++bank_recording::enclosing_frees;
        }
        ::operator delete(p);
    }
    const ModelGeometry* g_=nullptr;SessionState* ss_=nullptr;int device_=0;
    int64_t lb_=0,le_=0,n_vocab_=1000;Verifier* next_=nullptr;int cs_=11,copy_=12;bool strict_retirement_=false;
    std::shared_ptr<const NativeBatchSetupAdmission> owned_batch_setup_;
    int max_t_=2;void* arena_b_=nullptr;
    std::map<std::vector<int>,int> exec_bm_,commit_bm_;
    bool init_slots(const std::vector<SessionState*>& slots,std::string& err){
        if(!begin_inventory(err))return false;
        slots_=slots;arena_b_=this;seam::native_used[device_]+=20;return publish_inventory();}
    bool ar_off_=false;
    ACTUAL_BANK_KEY
    ACTUAL_BATCH_KEY
    bool capture_batch(const int* rows,int count,int,std::string& err){
        exec_bm_[bkey(rows,count,0)]=1;seam::native_used[device_]+=10;
        const int ex=1;ACTUAL_GRAPH_UPLOAD_CHECK
        return true;
    }
    bool capture_commit_batch(const int* rows,int count,int,std::string&){commit_bm_[bkey(rows,count,0)]=1;seam::native_used[device_]+=10;return true;}
    enum class SnapshotDrain{pending,drained,unhealthy,unsupported};
    SnapshotDrain poll_snapshot_drain(std::string&){return cudaStreamSynchronize(cs_)==cudaSuccess&&cudaStreamSynchronize(copy_)==cudaSuccess?SnapshotDrain::drained:SnapshotDrain::unhealthy;}
    bool nonfinite_in(int)const{return false;}
    bool prepare_owned_batch_slots(WindowBudget&,std::shared_ptr<const NativeBatchSlotBindings>,uint64_t,std::string&);
    uint64_t owned_group_last_operation_=0;bool owned_group_busy_=false,dispatch_fail_=false,commit_fail_=false;
    bool run_slot_rows(const int* rows,int count,const int32_t* tokens,const int64_t* pos,PoolMultiFn pool,void* user,int32_t* output,std::string& err){
        if(refuse_copy_query_failure(err))return false;
        if(position_banks_){
            if(!owned_position_begin_batch(rows,count,0)){err="recording dispatch rejected actual bank latch";return false;}
            last_t_=count;row_base_=0;last_batch_=true;
            for(int i=0;i<count;++i)last_rows_[i]=rows[i];
            // Actual matcher, authentic operation ticket supplied by the real
            // group method. Perturb only recording stage-output DATA, then restore.
            if(!owned_position_commit_matches())std::abort();
            row_base_=1;if(owned_position_commit_matches())std::abort();row_base_=0;
            ar_off_=!ar_off_;if(owned_position_commit_matches())std::abort();ar_off_=!ar_off_;
            const auto epoch=owned_position_replay_.epoch;++owned_position_replay_.epoch;
            if(owned_position_commit_matches())std::abort();
            owned_position_replay_.epoch=epoch;
            const int original=last_rows_[0];last_rows_[0]=1-original;
            if(owned_position_commit_matches())std::abort();
            last_rows_[0]=original;
            if(!owned_position_commit_matches())std::abort();
        }
        seam::events.push_back("dispatch");if(dispatch_fail_)return false;
        for(int row=0;row<count;++row)output[row]=tokens[row]+1;
        ACTUAL_BATCH_COPY_CHECK
        return !next_||next_->run_slot_rows(rows,count,tokens,pos,pool,user,output,err);
    }
    bool commit_slot_prefixes(const int* keep,std::string& err){
        if(position_banks_&&!owned_position_commit_matches()){err="actual bank commit mismatch";return false;}
        seam::events.push_back("commit");if(commit_fail_)return false;
        if(position_banks_)owned_position_replay_.live=false;
        return !next_||next_->commit_slot_prefixes(keep,err);
    }
    bool run_owned_slot_group(const NativeBatchSlotInspection*,size_t,uint64_t,const int*,int,const int32_t*,const int64_t*,const int*,PoolMultiFn,void*,int32_t*,std::shared_ptr<const ReservationLease>,std::shared_ptr<const NativeBatchGroupCompletion>&,std::string&);
    std::array<std::weak_ptr<const NativeBatchSlotOccupancy>,2> owned_slot_occupancies_;
    const uint64_t slot_verifier_incarnation_=next_verifier_incarnation();
    std::vector<SessionState*> slots_;uint64_t slot_inventory_generation_=0;bool slot_inventory_ready_=false;
    std::atomic<bool> released_{false},terminal_copy_query_failed_{false};bool batch_copy_failed_=false;
    std::string snapshot_failure_,commit_failure_;
    bool inspect_owned_batch_slot(std::shared_ptr<const NativeBatchSlotBindings>,NativeBatchSlotToken,size_t,NativeBatchSlotInspection&)const;
    bool owned_batch_slot_matches(const NativeBatchSlotInspection&)const noexcept;
    std::shared_ptr<const strata::program::OwnedPositionBankBinding> position_banks_;
    bool batch_rec_=false;int brow_[8]{};int exec_[17]{},exec_nr_[17]{},commit_exec_=0;
    ACTUAL_BANK_REPLAY owned_position_replay_;
    int last_t_=0,row_base_=0,last_rows_[8]{};bool last_batch_=false;
    bool owned_position_begin_batch(const int*,int,int)noexcept;
    bool owned_position_commit_matches()const noexcept;
    const int32_t* owned_position_table(int)const;
    bool owned_position_binding_matches()const noexcept;
    ACTUAL_COMMON_REFUSAL
    ACTUAL_SOLO_BINDING
    bool begin_inventory(std::string& err){ACTUAL_INVENTORY_BEGIN return true;}
    bool publish_inventory(){ACTUAL_INVENTORY_READY return true;}
};
ACTUAL_POSITION_BEGIN
ACTUAL_POSITION_COMMIT
ACTUAL_POSITION_MATCH
ACTUAL_POSITION_TABLE
ACTUAL_SLOT_MATCH
ACTUAL_SLOT_INSPECT
ACTUAL_OWNED_SETUP
ACTUAL_OWNED_GROUP
}

namespace strata::prefill {
ACTUAL_PREFILL_IMAGE_PRODUCER
[[maybe_unused]] static int64_t MAXBLOB(){return 4096;} // recording expert backend size, NOT a native/production ABI proof
struct Stager {struct Job{const uint8_t* src;size_t bytes;core::ExpertSource* from=nullptr;int32_t l=0,e=0;};};
// Other prefill state is a representation boundary; target predicate is actual.
// The separate workspace producer fixture still executes the real paid issuer.
class Prefill {public:
    static void* operator new(size_t n){return ::operator new(n);}
    static void operator delete(void* p)noexcept{
        if(bank_recording::require_enclosing_payment){
            bank_recording::paid_bank();
            ++bank_recording::enclosing_frees;
        }
        ::operator delete(p);
    }
    struct RecordedBorrower{bool drain_owned(){return true;}bool valid()const{return false;}};
    struct Impl{const core::ModelGeometry* g=nullptr;const core::SessionState* ss=nullptr;int device=-1;bool pp=false,help_pp=false;core::PinnedRing up_ring;int cs=31;};
    bool strict_retirement_=true;std::shared_ptr<const void> workspace_credit_;
    std::unique_ptr<Impl> impl_=std::make_unique<Impl>();
    void* cpu_pool_=nullptr;Prefill* helper_=nullptr;const float* const* embd_rows=nullptr;void* on_routes=nullptr;void* on_taps=nullptr;
    std::shared_ptr<const program::OwnedPositionBankBinding> position_banks_;
    std::shared_ptr<const program::OwnedImageRequestOperation> image_operation_;
    uint64_t image_generation_=0;RecordedBorrower next_run_;
    const core::ModelGeometry* workspace_geometry_=nullptr;const core::SessionState* workspace_session_=nullptr;
    const core::ExpertSource* workspace_source_=nullptr;int64_t workspace_chunk_=0;bool workspace_borrowed_=false,workspace_successor_=false;
    Prefill* next_=nullptr;
    using WorkspaceAdmitter=std::function<WorkspaceAdmission(const OwnedWorkspaceBytes&,std::shared_ptr<const void>&)>;
    WorkspaceAdmission admit_owned_workspace(const core::ModelGeometry&,const core::SessionState&,core::ExpertSource*,int64_t,bool,const WorkspaceAdmitter&,std::string&);
    static uint64_t bytes_needed(const core::ModelGeometry&,const core::SessionState&,int64_t,bool){return 4096;}
    static uint64_t bytes_needed_owned(const core::ModelGeometry& g,const core::SessionState& ss,int64_t c,bool source){return bytes_needed(g,ss,c,source);}
    bool owned_image_target(const core::ModelGeometry*,const core::SessionState*,int)const noexcept;
    bool owned_image_rows_present(int64_t,int64_t)const noexcept;
    // Exact pre-kernel private token/row guard from run_impl, ending at the
    // actual native computation boundary. This does not claim kernel parity.
    bool recorded_rows(const int64_t* tokens,int64_t n,int64_t pos0,const float* const*& output,std::string& err){
        Impl& m=*impl_;
        ACTUAL_IMAGE_ROWS_PREFIX
        output=request_rows;return true;
    }
    ~Prefill(){
        if(!impl_)return;
        if(impl_->device<0){if(impl_->up_ring.owned_mode())std::terminate();impl_.reset();workspace_credit_.reset();return;}
        core::WindowDeviceScope device(impl_->device);
        if(impl_->up_ring.owned_mode()&&(!device.ok()||!impl_->up_ring.drain_owned()))std::terminate();
        impl_->up_ring.retire_strict();impl_.reset();workspace_credit_.reset();
        if(!device.finish())std::terminate();
    }

    bool owned_image_retirement_ready(const program::OwnedImageRequestOperation*)noexcept;
    bool owned_position_bank_target(const core::ModelGeometry*,const core::SessionState*,int)const noexcept;
};
ACTUAL_PREFILL_BANK_TARGET
ACTUAL_PREFILL_IMAGE_RETIREMENT
ACTUAL_PREFILL_IMAGE_TARGET
ACTUAL_PREFILL_IMAGE_ROWS
ACTUAL_PREFILL_WORKSPACE_ADMISSION
}

SLOT_RAW_FIXTURE
namespace strata::program {ACTUAL_IMAGE_DRIVER;}
using namespace strata::program;
namespace core=strata::core;
namespace program=strata::program;
namespace prefill=strata::prefill;
// Recording runtime allocation/free boundary. The concrete outer keeper body
// is copied from generate, not reimplemented as a lifetime predicate.
struct DriverWindowRuntime {
    std::unique_ptr<core::Verifier> primary,secondary;
    std::unique_ptr<prefill::Prefill> first_prefill,second_prefill;
};
ACTUAL_BANK_RETIREMENT;

void require(bool result,const char* why){if(!result){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
// Preserve the former recording declaration's DATA defaults explicitly while
// using the SINGLE real ModelGeometry type from geometry.hpp. No model load or
// production compiled-geometry/ABI claim is made by this small fixture shape.
core::ModelGeometry recording_geometry(){
    core::ModelGeometry g;
    g.n_layers=4;g.n_vocab=1000;g.n_embd=2560;g.qsa_interval=2;
    g.ssm_state_size=2;g.ssm_k_heads=1;g.ssm_v_heads=1;g.ssm_d_conv=2;
    g.ssm_conv_channels=4;g.ssm_value_dim=2;g.n_head=4;g.n_head_kv=2;
    g.head_dim=64;g.idx_q_heads=4;g.idx_key_dim=128;
    g.hc=4;g.hc_lr=320;g.n_expert=2;g.n_ff=4;
    return g;
}
core::WindowResources capacity(uint64_t each=1000){return {{each,each},100000,100000,0,0,4};}
std::vector<core::NativeSessionRequest> requests(){
    std::vector<core::NativeSessionRequest> values(2);
    values[0].device=0;values[0].layer_begin=0;values[0].layer_end=2;
    values[1].device=1;values[1].layer_begin=2;values[1].layer_end=4;
    return values;
}
void distinct_and_reuse(){
    core::WindowBudget budget(capacity(),{{0,0},0,0,0,0,0});auto geometry=recording_geometry();
    OwnedBatchSlotSessions slots;std::string error;
    require(slots.initialize(geometry,128,2,requests(),budget,error),"two actual owned gang calls publish together");
    auto first=slots.handle(0),second=slots.handle(1);
    require(first.generation==1&&second.generation==1&&first.slot!=second.slot,"distinct physical-slot handle identities");
    require(seam::sessions_alive==4&&seam::allocation_calls==2&&slots.stage_count()==2,"two full stage gangs retained");
    require(slots.session(0,first)!=slots.session(0,second)&&slots.session(1,first)!=slots.session(1,second),
            "each stage/slot has its own actual fresh session object");
    require(slots.allocation(first)->reservation()!=slots.allocation(second)->reservation()&&
            slots.allocation(first)->reservation()->purpose()==core::ReservationPurpose::private_work,
            "each gang holds a distinct genuine production budget lease");
    require(budget.snapshot().reserved.device_bytes==std::vector<uint64_t>({200,200}),"real ledger charges both gangs");
    auto alias=*slots.allocation(first);auto single_stage_owner=alias.stages()[0].owner;
    std::weak_ptr<const void> whole_gang=single_stage_owner;
    auto stale=first;stale.incarnation++;
    require(!slots.allocation(stale),"foreign runtime incarnation refuses handle");
    seam::events.clear();
    require(slots.retire_after_consumers_joined({},error),"actual terminal retirement of both complete gangs");
    require(!slots.session(0,first)&&!alias.healthy(),"all old handles and allocation aliases revoked before reuse");
    auto first_free=std::find(seam::events.begin(),seam::events.end(),"free0");
    require(first_free!=seam::events.end()&&
            std::find(seam::events.begin(),first_free,"drain0")!=first_free&&
            std::find(seam::events.begin(),first_free,"drain1")!=first_free,
            "actual retirement drains every stage device before any stage free");
    require(!whole_gang.expired()&&seam::sessions_alive==2,"a retained stage owner pins the whole retired gang object/lease");
    alias=core::NativeSessionAllocation{};
    require(!whole_gang.expired()&&seam::sessions_alive==2,"one retained stage owner keeps every gang stage and reservation alive");
    single_stage_owner.reset();
    require(whole_gang.expired()&&seam::sessions_alive==0&&budget.snapshot().reserved.host_bytes==0,
            "last actual gang alias releases all stage objects and real reservation");
    require(slots.initialize(geometry,128,2,requests(),budget,error),"physical slot can reconstruct only after checked retirement");
    require(slots.handle(0).generation==2&&!slots.session(0,first),"reconstruction increments generation and rejects old handles");
    require(slots.retire_after_consumers_joined({},error),"reused gangs explicitly retire");
}
void second_gang_refusal(){
    core::WindowBudget budget(capacity(),{{0,0},0,0,0,0,0});auto geometry=recording_geometry();
    OwnedBatchSlotSessions slots;std::string error;seam::allocation_fail=2;
    require(!slots.initialize(geometry,128,2,requests(),budget,error)&&!slots.published(),"second allocation refusal publishes neither slot");
    require(slots.handle(0).generation==0&&slots.session(0,slots.handle(0))==nullptr,"partial shape cannot execute");
    require(seam::sessions_alive==2&&budget.snapshot().reserved.device_bytes==std::vector<uint64_t>({100,100}),
            "first complete all-stage gang and its genuine lease retained after second refusal");
    require(slots.retire_after_consumers_joined({},error)&&seam::sessions_alive==0&&budget.snapshot().reserved.host_bytes==0,
            "startup refusal uses actual terminal retirement without partial shape leak");
}
void partial_stage_refusal(){
    core::WindowBudget budget(capacity(),{{0,0},0,0,0,0,0});auto geometry=recording_geometry();
    OwnedBatchSlotSessions slots;std::string error;seam::partial_fail=2;
    require(!slots.initialize(geometry,128,2,requests(),budget,error)&&!slots.published(),"actual API unhealthy result never publishes slots");
    require(seam::sessions_alive==4&&budget.snapshot().reserved.device_bytes==std::vector<uint64_t>({200,200}),
            "earlier complete gang plus failing all-stage gang remain retained by native boundary/quarantine");
    require(!slots.retire_after_consumers_joined({},error),"uncertain native allocator refuses a guessed retirement proof");
    require(seam::sessions_alive==4&&!slots.initialize(geometry,128,2,requests(),budget,error),
            "failed retirement cannot recycle quarantined physical addresses");
}
void actual_owned_verifier_inspection(){
    core::WindowBudget budget(capacity(),{{0,0},0,0,0,0,0});auto geometry=std::make_shared<core::ModelGeometry>(recording_geometry());
    OwnedBatchSlotSessions slots;std::string error;
    require(slots.initialize(*geometry,128,2,requests(),budget,error),"actual slot owner publishes complete gangs for binding");
    std::array<const core::NativeSessionAllocation*,2> allocations{};std::array<core::NativeBatchSlotToken,2> tokens{};
    for(size_t slot=0;slot<2;++slot){auto handle=slots.handle(slot);allocations[slot]=slots.allocation(handle);
        tokens[slot]={handle.incarnation,handle.generation,handle.slot};}
    const auto binding_bytes=core::NativeBatchSlotBindings::metadata_bytes();
    auto create=[&](uint64_t bytes,const std::array<const core::NativeSessionAllocation*,2>& sources,
                    const std::array<core::NativeBatchSlotToken,2>& identities){
        auto quote=budget.reserve(core::WindowResources{{0,0},bytes,0,0,0,0},core::ReservationPurpose::foreground);
        require(bool(quote),"actual foreground metadata quota available");
        return core::NativeBatchSlotBindings::create(sources,identities,geometry,std::move(quote.lease),error);
    };
    require(!create(1,allocations,tokens),"binding refuses underpriced metadata owner before copying gang handles");
    auto alias=allocations;alias[1]=allocations[0];
    require(!create(binding_bytes,alias,tokens),"typed binding rejects two logical slots aliased to one physical gang");
    auto wrong=tokens;wrong[1].incarnation++;
    require(!create(binding_bytes,allocations,wrong),"mixed allocation incarnations refuse before publication");
    auto bindings=create(binding_bytes,allocations,tokens);require(bool(bindings),"bounded actual owned binding map created");
    core::SessionState main;main.layer_hi=2;main.max_cells=128;main.k=2;
    core::Verifier verifier;verifier.g_=geometry.get();verifier.ss_=&main;
    for(size_t slot=0;slot<2;++slot)verifier.slots_.push_back(slots.session(0,slots.handle(slot)));
    require(verifier.begin_inventory(error)&&!verifier.slot_inventory_ready_,"actual init epoch revokes uncompleted inventory");
    core::NativeBatchSlotInspection observation;
    require(!verifier.inspect_owned_batch_slot(bindings,tokens[0],0,observation),"incomplete actual verifier slot init cannot issue observation");
    require(verifier.publish_inventory()&&verifier.inspect_owned_batch_slot(bindings,tokens[0],0,observation)&&
        verifier.owned_batch_slot_matches(observation),"actual verifier inspects real slot pointer/device/layout/generation");
    require(!verifier.snapshot_binding_matches(verifier.slots_[0],0),"solo RAW verifier binding guard remains strict and rejects slot session");
    core::NativeBatchSlotInspection refused;
    auto stale=tokens[0];stale.generation++;
    require(!verifier.inspect_owned_batch_slot(bindings,stale,0,refused)&&!refused.present(),"stale physical allocation generation refuses");
    require(!verifier.inspect_owned_batch_slot(bindings,tokens[0],1,refused),"wrong stage/device refuses");
    auto* original=verifier.slots_[1];verifier.slots_[1]=verifier.slots_[0];
    require(!verifier.owned_batch_slot_matches(observation),"full actual slot table must match both owned gangs");verifier.slots_[1]=original;
    verifier.device_=1;require(!verifier.owned_batch_slot_matches(observation),"wrong actual verifier device refuses");verifier.device_=0;
    core::ModelGeometry substituted=*geometry;verifier.g_=&substituted;
    require(!verifier.owned_batch_slot_matches(observation),"equal-looking copied geometry is not actual retained owner");verifier.g_=geometry.get();
    ++verifier.slots_[0]->max_cells;
    require(!verifier.owned_batch_slot_matches(observation),"changed slot allocation layout refuses");--verifier.slots_[0]->max_cells;
    verifier.batch_copy_failed_=true;require(!verifier.owned_batch_slot_matches(observation),"uncertain verifier copy state refuses observation");verifier.batch_copy_failed_=false;
    require(verifier.begin_inventory(error)&&!verifier.owned_batch_slot_matches(observation),"actual init generation invalidates old observations before allocations");
    require(verifier.publish_inventory()&&!verifier.owned_batch_slot_matches(observation),"old observation remains stale after successful rebind");
    core::NativeBatchSlotInspection current;
    require(verifier.inspect_owned_batch_slot(bindings,tokens[1],0,current),"new actual inventory gets distinct current observation");
    // Destroy/reconstruct at the identical address with identical gangs and epoch.
    const auto previous_incarnation=verifier.slot_verifier_incarnation_;
    verifier.~Verifier();new (&verifier) core::Verifier;
    verifier.g_=geometry.get();verifier.ss_=&main;
    for(size_t slot=0;slot<2;++slot)verifier.slots_.push_back(slots.session(0,slots.handle(slot)));
    require(verifier.begin_inventory(error)&&verifier.begin_inventory(error)&&verifier.publish_inventory(),"replacement inventory has identical local epoch");
    require(verifier.slot_verifier_incarnation_!=previous_incarnation&&!verifier.owned_batch_slot_matches(current),"same-address verifier reconstruction cannot revive stale inspection");
    std::weak_ptr<const core::NativeBatchSlotBindings> weak=bindings;bindings.reset();
    require(slots.retire_after_consumers_joined({},error)&&!verifier.owned_batch_slot_matches(current),"terminal native retirement revokes binding health before reuse");
    require(seam::sessions_alive==5,"bounded observations retain actual complete gang owners until released");
    observation={};current={};
    require(weak.expired()&&seam::sessions_alive==1,"last observation destroys complete retained slot gangs, only main fixture remains");
    require(budget.snapshot().reserved.host_bytes==binding_bytes,"weak control block keeps metadata lease until deallocation");
    weak.reset();require(budget.snapshot().reserved.host_bytes==0,"final weak metadata owner releases its actual ledger reservation");
}
void actual_slot_raw_exercise(const std::shared_ptr<core::ModelGeometry>& geometry,OwnedBatchSlotSessions& slots,
        core::Verifier& root,core::Verifier& tail,const std::array<core::NativeBatchSlotInspection,4>& observations,
        core::WindowBudget& budget){
    std::string error;auto runtime=std::make_shared<int>(9);
    core::NativeSessionAllocation main_allocation;
    require(bool(core::allocate_native_sessions(*geometry,128,2,requests(),budget,main_allocation)),"genuine main cap owner gets distinct actual allocator gang");
    core::Verifier main_root,main_tail;main_root.g_=main_tail.g_=geometry.get();
    main_root.ss_=main_allocation.stages()[0].session;main_tail.ss_=main_allocation.stages()[1].session;
    main_root.le_=2;main_root.next_=&main_tail;main_tail.lb_=2;main_tail.le_=4;main_tail.device_=1;
    auto aggregate_main=ServingRawSession::open(*geometry,main_allocation,runtime,{&main_root,&main_tail},budget,128,1u<<26,error);
    require(bool(aggregate_main),"actual main RAW factory admits one real aggregate H owner");
    require(&aggregate_main->tx_->geometry_!=geometry.get()&&aggregate_main->tx_->geometry_identity_==geometry.get(),"actual constructor copies shape while retaining original geometry authority");
    auto factory=[&](size_t slot,uint64_t cap,const core::ModelGeometry& actual){
        return ServingRawSession::open_owned_slot(actual,*slots.allocation(slots.handle(slot)),runtime,
            {&root,&tail},{observations[slot],observations[2+slot]},budget,cap,*aggregate_main,error);};
    std::shared_ptr<const core::NativeBatchGroupCompletion> stale_physical;
    {
        const int rows[]={0,1},keep[]={1,1};const int32_t input[]={17,23};const int64_t positions[]={0,0};int32_t output[2]{};
        auto credit=budget.reserve({{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
        require(root.run_owned_slot_group(observations.data(),2,1,rows,2,input,positions,keep,nullptr,nullptr,output,std::move(credit.lease),stale_physical,error),"real earlier physical operation has no later conversation occupancy authority");
    }
    const auto before_reset=seam::zero_calls;
    std::shared_ptr<const void> foreign_control(runtime.get(),[](const void*){});
    require(!ServingRawSession::open_owned_slot(*geometry,*slots.allocation(slots.handle(0)),foreign_control,
        {&root,&tail},{observations[0],observations[2]},budget,128,*aggregate_main,error)&&
        error.find("control=0")!=std::string::npos&&seam::zero_calls==before_reset,"same runtime address with foreign ownership control refuses before reset");

    require(!factory(0,129,*geometry)&&seam::zero_calls==before_reset,"slot factory refuses capacity above actual session before reset");
    require(error.find("actual binding refused")!=std::string::npos&&error.find("cells=128 tokens=129")!=std::string::npos,"actual capacity refusal names physical and requested bounds");
    auto copied_geometry=*geometry;
    require(!factory(0,128,copied_geometry)&&seam::zero_calls==before_reset,"equal-looking caller geometry cannot replace actual owned geometry");
    require(error.find("aggregate owner guard")!=std::string::npos&&error.find("geometry=0")!=std::string::npos,"actual geometry refusal names authenticated owner mismatch");
    auto a=factory(0,128,*geometry),b=factory(1,128,*geometry);
    require(a&&b&&a->tx_->host_cap_==aggregate_main->tx_->host_cap_&&b->tx_->host_cap_==aggregate_main->tx_->host_cap_,"both slot actors retain same actual aggregate H object as main");
    require(&a->tx_->geometry_!=&b->tx_->geometry_&& &a->tx_->geometry_!=geometry.get()&&
        a->tx_->geometry_identity_==geometry.get()&&b->tx_->geometry_identity_==geometry.get(),"two actual copied snapshots retain same original geometry authority");
    require(aggregate_main->open_request(error)&&aggregate_main->begin_work(error),"main source actual transfer preflight fence");
    core::StateCut source_cut;source_cut.id=1;source_cut.history=core::window_history_digest(std::vector<int32_t>{});
    const std::vector<core::StateCut> source_stages(2,source_cut);
    require(bool(aggregate_main->tx_->complete_driver_work(true,source_cut,source_stages)),"main transfer preflight actual checked completion");
    require(a->open_request(error),"actual destination transfer request");
    core::SnapshotDescriptor source_metadata;
    require(!a->transfer_owned_slot_from(*aggregate_main,source_metadata,error)&&aggregate_main->transfer_capture_reached&&
        error=="recorded refusal at native capture boundary","actual transfer passes same-authority value copies and stops at recorded native capture boundary");
    require(a->close_request(error)&&aggregate_main->close_request(error),"transfer preflight leaves both requests closable");
    {
        core::NativeSessionAllocation allocation;
        require(bool(core::allocate_native_sessions(*geometry,128,2,requests(),budget,allocation)),"private positive gets distinct actual gang");
        core::Verifier private_root,private_tail;private_root.g_=private_tail.g_=geometry.get();
        private_root.ss_=allocation.stages()[0].session;private_tail.ss_=allocation.stages()[1].session;
        private_root.le_=2;private_root.next_=&private_tail;private_tail.lb_=2;private_tail.le_=4;private_tail.device_=1;
        auto private_runtime=std::make_shared<int>(10);
        strata::program::SuffixRawTransaction private_tx(*geometry,a->tx_->identity_,allocation,private_runtime,
            {&private_root,&private_tail},budget,128,a->tx_->max_bytes_,a->tx_->host_limit_,false);
        require(bool(private_tx.initialize_resident_shared_host(*a->tx_))&&private_tx.host_cap_==a->tx_->host_cap_,
            "actual private shared-host initializer accepts original service geometry despite distinct shape copies");
    }

    require(a&&b&&a->tx_->cut_.forwarded==0&&b->tx_->cut_.forwarded==0&&seam::zero_calls==before_reset+4,"actual two slot factories reset every real stage before initial zero cuts");
    require(!factory(0,128,*geometry),"active conversation occupancy cannot be replaced on same physical slot");
    require(error.find("occupied/running slot=0")!=std::string::npos,"actual occupied slot refusal cannot return blank reason");
    auto* real_slot=root.slots_[0];root.slots_[0]=root.slots_[1];
    std::vector<core::NativeWindowStage> actual_config;
    require(!a->tx_->resolve(actual_config,error),"actual RAW resolution rechecks current verifier slot inventory before capture/restore");
    root.slots_[0]=real_slot;

    require(a->open_request(error)&&b->open_request(error)&&a->queue_owned_slot_input(17,error)&&b->queue_owned_slot_input(23,error),"actual slot owners admit distinct known pending input");
    require(a->begin_work(error)&&b->begin_work(error),"actual RAW and occupancy fences open before group");
    const int rows[]={0,1},accepted[]={1,1};const int32_t input[]={17,23};const int64_t positions[]={0,0};int32_t output[2]{};
    auto* emitted=std::tmpfile();require(emitted!=nullptr,"open recording output file");
    require(!a->drain_owned_slot_group(*stale_physical,emitted,error)&&std::ftell(emitted)==0,"pre-reset unregistered physical receipt cannot certify or emit current occupant");
    auto quote=budget.reserve({{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
    std::shared_ptr<const core::NativeBatchGroupCompletion> receipt;
    require(root.run_owned_slot_group(observations.data(),2,2,rows,2,input,positions,accepted,nullptr,nullptr,output,std::move(quote.lease),receipt,error),"actual registered owners dispatch genuine checked group receipt");
    const auto before_duplicate=seam::events.size();
    auto duplicate_quote=budget.reserve({{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
    std::shared_ptr<const core::NativeBatchGroupCompletion> duplicate_receipt;
    require(!root.run_owned_slot_group(observations.data(),2,3,rows,2,input,positions,accepted,nullptr,nullptr,output,std::move(duplicate_quote.lease),duplicate_receipt,error)&&
        !duplicate_receipt&&seam::events.size()==before_duplicate,"new operation cannot resubmit actual pending input before its returned receipt is consumed");

    require(a->drain_owned_slot_group(*receipt,emitted,error)&&b->drain_owned_slot_group(*receipt,emitted,error),"actual once-only group consumers emit their actual selected outputs and drain RAW");
    require(a->tx_->cut_.accepted==2&&a->tx_->cut_.forwarded==1&&a->tx_->cut_.exact_frontier==1&&a->tx_->cut_.output_watermark==1&&a->tx_->cut_.history==core::window_history_digest({17,18}),"slot A truthful consumed prefix, real pending sample, emission watermark and authentic history");
    require(b->tx_->cut_.history==core::window_history_digest({23,24}),"slot B keeps independent actual history");
    const auto bytes=std::ftell(emitted);require(!a->drain_owned_slot_group(*receipt,emitted,error)&&std::ftell(emitted)==bytes,"duplicate receipt cannot emit a second output");
    require(a->close_request(error)&&b->close_request(error)&&a->retire_owned_slot(error)&&b->retire_owned_slot(error),"actual drained request retirement revokes occupancies");
    const auto old=a->slot_occupancy_->incarnation();auto next=factory(0,128,*geometry);
    require(next&&next->slot_occupancy_->incarnation()!=old,"physical gang reuse gets distinct actual conversation incarnation");
    require(next->open_request(error)&&next->queue_owned_slot_input(41,error)&&next->begin_work(error),"replacement owns new pending token");
    const int row0[]={0},keep0[]={1,0};const int32_t token0[]={41};const int64_t pos0[]={0};
    auto next_quote=budget.reserve({{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
    std::shared_ptr<const core::NativeBatchGroupCompletion> next_receipt;
    require(root.run_owned_slot_group(observations.data(),2,3,row0,1,token0,pos0,keep0,nullptr,nullptr,output,std::move(next_quote.lease),next_receipt,error),"replacement group issues actual new occupancy receipt");
    require(!next->drain_owned_slot_group(*receipt,emitted,error)&&std::ftell(emitted)==bytes,"old occupant receipt cannot commit or emit for replacement");
    require(next->drain_owned_slot_group(*next_receipt,emitted,error)&&next->tx_->cut_.history==core::window_history_digest({41,42}),"replacement consumes only genuine current receipt");
    require(next->close_request(error)&&next->retire_owned_slot(error),"replacement retires at checked cut");std::fclose(emitted);
    require(slots.retire_after_consumers_joined({},error),"slot fixture final physical checked retirement");
}
void actual_group_completion(bool failupload=false,bool rawslots=false){
    auto cap=capacity();if(rawslots){cap.host_bytes=cap.pinned_bytes=1u<<28;}
    core::WindowBudget budget(cap,{{0,0},0,0,0,0,0});std::string error;auto geometry=std::make_shared<core::ModelGeometry>(recording_geometry());geometry->n_layers=4;
    OwnedBatchSlotSessions slots;require(slots.initialize(*geometry,128,2,requests(),budget,error),"group gangs construct");
    std::array<const core::NativeSessionAllocation*,2> allocations{};std::array<core::NativeBatchSlotToken,2> tokens{};
    for(size_t slot=0;slot<2;++slot){allocations[slot]=slots.allocation(slots.handle(slot));const auto h=slots.handle(slot);tokens[slot]={h.incarnation,h.generation,h.slot};}
    auto quote=budget.reserve(core::WindowResources{{0,0},core::NativeBatchSlotBindings::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
    auto bindings=core::NativeBatchSlotBindings::create(allocations,tokens,geometry,std::move(quote.lease),error);require(bool(bindings),"group bindings");
    core::SessionState main0,main1;main0.layer_hi=2;main1.layer_lo=2;main1.layer_hi=4;
    for(auto* main:{&main0,&main1}){main->max_cells=128;main->k=2;}
    core::Verifier root,tail;root.g_=tail.g_=geometry.get();root.ss_=&main0;tail.ss_=&main1;
    root.le_=2;root.next_=&tail;tail.lb_=2;tail.le_=4;tail.device_=1;tail.cs_=21;tail.copy_=22;
    std::array<core::NativeBatchSlotInspection,4> inspections{};
    root.strict_retirement_=tail.strict_retirement_=true;
    if(failupload){
        seam::upload_fail_at=1;
        require(!root.prepare_owned_batch_slots(budget,bindings,32768,error)&&root.batch_copy_failed_&&!root.slot_inventory_ready_,"first actual graph upload error cannot be healed by successful later drain");
        require(budget.snapshot().reserved.device_bytes==std::vector<uint64_t>({1000,1000}),"failed graph setup retains full conservative device admission");return;
    }
    require(root.prepare_owned_batch_slots(budget,bindings,32768,error),"actual group setup pre-admits and prewarms both stages");
    require(budget.snapshot().reserved.device_bytes==std::vector<uint64_t>({280,280}),"actual measured setup charge shrinks without uncharged live interval");
    for(size_t stage=0;stage<2;++stage){auto& v=stage?tail:root;
        for(size_t slot=0;slot<2;++slot)require(v.inspect_owned_batch_slot(bindings,tokens[slot],stage,inspections[stage*2+slot]),"group observed stage slots");}
    if(rawslots){actual_slot_raw_exercise(geometry,slots,root,tail,inspections,budget);return;}
    const int rows[]={0,1},accepted[]={1,1};const int32_t input[]={17,23};const int64_t pos[]={7,11};int32_t output[2]{};
    auto invoke=[&](uint64_t operation,std::shared_ptr<const core::NativeBatchGroupCompletion>& result){
        auto credit=budget.reserve(core::WindowResources{{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
        require(bool(credit),"group metadata admission");
        return root.run_owned_slot_group(inspections.data(),2,operation,rows,2,input,pos,accepted,nullptr,nullptr,output,std::move(credit.lease),result,error);};
    std::shared_ptr<const core::NativeBatchGroupCompletion> result;
    root.next_=nullptr;const auto prior=seam::stream_calls;
    require(!invoke(1,result)&&!result&&seam::stream_calls==prior,"missing actual tail refuses before dispatch");root.next_=&tail;
    require(invoke(1,result)&&result&&result->rows()==2&&result->slot(1)==1&&result->position(1)==11&&seam::stream_calls==prior+6,"actual owned group publishes only after compute plus copy for both stages");
    result.reset();require(!invoke(1,result)&&!result,"consumed operation cannot issue another receipt");
    seam::stream_fail=tail.copy_;seam::stream_fail_once=true;
    require(!invoke(2,result)&&!result&&tail.batch_copy_failed_,"first downstream copy error stays poisoned despite later successful drain");
    const auto after_failure=seam::events.size();
    require(!root.run_slot_rows(rows,2,input,pos,nullptr,nullptr,output,error)&&seam::events.size()==after_failure,"public actual common native admission refuses poisoned owned group before dispatch");
    seam::stream_fail=0;require(slots.retire_after_consumers_joined({},error),"group fixture checked retirement");
}
void actual_group_success(){actual_group_completion();}
void actual_setup_upload_refusal(){actual_group_completion(true);}
void actual_slot_raw(){actual_group_completion(false,true);}
void actual_position_bank_startup_case(bool native_refusal){
    auto cap=capacity(32768);cap.host_bytes=cap.pinned_bytes=1u<<28;
    core::WindowBudget ledger(cap,{{0,0},0,0,0,0,0});bank_recording::ledger=&ledger;
    std::weak_ptr<const program::OwnedPositionBankBinding> tail;
    uint64_t retained=0;
    {
        // This outer state persists AFTER every enlarged fixture consumer free.
        OwnedPositionBankRetirement retirement;
        auto& setup=retirement.setup;
        {
            std::string error;auto geometry=std::make_shared<core::ModelGeometry>(recording_geometry());
            OwnedBatchSlotSessions slots;require(slots.initialize(*geometry,128,2,requests(),ledger,error),"bank physical slot owner");
            std::array<const core::NativeSessionAllocation*,2> allocations{};std::array<core::NativeBatchSlotToken,2> tokens{};
            for(size_t slot=0;slot<2;++slot){const auto h=slots.handle(slot);allocations[slot]=slots.allocation(h);tokens[slot]={h.incarnation,h.generation,h.slot};}
            auto metadata=ledger.reserve({{0,0},core::NativeBatchSlotBindings::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
            auto bindings=core::NativeBatchSlotBindings::create(allocations,tokens,geometry,std::move(metadata.lease),error);
            require(bool(bindings),"bank genuine slot bindings");
            core::NativeSessionAllocation main;
            require(bool(core::allocate_native_sessions(*geometry,128,2,requests(),ledger,main)),"bank genuine main allocation");
            auto runtime=std::make_shared<DriverWindowRuntime>();retirement.runtime=runtime;
            runtime->primary=std::make_unique<core::Verifier>();runtime->secondary=std::make_unique<core::Verifier>();
            auto& root=*runtime->primary;auto& last=*runtime->secondary;
            root.g_=last.g_=geometry.get();root.ss_=main.stages()[0].session;last.ss_=main.stages()[1].session;
            root.le_=2;root.next_=&last;last.lb_=2;last.le_=4;last.device_=1;root.strict_retirement_=last.strict_retirement_=true;
            std::shared_ptr<ServingRawSession> session;
            const auto zero_calls_before_startup=seam::zero_calls;
            require(ServingRawSession::open_owned_startup(*geometry,main,runtime,{&root,&last},ledger,128,1u<<27,session,error),"actual reset/drained request-closed startup");
            require(!session->request_open()&&session->phase()==ServingRawSession::Phase::drained&&seam::zero_calls-zero_calls_before_startup==static_cast<int>(main.stages().size()),"real startup common empty cut");
            runtime->first_prefill=std::make_unique<prefill::Prefill>();runtime->second_prefill=std::make_unique<prefill::Prefill>();
            auto& p0=*runtime->first_prefill;auto& p1=*runtime->second_prefill;
            auto workspace=ledger.reserve({{0,0},128,0,0,0,0},core::ReservationPurpose::foreground);require(bool(workspace),"recording prefill workspace ledger payment");
            for(auto* p:{&p0,&p1})p->workspace_credit_=workspace.lease;
            p0.impl_->g=p1.impl_->g=geometry.get();p0.impl_->ss=root.ss_;p1.impl_->ss=last.ss_;p0.impl_->device=0;p1.impl_->device=1;
            prefill::Prefill* ps[]={&p0,&p1};
            uint64_t quoted_cells=0,quoted_bank=0;
            require(strata::kernels::owned_mrope_bank_layout(128,quoted_cells,quoted_bank),"actual checked bank footprint");
            bank_recording::required_bank_device=3*quoted_bank;
            auto before=ledger.snapshot().reserved;const auto claims=session->tx_->host_cap_->claims.load();
            require(session->open_request(error),"real request opens");
            require(!session->prepare_owned_position_banks(bindings,ps,512,setup,error)&&!setup.ready()&&bank_recording::allocations==0&&ledger.snapshot().reserved==before,"open request refuses without allocation/claim/mutation");
            require(session->close_request(error),"real request closes");
            require(!session->prepare_owned_position_banks(bindings,ps,1u<<28,setup,error)&&
                ledger.snapshot().reserved==before&&session->tx_->host_cap_->claims.load()==claims&&bank_recording::allocations==0,
                "aggregate H refusal rolls back the single claim before payload/install");
            auto blocker=ledger.reserve({{30000,30000},0,0,0,0,0},core::ReservationPurpose::foreground);
            require(bool(blocker),"bounded real device ledger blocker");const auto blocked=ledger.snapshot().reserved;
            require(!session->prepare_owned_position_banks(bindings,ps,512,setup,error)&&
                ledger.snapshot().reserved==blocked&&session->tx_->host_cap_->claims.load()==claims&&bank_recording::allocations==0,
                "global vector refusal unwinds H with no payload or publication");blocker.lease.reset();
            if(native_refusal){
                int fds[2];require(::pipe(fds)==0,"poison free-observer pipe");const auto child=::fork();require(child>=0,"poison child");
                if(!child){
                    ::close(fds[0]);bank_recording::free_receipt=fds[1];bank_recording::copy_fail=true;
                    std::set_terminate([]{std::_Exit(73);});
                    (void)session->prepare_owned_position_banks(bindings,ps,512,setup,error);
                    std::_Exit(74); // neither return nor ordinary unwind qualifies
                }
                ::close(fds[1]);int status=0;require(::waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==73,
                    "actual copy uncertainty reaches failstop without returning");
                char freed=0;require(::read(fds[0],&freed,1)==0,"uncertain copy failstops BEFORE any bank free");::close(fds[0]);
                bank_recording::fail_at=2;
                require(!session->prepare_owned_position_banks(bindings,ps,512,setup,error)&&!setup.ready()&&
                    !root.position_banks_&&!last.position_banks_&&bank_recording::frees==2,
                    "partial native allocation refusal frees only after checked drain and publishes no binding");
                require(session->tx_->host_cap_->claims.load()==claims+1&&ledger.snapshot().reserved.host_bytes>before.host_bytes,
                    "post-admission failure retains fixed-enclosing payment until actual free");
                bank_recording::require_enclosing_payment=true;
            }else{
            require(session->prepare_owned_position_banks(bindings,ps,512,setup,error)&&setup.ready(),"actual private bank admission and install");
            bank_recording::require_enclosing_payment=true;
            require(bank_recording::allocations==6&&session->tx_->host_cap_->claims.load()==claims+1,"THREE banks on BOTH stages, ONE actual H claim");
            const uint64_t bank_bytes=(128+64)*3*sizeof(int32_t);
            require(ledger.snapshot().reserved.device_bytes==std::vector<uint64_t>({before.device_bytes[0]+3*bank_bytes,before.device_bytes[1]+3*bank_bytes}),"one exact all-stage device vector");
            const auto* main0=root.owned_position_table(0);const auto* main1=last.owned_position_table(0);
            require(main0!=main1&&main0[3*127]==127&&main1[3*191+2]==191,"actual identity upload/stage isolation/lookahead");
            require(root.prepare_owned_batch_slots(ledger,bindings,32768,error),"real graph setup follows immutable bank install");
            require(root.owned_position_binding_matches()&&last.owned_position_binding_matches(),"actual stage/inventory bank match");
            root.batch_rec_=true;root.brow_[0]=0;root.brow_[1]=1;
            const auto* a=root.owned_position_table(0);const auto* b=root.owned_position_table(1);
            require(a!=b&&a!=main0&&b!=main0&&a[3*7+1]==7&&b[3*11+2]==11,"actual per-token ordered slot bank selection");
            root.batch_rec_=false;
            ++root.slot_inventory_generation_;require(!root.owned_position_binding_matches(),"stale actual inventory refuses");--root.slot_inventory_generation_;
            std::array<core::NativeBatchSlotInspection,4> observations{};
            for(size_t stage=0;stage<2;++stage)for(size_t slot=0;slot<2;++slot){auto& v=stage?last:root;
                require(v.inspect_owned_batch_slot(bindings,tokens[slot],stage,observations[2*stage+slot]),"bank actual native slot observation");}
            auto actor=[&](size_t slot){return ServingRawSession::open_owned_slot(*geometry,
                *slots.allocation(slots.handle(slot)),runtime,{&root,&last},{observations[slot],observations[2+slot]},
                ledger,128,*session,error);};
            auto actor0=actor(0),actor1=actor(1);require(actor0&&actor1,"bank genuine slot request owners");
            require(actor0->open_request(error)&&actor1->open_request(error)&&
                actor0->queue_owned_slot_input(17,error)&&actor1->queue_owned_slot_input(23,error)&&
                actor0->begin_work(error)&&actor1->begin_work(error),"bank real pending request/occupancy latches");
            const int reverse[]={1,0},duplicate[]={0,0},ordered[]={0,1},accepted[]={1,1};
            const int32_t input[]={17,23};const int64_t pos[]={0,0};int32_t output[2]{};
            auto dispatch=[&](const int* rows,std::shared_ptr<const core::NativeBatchGroupCompletion>& result){
                auto credit=ledger.reserve({{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
                require(bool(credit),"bank group real metadata lease");
                return root.run_owned_slot_group(observations.data(),2,1,rows,2,input,pos,accepted,nullptr,nullptr,output,std::move(credit.lease),result,error);};
            std::shared_ptr<const core::NativeBatchGroupCompletion> receipt;
            const auto submissions=seam::events.size();
            require(!dispatch(reverse,receipt)&&!dispatch(duplicate,receipt)&&!receipt&&seam::events.size()==submissions,
                "actual banked group refuses reverse/duplicate ordering before submission");
            require(dispatch(ordered,receipt)&&receipt,"actual banked group accepts both genuine current requests");
            auto* emitted=std::tmpfile();require(emitted,"bank recording output");
            require(actor0->drain_owned_slot_group(*receipt,emitted,error)&&actor1->drain_owned_slot_group(*receipt,emitted,error),
                "actual banked receipt is consumed under both real RAW request fences");
            require(actor0->close_request(error)&&actor1->close_request(error)&&actor0->retire_owned_slot(error)&&actor1->retire_owned_slot(error),
                "actual banked occupancies retire after checked consumption");std::fclose(emitted);
            const auto paid=ledger.snapshot().reserved;
            require(!session->prepare_owned_position_banks(bindings,ps,512,setup,error)&&ledger.snapshot().reserved==paid,"published install cannot be replaced or charged twice");
            tail=root.position_banks_;retained=ledger.snapshot().reserved.host_bytes;
            }
            root.exec_bm_.clear();root.commit_bm_.clear();last.exec_bm_.clear();last.commit_bm_.clear();
            require(slots.retire_after_consumers_joined({},error),"recording slots retire after named graph boundary");
        }
        require(ledger.snapshot().reserved.host_bytes>0&&bank_recording::frees==(native_refusal?2:0),"outer setup retains payment before synchronous enclosing retirement");
    }
    require(bank_recording::enclosing_frees==4,"actual outer keeper paid through all Verifier and Prefill enclosing operator deletes");
    if(native_refusal){
        require(bank_recording::frees==2&&ledger.snapshot().reserved.host_bytes==0,"failed setup releases paid envelope only after enclosing frees");return;
    }
    require(tail.expired()&&bank_recording::frees==6&&ledger.snapshot().reserved.host_bytes>0&&retained>0,"weak binding-control tail retains real credit AFTER actual native frees");
    tail.reset();require(ledger.snapshot().reserved.host_bytes==0,"final actual typed weak-control deallocation releases roots and claim");
}
void actual_position_bank_startup(){actual_position_bank_startup_case(false);}
void actual_position_bank_refusal(){actual_position_bank_startup_case(true);}

// New image groups use real startup, private issuer/capture/CAS/reset/transfer,
// ledger and immutable parser; native compute alone writes recording byte DATA.
// No sampled-model output, GPU DMA, production ABI or memory-fit conclusion.
void actual_image_private_pair(){
    seam::image_mode=true;
    auto cap=capacity(64ULL<<20);cap.host_bytes=cap.pinned_bytes=512ULL<<20;
    core::WindowBudget ledger(cap,{{0,0},0,0,0,0,0});bank_recording::ledger=&ledger;
    std::array<std::weak_ptr<const void>,2> workspace_tails;
    std::weak_ptr<const void> operation_tail;
    std::weak_ptr<program::OwnedImagePeerObservation> observation_tail;
    {
        OwnedPositionBankRetirement retirement;
        {
            std::string error;auto geometry=std::make_shared<core::ModelGeometry>(recording_geometry());geometry->n_vocab=248320;geometry->head_dim=256;geometry->n_head=24;
            OwnedBatchSlotSessions slots;require(slots.initialize(*geometry,128,10,requests(),ledger,error),"image genuine two-slot allocation");
            std::array<const core::NativeSessionAllocation*,2> allocations{};std::array<core::NativeBatchSlotToken,2> tokens{};
            for(size_t i=0;i<2;++i){auto handle=slots.handle(i);allocations[i]=slots.allocation(handle);tokens[i]={handle.incarnation,handle.generation,handle.slot};}
            auto metadata=ledger.reserve({{0,0},core::NativeBatchSlotBindings::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
            auto bindings=core::NativeBatchSlotBindings::create(allocations,tokens,geometry,std::move(metadata.lease),error);
            require(bool(bindings),"image canonical physical binding owner");
            core::NativeSessionAllocation main;
            require(bool(core::allocate_native_sessions(*geometry,128,10,requests(),ledger,main)),"image genuine main allocation");
            auto runtime=std::make_shared<DriverWindowRuntime>();retirement.runtime=runtime;
            runtime->primary=std::make_unique<core::Verifier>();runtime->secondary=std::make_unique<core::Verifier>();
            auto& root=*runtime->primary;auto& last=*runtime->secondary;
            root.g_=last.g_=geometry.get();root.ss_=main.stages()[0].session;last.ss_=main.stages()[1].session;
            root.le_=2;root.next_=&last;last.lb_=2;last.le_=4;last.device_=1;root.strict_retirement_=last.strict_retirement_=true;
            std::shared_ptr<ServingRawSession> source;
            require(ServingRawSession::open_owned_startup(*geometry,main,runtime,{&root,&last},ledger,128,128ULL<<20,source,error),"image startup actual reset/common empty cut");
            runtime->first_prefill=std::make_unique<prefill::Prefill>();runtime->second_prefill=std::make_unique<prefill::Prefill>();
            prefill::Prefill* parts[]={runtime->first_prefill.get(),runtime->second_prefill.get()};
            for(size_t i=0;i<2;++i){auto& p=*parts[i];cudaSetDevice(int(i));
                auto admit=[&](const prefill::OwnedWorkspaceBytes& q,std::shared_ptr<const void>& out){
                    core::WindowResources bytes{{0,0},q.host,q.pinned,0,0,0};bytes.device_bytes[i]=q.device;
                    auto result=ledger.reserve(bytes,core::ReservationPurpose::foreground);
                    if(!result)return prefill::WorkspaceAdmission::budget_refused;
                    out=std::move(result.lease);return prefill::WorkspaceAdmission::ready;
                };
                require(p.admit_owned_workspace(*geometry,*main.stages()[i].session,nullptr,4,true,admit,error)==prefill::WorkspaceAdmission::ready,"image actual workspace quote/admission");
                workspace_tails[i]=p.workspace_credit_;
                // Only arena/model-kernel construction is a representation.
                // Actual ring init uses the actual admitted stage/shape/device.
                p.impl_->g=geometry.get();p.impl_->ss=main.stages()[i].session;
                require(prefill::prepare_paid_uploads(p.impl_->up_ring,4,uint64_t(geometry->n_expert),reinterpret_cast<void*>(uintptr_t(p.impl_->cs)),error),"image actual four-slot/event paid preparation");
            }
            cudaSetDevice(0);
            const uint64_t fixed=OwnedImageRequestDriver::reactor_quote()+OwnedImageRequestDriver::choice_quote()+1024;
            uint64_t cells=0,bank=0;require(strata::kernels::owned_mrope_bank_layout(128,cells,bank),"image actual bank layout");bank_recording::required_bank_device=3*bank;
            require(source->prepare_owned_position_banks(bindings,parts,fixed,retirement.setup,error),"image real bank/directory/root preparation");
            require(root.prepare_owned_batch_slots(ledger,bindings,32768,error),"image real graph/setup/inspection producer");
            require(source->tx_->host_cap_->claim_limit==16,"image retains original MAIN sixteen-claim lane before actual sharing");
            std::array<core::NativeBatchSlotInspection,4> inspections;
            for(size_t stage=0;stage<2;++stage){auto* verifier=stage?&last:&root;
                for(size_t slot=0;slot<2;++slot)require(verifier->inspect_owned_batch_slot(bindings,tokens[slot],stage,
                    inspections[stage*2+slot]),"image actual verifier observation");}
            auto actor=[&](size_t slot){std::vector<core::NativeBatchSlotInspection> stages;
                for(size_t i=0;i<2;++i){core::NativeBatchSlotInspection observation;auto* v=i?&last:&root;
                    require(v->inspect_owned_batch_slot(bindings,tokens[slot],i,observation),"image real per-stage slot observation");stages.push_back(std::move(observation));}
                return ServingRawSession::open_owned_slot(*geometry,*allocations[slot],runtime,{&root,&last},std::move(stages),ledger,128,*source,error);};
            auto destination=actor(0),foreign=actor(1);require(destination&&foreign,"image genuine vacant request actors");
            require(source->tx_->host_cap_->claim_limit==48&&destination->tx_->host_cap_==source->tx_->host_cap_,"image actual two-slot sharing retains original forty-eight-claim family");
            auto input=OwnedImageRequestDriver::reactor(*source);input->max_queue_lines=4;
            require(OwnedImageRequestDriver::bind(*source,input,nullptr,error),"image real controller binding");
            constexpr uint64_t request=1;
            require(input->admissions.prepare_image(0,request),"image same canonical preparation ticket");
            const auto incarnation=input->admissions.incarnation(0,request);
            OwnedImageRequestDriver::Preparation prep;prep.slot=0;prep.request=request;prep.output=4;prep.tokens=2;prep.bytes=20+2560*4;prep.deadline=UINT64_MAX;
            const auto before=source->tx_->host_cap_->live.load();
            const auto original_claims=source->tx_->host_cap_->claims.load();const auto original_cut=source->cut();
            std::shared_ptr<SuffixRawTransaction::HostCredit> h_block;
            require(bool(source->tx_->admit_host(source->tx_->host_cap_->limit-before-1,true,h_block)),"image genuine aggregate H blocker");
            const auto blocked_h=source->tx_->host_cap_->live.load();const auto blocked_claims=source->tx_->host_cap_->claims.load();
            require(!OwnedImageRequestDriver::prepare(*source,*destination,input,nullptr,prep,error)&&
                source->tx_->host_cap_->live.load()==blocked_h&&source->tx_->host_cap_->claims.load()==blocked_claims&&
                source->cut()==original_cut&&!source->request_open(),"image aggregate refusal preserves live bytes/claims/cut/request state");h_block.reset();
            require(source->tx_->host_cap_->live.load()==before&&source->tx_->host_cap_->claims.load()==original_claims,"actual H blocker returns exactly to entry");
            // A genuine global exhaustion refuses before any record/payload.
            auto block=ledger.reserve({{0,0},cap.host_bytes-ledger.snapshot().reserved.host_bytes,0,0,0,0},core::ReservationPurpose::foreground);
            require(bool(block),"image bounded global host blocker");const auto blocked=ledger.snapshot().reserved;
            require(!OwnedImageRequestDriver::prepare(*source,*destination,input,nullptr,prep,error)&&
                ledger.snapshot().reserved==blocked&&source->tx_->host_cap_->live.load()==before&&source->tx_->host_cap_->claims.load()==original_claims&&source->cut()==original_cut&&!source->request_open(),"image second global admission rollback preserves entry");
            block.lease.reset();
            require(OwnedImageRequestDriver::prepare(*source,*destination,input,nullptr,prep,error),"image genuine positive transport payment/issuer");
            const auto paid=source->tx_->host_cap_->live.load();
            require(paid>before&&!OwnedImageRequestDriver::prepare(*source,*destination,input,nullptr,prep,error)&&source->tx_->host_cap_->live.load()==paid,"duplicate prepare neither reissues nor recharges");
            // Real sealed producer transport, allocated only AFTER permission.
            const int fd=::memfd_create("owned-image-recording",MFD_CLOEXEC|MFD_ALLOW_SEALING);require(fd>=0,"image paid real memfd");
            const int32_t header[]={0x31455653,1,1,1,2560};std::array<float,2560> row;row.fill(0.25f);
            require(::write(fd,header,sizeof(header))==ssize_t(sizeof(header))&&::write(fd,row.data(),sizeof(row))==ssize_t(sizeof(row)),"image real immutable SVE1 bytes");
            require(::fcntl(fd,F_ADD_SEALS,F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL)==0,"image real input seal");
            const std::vector<int64_t> text{248056,17};const auto path="/proc/self/fd/"+std::to_string(fd);
            require(input->admissions.deliver_image(0,request)&&!input->admissions.deliver_image(0,request),"image final arrival consumes same preparation ticket only once");
            require(OwnedImageRequestDriver::capture(*source,0,request,input,"BGENI 0 4 http_request=000000000000 "+path+" 248056,17",path,text,error),"image actual private sealed capture");
            require(::close(fd)==0,"image producer actual last descriptor closes before confirmation");
            require(OwnedImageRequestDriver::producer_close(*source,0,request,incarnation+1,false,input,error)==-1,"image foreign close cannot settle real record");
            require(OwnedImageRequestDriver::producer_close(*source,0,request,incarnation,false,input,error)==1&&OwnedImageRequestDriver::ready(*source,0,request),"image canonical close and same-ticket ready reentry");
            bool cancelled=false;require(input->admissions.begin(0,request,cancelled)&&!cancelled,"image real final admission begins once");
            require(OwnedImageRequestDriver::select(*source,0,request,input,error)&&OwnedImageRequestDriver::prefill(*source,parts,error),"image actual private MAIN reset/bank/view publication");
            operation_tail=source->image_operation();
            // Cancellation uses the real entry incarnation and actual image
            // reset/join path; parent continues the independent positive route.
            {const auto child=::fork();require(child>=0,"image actual cancel fork");
                if(!child){bool active=false;
                    require(input->admissions.cancel_image(0,request,incarnation,active)&&active,"actual image active cancel ticket");
                    require(OwnedImageRequestDriver::cancel(*source,error)&&!source->request_open()&&
                        source->phase()==ServingRawSession::Phase::drained&&!source->tx_->image_exact()&&
                        !parts[0]->image_operation_&&!parts[1]->image_operation_&&!seam::pending(),"actual canceled image joins/reset/closed fence before retirement");
                    std::_Exit(0);}
                int status=0;require(::waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"actual image cancellation boundary");}
            const auto accepted_before=source->cut();
            require(!source->tx_->identity_matches(source->key()),"image generic text identity refuses");
            const float* const* rows=nullptr;
            for(size_t i=0;i<2;++i){auto& p=*parts[i];cudaSetDevice(int(i));
                require(p.owned_image_rows_present(0,2)&&p.recorded_rows(text.data(),2,0,rows,error)&&rows&&rows[0]&&!rows[1],"actual pre-kernel private row/token checks positive");
                std::array<int64_t,2> changed{248056,18};require(!p.recorded_rows(changed.data(),2,0,rows,error),"actual private immutable token mismatch refuses");
                require(p.recorded_rows(text.data(),2,0,rows,error),"actual private unchanged row source remains usable");
                std::array<float,2560> uploaded{};
                require(p.impl_->up_ring.h2d(uploaded.data(),rows[0],sizeof(uploaded),reinterpret_cast<void*>(uintptr_t(p.impl_->cs)))==cudaSuccess&&
                    p.impl_->up_ring.drain_owned()&&uploaded==row,"real frozen paid ring transfers immutable image row at recording boundary");
            }
            cudaSetDevice(0);
            require(source->begin_work(error),"image actual native work fence");
            for(const auto& stage:main.stages()){
                auto& native=*stage.session;
                std::fill(native.recorded_gdn.begin(),native.recorded_gdn.end(),std::byte{0x35});
                std::fill(native.recorded_ple.begin(),native.recorded_ple.end(),std::byte{0x61});
                native.ple_prev[0]=248056;native.ple_prev[1]=17;native.ple_token=19;
                for(int64_t i=0;i<native.qsa_alloc;++i)for(auto& bytes:native.qsa_states[native.qsa_ord0+i].backing)std::fill(bytes.begin(),bytes.end(),std::byte{0x27});
            }
            // Recording kernel outputs above complete before the REAL common
            // cut validator. This is not a cut inferred from token identity.
            core::StateCut completed=accepted_before;completed.id++;completed.accepted=3;completed.forwarded=2;completed.exact_frontier=2;completed.output_watermark=0;
            const std::vector<int32_t> history{248056,17,19};completed.history=core::window_history_digest(history);
            std::array<core::StateCut,2> cuts{completed,completed};
            require(bool(source->tx_->complete_driver_work(true,completed,cuts)),"image real checked recording all-stage completion");
            source->lifecycle_.set_observed_phase(source->phase());source->lifecycle_.apply(core::serving_raw::Lifecycle::Event::drain_work);
            auto endpoint=ServingRawSession::endpoint_metadata(source->key(),source->cut(),history);
            core::BundleHandle generic;
            require(!source->tx_->recorded_capture_guard(endpoint,generic)&&!source->tx_->recorded_capture_native_reached&&!generic,"actual generic capture early guard refuses before native export");
            require(!OwnedImageRequestDriver::transfer(*source,*foreign,endpoint,error)&&!foreign->request_open()&&!foreign->cut().accepted,"private foreign destination refuses before issue/mutation");
            for(int fault=0;fault<2;++fault){const auto child=::fork();require(child>=0,"image uncertainty fork");
                if(!child){std::set_terminate([]{std::_Exit(73);});
                    if(fault==0)seam::enqueue_fail=seam::enqueue_calls+1;else seam::fail_queued_drain=true;
                    (void)OwnedImageRequestDriver::transfer(*source,*destination,endpoint,error);std::_Exit(74);}
                int status=0;require(::waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==73,"actual private submitted error/drain ambiguity failstops without unwind");
            }
            require(OwnedImageRequestDriver::transfer(*source,*destination,endpoint,error)&&!seam::pending(),"actual single-use same-record native pair transfer and checked all-stage drain");
            require(destination->cut().accepted==3&&destination->cut().forwarded==2&&destination->slot_tokens_==history,"actual destination accepted/pending/common cut installed");
            require(!OwnedImageRequestDriver::transfer(*source,*destination,endpoint,error),"actual source completion cannot replay");
            const auto image_frontier=destination->cut();
            require(!destination->tx_->recorded_capture_guard(endpoint,generic)&&!generic,"live image generic capture still refuses");
            std::shared_ptr<OwnedImagePeerObservation> witness;
            require(OwnedImageRequestDriver::observe_peer(*destination,witness,error)&&witness,"actual private observation pays both full peaks before native reads");
            observation_tail=witness;uint64_t tensors=0,bytes=0;
            require(!OwnedImageRequestDriver::compare_peer(*foreign,witness,tensors,bytes,error),"observation refuses foreign live actor/context");
            require(OwnedImageRequestDriver::compare_peer(*destination,witness,tensors,bytes,error)&&bytes>0&&tensors>0,"actual byte-exact image peer comparator positive");
            require(!OwnedImageRequestDriver::compare_peer(*destination,witness,tensors,bytes,error),"private peer observation is single use");witness.reset();
            std::shared_ptr<OwnedImagePeerObservation> changed;
            require(OwnedImageRequestDriver::observe_peer(*destination,changed,error),"new private before witness");
            auto& q=allocations[0]->stages()[0].session->qsa_states[0];q.backing[0][0]^=std::byte{1};
            require(!OwnedImageRequestDriver::compare_peer(*destination,changed,tensors,bytes,error)&&destination->cut()==image_frontier,"byte-exact comparator detects changed live native data without rewriting cut");
            q.backing[0][0]^=std::byte{1};changed.reset();
            std::shared_ptr<OwnedImagePeerObservation> isolation;
            require(OwnedImageRequestDriver::observe_peer(*destination,isolation,error),"image peer before real MAIN reset");
            require(OwnedImageRequestDriver::finish_source(*source,error)&&!source->request_open()&&!source->tx_->image_exact(),"actual source joins/reset/identity bank/closed fence before context retirement");
            require(OwnedImageRequestDriver::compare_peer(*destination,isolation,tensors,bytes,error),"actual MAIN reset and own-bank retirement preserve every image-peer native/metadata/bank byte");isolation.reset();
            require(foreign->open_request(error)&&foreign->queue_owned_slot_input(17,error)&&
                destination->begin_work(error)&&foreign->begin_work(error),"genuine image/text live owned slot work fences");
            const int rows2[]={0,1},accepted2[]={1,1};const int32_t input2[]={19,17};const int64_t positions2[]={2,0};int32_t outputs2[2]{};
            auto receipt_payment=ledger.reserve({{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
            require(bool(receipt_payment),"live image/text real group metadata payment");
            std::shared_ptr<const core::NativeBatchGroupCompletion> completion;
            require(root.run_owned_slot_group(inspections.data(),2,1,rows2,2,input2,positions2,accepted2,nullptr,nullptr,
                outputs2,std::move(receipt_payment.lease),completion,error)&&completion,"actual ordered image/text EXECUTE plus COMMIT context/epoch latch");
            auto* output=std::tmpfile();require(output,"image live recording output");
            require(destination->drain_owned_slot_group(*completion,output,error)&&foreign->drain_owned_slot_group(*completion,output,error),
                "actual image and text live receipt consumption advances both real RAW cuts");std::fclose(output);completion.reset();
            require(destination->cut().forwarded==3&&destination->slot_tokens_.size()==4&&destination->slot_tokens_.back()==20,
                "occupied image slot continues through actual owned control route with recording sample");

            require(!operation_tail.expired(),"occupied image slot still retains exact operation/data owner");
            require(destination->close_request(error)&&destination->retire_owned_slot(error),"actual image slot reset/closed/occupancy retirement");
            require(foreign->close_request(error)&&foreign->retire_owned_slot(error),"genuine text peer closed and retired");
            input->admissions.finish(0,request);input->lines.clear();
            require(operation_tail.expired()&&source->tx_->host_cap_->live.load()>before,"actual operation payload died while its weak typed-control tail still holds payment");
            const auto tail_paid=source->tx_->host_cap_->live.load();
            std::thread final_free([operation=std::move(operation_tail),observation=std::move(observation_tail)]()mutable{
                operation.reset();observation.reset();
            });final_free.join();
            require(source->tx_->host_cap_->live.load()<tail_paid,"actual typed operation/observation control deallocation releases payment off controller thread");witness.reset();
            // Required nonidentity grids and independent request generations.
            // This extends the SAME actual private group; the original 1x1,
            // mixed TEXT/IMAGE, failures and last-free assertions above remain.
            {
                using Triples=std::array<std::array<int32_t,3>,8>;
                const Triples grid_a{{{0,0,0},{1,1,1},{1,1,2},{1,1,3},
                    {1,2,1},{1,2,2},{1,2,3},{4,4,4}}}; // base1, 2 rows x 3 columns
                const Triples grid_b{{{0,0,0},{1,1,1},{1,1,2},{1,2,1},
                    {1,2,2},{1,3,1},{1,3,2},{4,4,4}}}; // base1, 3 rows x 2 columns
                require(grid_a!=grid_b&&grid_a[3]!=std::array<int32_t,3>{3,3,3}&&
                    grid_b[3]!=std::array<int32_t,3>{3,3,3},"explicit nonidentity expected grids are distinct");
                std::array<std::array<const int32_t*,3>,2> addresses{};
                const std::array bindings_before{root.position_banks_,last.position_banks_};
                const auto allocation_count=bank_recording::allocations;
                const auto setup_epoch=root.owned_position_replay_.epoch;
                require(allocation_count==6&&setup_epoch,"genuine original bank allocations and observed setup epoch");
                for(size_t stage=0;stage<2;++stage)for(size_t bank_index=0;bank_index<3;++bank_index){
                    const size_t at=3*stage+bank_index;
                    require(bank_recording::devices[at]==int(stage)&&bank_recording::pointers[at],"actual allocated stage/bank address");
                    addresses[stage][bank_index]=static_cast<const int32_t*>(bank_recording::pointers[at]);
                }
                auto stable_banks=[&](){
                    require(bank_recording::allocations==allocation_count&&root.position_banks_==bindings_before[0]&&
                        last.position_banks_==bindings_before[1]&&root.owned_position_binding_matches()&&
                        last.owned_position_binding_matches(),"same genuine bank allocations/bindings/inventory/setup epoch");
                    for(size_t stage=0;stage<2;++stage)for(size_t bank_index=0;bank_index<3;++bank_index)
                        require(bank_recording::pointers[3*stage+bank_index]==addresses[stage][bank_index],"no bank address replacement");
                };
                auto check_grid=[&](size_t bank_index,const Triples& expected){
                    stable_banks();
                    for(size_t stage=0;stage<2;++stage){const auto* actual=addresses[stage][bank_index];
                        for(size_t cell=0;cell<expected.size();++cell)for(size_t axis=0;axis<3;++axis)
                            require(actual[3*cell+axis]==expected[cell][axis],"actual all-stage nonidentity grid triples");
                        for(size_t cell:{size_t(8),size_t(127),size_t(191)})for(size_t axis=0;axis<3;++axis)
                            require(actual[3*cell+axis]==int32_t(cell-3),"actual image grid lookahead continues after max grid extent");
                    }
                };
                std::array<uint64_t,2> main_generations{};
                std::array<std::shared_ptr<OwnedImageRequestOperation>,2> contexts;
                const std::vector<int64_t> grid_tokens{11,248056,248056,248056,248056,248056,248056,17};
                auto install_grid=[&](size_t slot,uint64_t key,int32_t ny,int32_t nx,const Triples& expected){
                    auto target=actor(slot);require(bool(target),"nonidentity grid genuine fresh vacant actor");
                    require(input->admissions.prepare_image(int(slot),key),"nonidentity grid new real admission incarnation");
                    const auto serial=input->admissions.incarnation(int(slot),key);
                    OwnedImageRequestDriver::Preparation prepared;prepared.slot=int(slot);prepared.request=key;
                    prepared.output=4;prepared.tokens=grid_tokens.size();prepared.bytes=20+6*2560*4;prepared.deadline=UINT64_MAX;
                    require(OwnedImageRequestDriver::prepare(*source,*target,input,nullptr,prepared,error),"nonidentity grid genuine private prepare/payment");
                    const int transport=::memfd_create("owned-grid-recording",MFD_CLOEXEC|MFD_ALLOW_SEALING);
                    require(transport>=0,"nonidentity grid paid sealed transport");
                    const int32_t sve[]={0x31455653,6,nx,ny,2560};std::array<float,6*2560> image;
                    image.fill(slot==0?0.5f:1.5f);
                    require(::write(transport,sve,sizeof(sve))==ssize_t(sizeof(sve))&&
                        ::write(transport,image.data(),sizeof(image))==ssize_t(sizeof(image))&&
                        ::fcntl(transport,F_ADD_SEALS,F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL)==0,
                        "actual distinct nontrivial SVE1 data and immutable seals");
                    const auto descriptor="/proc/self/fd/"+std::to_string(transport);
                    char command[512];std::snprintf(command,sizeof(command),
                        "BGENI %zu 4 http_request=%012llx %s 11,248056,248056,248056,248056,248056,248056,17",
                        slot,(unsigned long long)(key-1),descriptor.c_str());
                    require(input->admissions.deliver_image(int(slot),key)&&
                        OwnedImageRequestDriver::capture(*source,slot,key,input,command,descriptor,grid_tokens,error),"actual nonidentity immutable private capture");
                    require(::close(transport)==0&&OwnedImageRequestDriver::producer_close(*source,slot,key,serial,false,input,error)==1,
                        "nonidentity producer actual close and canonical confirmation");
                    bool canceled=false;require(input->admissions.begin(int(slot),key,canceled)&&!canceled&&
                        OwnedImageRequestDriver::select(*source,slot,key,input,error)&&
                        OwnedImageRequestDriver::prefill(*source,parts,error),"actual independently issued MAIN image operation/bank install");
                    contexts[slot]=source->image_operation();main_generations[slot]=parts[0]->image_generation_;
                    require(contexts[slot]&&parts[1]->image_operation_==contexts[slot]&&
                        parts[1]->image_generation_==main_generations[slot]&&main_generations[slot],"actual all-stage issued MAIN context/content generation");
                    check_grid(0,expected);
                    for(size_t stage=0;stage<2;++stage){cudaSetDevice(int(stage));const float* const* current=nullptr;
                        require(parts[stage]->recorded_rows(grid_tokens.data(),int64_t(grid_tokens.size()),0,current,error)&&current&&
                            !current[0]&&!current[7],"actual current private nonidentity row/token guard");
                        for(size_t cell=1;cell<=6;++cell)require(current[cell]&&current[cell][0]==image[0],"actual sealed rows for every grid cell");
                        std::array<float,2560> uploaded{};
                        require(parts[stage]->impl_->up_ring.h2d(uploaded.data(),current[3],sizeof(uploaded),
                            reinterpret_cast<void*>(uintptr_t(parts[stage]->impl_->cs)))==cudaSuccess&&
                            parts[stage]->impl_->up_ring.drain_owned()&&uploaded[0]==image[0],"nonidentity current row uses actual paid ring");
                    }
                    cudaSetDevice(0);require(source->begin_work(error),"nonidentity actual MAIN native work fence");
                    for(const auto& stage:main.stages()){auto& native=*stage.session;
                        const auto value=std::byte(slot==0?0x4a:0x5b);
                        std::fill(native.recorded_gdn.begin(),native.recorded_gdn.end(),value);
                        std::fill(native.recorded_ple.begin(),native.recorded_ple.end(),value);
                        native.ple_prev[0]=248056;native.ple_prev[1]=17;native.ple_token=19;
                        for(int64_t i=0;i<native.qsa_alloc;++i)for(auto& bytes:native.qsa_states[native.qsa_ord0+i].backing)
                            std::fill(bytes.begin(),bytes.end(),value);
                    }
                    // Same declared recording-compute boundary as the original
                    // group, followed by the REAL per-stage cut validator.
                    auto end=source->cut();++end.id;end.forwarded=8;end.accepted=9;end.exact_frontier=8;end.output_watermark=0;
                    std::vector<int32_t> accepted(grid_tokens.begin(),grid_tokens.end());accepted.push_back(19);
                    end.history=core::window_history_digest(accepted);std::array<core::StateCut,2> stages{end,end};
                    require(bool(source->tx_->complete_driver_work(true,end,stages)),"nonidentity real recording all-stage completion");
                    source->lifecycle_.set_observed_phase(source->phase());source->lifecycle_.apply(core::serving_raw::Lifecycle::Event::drain_work);
                    const auto metadata=ServingRawSession::endpoint_metadata(source->key(),source->cut(),accepted);
                    require(OwnedImageRequestDriver::transfer(*source,*target,metadata,error),"actual independently issued nonidentity paired transfer");
                    require(target->image_operation()==contexts[slot]&&target->request_open()&&target->cut().forwarded==8,
                        "actual target retains its genuine context and completed native cut");
                    check_grid(slot+1,expected);
                    require(OwnedImageRequestDriver::finish_source(*source,error),"nonidentity actual MAIN join/reset/closed source");
                    input->admissions.finish(int(slot),key);input->lines.clear();
                    check_grid(slot+1,expected);
                    return target;
                };
                auto a=install_grid(0,2,2,3,grid_a);
                std::shared_ptr<OwnedImagePeerObservation> before_other;
                require(OwnedImageRequestDriver::observe_peer(*a,before_other,error),"nonidentity image A before real independent MAIN/slot B installation");
                const auto a_context=a->image_operation();const auto a_occupancy=a->slot_occupancy_incarnation();
                auto b=install_grid(1,3,3,2,grid_b);
                require(main_generations[1]>main_generations[0]&&contexts[0]!=contexts[1]&&
                    a->image_operation()==a_context&&a->slot_occupancy_incarnation()==a_occupancy,
                    "real later MAIN generation and distinct image operation leave live peer context/occupancy unchanged");
                require(OwnedImageRequestDriver::compare_peer(*a,before_other,tensors,bytes,error),"full actual nonidentity A byte/context/generation isolation across B install/reset");before_other.reset();
                check_grid(1,grid_a);check_grid(2,grid_b);
                auto run_current=[&](uint64_t operation,int32_t pending,int64_t position){
                    require(a->begin_work(error)&&b->begin_work(error),"two genuine nonidentity image slot work fences");
                    const int ordered[]={0,1},keep[]={1,1};const int32_t tokens_now[]={pending,pending};const int64_t pos_now[]={position,position};int32_t out[2]{};
                    auto payment=ledger.reserve({{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
                    require(bool(payment),"two current image group real metadata payment");
                    std::shared_ptr<const core::NativeBatchGroupCompletion> receipt;
                    require(root.run_owned_slot_group(inspections.data(),2,operation,ordered,2,tokens_now,pos_now,keep,nullptr,nullptr,
                        out,std::move(payment.lease),receipt,error)&&receipt,"actual current nonidentity image EXECUTE/COMMIT positive");
                    require(root.owned_position_replay_.epoch==setup_epoch&&last.owned_position_replay_.epoch==setup_epoch&&
                        root.owned_position_replay_.image_contexts[0]==contexts[0].get()&&root.owned_position_replay_.image_contexts[1]==contexts[1].get()&&
                        last.owned_position_replay_.content_generations==root.owned_position_replay_.content_generations,
                        "actual issued replay observes both real contexts and same per-bank generations under unchanged setup epoch");
                    auto* emitted=std::tmpfile();require(emitted,"nonidentity group recording output");
                    require(a->drain_owned_slot_group(*receipt,emitted,error)&&b->drain_owned_slot_group(*receipt,emitted,error),"actual current nonidentity image receipt consumption");std::fclose(emitted);
                    check_grid(1,grid_a);check_grid(2,grid_b);
                };
                run_current(2,19,8);
                const auto old_generation=root.owned_position_replay_.content_generations;
                std::shared_ptr<OwnedImagePeerObservation> stale,current_b;
                require(OwnedImageRequestDriver::observe_peer(*a,stale,error)&&OwnedImageRequestDriver::observe_peer(*b,current_b,error),
                    "retain genuine issued A old-generation witness and B isolation witness");
                const auto old_cut=a->cut();const auto old_context=a->image_operation();
                // Legal own-bank reinstall: same real consumed operation,
                // actual healthy drained request, joined groups and SAME bank.
                // ONLY the production installer advances content generation;
                // the test never assigns a generation, context or issuer field.
                require(a->phase()==ServingRawSession::Phase::drained&&a->request_open()&&
                    !root.owned_group_busy_&&!last.owned_group_busy_&&
                    a->image_install_bank(1,old_context,error),"actual legal drained same-context own-bank reinstall");
                check_grid(1,grid_a);check_grid(2,grid_b);
                require(a->image_operation()==old_context&&a->slot_occupancy_incarnation()==a_occupancy&&a->cut()==old_cut,
                    "reinstall changes content generation alone, not context/occupancy/cut");
                const auto streams_before=seam::stream_calls;const auto events_before=seam::events.size();
                const auto enqueues_before=seam::enqueue_calls;const auto allocations_before=bank_recording::allocations;
                error.clear();
                require(!OwnedImageRequestDriver::compare_peer(*a,stale,tensors,bytes,error)&&
                    error=="foreign or duplicate image observation"&&tensors==0&&bytes==0&&
                    seam::stream_calls==streams_before&&seam::events.size()==events_before&&seam::enqueue_calls==enqueues_before&&
                    bank_recording::allocations==allocations_before&&a->cut()==old_cut,
                    "actual stale-generation witness refuses BEFORE native read/dispatch or mutation with same context/occupancy");stale.reset();
                require(OwnedImageRequestDriver::compare_peer(*b,current_b,tensors,bytes,error),"other nonidentity image context/generation/native bytes survive legal A reinstall");current_b.reset();
                std::shared_ptr<OwnedImagePeerObservation> current_a;
                require(OwnedImageRequestDriver::observe_peer(*a,current_a,error)&&
                    OwnedImageRequestDriver::compare_peer(*a,current_a,tensors,bytes,error),"fresh current-generation witness succeeds after stale refusal");current_a.reset();
                run_current(3,20,9);
                require(root.owned_position_replay_.content_generations[0]==old_generation[0]+1&&
                    root.owned_position_replay_.content_generations[1]==old_generation[1],
                    "real installer advances only A content generation; positive current replay observes it");
                stable_banks();
                require(a->close_request(error)&&a->retire_owned_slot(error)&&b->close_request(error)&&b->retire_owned_slot(error),
                    "both genuine nonidentity image requests reset/close/retire before scope owners drop");
            }
            root.exec_bm_.clear();root.commit_bm_.clear();last.exec_bm_.clear();last.commit_bm_.clear();
            require(slots.retire_after_consumers_joined({},error),"image all physical slots checked retirement");
        }
    }
    require(ledger.snapshot().reserved.host_bytes==0&&ledger.snapshot().reserved.pinned_bytes==0&&!seam::pending(),"actual private roots and all typed weak/control/arena/consumer tails return to baseline");
    for(auto& weak:workspace_tails)weak.reset();
}
int main(){
    for(auto test:{distinct_and_reuse,second_gang_refusal,partial_stage_refusal,actual_owned_verifier_inspection,actual_group_success,actual_setup_upload_refusal,actual_slot_raw,actual_position_bank_startup,actual_position_bank_refusal,actual_image_private_pair}){
        auto pid=fork();require(pid>=0,"fork fixture");if(!pid){test();std::_Exit(0);}
        int status=0;require(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"actual ownership fixture child passes");
    }
    std::puts("PASS actual owned two-slot wrapper, production fresh-session factory/retirement/ledger; recording allocation+CUDA boundary only");
}
