// Actual two-slot production owner + native retirement methods + real budget.
// Allocation/CUDA boundaries record host lifetime/control only, never native fit/drain proof.
#include "owned_batch_slot_sessions.hpp"
#include "strata/core/native_batch_slot_binding.hpp"
#include "strata/core/native_batch_group_completion.hpp"
#include "strata/core/native_batch_setup_admission.hpp"
#include <map>
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

namespace strata::core {
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
        stage.session->max_cells=cells;stage.session->k=k;stage.session->qsa_states=new QsaState[2];}
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
    const ModelGeometry* g_=nullptr;SessionState* ss_=nullptr;int device_=0;
    int64_t lb_=0,le_=0,n_vocab_=1000;Verifier* next_=nullptr;int cs_=11,copy_=12;bool strict_retirement_=false;
    std::shared_ptr<const NativeBatchSetupAdmission> owned_batch_setup_;
    int max_t_=2;void* arena_b_=nullptr;
    std::map<std::vector<int>,int> exec_bm_,commit_bm_;
    bool init_slots(const std::vector<SessionState*>& slots,std::string& err){
        if(!begin_inventory(err))return false;
        slots_=slots;arena_b_=this;seam::native_used[device_]+=20;return publish_inventory();}
    std::vector<int> bkey(const int* rows,int count,int)const{std::vector<int> key(rows,rows+count);key.push_back(device_);return key;}
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
        seam::events.push_back("dispatch");if(dispatch_fail_)return false;
        for(int row=0;row<count;++row)output[row]=tokens[row]+1;
        ACTUAL_BATCH_COPY_CHECK
        return !next_||next_->run_slot_rows(rows,count,tokens,pos,pool,user,output,err);
    }
    bool commit_slot_prefixes(const int*,std::string&){seam::events.push_back("commit");return !commit_fail_;}
    bool run_owned_slot_group(const NativeBatchSlotInspection*,size_t,uint64_t,const int*,int,const int32_t*,const int64_t*,const int*,PoolMultiFn,void*,int32_t*,std::shared_ptr<const ReservationLease>,std::shared_ptr<const NativeBatchGroupCompletion>&,std::string&);
    std::array<std::weak_ptr<const NativeBatchSlotOccupancy>,2> owned_slot_occupancies_;
    const uint64_t slot_verifier_incarnation_=next_verifier_incarnation();
    std::vector<SessionState*> slots_;uint64_t slot_inventory_generation_=0;bool slot_inventory_ready_=false;
    std::atomic<bool> released_{false},terminal_copy_query_failed_{false};bool batch_copy_failed_=false;
    std::string snapshot_failure_,commit_failure_;
    bool inspect_owned_batch_slot(std::shared_ptr<const NativeBatchSlotBindings>,NativeBatchSlotToken,size_t,NativeBatchSlotInspection&)const;
    bool owned_batch_slot_matches(const NativeBatchSlotInspection&)const noexcept;
    ACTUAL_COMMON_REFUSAL
    ACTUAL_SOLO_BINDING
    bool begin_inventory(std::string& err){ACTUAL_INVENTORY_BEGIN return true;}
    bool publish_inventory(){ACTUAL_INVENTORY_READY return true;}
};
ACTUAL_SLOT_MATCH
ACTUAL_SLOT_INSPECT
ACTUAL_OWNED_SETUP
ACTUAL_OWNED_GROUP
}

SLOT_RAW_FIXTURE
using namespace strata::program;
namespace core=strata::core;
void require(bool result,const char* why){if(!result){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
core::WindowResources capacity(uint64_t each=1000){return {{each,each},100000,100000,0,0,4};}
std::vector<core::NativeSessionRequest> requests(){
    std::vector<core::NativeSessionRequest> values(2);
    values[0].device=0;values[0].layer_begin=0;values[0].layer_end=2;
    values[1].device=1;values[1].layer_begin=2;values[1].layer_end=4;
    return values;
}
void distinct_and_reuse(){
    core::WindowBudget budget(capacity(),{{0,0},0,0,0,0,0});core::ModelGeometry geometry;
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
    core::WindowBudget budget(capacity(),{{0,0},0,0,0,0,0});core::ModelGeometry geometry;
    OwnedBatchSlotSessions slots;std::string error;seam::allocation_fail=2;
    require(!slots.initialize(geometry,128,2,requests(),budget,error)&&!slots.published(),"second allocation refusal publishes neither slot");
    require(slots.handle(0).generation==0&&slots.session(0,slots.handle(0))==nullptr,"partial shape cannot execute");
    require(seam::sessions_alive==2&&budget.snapshot().reserved.device_bytes==std::vector<uint64_t>({100,100}),
            "first complete all-stage gang and its genuine lease retained after second refusal");
    require(slots.retire_after_consumers_joined({},error)&&seam::sessions_alive==0&&budget.snapshot().reserved.host_bytes==0,
            "startup refusal uses actual terminal retirement without partial shape leak");
}
void partial_stage_refusal(){
    core::WindowBudget budget(capacity(),{{0,0},0,0,0,0,0});core::ModelGeometry geometry;
    OwnedBatchSlotSessions slots;std::string error;seam::partial_fail=2;
    require(!slots.initialize(geometry,128,2,requests(),budget,error)&&!slots.published(),"actual API unhealthy result never publishes slots");
    require(seam::sessions_alive==4&&budget.snapshot().reserved.device_bytes==std::vector<uint64_t>({200,200}),
            "earlier complete gang plus failing all-stage gang remain retained by native boundary/quarantine");
    require(!slots.retire_after_consumers_joined({},error),"uncertain native allocator refuses a guessed retirement proof");
    require(seam::sessions_alive==4&&!slots.initialize(geometry,128,2,requests(),budget,error),
            "failed retirement cannot recycle quarantined physical addresses");
}
void actual_owned_verifier_inspection(){
    core::WindowBudget budget(capacity(),{{0,0},0,0,0,0,0});auto geometry=std::make_shared<core::ModelGeometry>();
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
    core::WindowBudget budget(cap,{{0,0},0,0,0,0,0});std::string error;auto geometry=std::make_shared<core::ModelGeometry>();geometry->n_layers=4;
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
int main(){
    for(auto test:{distinct_and_reuse,second_gang_refusal,partial_stage_refusal,actual_owned_verifier_inspection,actual_group_success,actual_setup_upload_refusal,actual_slot_raw}){
        auto pid=fork();require(pid>=0,"fork fixture");if(!pid){test();std::_Exit(0);}
        int status=0;require(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"actual ownership fixture child passes");
    }
    std::puts("PASS actual owned two-slot wrapper, production fresh-session factory/retirement/ledger; recording allocation+CUDA boundary only");
}
