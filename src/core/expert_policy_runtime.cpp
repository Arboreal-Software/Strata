#include "strata/core/expert_policy_runtime.hpp"
#include <algorithm>
#include <atomic>
#include <limits>
#include <exception>
#include <stdexcept>
namespace strata::core {
namespace {
std::atomic<uint64_t> next_tracker{1};
std::atomic<uint64_t> quarantined_count{0};
uint64_t mint() noexcept {auto n=next_tracker.load();while(n&&n!=UINT64_MAX){if(next_tracker.compare_exchange_weak(n,n+1))return n;}return 0;}
void need(bool v,const char* msg){if(!v)throw std::invalid_argument(msg);}
uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,"placement envelope overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,"placement envelope overflow");return a*b;}
bool empty(const ExpertPlacementValue& v){return v==ExpertPlacementValue{};}
}
struct ExpertPolicyRuntime::Retention {
    // Process-lifetime quarantine: no destructor/drain is attempted on uncertain
    // work. The first member is destroyed last on the ordinary unused path.
    std::shared_ptr<const ReservationLease> reservation;
    std::shared_ptr<const ExpertPolicyCapture> capture;
    std::shared_ptr<NativeRuntimeAttachment> owner;
    std::shared_ptr<const ExpertProfileCapture> ranking;
    std::vector<ExpertPlacementDependency> dependencies;
    Retention* next=nullptr;
    static std::atomic<Retention*> head;
};
std::atomic<ExpertPolicyRuntime::Retention*> ExpertPolicyRuntime::Retention::head{nullptr};
uint64_t expert_policy_runtime_quarantine_count() noexcept{return quarantined_count.load(std::memory_order_acquire);}
bool ExpertPolicyRuntime::unsafe_to_destroy()const noexcept {
    if(closed_)return false;
    if(unknown_)return true;
    for(const auto& t:transactions_)if(t.active&&(t.phase.unknown||(t.phase.attempted&&!t.phase.reclaim_complete)))return true;
    return false;
}
ExpertPolicyRuntime::~ExpertPolicyRuntime() noexcept {
    if(!unsafe_to_destroy())return;
    // Creation preallocates this node. All transfers are noexcept; dependency
    // destructors and device-owning runtime destructors cannot run here.
    auto* node=retention_.release();
    if(!node)std::terminate(); // impossible for a successfully created tracker
    node->reservation=std::move(reservation_);node->capture=std::move(capture_);
    node->owner=std::move(owner_);node->ranking=std::move(ranking_);
    node->dependencies.swap(dependencies_);
    auto* head=Retention::head.load(std::memory_order_relaxed);
    do{node->next=head;}while(!Retention::head.compare_exchange_weak(head,node,std::memory_order_release,std::memory_order_relaxed));
    auto n=quarantined_count.load(std::memory_order_relaxed);
    while(n!=UINT64_MAX&&!quarantined_count.compare_exchange_weak(n,n+1,std::memory_order_release,std::memory_order_relaxed)){}
}
bool ExpertPolicyRuntime::create(std::shared_ptr<const ExpertPolicyCapture> capture,
    std::shared_ptr<NativeRuntimeAttachment> owner,uint64_t owner_id,const ExpertPolicyRuntimeLimits& l,
    WindowBudget budget,std::unique_ptr<ExpertPolicyRuntime>& out,std::string& error){
    try{
        need(!expert_policy_runtime_quarantine_count(),"placement worker quarantined; restart required");
        need(!out||!out->unsafe_to_destroy(),"cannot replace tracker with unresolved attempted work");
        const auto* initial=capture?capture->completed_inventory():nullptr;
        need(initial&&initial->initial_only&&initial->generation&&initial->reservation&&owner&&owner_id,"completed initial policy and runtime owner required");
        const auto* ranking=initial->input_ranking?initial->input_ranking->completed_inventory():nullptr;
        need(ranking,"completed ranking authority required");
        need(l.max_tiers&&l.max_slots&&l.max_transactions&&l.max_changes&&l.max_dependencies&&l.max_host_bytes,"zero runtime bounds");
        need(initial->tiers.size()<=l.max_tiers&&initial->current_slots.size()<=l.max_slots,"initial placement exceeds bounds");
        uint64_t cells=0;for(size_t i=0;i<initial->tiers.size();++i){const auto& t=initial->tiers[i];
            need(t.slots<=uint64_t(INT32_MAX)&&(!t.disabled||!t.slots)&&
                (t.disabled||(t.device>=0&&t.layer_begin>=0&&t.layer_end>t.layer_begin&&uint64_t(t.layer_end)<=ranking->layers())),"invalid tier capacity/geometry");
            for(size_t j=0;j<i;++j)need(initial->tiers[j].id!=t.id,"duplicate tier identity");
            cells=add(cells,t.slots);
        }need(cells<=l.max_slots,"tier slot envelope exceeded");
        const auto nc=mul(l.max_transactions,l.max_changes),nd=mul(l.max_transactions,l.max_dependencies);
        uint64_t bytes=add(4096,sizeof(ExpertPolicyRuntime)+sizeof(Retention));
        bytes=add(bytes,mul(initial->tiers.size(),sizeof(ExpertPolicyTier)));
        bytes=add(bytes,mul(cells,2*sizeof(ExpertPlacementCell)));
        bytes=add(bytes,mul(l.max_transactions,sizeof(Transaction)));
        bytes=add(bytes,mul(nc,sizeof(ExpertPlacementChange)));
        bytes=add(bytes,mul(nd,sizeof(ExpertPlacementDependency)+sizeof(uint8_t)));
        need(bytes<=l.max_host_bytes&&bytes<=SIZE_MAX,"runtime metadata exceeds host bound");
        auto snapshot=budget.snapshot();WindowResources charge;charge.device_bytes.resize(snapshot.capacity.device_bytes.size());charge.host_bytes=bytes;
        auto reserved=budget.reserve(charge,ReservationPurpose::foreground);need(bool(reserved)&&reserved.lease,"placement budget refused");
        auto p=std::unique_ptr<ExpertPolicyRuntime>(new ExpertPolicyRuntime);
        p->retention_=std::make_unique<Retention>();
        p->reservation_=std::move(reserved.lease);p->capture_=capture;p->owner_=std::move(owner);p->owner_incarnation_=owner_id;p->limits_=l;
        p->ranking_=initial->input_ranking;p->ranking_receipt_=ranking;p->ranking_generation_=p->ranking_->attempt_generation();p->initial_generation_=initial->generation;
        p->tiers_=initial->tiers;p->host_.reserve(size_t(cells));
        for(const auto& t:p->tiers_)for(uint64_t s=0;s<t.slots;++s)p->host_.push_back({t.id,int32_t(s),{}});
        for(const auto& s:initial->current_slots){const auto n=p->cell(s.tier_id,s.slot);need(n<p->host_.size()&&!p->host_[n].value.present,"invalid initial slot identity");
            const auto tier=std::find_if(p->tiers_.begin(),p->tiers_.end(),[&](const auto& t){return t.id==s.tier_id;});
            need(tier!=p->tiers_.end()&&!tier->disabled&&s.bytes&&s.layer>=tier->layer_begin&&s.layer<tier->layer_end&&s.expert>=0&&uint64_t(s.expert)<ranking->experts(),"invalid initial placement extent");
            for(const auto& other:p->host_)need(!(other.tier==s.tier_id&&other.value.present&&other.value.layer==s.layer&&other.value.expert==s.expert),"duplicate initial expert");
            p->host_[n].value={true,s.layer,s.expert,s.source_ordinal,s.bytes};
        }
        p->completed_=p->host_;p->transactions_.resize(size_t(l.max_transactions));p->changes_.resize(size_t(nc));p->dependencies_.resize(size_t(nd));p->ordered_.resize(size_t(nd));
        p->incarnation_=mint();need(p->incarnation_!=0,"placement tracker incarnation exhausted");
        error.clear();out.swap(p);return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
    catch(const std::exception& e){try{error=e.what();}catch(...){error.clear();}return false;}
    catch(...){error.clear();return false;}
}
bool ExpertPolicyRuntime::authority()const noexcept {
    if(expert_policy_runtime_quarantine_count()||closed_||!owner_||!capture_||!ranking_||ranking_->completed_inventory()!=ranking_receipt_||ranking_->attempt_generation()!=ranking_generation_)return false;
    const auto state=capture_->state();
    if(state==ExpertPolicyState::stale)return true; // expected one-way startup invalidation
    const auto* initial=capture_->completed_inventory();
    return state==ExpertPolicyState::initial_complete&&initial&&initial->generation==initial_generation_;
}
size_t ExpertPolicyRuntime::cell(uint64_t tier,int32_t slot)const noexcept {for(size_t i=0;i<host_.size();++i)if(host_[i].tier==tier&&host_[i].slot==slot)return i;return host_.size();}
size_t ExpertPolicyRuntime::index(const Transaction& t)const noexcept{return size_t(&t-transactions_.data());}
const ExpertPolicyRuntime::Transaction* ExpertPolicyRuntime::find(ExpertPlacementToken token)const noexcept {
    if(!token.operation||token.tracker!=incarnation_||token.owner!=owner_incarnation_)return nullptr;
    for(const auto& t:transactions_)if(t.active&&!t.retiring&&t.token==token)return &t;return nullptr;
}
ExpertPolicyRuntime::Transaction* ExpertPolicyRuntime::find(ExpertPlacementToken token)noexcept{return const_cast<Transaction*>(static_cast<const ExpertPolicyRuntime*>(this)->find(token));}
bool ExpertPolicyRuntime::tick()noexcept {if(revision_==UINT64_MAX){unknown_=true;return false;}++revision_;return true;}
bool ExpertPolicyRuntime::begin(std::span<const ExpertPlacementChange> changes,std::span<const ExpertPlacementDependency> deps,ExpertPlacementToken& out)noexcept {
    if(!authority()||unknown_||changes.empty()||changes.size()>limits_.max_changes||deps.size()>limits_.max_dependencies||!next_operation_||next_operation_==UINT64_MAX)return false;
    Transaction* free=nullptr;for(auto& t:transactions_)if(!t.active&&!t.retiring){free=&t;break;}if(!free)return false;
    for(size_t i=0;i<changes.size();++i){const auto& c=changes[i];const auto n=cell(c.tier,c.slot);if(n==host_.size()||host_[n].value!=c.expected||completed_[n].value!=c.expected)return false;
        const auto tier=std::find_if(tiers_.begin(),tiers_.end(),[&](const auto& t){return t.id==c.tier;});
        if(tier==tiers_.end()||tier->disabled)return false;
        const auto& v=c.proposed;if(v.present&&!c.requires_copy&&v!=c.expected)return false;
        if(v.present){if(!v.bytes||v.layer<tier->layer_begin||v.layer>=tier->layer_end||v.expert<0||uint64_t(v.expert)>=ranking_receipt_->experts())return false;
            if(v.source_ordinal!=expert_policy_synthetic_ordinal&&(v.source_ordinal>=ranking_receipt_->ranked.size()||ranking_receipt_->ranked[size_t(v.source_ordinal)]!=std::pair<int32_t,int32_t>{v.layer,v.expert}))return false;
        }else if(!empty(v))return false;
        for(size_t j=0;j<i;++j)if(changes[j].tier==c.tier&&(changes[j].slot==c.slot||(v.present&&changes[j].proposed.present&&changes[j].proposed.layer==v.layer&&changes[j].proposed.expert==v.expert)))return false;
        for(const auto& t:transactions_)if(t.active){const auto base=index(t)*size_t(limits_.max_changes);for(size_t j=0;j<t.changes;++j){const auto& other=changes_[base+j];if(other.tier==c.tier&&(other.slot==c.slot||(v.present&&other.proposed.present&&other.proposed.layer==v.layer&&other.proposed.expert==v.expert)))return false;}}
        if(v.present)for(const auto& occupied:host_)if(occupied.tier==c.tier&&occupied.slot!=c.slot&&occupied.value.present&&occupied.value.layer==v.layer&&occupied.value.expert==v.expert){bool removed=false;for(const auto& other:changes)if(other.tier==occupied.tier&&other.slot==occupied.slot&&(!other.proposed.present||other.proposed.layer!=v.layer||other.proposed.expert!=v.expert))removed=true;if(!removed)return false;}
    }
    for(size_t i=0;i<deps.size();++i){const auto& d=deps[i];if(!d.id||!d.generation||!d.work_scope||!d.owner||uint8_t(d.kind)>uint8_t(ExpertPlacementOrder::per_layer_gate))return false;
        const auto t=std::find_if(tiers_.begin(),tiers_.end(),[&](const auto& t){return t.id==d.tier;});if(t==tiers_.end()||t->device!=d.device)return false;
        for(size_t j=0;j<i;++j)if(deps[j].id==d.id)return false;
    }
    if(!tick())return false;
    const auto at=index(*free),cb=at*size_t(limits_.max_changes),db=at*size_t(limits_.max_dependencies);
    for(size_t i=0;i<changes.size();++i)changes_[cb+i]=changes[i];
    for(size_t i=0;i<deps.size();++i){dependencies_[db+i]=deps[i];ordered_[db+i]=0;}
    free->token={incarnation_,owner_incarnation_,next_operation_++};free->changes=changes.size();free->dependencies=deps.size();free->phase={};free->active=true;out=free->token;return true;
}
bool ExpertPolicyRuntime::covered(const Transaction& t)const noexcept {
    const auto cb=index(t)*size_t(limits_.max_changes),db=index(t)*size_t(limits_.max_dependencies);
    for(size_t i=0;i<t.changes;++i)if(changes_[cb+i].requires_copy){bool found=false;for(size_t j=0;j<t.dependencies;++j)if(dependencies_[db+j].tier==changes_[cb+i].tier)found=true;if(!found)return false;}return true;
}
void ExpertPolicyRuntime::commit_if_ready(Transaction& t)noexcept {
    if(t.phase.unknown||t.phase.placement_committed||!t.phase.copy_complete||!t.phase.host_published||!t.phase.device_ordered)return;
    const auto base=index(t)*size_t(limits_.max_changes);for(size_t i=0;i<t.changes;++i){const auto& c=changes_[base+i];completed_[cell(c.tier,c.slot)].value=c.proposed;}t.phase.placement_committed=true;
}
bool ExpertPolicyRuntime::mark_attempted(ExpertPlacementToken token)noexcept {auto* t=find(token);if(!t||!authority()||t->phase.unknown||t->phase.attempted||!tick())return false;t->phase.attempted=true;return true;}
bool ExpertPolicyRuntime::copy_completed(ExpertPlacementToken token)noexcept {auto* t=find(token);if(!t||!authority()||!t->phase.attempted||t->phase.unknown||!tick())return false;t->phase.copy_complete=true;commit_if_ready(*t);return true;}
bool ExpertPolicyRuntime::publish_host(ExpertPlacementToken token)noexcept {
    auto* t=find(token);if(!t||!authority()||!t->phase.attempted||t->phase.unknown||t->phase.host_published||(!t->phase.copy_complete&&!covered(*t))||!tick())return false;
    const auto base=index(*t)*size_t(limits_.max_changes);for(size_t i=0;i<t->changes;++i){const auto& c=changes_[base+i];host_[cell(c.tier,c.slot)].value=c.proposed;}t->phase.host_published=true;commit_if_ready(*t);return true;
}
bool ExpertPolicyRuntime::dependency_ordered(ExpertPlacementToken token,uint64_t id,uint64_t generation)noexcept {
    auto* t=find(token);if(!t||!authority()||!t->phase.attempted||t->phase.unknown)return false;
    const auto base=index(*t)*size_t(limits_.max_dependencies);for(size_t j=0;j<t->dependencies;++j)if(dependencies_[base+j].id==id&&dependencies_[base+j].generation==generation){if(!tick())return false;ordered_[base+j]=1;return true;}return false;
}
bool ExpertPolicyRuntime::device_tables_ordered(ExpertPlacementToken token)noexcept {
    auto* t=find(token);if(!t||!authority()||!t->phase.host_published||t->phase.unknown||(!t->phase.copy_complete&&!covered(*t)))return false;
    const auto base=index(*t)*size_t(limits_.max_dependencies);for(size_t j=0;j<t->dependencies;++j)if(!ordered_[base+j])return false;
    if(!tick())return false;t->phase.device_ordered=true;commit_if_ready(*t);return true;
}
bool ExpertPolicyRuntime::reclamation_completed(ExpertPlacementToken token)noexcept {auto* t=find(token);if(!t||!authority()||t->phase.unknown||!t->phase.placement_committed||!tick())return false;t->phase.reclaim_complete=true;return true;}
void ExpertPolicyRuntime::clear(Transaction& t)noexcept {
    t.retiring=true; // callbacks may reenter, but cannot reuse this arena yet
    const auto base=index(t)*size_t(limits_.max_dependencies);for(size_t j=0;j<t.dependencies;++j){auto owner=std::move(dependencies_[base+j].owner);dependencies_[base+j]={};ordered_[base+j]=0;owner.reset();}
    t={};
}
bool ExpertPolicyRuntime::release_completed(ExpertPlacementToken token)noexcept {auto* t=find(token);if(!t||!authority()||t->phase.unknown||!t->phase.reclaim_complete||!tick())return false;clear(*t);return true;}
bool ExpertPolicyRuntime::abandon_unattempted(ExpertPlacementToken token)noexcept {auto* t=find(token);if(!t||t->phase.attempted||t->phase.unknown||!tick())return false;clear(*t);return true;}
bool ExpertPolicyRuntime::fail_unknown(ExpertPlacementToken token)noexcept {auto* t=find(token);if(!t)return false;t->phase.unknown=true;unknown_=true;(void)tick();return true;}
bool ExpertPolicyRuntime::progress(ExpertPlacementToken token,ExpertPlacementProgress& out)const noexcept {const auto* t=find(token);if(!t)return false;out=t->phase;return true;}
bool ExpertPolicyRuntime::transaction(ExpertPlacementToken token,ExpertPlacementTransactionView& out)const noexcept {
    const auto* t=find(token);if(!t)return false;const auto cb=index(*t)*size_t(limits_.max_changes),db=index(*t)*size_t(limits_.max_dependencies);
    out={std::span<const ExpertPlacementChange>(changes_).subspan(cb,t->changes),std::span<const ExpertPlacementDependency>(dependencies_).subspan(db,t->dependencies),std::span<const uint8_t>(ordered_).subspan(db,t->dependencies),t->phase};return true;
}
ExpertPlacementView ExpertPolicyRuntime::view()const noexcept {
    ExpertPlacementView v{host_,completed_,revision_,0,false,unknown_||!authority(),closed_};
    bool complete=true;for(const auto& t:transactions_)if(t.active){++v.active_transactions;complete=complete&&t.phase.placement_committed&&!t.phase.unknown;}
    v.stable=!v.unknown&&!v.closed&&complete;return v;
}
bool ExpertPolicyRuntime::resolve_after_teardown(const std::shared_ptr<NativeRuntimeAttachment>& owner,uint64_t incarnation)noexcept {
    if(closed_||owner.get()!=owner_.get()||owner.owner_before(owner_)||owner_.owner_before(owner)||incarnation!=owner_incarnation_)return false;
    closed_=true;unknown_=true;(void)tick();for(auto& t:transactions_)if(t.active&&!t.retiring)clear(t);return true;
}
}
