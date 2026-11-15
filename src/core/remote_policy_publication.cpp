#include "strata/core/remote_policy_publication.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/window_restore.hpp"
#include <algorithm>
#include <limits>
#include <new>

namespace strata::core {
namespace {
bool add_product(uint64_t& total,uint64_t count,uint64_t size) noexcept {
    if(size && count>(UINT64_MAX-total)/size)return false;
    total+=count*size;return total<=SIZE_MAX;
}
bool same_plan(const RemoteExpertOpt::AdaptPlanEntry& a,const RemoteExpertOpt::AdaptPlanEntry& b) noexcept {
    return a.peer==b.peer&&a.slot==b.slot&&a.layer==b.layer&&a.out==b.out&&a.in==b.in&&a.bytes==b.bytes;
}
}
bool RemotePolicyPublication::required_bytes(const RemotePolicyPublicationLimits& l,uint64_t& out) noexcept {
    if(!l.max_peers||l.max_peers>3||!l.max_changes_per_peer||!l.max_total_changes||
        l.max_changes_per_peer>l.max_total_changes)return false;
    uint64_t bytes=4096+sizeof(RemotePolicyPublication);
    if(!add_product(bytes,l.max_peers,sizeof(Peer)+sizeof(RemotePolicyPeerBinding))||
       !add_product(bytes,l.max_total_changes,sizeof(RemoteExpertOpt::AdaptPlanEntry)+sizeof(ExpertPlacementChange)))return false;
    out=bytes;return true;
}
const ExpertSourceInventory* RemotePolicyPublication::source_inventory() const noexcept {
    if(auto* file=dynamic_cast<const FileExpertSource*>(source_.get()))return file->completed_inventory();
    if(auto* arena=dynamic_cast<const ArenaExpertSource*>(source_.get()))return arena->completed_inventory();
    return nullptr;
}
std::shared_ptr<const ExpertSourceCapture> RemotePolicyPublication::source_capture_owner() const noexcept {
    if(auto* file=dynamic_cast<const FileExpertSource*>(source_.get()))return file->completed_inventory_owner();
    if(auto* arena=dynamic_cast<const ArenaExpertSource*>(source_.get()))return arena->completed_inventory_owner();
    return {};
}
bool RemotePolicyPublication::authority() const noexcept {
    const auto current_capture=source_capture_owner();
    return current_capture&&current_capture.get()==source_capture_.get()&&
        !current_capture.owner_before(source_capture_)&&!source_capture_.owner_before(current_capture)&& !poisoned_&&runtime_&&source_&&opt_&&owns_&&
        owns_(*runtime_,*source_,*opt_,{bindings_.get(),peer_count_})&&
        source_inventory()==source_receipt_&&source_receipt_&&source_capture_&&
        source_capture_->completed_inventory()==source_receipt_&&layout_capture_&&
        layout_capture_->completed_inventory()==layout_receipt_&&layout_receipt_&&
        layout_receipt_->layout_generation==layout_generation_&&layout_receipt_->hadamard_generation==hadamard_generation_&&ranking_&&
        ranking_->completed_inventory()==ranking_receipt_&&ranking_->attempt_generation()==ranking_generation_;
}
bool RemotePolicyPublication::create(std::shared_ptr<const ExpertPolicyCapture> capture,
    std::shared_ptr<NativeRuntimeAttachment> runtime,std::shared_ptr<ExpertSource> source,
    std::shared_ptr<const ExpertSourceCapture> source_capture,
    RemoteExpertOpt& opt,std::span<const RemotePolicyPeerBinding> bindings,uint64_t incarnation,
    const RemotePolicyPublicationLimits& limits,WindowBudget budget,RemotePolicyOwnershipValidator owns,
    std::unique_ptr<RemotePolicyPublication>& out,std::string& error) {
    try {
        // Refuse replacing any pending/unknown adapter; destruction is not proof of drain.
        if(out&&(out->in_call_||out->poisoned_||out->tracker_->view().active_transactions))return false;
        uint64_t bytes=0;
        const auto* initial=capture?capture->completed_inventory():nullptr;
        if(!runtime||!source||!source_capture||!owns||!incarnation||runtime.owner_before(source)||source.owner_before(runtime)||
           !initial||!initial->initial_only||!initial->input_ranking||bindings.empty()||
           bindings.size()>limits.max_peers||!required_bytes(limits,bytes)||bytes>limits.max_host_bytes||
           limits.tracker.max_transactions<bindings.size()||limits.tracker.max_changes<limits.max_changes_per_peer||
           !owns(*runtime,*source,opt,bindings))return false;
        const auto* ranking=initial->input_ranking->completed_inventory();if(!ranking)return false;
        for(size_t i=0;i<bindings.size();++i) {
            const auto& b=bindings[i];
            const auto t=std::find_if(initial->tiers.begin(),initial->tiers.end(),[&](const auto& t){return t.id==b.tier;});
            if(!b.helper||t==initial->tiers.end()||t->kind!=ExpertPolicyTierKind::remote||t->disabled||t->device!=b.device)return false;
            for(size_t j=0;j<i;++j)if(bindings[j].tier==b.tier||bindings[j].helper==b.helper)return false;
        }
        auto snap=budget.snapshot();WindowResources charge;charge.device_bytes.resize(snap.capacity.device_bytes.size());charge.host_bytes=bytes;
        auto admission=budget.reserve(charge,ReservationPurpose::foreground);if(!admission||!admission.lease)return false;
        auto p=std::unique_ptr<RemotePolicyPublication>(new RemotePolicyPublication);
        p->reservation_=std::move(admission.lease);p->capture_=capture;p->runtime_=runtime;p->source_=source;
        p->opt_=&opt;p->owns_=owns;p->limits_=limits;p->owner_incarnation_=incarnation;p->peer_count_=bindings.size();
        p->bindings_=std::make_unique<RemotePolicyPeerBinding[]>(size_t(limits.max_peers));
        p->peers_=std::make_unique<Peer[]>(size_t(limits.max_peers));
        p->plans_=std::make_unique<RemoteExpertOpt::AdaptPlanEntry[]>(size_t(limits.max_total_changes));
        p->changes_=std::make_unique<ExpertPlacementChange[]>(size_t(limits.max_total_changes));
        std::copy(bindings.begin(),bindings.end(),p->bindings_.get());
        p->ranking_=initial->input_ranking;p->ranking_receipt_=ranking;p->ranking_generation_=p->ranking_->attempt_generation();
        p->source_capture_=std::move(source_capture);p->source_receipt_=p->source_inventory();
        if(!p->source_receipt_||p->source_capture_->completed_inventory()!=p->source_receipt_)return false;
        p->layout_capture_=p->source_receipt_->retained_layout;
        p->layout_receipt_=p->layout_capture_?p->layout_capture_->completed_inventory():nullptr;
        if(!p->layout_receipt_)return false;
        p->layout_generation_=p->layout_receipt_->layout_generation;p->hadamard_generation_=p->layout_receipt_->hadamard_generation;
        if(p->layout_generation_!=p->source_receipt_->layout_generation||p->hadamard_generation_!=p->source_receipt_->hadamard_generation)return false;
        if(!p->source_receipt_||p->source_receipt_->layers!=ranking->layers()||p->source_receipt_->experts!=ranking->experts())return false;
        const auto& source_inputs=p->source_receipt_->retained_input_bindings;
        const auto& ranking_inputs=ranking->retained_input_bindings;
        if(!source_inputs||source_inputs.get()!=ranking_inputs.get()||source_inputs.owner_before(ranking_inputs)||ranking_inputs.owner_before(source_inputs))return false;
        if(!ExpertPolicyRuntime::create(capture,runtime,incarnation,limits.tracker,budget,p->tracker_,error))return false;
        if(!p->authority())return false;
        error.clear();out.swap(p);return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
    catch(...){error.clear();return false;}
}
RemotePolicyPublication::~RemotePolicyPublication() noexcept {
    // Buffers contain only host plans. Tracker quarantines the actual runtime
    // (which genuinely owns source/helpers) for any unresolved attempted work.
    abandon_unattempted();
}
bool RemotePolicyPublication::prepare_all(const RemoteExpertOpt::AdaptPlanEntry* plans,size_t count) {
    if(!in_call_||prepared_||!authority()||!plans||!count||count>limits_.max_total_changes)return false;
    auto view=tracker_->view();if(view.unknown||view.closed||view.active_transactions||!view.stable)return false;
    for(size_t p=0;p<peer_count_;++p)peers_[p]={};
    // Validate all peers before admitting any transaction. Actual caller groups
    // compact attach-order peer plans; do not reinterpret peer as tier ordinal.
    for(size_t i=0;i<count;++i) {
        const auto& plan=plans[i];
        if(plan.peer>=peer_count_||(i&&plan.peer<plans[i-1].peer)||plan.slot<0||plan.layer<0||plan.in<0||plan.out<0||plan.in==plan.out||!plan.bytes)return false;
        auto& peer=peers_[plan.peer];if(!peer.count)peer.start=i;
        if(++peer.count>limits_.max_changes_per_peer)return false;
        const auto tier=bindings_[plan.peer].tier;
        const auto cell=std::find_if(view.completed.begin(),view.completed.end(),[&](const auto& c){return c.tier==tier&&c.slot==plan.slot;});
        if(cell==view.completed.end()||!cell->value.present||cell->value.layer!=plan.layer||cell->value.expert!=plan.out||cell->value.bytes!=plan.bytes)return false;
        if(uint64_t(plan.layer)>=source_receipt_->layers||uint64_t(plan.in)>=source_receipt_->experts)return false;
        const ExpertLayerLoadRecord* actual_layer=nullptr;
        for(const auto& layer:layout_receipt_->layers)if(layer.layer==uint64_t(plan.layer)) {
            if(actual_layer)return false;actual_layer=&layer;
        }
        if(!actual_layer||!actual_layer->blob_bytes||actual_layer->blob_bytes!=plan.bytes)return false;
        for(size_t j=peer.start;j<i;++j)if(plans[j].slot==plan.slot)return false;
        uint64_t ordinal=expert_policy_synthetic_ordinal;
        for(size_t j=0;j<ranking_receipt_->ranked.size();++j)
            if(ranking_receipt_->ranked[j]==std::pair<int32_t,int32_t>{plan.layer,plan.in}){ordinal=j;break;}
        plans_[i]=plan;changes_[i]={tier,plan.slot,cell->value,{true,plan.layer,plan.in,ordinal,plan.bytes},true};
    }
    prepared_=true;
    for(size_t p=0;p<peer_count_;++p)if(peers_[p].count) {
        auto& peer=peers_[p];
        if(!tracker_->begin({changes_.get()+peer.start,peer.count},{},peer.token)) {abandon_unattempted();return false;}
    }
    return true;
}
bool RemotePolicyPublication::begin_attempt(uint32_t p) {
    if(!in_call_||!prepared_||!authority()||p>=peer_count_)return false;
    auto& peer=peers_[p];
    if(!peer.token.operation||peer.attempted||peer.unknown||!tracker_->mark_attempted(peer.token))return false;
    peer.attempted=true;return true;
}
bool RemotePolicyPublication::publish_after_copies(uint32_t p,const RemoteExpertOpt::AdaptPlanEntry* plans,size_t count) {
    if(!in_call_||!prepared_||!authority()||p>=peer_count_||!plans)return false;
    auto& peer=peers_[p];if(!peer.attempted||peer.published||peer.unknown||count!=peer.count)return false;
    for(size_t i=0;i<count;++i)if(!same_plan(plans[i],plans_[peer.start+i]))return false;
    if(!tracker_->copy_completed(peer.token)||!tracker_->publish_host(peer.token)||
       !tracker_->device_tables_ordered(peer.token)||!tracker_->reclamation_completed(peer.token)) {
        fail_unknown(p);return false;
    }
    peer.published=true;
    // Retain token until adapt_owned finishes writer publication. Its cleanup
    // must still be able to fail_unknown if finish_publication refuses afterward.
    return true;
}
void RemotePolicyPublication::fail_unknown(uint32_t p) noexcept {
    poisoned_=true;
    if(p<peer_count_) {auto& peer=peers_[p];peer.unknown=true;if(peer.token.operation)(void)tracker_->fail_unknown(peer.token);}
}
void RemotePolicyPublication::abandon_unattempted() noexcept {
    if(!tracker_||!peers_)return;
    for(size_t p=0;p<peer_count_;++p) {auto& peer=peers_[p];
        if(peer.token.operation&&!peer.attempted&&!peer.unknown) {
            if(tracker_->abandon_unattempted(peer.token))peer.token={};else fail_unknown(uint32_t(p));
        }
    }
}
void RemotePolicyPublication::finish_call() noexcept {
    abandon_unattempted();
    for(size_t p=0;p<peer_count_;++p) {auto& peer=peers_[p];
        if(peer.token.operation&&peer.attempted&&!peer.unknown) {
            if(peer.published&&tracker_->release_completed(peer.token))peer.token={};
            else fail_unknown(uint32_t(p));
        }
    }
    prepared_=false;in_call_=false;
}
RemoteExpertOpt::AdaptResult RemotePolicyPublication::adapt(const std::vector<float>& usage,
    const std::vector<int32_t>& primary,const std::vector<std::pair<int32_t,int32_t>>& pending,
    int swaps,RemoteExpertOpt::OwnedAdaptWorkspace& workspace,std::string& error) {
    if(in_call_||!authority()||swaps<0||uint64_t(swaps)>limits_.max_changes_per_peer)return RemoteExpertOpt::AdaptResult::failed;
    auto v=tracker_->view();if(v.unknown||v.closed||v.active_transactions)return RemoteExpertOpt::AdaptResult::failed;
    in_call_=true;
    // No potentially allocating completion receipt: all state already admitted.
    auto result=RemoteExpertOpt::AdaptResult::failed;
    try {result=opt_->adapt_owned(usage,primary,pending,swaps,*source_,workspace,*this,error);}
    catch(...) {for(size_t p=0;p<peer_count_;++p)if(peers_[p].attempted&&peers_[p].token.operation)fail_unknown(uint32_t(p));}
    finish_call();return poisoned_?RemoteExpertOpt::AdaptResult::failed:result;
}
RemotePolicyPublicationView RemotePolicyPublication::view() const noexcept {
    auto v=tracker_->view();return {v.revision,v.active_transactions,v.unknown||!authority()||in_call_,v.closed};
}
bool RemotePolicyPublication::remote_cell(uint32_t peer,int32_t slot,ExpertPlacementValue& host,ExpertPlacementValue& completed) const noexcept {
    if(in_call_||!authority()||peer>=peer_count_||slot<0)return false;
    auto v=tracker_->view();if(v.unknown||v.closed)return false;
    for(size_t i=0;i<v.host.size();++i)if(v.host[i].tier==bindings_[peer].tier&&v.host[i].slot==slot) {
        host=v.host[i].value;completed=v.completed[i].value;return true;
    }
    return false;
}
bool RemotePolicyPublication::resolve_after_teardown(const std::shared_ptr<NativeRuntimeAttachment>& runtime,uint64_t incarnation) noexcept {
    if(in_call_)return false;
    return tracker_->resolve_after_teardown(runtime,incarnation);
}
}
