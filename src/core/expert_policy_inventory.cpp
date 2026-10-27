#include "strata/core/expert_policy_inventory.hpp"
#include <limits>
#include <new>

namespace strata::core {
bool expert_policy_inventory_host_bound(const ExpertPolicyLimits& l,uint64_t& out,std::string& error) {
    if(!l.max_tiers||l.max_tiers>256||!l.max_transforms||l.max_transforms>4096||
       !l.max_order_entries||l.max_order_entries>(1ull<<27)||!l.max_decisions||l.max_decisions>(1ull<<27)||
       !l.max_slots||l.max_slots>(1ull<<26)) {error="policy receipt bounds outside supported limits";return false;}
    uint64_t bytes=65536+sizeof(ExpertPolicyCapture)+sizeof(ExpertPolicyInventory);
    const auto add=[&](uint64_t n,uint64_t stride){if(n>(UINT64_MAX-bytes)/stride)return false;bytes+=n*stride;return true;};
    if(!add(l.max_tiers,sizeof(ExpertPolicyTier)+sizeof(uint64_t)+2*sizeof(uint8_t))||
       !add(l.max_transforms,sizeof(ExpertPolicyTransform))||!add(l.max_order_entries,sizeof(uint64_t)+sizeof(ExpertPolicyOrderedPair))||
       !add(l.max_decisions,sizeof(ExpertPolicyPlacement))||!add(l.max_slots,sizeof(ExpertPolicySlot))) {
        error="policy receipt envelope overflow";return false;
    }
    out=bytes;return true;
}
bool ExpertPolicyCapture::create(const ExpertPolicyLimits& limits,std::shared_ptr<const ReservationLease> lease,
    std::shared_ptr<const ExpertProfileCapture> ranking,std::unique_ptr<ExpertPolicyCapture>& out,std::string& error) {
    uint64_t bytes=0;if(!expert_policy_inventory_host_bound(limits,bytes,error))return false;
    const auto* input=ranking?ranking->completed_inventory():nullptr;
    if(!lease||lease->purpose()!=ReservationPurpose::foreground||lease->resources().host_bytes<bytes||!input||
       lease==input->metadata_reservation) {error="policy capture requires distinct prepaid storage and completed input ranking";return false;}
    try {
        auto c=std::make_unique<ExpertPolicyCapture>();c->limits_=limits;
        c->inventory_.reservation=std::move(lease);c->inventory_.input_ranking=std::move(ranking);c->input_receipt_=input;c->input_generation_=c->inventory_.input_ranking->attempt_generation();
        c->inventory_.tiers.reserve(size_t(limits.max_tiers));c->decision_counts_.resize(size_t(limits.max_tiers));
        c->tier_complete_.resize(size_t(limits.max_tiers));c->tier_configured_.resize(size_t(limits.max_tiers));c->inventory_.transforms.reserve(size_t(limits.max_transforms));
        c->inventory_.source_ordinals.reserve(size_t(limits.max_order_entries));
        c->inventory_.ordered_pairs.reserve(size_t(limits.max_order_entries));
        c->inventory_.decisions.reserve(size_t(limits.max_decisions));c->inventory_.current_slots.reserve(size_t(limits.max_slots));
        out=std::move(c);return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
}
bool ExpertPolicyCapture::building() noexcept {
    if(state_!=ExpertPolicyState::building)return false;
    if((inventory_.input_ranking->completed_inventory()!=input_receipt_||inventory_.input_ranking->attempt_generation()!=input_generation_)){fail();return false;}
    return true;
}
size_t ExpertPolicyCapture::tier_index(uint64_t id)const noexcept {
    for(size_t i=0;i<inventory_.tiers.size();++i)if(inventory_.tiers[i].id==id)return i;
    return inventory_.tiers.size();
}
bool ExpertPolicyCapture::declare_tiers(const ExpertPolicyTierIdentity* tiers,size_t n) noexcept {
    if(!building())return false;
    if(configured_||!tiers||!n||n>limits_.max_tiers){fail();return false;}
    for(size_t i=0;i<n;++i){
        for(size_t j=0;j<i;++j)if(tiers[i].id==tiers[j].id){fail();return false;}
        ExpertPolicyTier t;t.id=tiers[i].id;t.kind=tiers[i].kind;inventory_.tiers.push_back(t);
    }
    configured_=true;return true;
}
bool ExpertPolicyCapture::configure_tier(const ExpertPolicyTier& t) noexcept {
    if(!building())return false;
    const size_t i=tier_index(t.id);
    if(!configured_||i==inventory_.tiers.size()||tier_configured_[i]||t.kind!=inventory_.tiers[i].kind||
       (!t.disabled&&(t.device<0||t.layer_begin<0||t.layer_end<=t.layer_begin||uint64_t(t.layer_end)>input_receipt_->layers()))||
       (t.disabled&&(t.slots||t.decision_limit))||t.slots>limits_.max_slots-configured_slots_||
       t.decision_limit>limits_.max_decisions-configured_decisions_){fail();return false;}
    configured_slots_+=t.slots;configured_decisions_+=t.decision_limit;
    inventory_.tiers[i]=t;inventory_.tiers[i].actual_attempts=0;tier_configured_[i]=1;return true;
}
bool ExpertPolicyCapture::record_transform(ExpertPolicyTransformKind kind,uint64_t tier,const uint64_t* ord,size_t n) noexcept {
    if(!building())return false;
    if((!configured_||tier_index(tier)==inventory_.tiers.size())||(!ord&&n)||
       inventory_.transforms.size()>=limits_.max_transforms||n>limits_.max_order_entries-inventory_.source_ordinals.size()-inventory_.ordered_pairs.size()) {fail();return false;}
    for(size_t i=0;i<n;++i)if(ord[i]>=input_receipt_->ranked.size()){fail();return false;}
    const uint64_t offset=inventory_.source_ordinals.size();
    for(size_t i=0;i<n;++i)inventory_.source_ordinals.push_back(ord[i]);
    inventory_.transforms.push_back({kind,tier,offset,uint64_t(n)});return true;
}
bool ExpertPolicyCapture::record_transform_pairs(ExpertPolicyTransformKind kind,uint64_t tier,const uint64_t* ord,
    const std::pair<int32_t,int32_t>* pairs,size_t n) noexcept {
    if(!building())return false;
    if(!configured_||tier_index(tier)==inventory_.tiers.size()||((!ord||!pairs)&&n)||
       inventory_.transforms.size()>=limits_.max_transforms||
       n>limits_.max_order_entries-inventory_.source_ordinals.size()-inventory_.ordered_pairs.size()){fail();return false;}
    for(size_t i=0;i<n;++i){
        if(pairs[i].first<0||pairs[i].second<0||uint64_t(pairs[i].first)>=input_receipt_->layers()||
           uint64_t(pairs[i].second)>=input_receipt_->experts()){fail();return false;}
        if(ord[i]!=expert_policy_synthetic_ordinal&&(ord[i]>=input_receipt_->ranked.size()||
           input_receipt_->ranked[size_t(ord[i])]!=pairs[i])){fail();return false;}
    }
    const uint64_t offset=inventory_.ordered_pairs.size();
    for(size_t i=0;i<n;++i)inventory_.ordered_pairs.push_back({ord[i],pairs[i].first,pairs[i].second});
    inventory_.transforms.push_back({kind,tier,offset,uint64_t(n),true});return true;
}
bool ExpertPolicyCapture::record_placement(const ExpertPolicyPlacement& p) noexcept {
    if(!building())return false;
    const size_t i=tier_index(p.tier_id);
    if(!configured_||i==inventory_.tiers.size()||!tier_configured_[i]||tier_complete_[i]||
       decision_counts_[i]>=inventory_.tiers[i].decision_limit||inventory_.decisions.size()>=limits_.max_decisions){fail();return false;}
    const auto&t=inventory_.tiers[i];
    if(p.layer<t.layer_begin||p.layer>=t.layer_end||p.expert<0||uint64_t(p.expert)>=input_receipt_->experts()){fail();return false;}
    if(p.source_ordinal!=expert_policy_synthetic_ordinal){
        if(p.source_ordinal>=input_receipt_->ranked.size()||input_receipt_->ranked[size_t(p.source_ordinal)]!=std::pair<int32_t,int32_t>{p.layer,p.expert}){fail();return false;}
    }
    const bool resident=p.outcome==ExpertPolicyOutcome::copied||p.outcome==ExpertPolicyOutcome::already_resident;
    if(resident){
        if(p.slot<0||uint64_t(p.slot)>=t.slots||!p.bytes){fail();return false;}
        const ExpertPolicySlot* prior=nullptr;
        for(const auto& s:inventory_.current_slots)if(s.tier_id==p.tier_id&&(s.slot==p.slot||(s.layer==p.layer&&s.expert==p.expert))){prior=&s;break;}
        if(p.outcome==ExpertPolicyOutcome::already_resident){
            if(!prior||prior->slot!=p.slot||prior->layer!=p.layer||prior->expert!=p.expert||prior->bytes!=p.bytes){fail();return false;}
        }else{
            if(prior||inventory_.current_slots.size()>=limits_.max_slots){fail();return false;}
            inventory_.current_slots.push_back({p.tier_id,p.bytes,p.source_ordinal,p.slot,p.layer,p.expert});
        }
    }else if((p.outcome!=ExpertPolicyOutcome::skipped_full&&p.outcome!=ExpertPolicyOutcome::skipped_other)||p.slot!=-1||p.bytes){fail();return false;}
    inventory_.decisions.push_back(p);++decision_counts_[i];return true;
}
bool ExpertPolicyCapture::complete_tier(uint64_t id,uint64_t actual,ExpertPolicyTermination why) noexcept {
    if(!building())return false;
    const size_t i=tier_index(id);
    if(!configured_||i==inventory_.tiers.size()||!tier_configured_[i]||tier_complete_[i]||decision_counts_[i]!=actual){fail();return false;}
    const auto&t=inventory_.tiers[i];
    if(t.disabled ? (why!=ExpertPolicyTermination::disabled||actual!=0) :
       (why==ExpertPolicyTermination::disabled||
        (why==ExpertPolicyTermination::ranking_end&&actual!=t.decision_limit)||
        (why!=ExpertPolicyTermination::ranking_end&&why!=ExpertPolicyTermination::capacity_full))){fail();return false;}
    inventory_.tiers[i].actual_attempts=actual;inventory_.tiers[i].termination=why;
    tier_complete_[i]=1;return true;
}
bool ExpertPolicyCapture::publish_initial() noexcept {
    if(!building())return false;
    if(!configured_){fail();return false;}
    for(size_t i=0;i<inventory_.tiers.size();++i)if(!tier_complete_[i]){fail();return false;}
    inventory_.generation=1;state_=ExpertPolicyState::initial_complete;return true;
}
void ExpertPolicyCapture::begin_dynamic_change() noexcept {
    if(state_==ExpertPolicyState::failed)return;
    state_=ExpertPolicyState::stale;
    if(inventory_.generation!=UINT64_MAX)++inventory_.generation;
}
const ExpertPolicyInventory* ExpertPolicyCapture::completed_inventory()const noexcept {
    if(state_!=ExpertPolicyState::initial_complete||(inventory_.input_ranking->completed_inventory()!=input_receipt_||inventory_.input_ranking->attempt_generation()!=input_generation_))return nullptr;
    return &inventory_;
}
}
