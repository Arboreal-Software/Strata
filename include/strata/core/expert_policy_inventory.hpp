#pragma once
#include "strata/core/expert_profile_inventory.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::core {
inline constexpr uint64_t expert_policy_synthetic_ordinal=UINT64_MAX;
enum class ExpertPolicyTierKind : uint8_t { primary, stage, peer, remote };
enum class ExpertPolicyTransformKind : uint8_t { peer_order, stage_partition, remote_order };
enum class ExpertPolicyOutcome : uint8_t { copied, already_resident, skipped_full, skipped_other };
enum class ExpertPolicyState : uint8_t { building, initial_complete, stale, failed };
struct ExpertPolicyLimits {
    uint64_t max_tiers=0,max_transforms=0,max_order_entries=0,max_decisions=0,max_slots=0;
};
enum class ExpertPolicyTermination : uint8_t { ranking_end, capacity_full, disabled };
struct ExpertPolicyTierIdentity {uint64_t id=0;ExpertPolicyTierKind kind=ExpertPolicyTierKind::primary;};
struct ExpertPolicyTier {
    uint64_t id=0,slots=0,decision_limit=0;
    int device=-1;
    int64_t layer_begin=0,layer_end=0;
    uint64_t slot_guard_bytes=0;
    ExpertPolicyTierKind kind=ExpertPolicyTierKind::primary;
    bool per_layer=false,disabled=false;
    uint64_t actual_attempts=0;
    ExpertPolicyTermination termination=ExpertPolicyTermination::ranking_end;
};
struct ExpertPolicyTransform {
    ExpertPolicyTransformKind kind=ExpertPolicyTransformKind::peer_order;
    uint64_t tier_id=0,offset=0,count=0;
    bool pair_bearing=false;
};
struct ExpertPolicyOrderedPair {uint64_t source_ordinal=0;int32_t layer=0,expert=0;};
struct ExpertPolicyPlacement {
    uint64_t tier_id=0,source_ordinal=0,bytes=0;
    int32_t layer=0,expert=0,slot=-1;
    ExpertPolicyOutcome outcome=ExpertPolicyOutcome::skipped_other;
};
struct ExpertPolicySlot {
    uint64_t tier_id=0,bytes=0,source_ordinal=0;
    int32_t slot=-1,layer=0,expert=0;
};
// Immutable view while the retained capture is unchanged. Not concurrent
// execution authority; caller serializes all producer/accessor operations.
struct ExpertPolicyInventory {
    std::shared_ptr<const ReservationLease> reservation;
    std::shared_ptr<const ExpertProfileCapture> input_ranking;
    std::vector<ExpertPolicyTier> tiers;
    std::vector<ExpertPolicyTransform> transforms;
    std::vector<uint64_t> source_ordinals;
    std::vector<ExpertPolicyOrderedPair> ordered_pairs;
    std::vector<ExpertPolicyPlacement> decisions;
    std::vector<ExpertPolicySlot> current_slots;
    uint64_t generation=0;
    bool initial_only=true;
};
bool expert_policy_inventory_host_bound(const ExpertPolicyLimits&,uint64_t&,std::string&);
class ExpertPolicyCapture {
public:
    static bool create(const ExpertPolicyLimits&,std::shared_ptr<const ReservationLease>,
        std::shared_ptr<const ExpertProfileCapture>,std::unique_ptr<ExpertPolicyCapture>&,std::string&);
    bool declare_tiers(const ExpertPolicyTierIdentity*,size_t) noexcept;
    // Each tier is configured after its own successful capacity selection.
    // decision_limit bounds loop attempts, including duplicate ranking entries.
    bool configure_tier(const ExpertPolicyTier&) noexcept;
    // Transform hooks need declared IDs, but may precede capacity selection.
    bool record_transform(ExpertPolicyTransformKind,uint64_t tier_id,const uint64_t* ordinals,size_t) noexcept;
    // Required for synthetic tails; captures actual coordinates even for entries
    // the helper never scans. Known ordinals must match the input receipt pair.
    bool record_transform_pairs(ExpertPolicyTransformKind,uint64_t tier_id,const uint64_t* ordinals,
                                const std::pair<int32_t,int32_t>* pairs,size_t) noexcept;
    // copied means actual copy completion was checked, not merely enqueued/admitted.
    // Any partial/uncertain copy must call fail() rather than record a placement.
    bool record_placement(const ExpertPolicyPlacement&) noexcept;
    bool complete_tier(uint64_t tier_id,uint64_t actual_attempt_count,ExpertPolicyTermination) noexcept;
    bool publish_initial() noexcept;
    void fail() noexcept {state_=ExpertPolicyState::failed;}
    // Invoke BEFORE any dynamic mutation, including direct host_res updates.
    // This initial-only slice never certifies dynamic completion.
    void begin_dynamic_change() noexcept;
    const ExpertPolicyInventory* completed_inventory()const noexcept;
    ExpertPolicyState state()const noexcept{return state_;}
private:
    bool building() noexcept;
    size_t tier_index(uint64_t)const noexcept;
    ExpertPolicyLimits limits_;
    ExpertPolicyInventory inventory_;
    std::vector<uint64_t> decision_counts_;
    std::vector<uint8_t> tier_complete_,tier_configured_;
    uint64_t configured_slots_=0,configured_decisions_=0;
    const ExpertProfileInventory* input_receipt_=nullptr;
    uint64_t input_generation_=0;
    ExpertPolicyState state_=ExpertPolicyState::building;
    bool configured_=false;
};
}
