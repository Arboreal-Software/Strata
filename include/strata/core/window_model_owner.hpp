#pragma once
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/window_budget.hpp"

namespace strata::core {
struct ModelStageResources {
    // Retain actual immutable input bindings even when this stage outlives its gang.
    // First member: released after weights/native/head and retained by quarantine.
    std::shared_ptr<const void> input_owner;
    // Actual layout/sign receipt authority, retained through independent stages.
    // Retention does not serialize process-global layout mutations.
    std::shared_ptr<const void> expert_layout_owner;
    std::shared_ptr<const void> expert_profile_owner;
    int device=-1;
    void* arena=nullptr;
    uint64_t arena_bytes=0;
    std::unique_ptr<WeightTable> weights;
    std::unique_ptr<NativeDense> native;
    std::unique_ptr<NativeHead> head;
    std::shared_ptr<const ReservationLease> reservation;
};
class OwnedModelStage {
public:
    const WeightTable& weights() const {return *resources_.weights;}
    const NativeHead* head() const {return resources_.head.get();}
    const NativeDense* native() const {return resources_.native.get();}
    const std::shared_ptr<const ReservationLease>& reservation() const {return resources_.reservation;}
    int device() const {return resources_.device;}
    bool healthy() const;
    bool owns_weight_span(const void*,uint64_t bytes,bool native) const;
private:
    friend bool adopt_model_stage(ModelStageResources&,std::shared_ptr<const OwnedModelStage>&,std::string&);
    static void destroy(OwnedModelStage*) noexcept;
    ModelStageResources resources_;
    OwnedModelStage* next_=nullptr;
};
// Internal runtime API. Caller exclusively owns all input allocations, has
// finished loading, and reserved their full charge (not duplicated in baseline).
// This checks only a lower bound; pricing, model identity and device provenance
// are trusted runtime responsibilities. Refusal leaves input/output untouched.
bool adopt_model_stage(ModelStageResources&,std::shared_ptr<const OwnedModelStage>&,std::string&);
uint64_t model_stage_quarantine_count();

struct ModelGangResources {
    // Genuine verified loader binding owner; destroyed after every stage/map.
    std::shared_ptr<const void> input_owner;
    std::shared_ptr<const void> expert_layout_owner;
    std::shared_ptr<const void> expert_profile_owner;
    std::vector<std::shared_ptr<const OwnedModelStage>> stages;
    std::unique_ptr<NativeEmbed> embedding;
    std::shared_ptr<const ReservationLease> embedding_reservation;
};
// Every session, graph, and asynchronous user must retain this gang. The shared
// mapped embedding can be read by any stage and must outlive all stage work.
class OwnedModelGang {
public:
    const auto& stages() const { return resources_.stages; }
    const NativeEmbed* embedding() const { return resources_.embedding.get(); }
    bool healthy() const;
private:
    friend bool adopt_model_gang(ModelGangResources&,std::shared_ptr<const OwnedModelGang>&,std::string&);
    static void destroy(OwnedModelGang*) noexcept;
    ModelGangResources resources_;
    OwnedModelGang* next_=nullptr;
};
// Caller has stopped external producers and transfers exclusive embedding
// ownership. Stage owners may be shared; the gang retains them through its
// all-device drain. This does not publish a global embedding or execution state.
bool adopt_model_gang(ModelGangResources&,std::shared_ptr<const OwnedModelGang>&,std::string&);
uint64_t model_gang_quarantine_count();
// Caller seals submissions and joins enqueue-capable host producers first.
// Drains device users while the caller still retains borrowed callback state.
bool drain_model_gang_users(const OwnedModelGang&,std::string& error);
} // namespace strata::core
