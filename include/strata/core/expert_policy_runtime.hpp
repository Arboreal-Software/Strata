#pragma once
#include "strata/core/expert_policy_inventory.hpp"
#include <span>
namespace strata::core {
struct NativeRuntimeAttachment;
struct ExpertPolicyRuntimeLimits {
    uint64_t max_tiers=0,max_slots=0,max_transactions=0,max_changes=0,max_dependencies=0,max_host_bytes=0;
};
struct ExpertPlacementToken {
    uint64_t tracker=0,owner=0,operation=0;
    bool operator==(const ExpertPlacementToken&)const=default;
};
struct ExpertPlacementValue {
    bool present=false;
    int32_t layer=0,expert=0;
    uint64_t source_ordinal=0,bytes=0;
    bool operator==(const ExpertPlacementValue&)const=default;
};
struct ExpertPlacementCell {uint64_t tier=0;int32_t slot=-1;ExpertPlacementValue value;};
struct ExpertPlacementChange {
    uint64_t tier=0;int32_t slot=-1;
    ExpertPlacementValue expected,proposed;
    bool requires_copy=true;
};
enum class ExpertPlacementOrder : uint8_t { consuming_stream, per_layer_gate };
// Trusted driver receipt for a real retained event/gate/stream dependency. No
// device operation is performed or completion inferred by this component.
struct ExpertPlacementDependency {
    uint64_t id=0,generation=0,tier=0,work_scope=0;
    int device=-1;
    ExpertPlacementOrder kind=ExpertPlacementOrder::consuming_stream;
    std::shared_ptr<const void> owner;
};
struct ExpertPlacementProgress {
    bool attempted=false,copy_complete=false,host_published=false,device_ordered=false;
    bool placement_committed=false,reclaim_complete=false,unknown=false;
};
struct ExpertPlacementTransactionView {
    std::span<const ExpertPlacementChange> changes;
    std::span<const ExpertPlacementDependency> dependencies;
    std::span<const uint8_t> dependencies_ordered;
    ExpertPlacementProgress phase;
};
struct ExpertPlacementView {
    std::span<const ExpertPlacementCell> host,completed;
    uint64_t revision=0,active_transactions=0;
    bool stable=false,unknown=false,closed=false;
};
// External to retained runtime: runtime must not own this tracker. All calls,
// reads and underlying cache mutations are serialized by one trusted reactor.
// Borrowed views expire at next mutation. Not per-layer execution authority.
// Destruction with attempted/unknown unreclaimed work quarantines retained
// owners and budget for process lifetime, and refuses new trackers in worker.
// Dependency-release callbacks may query or use other non-retiring records,
// but must not delete/replace this tracker during a method or reenter its
// destructor. The external reactor retains unique ownership across each call.
uint64_t expert_policy_runtime_quarantine_count() noexcept;
class ExpertPolicyRuntime {
public:
    ~ExpertPolicyRuntime() noexcept;
    ExpertPolicyRuntime(const ExpertPolicyRuntime&)=delete;
    ExpertPolicyRuntime& operator=(const ExpertPolicyRuntime&)=delete;
    ExpertPolicyRuntime(ExpertPolicyRuntime&&)=delete;
    ExpertPolicyRuntime& operator=(ExpertPolicyRuntime&&)=delete;
    static bool create(std::shared_ptr<const ExpertPolicyCapture>,std::shared_ptr<NativeRuntimeAttachment>,
        uint64_t owner_incarnation,const ExpertPolicyRuntimeLimits&,WindowBudget,
        std::unique_ptr<ExpertPolicyRuntime>&,std::string&);
    bool begin(std::span<const ExpertPlacementChange>,std::span<const ExpertPlacementDependency>,ExpertPlacementToken&) noexcept;
    bool mark_attempted(ExpertPlacementToken) noexcept; // BEFORE first external effect
    bool copy_completed(ExpertPlacementToken) noexcept; // ALL copies actually complete
    bool publish_host(ExpertPlacementToken) noexcept;
    bool dependency_ordered(ExpertPlacementToken,uint64_t id,uint64_t generation) noexcept;
    bool device_tables_ordered(ExpertPlacementToken) noexcept;
    bool reclamation_completed(ExpertPlacementToken) noexcept;
    bool release_completed(ExpertPlacementToken) noexcept;
    bool abandon_unattempted(ExpertPlacementToken) noexcept;
    bool fail_unknown(ExpertPlacementToken) noexcept;
    bool progress(ExpertPlacementToken,ExpertPlacementProgress&)const noexcept;
    ExpertPlacementView view()const noexcept;
    bool transaction(ExpertPlacementToken,ExpertPlacementTransactionView&)const noexcept;
    // Exact owner identity (get AND control block) and incarnation required.
    // Caller has joined producers and physically torn down every device user.
    // Permanently closes tracker; never restores a healthy placement claim.
    bool resolve_after_teardown(const std::shared_ptr<NativeRuntimeAttachment>&,uint64_t) noexcept;
private:
    struct Retention;
    std::unique_ptr<Retention> retention_; // allocated before any externally attempted work
    bool unsafe_to_destroy()const noexcept;
    struct Transaction {ExpertPlacementToken token;ExpertPlacementProgress phase;size_t changes=0,dependencies=0;bool active=false,retiring=false;};
    ExpertPolicyRuntime()=default;
    bool authority()const noexcept;
    Transaction* find(ExpertPlacementToken) noexcept;
    const Transaction* find(ExpertPlacementToken)const noexcept;
    size_t cell(uint64_t,int32_t)const noexcept;
    size_t index(const Transaction&)const noexcept;
    bool tick() noexcept;
    bool covered(const Transaction&)const noexcept;
    void commit_if_ready(Transaction&) noexcept;
    void clear(Transaction&) noexcept;
    std::shared_ptr<const ReservationLease> reservation_;
    std::shared_ptr<const ExpertPolicyCapture> capture_;
    std::shared_ptr<NativeRuntimeAttachment> owner_;
    std::shared_ptr<const ExpertProfileCapture> ranking_;
    const ExpertProfileInventory* ranking_receipt_=nullptr;
    uint64_t ranking_generation_=0,initial_generation_=0;
    ExpertPolicyRuntimeLimits limits_;
    uint64_t incarnation_=0,owner_incarnation_=0,next_operation_=1,revision_=1;
    bool unknown_=false,closed_=false;
    std::vector<ExpertPolicyTier> tiers_;
    std::vector<ExpertPlacementCell> host_,completed_;
    std::vector<Transaction> transactions_;
    std::vector<ExpertPlacementChange> changes_;
    std::vector<ExpertPlacementDependency> dependencies_;
    std::vector<uint8_t> ordered_;
#if defined(STRATA_EXPERT_POLICY_RUNTIME_TEST)
    friend struct ExpertPolicyRuntimeTestAccess;
#endif
};
}
