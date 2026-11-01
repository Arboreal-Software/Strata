#pragma once
#include "strata/core/expert_policy_runtime.hpp"
#include "strata/core/remote_expert_opt.hpp"

namespace strata::core {
struct ExpertSourceInventory;
class ExpertSourceCapture;
class ExpertLayoutCapture;
struct ExpertLayoutInventory;
struct RemotePolicyPeerBinding { RemoteExperts* helper=nullptr; uint64_t tier=0; int device=-1; };
struct RemotePolicyPublicationView { uint64_t revision=0,active_transactions=0; bool unknown=false,closed=false; };
struct RemotePolicyPublicationLimits {
    uint64_t max_peers=0,max_changes_per_peer=0,max_total_changes=0,max_host_bytes=0;
    ExpertPolicyRuntimeLimits tracker;
};
// Trusted internal driver validator: checks the concrete runtime's actual source,
// optimizer and helper ownership AND optimizer attach-order -> policy-tier map.
// Not a user-provided boolean or authentication mechanism. No allocation/effects.
using RemotePolicyOwnershipValidator = bool (*)(const NativeRuntimeAttachment&,
    const ExpertSource&,const RemoteExpertOpt&,std::span<const RemotePolicyPeerBinding>) noexcept;

// External to the runtime. One reactor transfers exclusive ownership to its adapt
// worker; no simultaneous view/mutation/destruction or self-delete callbacks.
// This only updates existing remote placement cells, NOT whole-policy readiness.
// While attempted or unknown work remains, callers must not close/reopen the
// source, replace layout/Hadamard state, or retry ranking capture. Keep this
// configuration frozen until genuine teardown resolves the transactions.
// Quarantine retains the actual current runtime; it does not independently
// preserve arbitrary old captures detached by forbidden reconfiguration.
class RemotePolicyPublication final : private RemoteExpertOpt::AdaptPublication {
public:
    ~RemotePolicyPublication() noexcept;
    RemotePolicyPublication(const RemotePolicyPublication&)=delete;
    RemotePolicyPublication& operator=(const RemotePolicyPublication&)=delete;
    RemotePolicyPublication(RemotePolicyPublication&&)=delete;
    RemotePolicyPublication& operator=(RemotePolicyPublication&&)=delete;
    static bool required_bytes(const RemotePolicyPublicationLimits&,uint64_t&) noexcept;
    static bool create(std::shared_ptr<const ExpertPolicyCapture>,
        std::shared_ptr<NativeRuntimeAttachment>,std::shared_ptr<ExpertSource>,
        std::shared_ptr<const ExpertSourceCapture>,
        RemoteExpertOpt&,std::span<const RemotePolicyPeerBinding>,uint64_t owner_incarnation,
        const RemotePolicyPublicationLimits&,WindowBudget,RemotePolicyOwnershipValidator,
        std::unique_ptr<RemotePolicyPublication>&,std::string&);
    RemoteExpertOpt::AdaptResult adapt(const std::vector<float>& usage,
        const std::vector<int32_t>& primary,const std::vector<std::pair<int32_t,int32_t>>& pending,
        int max_swaps,RemoteExpertOpt::OwnedAdaptWorkspace&,std::string&);
    RemotePolicyPublicationView view() const noexcept;
    bool remote_cell(uint32_t peer,int32_t slot,ExpertPlacementValue& host,ExpertPlacementValue& completed) const noexcept;
    // Explicit checked-teardown trust boundary inherited from tracker. Never
    // resets healthy policy; actual runtime and incarnation must match exactly.
    bool resolve_after_teardown(const std::shared_ptr<NativeRuntimeAttachment>&,uint64_t) noexcept;
private:
    RemotePolicyPublication()=default;
    struct Peer { ExpertPlacementToken token;size_t start=0,count=0;bool attempted=false,published=false,unknown=false; };
    bool authority() const noexcept;
    const ExpertSourceInventory* source_inventory() const noexcept;
    std::shared_ptr<const ExpertSourceCapture> source_capture_owner() const noexcept;
    bool prepare_all(const RemoteExpertOpt::AdaptPlanEntry*,size_t) override;
    bool begin_attempt(uint32_t) override;
    bool publish_after_copies(uint32_t,const RemoteExpertOpt::AdaptPlanEntry*,size_t) override;
    void abandon_unattempted() noexcept override;
    void fail_unknown(uint32_t) noexcept override;
    void finish_call() noexcept;
    std::shared_ptr<const ReservationLease> reservation_; // outlives all arrays
    std::shared_ptr<const ExpertPolicyCapture> capture_;
    std::shared_ptr<NativeRuntimeAttachment> runtime_;
    std::shared_ptr<ExpertSource> source_;
    std::unique_ptr<ExpertPolicyRuntime> tracker_;
    std::shared_ptr<const ExpertProfileCapture> ranking_;
    const ExpertProfileInventory* ranking_receipt_=nullptr;
    std::shared_ptr<const ExpertSourceCapture> source_capture_;
    std::shared_ptr<const ExpertLayoutCapture> layout_capture_;
    const ExpertSourceInventory* source_receipt_=nullptr;
    const ExpertLayoutInventory* layout_receipt_=nullptr;
    uint64_t layout_generation_=0,hadamard_generation_=0;
    uint64_t ranking_generation_=0,owner_incarnation_=0;
    RemoteExpertOpt* opt_=nullptr;
    RemotePolicyOwnershipValidator owns_=nullptr;
    RemotePolicyPublicationLimits limits_;
    size_t peer_count_=0;
    std::unique_ptr<RemotePolicyPeerBinding[]> bindings_;
    std::unique_ptr<Peer[]> peers_;
    std::unique_ptr<RemoteExpertOpt::AdaptPlanEntry[]> plans_;
    std::unique_ptr<ExpertPlacementChange[]> changes_;
    bool in_call_=false,prepared_=false,poisoned_=false;
};
}
