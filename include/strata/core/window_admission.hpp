#pragma once
#include "strata/core/thread_dependencies.hpp"
#include "strata/core/window_manager.hpp"
#include <array>
#include <span>

namespace strata::core {
enum class AdmissionUnit { metadata, setup, staging, prefill, decode, repair, save, return_work, count };
enum class AdmissionStatus { ready, no_work, routing_needed, invalid, stale, capacity, exhausted,
    budget, allocation_failed, acknowledgement_pending, acknowledgement_blocked, unresolved, released };
struct AdmissionLimits {
    size_t max_candidates=0, max_outstanding=0;
    // Conservative caller-priced envelopes, not exact allocator/RSS caps.
    uint64_t workspace_bytes=0, record_bytes=0, last_admission_cut=0;
    std::array<uint64_t,static_cast<size_t>(AdmissionUnit::count)> max_units{};
};
struct AdmissionCandidate {
    OperationKey operation;
    ManagedWindowToken window;
    WindowLaneToken lane;
    uint64_t unit_serial=0, amount=0;
    AdmissionUnit unit=AdmissionUnit::metadata;
    bool demand=false, resource_ready=false, idle_authorized=false, foreground_return_protected=false;
    // Trusted caller's actual capacity/return authorization owner, not authentication.
    std::shared_ptr<const void> authorization;
};
struct AdmissionDecision {
    AdmissionStatus status=AdmissionStatus::no_work;
    size_t candidate=0;
    uint64_t graph_revision=0;
    std::optional<uint64_t> next_expiry_ms;
};
struct WindowAdmissionRecord;
class WindowAdmissionHandle {
public:
    AdmissionStatus status() const;
    uint64_t cut() const;
    const DependencyOperation* priority() const;
    const ThreadPrioritySnapshot* inherited_source() const;
    // Only ready units expose selected borrowing/execution. Recovery never does.
    const ManagedWindowWork* work() const;
    WindowWork recovery_lease() const;
    const ManagedWindowToken* token() const;
private:
    friend class WindowAdmission;
    std::shared_ptr<WindowAdmissionRecord> record_;
};
// One externally serialized reactor owns ALL priorities/graph/manager/ledger calls.
// Strict priority is an explicit initial adapter, not a fairness guarantee.
// No completion, authentication, resource-pricing or physical preemption is inferred.
class WindowAdmission {
public:
    WindowAdmission(AdmissionLimits, ThreadPriorities&, ThreadDependencies&, WindowManager&, WindowBudget&);
    // Records bind to this exact reactor address; ownership cannot be copied or relocated.
    WindowAdmission(const WindowAdmission&)=delete;
    WindowAdmission& operator=(const WindowAdmission&)=delete;
    WindowAdmission(WindowAdmission&&)=delete;
    WindowAdmission& operator=(WindowAdmission&&)=delete;
    AdmissionDecision refresh(uint64_t now_ms);
    AdmissionDecision admit(uint64_t now_ms, std::span<const AdmissionCandidate>,
                            bool other_nonmaintenance_demand, WindowAdmissionHandle&);
    AdmissionStatus recover_acknowledgement(const WindowAdmissionHandle&);
    AdmissionStatus release(const WindowAdmissionHandle&);
    bool outstanding(size_t slot, WindowAdmissionHandle&) const;
private:
    AdmissionStatus acknowledge(const std::shared_ptr<WindowAdmissionRecord>&);
    bool pending() const;
    AdmissionLimits limits_;
    ThreadPriorities& priorities_;
    ThreadDependencies& dependencies_;
    WindowManager& manager_;
    WindowBudget& budget_;
    std::shared_ptr<const ReservationLease> workspace_;
    std::vector<std::shared_ptr<WindowAdmissionRecord>> records_;
    DependencySnapshot graph_;
    uint64_t cut_=0, now_=0;
    std::optional<uint64_t> next_expiry_;
};
} // namespace strata::core
