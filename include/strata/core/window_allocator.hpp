#pragma once
#include "strata/core/session.hpp"
#include "strata/core/window_budget.hpp"
#include <memory>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {
struct NativeSessionRequest {
    int device = -1;
    int64_t layer_begin = 0, layer_end = 0;
    bool ple_buffers = false;
    bool ple_native_key = false;
    // Exact quote before parent admission; factory is trusted HOST ONLY. It may
    // reserve metadata and physical private_attachment leases under the supplied
    // genuine arena parent, but cannot discover/create/map any device resources.
    std::vector<QsaElasticPlan> elastic_plans;
    std::function<bool(const std::vector<QsaElasticPlan>&,WindowBudget&,
        const std::shared_ptr<const ReservationLease>&,
        std::vector<std::shared_ptr<QsaElasticView>>&,std::string&)> prepare_elastic;

};
struct OwnedNativeSession {
    int device = -1;
    SessionState* session = nullptr;
    std::shared_ptr<const void> owner;
};
struct NativeAllocationResult;
class NativeSessionAllocation {
public:
    NativeSessionAllocation() = default;
    const std::vector<OwnedNativeSession>& stages() const { return stages_; }
    const std::shared_ptr<const ReservationLease>& reservation() const { return reservation_; }
    bool healthy() const { return healthy_ && healthy_(); }
    void mark_unhealthy() const { if (poison_) poison_(); }
private:
    friend NativeAllocationResult allocate_native_sessions(const ModelGeometry&,int64_t,int64_t,
        const std::vector<NativeSessionRequest>&,WindowBudget&,NativeSessionAllocation&);
    friend NativeAllocationResult allocate_native_sessions(const ModelGeometry&,int64_t,int64_t,
        const std::vector<NativeSessionRequest>&,std::vector<std::unique_ptr<SessionState>>&,WindowBudget&,NativeSessionAllocation&);
    std::vector<OwnedNativeSession> stages_;
    std::shared_ptr<const ReservationLease> reservation_;
    std::function<bool()> healthy_;
    std::function<void()> poison_;
};
enum class NativeAllocationStatus { allocated, invalid, budget_refused, unhealthy };
struct NativeAllocationResult {
    NativeAllocationStatus status = NativeAllocationStatus::invalid;
    std::string detail;
    explicit operator bool() const { return status == NativeAllocationStatus::allocated; }
};
// Internal runtime API: validated geometry and stable global KV configuration.
// Reserves the entire gang before allocating. Creates fresh session arenas,
// pinned QSA resources and optional PLE embedding/scratch buffers. PLE readers,
// immutable weights, graphs and expert bindings must be separately reserved and
// installed before execution; these sessions cannot yet execute a model.
NativeAllocationResult allocate_native_sessions(const ModelGeometry&, int64_t max_cells, int64_t k,
    const std::vector<NativeSessionRequest>&, WindowBudget&, NativeSessionAllocation& out);
// Transfers caller-created, fresh unbound sessions without changing their addresses.
// Validation/reservation/host-handle refusal leaves every input and out unchanged.
// Once all handles are prepared, all input elements become null before device work;
// later failure retains every transferred session and reservation in quarantine.
// No existing device buffers, IO readers, PLE weights or scratch may be bound.
NativeAllocationResult allocate_native_sessions(const ModelGeometry&, int64_t max_cells, int64_t k,
    const std::vector<NativeSessionRequest>&, std::vector<std::unique_ptr<SessionState>>& fresh,
    WindowBudget&, NativeSessionAllocation& out);
// Failed initialization or failed release retains allocations AND accounting
// until worker restart. This count is an operational fault signal, not a cache.
uint64_t native_session_quarantine_count();
} // namespace strata::core
