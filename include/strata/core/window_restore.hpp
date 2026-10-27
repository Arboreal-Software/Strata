#pragma once
#include "strata/core/window_native.hpp"
#include "strata/core/window_budget.hpp"
#include <functional>
#include <mutex>

namespace strata::core {
struct SessionState;
class NativeSessionAllocation;
// Internal allocator contract: all mutable session buffers and graph bindings are
// fresh, exclusively owned by owner, and cannot alias any selected/shared window.
// Shared immutable weights are allowed. This is not a client-supplied descriptor.
struct NativePrivateStage {
    int device = -1;
    SessionState* session = nullptr;
    std::shared_ptr<const void> owner;
    // Rebuild graph/maps/residency bindings for THIS private session, never active
    // bindings. Must join host producers before returning; GPU work is drained below.
    std::function<bool(std::string&)> bind_private;
};
// Core-neutral runtime owner; derived driver objects retain all bound consumers,
// model ownership and their reservations. Access only through selected admission.
// Must never retain its PreparedNativeWindow/storage (an ownership cycle).
// If retained independently, must itself retain every session allocation it borrows.
struct NativeRuntimeAttachment { virtual ~NativeRuntimeAttachment() = default; };
struct NativeSessionView { int device = -1; SessionState* session = nullptr; };
struct NativeReadSessionView { int device = -1; const SessionState* session = nullptr; };
struct NativeRestoreStorage;
class WindowExecution;
class PreparedNativeWindow {
public:
    explicit operator bool() const { return bool(storage_); }
    // Null for fresh initial windows: no serialized snapshot has been produced.
    const SnapshotDescriptor* descriptor() const;
private:
    friend struct NativeRestoreAccess;
    friend class WindowSelection;
    std::shared_ptr<WindowExecution> execution() const;
    std::vector<NativeSessionView> session_views() const;
    std::shared_ptr<NativeRuntimeAttachment> runtime_attachment() const;
    bool claim_selection() const;
    void release_selection_claim() const;
    std::shared_ptr<NativeRestoreStorage> storage_;
};
class NativeRestoreFailure {
public:
    void retire_after_teardown(); // caller confirms all affected contexts torn down and host/IO joined
private:
    friend struct NativeRestoreAccess;
    std::mutex mutex_;
    std::shared_ptr<NativeRestoreStorage> storage_;
};
enum class NativeRestoreStatus { prepared, invalid, budget_refused, unhealthy };
struct NativeRestoreResult {
    NativeRestoreStatus status = NativeRestoreStatus::invalid;
    std::string detail;
    std::shared_ptr<NativeRestoreFailure> failure{};
    explicit operator bool() const { return status == NativeRestoreStatus::prepared; }
};
// Does not publish or write selected state. Caller has already reserved all private
// device/host allocations in destination_budget and built stages from its allocator.
// A separate host staging reservation is acquired before materializing source bytes.
NativeRestoreResult prepare_native_restore(const BundleHandle&, const WindowKey& expected,
    const ModelGeometry&, std::vector<NativePrivateStage>,
    std::shared_ptr<const ReservationLease> destination_budget, WindowBudget& host_budget,
    uint64_t max_tokens, uint64_t max_payload_bytes, bool steering_enabled, PreparedNativeWindow& out,
    std::shared_ptr<NativeRuntimeAttachment> runtime_attachment = {});
// Factory-backed entry point. Each binder receives the corresponding fresh
// session; it must own/reserve its runtime resources and must not change global
// selected bindings. The allocation and all stage owners survive preparation.
using NativeSessionBinder = std::function<bool(SessionState&,std::string&)>;
NativeRestoreResult prepare_owned_native_restore(const BundleHandle&, const WindowKey& expected,
    const ModelGeometry&, const NativeSessionAllocation&, const std::vector<NativeSessionBinder>&,
    WindowBudget& host_budget, uint64_t max_tokens, uint64_t max_payload_bytes,
    bool steering_enabled, PreparedNativeWindow& out,
    std::shared_ptr<NativeRuntimeAttachment> runtime_attachment = {});
// Fresh initial selection only: first cut id=1, zero token counts/frontiers,
// canonical empty-history digest; no fabricated snapshot/continuation.
// Factory destinations must be private and unclaimed. Reserves metadata before
// zero/bind/drain; uncertain work poisons allocation and retains failure owners.
// Descriptorless windows cannot be used as WindowSelection replacements.
NativeRestoreResult prepare_owned_native_initial(const WindowKey&, const ModelGeometry&,
    const NativeSessionAllocation&, const std::vector<NativeSessionBinder>&,
    WindowBudget& host_budget, uint64_t max_tokens, uint64_t max_payload_bytes,
    bool steering_enabled, PreparedNativeWindow& out,
    std::shared_ptr<NativeRuntimeAttachment> runtime_attachment = {});
} // namespace strata::core
