#pragma once
#include "strata/core/window_manager.hpp"
#include <span>

namespace strata::core {
struct WindowTokenLedgerState;
struct WindowTokenCandidateState;
struct WindowTokenImportState;
struct WindowTokenCaptureStorage;
enum class WindowTokenStatus { ok, invalid, stale, busy, limit, exhausted, allocation_failed,
                               awaiting_sample, unresolved, cancelled, inactive, budget_refused };
enum class WindowTokenPhase { awaiting_sample, ready, unresolved, committed, cancelled };
struct WindowTokenLedgerLimits {
    uint64_t max_tokens=0;
    uint32_t stages=0;
    // Monotonic serial high-water seed, never wrap or reuse within a ledger.
    uint64_t candidate_serial_seed=0;
};
struct WindowTokenLedgerView {
    WindowKey key;
    StateCut cut;
    uint64_t pending_serial=0;
    std::optional<WindowTokenPhase> phase;
    bool active=true;
    uint64_t import_serial=0;
};
struct WindowTokenCandidateView {
    uint64_t serial=0, operation=0;
    ManagedWindowToken base;
    WindowKey proposed_key;
    StateCut proposed_cut;
    WindowTokenPhase phase=WindowTokenPhase::unresolved;
    std::optional<int32_t> output;
};
class WindowTokenCandidate {
public:
    explicit operator bool() const { return bool(state_); }
private:
    friend class WindowTokenLedger;
    std::shared_ptr<WindowTokenCandidateState> state_;
};
// External transaction ownership: never retained by a runtime-owned ledger.
// Dropping a published unresolved ticket does NOT reactivate either ledger.
class WindowTokenImport {
public:
    WindowTokenImport()=default;
    ~WindowTokenImport();
    WindowTokenImport(WindowTokenImport&&) noexcept;
    WindowTokenImport& operator=(WindowTokenImport&&) noexcept;
    WindowTokenImport(const WindowTokenImport&)=delete;
    WindowTokenImport& operator=(const WindowTokenImport&)=delete;
    explicit operator bool() const { return bool(state_); }
    const WindowRead& old_read() const;
    const BundleHandle& source() const;
private:
    friend class WindowTokenLedger;
    void reset() noexcept;
    std::shared_ptr<WindowTokenImportState> state_;
};
// External immutable history/read owner for capture_native_window. Do not store
// this handle inside the runtime retained by its own selected read.
class WindowTokenCapture {
public:
    explicit operator bool() const { return bool(storage_); }
    bool usable() const;
    const ManagedWindowToken& token() const;
    const SnapshotDescriptor& metadata() const;
    const SelectedWindowRead& selected_read() const;
private:
    friend class WindowTokenLedger;
    std::shared_ptr<const WindowTokenCaptureStorage> storage_;
};
struct WindowTokenReceipt {
    WindowKey key;
    StateCut cut;
    std::optional<int32_t> output;
};
// Exact initial or validated same-window restored ledger.
// One reactor MUST serialize manager admission/control and ledger finish. An
// unexpectedly advanced post-finish snapshot retains an unresolved candidate;
// it never rolls back device state or authorizes output. No method emits tokens.
class WindowTokenLedger {
public:
    WindowTokenLedger(const WindowKey&, const StateCut&, WindowTokenLedgerLimits, WindowBudget&);
    WindowTokenLedger(const WindowTokenLedger&)=delete;
    WindowTokenLedger& operator=(const WindowTokenLedger&)=delete;
    WindowTokenLedgerView view() const;
    WindowTokenStatus prepare_import(WindowManager&, const ManagedWindowToken& old_entry,
        const BundleHandle&, const WindowKey& expected_snapshot, const WindowStateLayout&,
        WindowTokenLedgerLimits, WindowBudget&, std::unique_ptr<WindowTokenLedger>& new_ledger,
        WindowTokenImport&);
    WindowTokenStatus activate_import(WindowTokenImport&, WindowManager&,
        const ManagedWindowToken& selected, const PreparedNativeWindow&);
    WindowTokenStatus abort_import(WindowTokenImport&, WindowManager&);
    WindowTokenStatus export_capture(WindowManager&, const ManagedWindowToken&, bool steering_enabled,
        WindowBudget&, WindowTokenCapture&, const BundleHandle& approximation_provenance = {});
    WindowTokenStatus inspect(const WindowTokenCandidate&, WindowTokenCandidateView&) const;
    WindowTokenStatus pending_candidate(WindowTokenCandidate&) const;
    // Synchronous spans only; no pointer retention. Callback executes outside
    // mutex; mutation refuses while any borrow is active, including unwinding.
    WindowTokenStatus borrow_published(const std::function<void(std::span<const int32_t>,size_t pending_begin)>&) const;
    WindowTokenStatus borrow_candidate(const WindowTokenCandidate&,
        const std::function<void(std::span<const int32_t>,size_t pending_begin)>&) const;
    bool matches_history_prefix(std::span<const int32_t>) const;
    WindowTokenStatus stage_accept(const ManagedWindowToken&, const ManagedWindowWork&,
                                  std::span<const int32_t> suffix, WindowTokenCandidate&);
    WindowTokenStatus stage_forward(const ManagedWindowToken&, const ManagedWindowWork&,
                                   uint64_t count, WindowTokenCandidate&);
    WindowTokenStatus stage_decode(const ManagedWindowToken&, const ManagedWindowWork&, WindowTokenCandidate&);
    // Called after sampling, before finish. Failure after device work retains
    // the candidate; retrying the SAME sample can finish its canonical hash.
    WindowTokenStatus record_sample(const WindowTokenCandidate&, int32_t);
    WindowTokenStatus finish(const WindowTokenCandidate&, WindowManager&, const std::vector<StateCut>& actual_stage_cuts,
                             bool host_joined, bool devices_io_drained, WindowTokenReceipt&);
    // Cancellation/teardown is performed on the original work by the driver.
    // This verifies terminal unchanged state before discarding unpublished suffix.
    WindowTokenStatus reconcile_terminal(const WindowTokenCandidate&, WindowManager&);
private:
    explicit WindowTokenLedger(std::shared_ptr<WindowTokenLedgerState> state):state_(std::move(state)){}
    WindowTokenStatus stage(const ManagedWindowToken&, const ManagedWindowWork&, std::span<const int32_t>,
                            uint64_t forwarded, bool decode, WindowTokenCandidate&);
    std::shared_ptr<WindowTokenLedgerState> state_;
};
} // namespace strata::core
