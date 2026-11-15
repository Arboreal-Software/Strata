#pragma once

#include "strata/core/window_budget.hpp"
#include "strata/core/window_state.hpp"

namespace strata::core {

enum class WindowExecutionPhase { running, quiescing, quiesced, cancelling, aborted, unhealthy };
struct WindowExecutionView {
    WindowExecutionPhase phase;
    WindowKey key;
    StateCut cut;
    uint64_t operation;
    bool in_flight;
};
struct WindowExecutionState;
struct WindowReadHold;

// A synchronous capture hold. Copies keep execution quiesced until every copy
// is gone. Keep one alive through the final copy/event if capture is asynchronous.
class WindowRead {
public:
    explicit operator bool() const { return bool(hold_); }
    const WindowKey& key() const { return key_; }
    const StateCut& cut() const { return cut_; }
    bool usable() const;
    void mark_unhealthy() const;
private:
    friend class WindowExecution;
    std::shared_ptr<WindowReadHold> hold_;
    WindowKey key_;
    StateCut cut_;
};

// Copyable so both the execution owner and completion reactor can retain it.
// Destruction is NOT completion. An abandoned outstanding use retains its owner
// and budget until a completion handle drains it or confirms device teardown.
class WindowWork {
public:
    bool finish(uint64_t operation, const WindowKey& completed_key,
                const std::vector<StateCut>& stage_cuts,
                bool host_joined, bool devices_and_io_drained);
    bool mark_unhealthy(uint64_t operation);
    // Only after the runtime has destroyed/reset the affected device contexts and
    // joined CPU/IO producers. Never use a CUDA error itself as proof of teardown.
    bool retire_after_teardown(uint64_t operation);
    // Exact retained execution only; a newer operation is conservatively unresolved.
    bool resolved() const;
    uint64_t operation() const { return operation_; }
    explicit operator bool() const { return bool(state_); }
private:
    friend class WindowExecution;
    std::shared_ptr<WindowExecutionState> state_;
    uint64_t operation_ = 0;
};

class WindowExecution {
public:
    // allocation_owner owns every session/graph/buffer touched by submitted work;
    // reservations account for those allocations. Both remain held through drain.
    WindowExecution(WindowKey, StateCut, uint32_t stages,
                    std::shared_ptr<const void> allocation_owner,
                    std::shared_ptr<const ReservationLease> reservation);
    // Call before setup, first transfer or kernel. At most one mutation is admitted.
    bool admit(const WindowKey& expected, WindowWork& out, const StateCut* expected_cut = nullptr);
    bool begin_quiesce(const WindowKey& expected, const StateCut* expected_cut = nullptr);
    bool cancel(const WindowKey& expected, const StateCut* expected_cut = nullptr);
    bool acquire_read(const WindowKey& expected, WindowRead& out, const StateCut* expected_cut = nullptr);
    bool matches(const WindowKey&, const StateCut&) const;
    // Used only by selection's serialized replacement; atomically fences old
    // admission while verifying that the supplied read belongs to this gate.
    bool retire_for_replacement(const WindowRead&, const WindowKey&, const StateCut&);
    // Quiesced callers may read/capture only while retaining this gate, and must
    // finish those reads before resume. No asynchronous read may outlive that hold.
    bool resume(const WindowKey& expected);
    WindowExecutionView view() const;
private:
    std::shared_ptr<WindowExecutionState> state_;
};
} // namespace strata::core
