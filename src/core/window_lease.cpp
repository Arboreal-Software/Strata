#include "strata/core/window_lease.hpp"

#include <limits>
#include <mutex>
#include <stdexcept>

namespace strata::core {
struct WindowExecutionState {
    mutable std::mutex mutex;
    WindowKey key;
    StateCut cut;
    uint32_t stages = 0;
    WindowExecutionPhase phase = WindowExecutionPhase::running;
    uint64_t operation = 0;
    bool in_flight = false;
    uint64_t readers = 0;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const ReservationLease> reservation;
    // Deliberate quarantine while work may reference raw pointers. A forgotten
    // callback may leak accounted capacity, but cannot release an in-flight buffer.
    std::shared_ptr<WindowExecutionState> pending_owner;
};
struct WindowReadHold {
    std::shared_ptr<WindowExecutionState> state;
    ~WindowReadHold() {
        if (state) { std::lock_guard lock(state->mutex); --state->readers; }
    }
};
bool WindowRead::usable() const {
    if (!hold_) return false;
    std::lock_guard lock(hold_->state->mutex);
    return hold_->state->phase == WindowExecutionPhase::quiesced &&
           hold_->state->key == key_ && hold_->state->cut == cut_;
}
void WindowRead::mark_unhealthy() const {
    if (!hold_) return;
    std::lock_guard lock(hold_->state->mutex);
    hold_->state->phase = WindowExecutionPhase::unhealthy;
}
namespace {
bool valid(const StateCut& c) {
    return c.id && c.exact_frontier <= c.forwarded && c.forwarded <= c.accepted;
}
bool matches(const WindowExecutionState& s, uint64_t operation) {
    return s.in_flight && s.operation == operation;
}
}
WindowExecution::WindowExecution(WindowKey key, StateCut cut, uint32_t stages,
                                 std::shared_ptr<const void> owner,
                                 std::shared_ptr<const ReservationLease> reservation)
    : state_(std::make_shared<WindowExecutionState>()) {
    if (!valid(cut) || !stages || !owner || !reservation || !key.engine_generation ||
        !key.window_generation || key.window_id.empty())
        throw std::invalid_argument("window execution requires identity, cut, stages and owned capacity");
    state_->key = std::move(key); state_->cut = cut; state_->stages = stages;
    state_->owner = std::move(owner); state_->reservation = std::move(reservation);
}
bool WindowExecution::admit(const WindowKey& expected, WindowWork& out, const StateCut* expected_cut) {
    WindowWork work;
    {
        std::lock_guard lock(state_->mutex);
        if (state_->key != expected || (expected_cut && state_->cut != *expected_cut) || state_->phase != WindowExecutionPhase::running ||
            state_->in_flight || state_->operation == std::numeric_limits<uint64_t>::max()) return false;
        ++state_->operation; state_->in_flight = true; state_->pending_owner = state_;
        work.state_ = state_; work.operation_ = state_->operation;
    }
    out = std::move(work);
    return true;
}
bool WindowExecution::begin_quiesce(const WindowKey& expected,const StateCut* expected_cut) {
    std::lock_guard lock(state_->mutex);
    if (state_->key != expected || (expected_cut && state_->cut != *expected_cut) || (state_->phase != WindowExecutionPhase::running &&
        state_->phase != WindowExecutionPhase::quiescing && state_->phase != WindowExecutionPhase::quiesced)) return false;
    state_->phase = state_->in_flight ? WindowExecutionPhase::quiescing : WindowExecutionPhase::quiesced;
    return true;
}
bool WindowExecution::cancel(const WindowKey& expected,const StateCut* expected_cut) {
    std::lock_guard lock(state_->mutex);
    if (state_->key != expected || (expected_cut && state_->cut != *expected_cut) || state_->phase == WindowExecutionPhase::unhealthy) return false;
    state_->phase = state_->in_flight ? WindowExecutionPhase::cancelling : WindowExecutionPhase::aborted;
    return true;
}
bool WindowExecution::resume(const WindowKey& expected) {
    std::lock_guard lock(state_->mutex);
    if (state_->key != expected || state_->phase != WindowExecutionPhase::quiesced || state_->in_flight || state_->readers) return false;
    state_->phase = WindowExecutionPhase::running;
    return true;
}
bool WindowExecution::retire_for_replacement(const WindowRead& read,const WindowKey& key,const StateCut& cut) {
    std::lock_guard lock(state_->mutex);
    if (!read.hold_ || read.hold_->state != state_ || state_->key != key || state_->cut != cut ||
        read.key_ != key || read.cut_ != cut || state_->phase != WindowExecutionPhase::quiesced || state_->in_flight)
        return false;
    state_->phase = WindowExecutionPhase::aborted;
    return true;
}
bool WindowExecution::matches(const WindowKey& key,const StateCut& cut) const {
    std::lock_guard lock(state_->mutex);
    return state_->key == key && state_->cut == cut;
}
bool WindowExecution::acquire_read(const WindowKey& expected, WindowRead& out, const StateCut* expected_cut) {
    // Replace the caller's prior hold outside the mutex: its destructor may acquire
    // this same mutex. Allocation/copy failures leave the existing output intact.
    WindowRead read;
    read.key_ = expected;
    read.hold_ = std::make_shared<WindowReadHold>();
    {
        std::lock_guard lock(state_->mutex);
        if (state_->key != expected || (expected_cut && state_->cut != *expected_cut) || state_->phase != WindowExecutionPhase::quiesced || state_->in_flight ||
            state_->readers == std::numeric_limits<uint64_t>::max()) return false;
        read.cut_ = state_->cut;
        read.hold_->state = state_;
        ++state_->readers;
    }
    out = std::move(read);
    return true;
}
WindowExecutionView WindowExecution::view() const {
    std::lock_guard lock(state_->mutex);
    return {state_->phase, state_->key, state_->cut, state_->operation, state_->in_flight};
}
bool WindowWork::finish(uint64_t op, const WindowKey& completed_key, const std::vector<StateCut>& cuts, bool host_joined, bool drained) {
    if (!state_ || op != operation_) return false;
    std::lock_guard lock(state_->mutex);
    if (!matches(*state_, op) || state_->phase == WindowExecutionPhase::unhealthy || !host_joined || !drained)
        return false;
    WindowKey same = completed_key;
    same.append_epoch = state_->key.append_epoch;
    if (same != state_->key) return false;
    // Cancellation can retire an invalid partial unit once all work drained. It
    // cannot publish that unit's cut or return to running.
    if (state_->phase != WindowExecutionPhase::cancelling) {
        if (cuts.size() != state_->stages || !valid(cuts.front()) || cuts.front().id <= state_->cut.id)
            return false;
        for (const auto& cut : cuts) if (cut != cuts.front()) return false;
        if (completed_key.append_epoch < state_->key.append_epoch ||
            cuts.front().accepted < state_->cut.accepted || cuts.front().forwarded < state_->cut.forwarded ||
            cuts.front().exact_frontier < state_->cut.exact_frontier ||
            cuts.front().output_watermark < state_->cut.output_watermark) return false;
        const bool appended = cuts.front().history != state_->cut.history || cuts.front().accepted != state_->cut.accepted;
        // One completed unit publishes at most one append transaction, even if
        // that transaction contains multiple accepted tokens.
        if (appended) {
            if (state_->key.append_epoch == std::numeric_limits<uint64_t>::max() ||
                completed_key.append_epoch != state_->key.append_epoch + 1) return false;
        } else if (completed_key.append_epoch != state_->key.append_epoch) return false;
        state_->key.append_epoch = completed_key.append_epoch;
        state_->cut = cuts.front();
    } else if (completed_key.append_epoch != state_->key.append_epoch) return false;
    state_->in_flight = false;
    if (state_->phase == WindowExecutionPhase::cancelling) state_->phase = WindowExecutionPhase::aborted;
    else if (state_->phase == WindowExecutionPhase::quiescing) state_->phase = WindowExecutionPhase::quiesced;
    state_->pending_owner.reset();
    return true;
}
bool WindowWork::resolved() const {
    if (!state_ || !operation_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->operation == operation_ && !state_->in_flight &&
           state_->phase != WindowExecutionPhase::unhealthy;
}
bool WindowWork::mark_unhealthy(uint64_t op) {
    if (!state_ || op != operation_) return false;
    std::lock_guard lock(state_->mutex);
    if (!matches(*state_, op)) return false;
    state_->phase = WindowExecutionPhase::unhealthy;
    return true;
}
bool WindowWork::retire_after_teardown(uint64_t op) {
    if (!state_ || op != operation_) return false;
    std::lock_guard lock(state_->mutex);
    if (!matches(*state_, op) || state_->phase != WindowExecutionPhase::unhealthy) return false;
    state_->in_flight = false;
    state_->phase = WindowExecutionPhase::aborted;
    state_->pending_owner.reset();
    return true;
}
} // namespace strata::core
