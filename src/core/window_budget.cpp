#include "strata/core/window_budget.hpp"

#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>

namespace strata::core {
struct WindowBudgetState {
    mutable std::mutex mutex;
    WindowResources capacity, baseline, reserved;
};
namespace {
bool valid(const WindowResources& r, size_t devices) {
    return r.device_bytes.size() == devices && r.pinned_bytes <= r.host_bytes;
}
bool fits(uint64_t capacity, uint64_t baseline, uint64_t held, uint64_t request) {
    return baseline <= capacity && held <= capacity - baseline && request <= capacity - baseline - held;
}
bool covers(const WindowResources& held, const WindowResources& required) {
    if (!valid(required,held.device_bytes.size())) return false;
    bool nonzero = false;
    for (size_t i = 0; i < held.device_bytes.size(); ++i) {
        if (held.device_bytes[i] < required.device_bytes[i]) return false;
        nonzero |= required.device_bytes[i] != 0;
    }
    for (auto field : {&WindowResources::host_bytes, &WindowResources::pinned_bytes, &WindowResources::disk_bytes,
                       &WindowResources::io_slots, &WindowResources::shadows}) {
        if (held.*field < required.*field) return false;
        nonzero |= required.*field != 0;
    }
    return nonzero;
}
ReservationResult check(const WindowBudgetState& s, const WindowResources& r) {
    if (!valid(r, s.capacity.device_bytes.size())) return {ReservationError::invalid_vector, -1, {}};
    for (size_t i = 0; i < r.device_bytes.size(); ++i)
        if (!fits(s.capacity.device_bytes[i], s.baseline.device_bytes[i], s.reserved.device_bytes[i], r.device_bytes[i]))
            return {ReservationError::device_limit, int(i), {}};
    if (!fits(s.capacity.host_bytes,s.baseline.host_bytes,s.reserved.host_bytes,r.host_bytes))
        return {ReservationError::host_limit,-1,{}};
    if (!fits(s.capacity.pinned_bytes,s.baseline.pinned_bytes,s.reserved.pinned_bytes,r.pinned_bytes))
        return {ReservationError::pinned_limit,-1,{}};
    if (!fits(s.capacity.disk_bytes,s.baseline.disk_bytes,s.reserved.disk_bytes,r.disk_bytes))
        return {ReservationError::disk_limit,-1,{}};
    if (!fits(s.capacity.io_slots,s.baseline.io_slots,s.reserved.io_slots,r.io_slots))
        return {ReservationError::io_limit,-1,{}};
    if (!fits(s.capacity.shadows,s.baseline.shadows,s.reserved.shadows,r.shadows))
        return {ReservationError::shadow_limit,-1,{}};
    return {};
}
void charge(WindowResources& held, const WindowResources& r, bool add) {
    // Addition only follows subtraction-based fit checks under the same lock;
    // subtraction only occurs once, at the last shared lease's destruction.
    for (size_t i = 0; i < r.device_bytes.size(); ++i)
        if (add) held.device_bytes[i] += r.device_bytes[i]; else held.device_bytes[i] -= r.device_bytes[i];
    for (auto field : {&WindowResources::host_bytes, &WindowResources::pinned_bytes, &WindowResources::disk_bytes,
                       &WindowResources::io_slots, &WindowResources::shadows})
        if (add) held.*field += r.*field; else held.*field -= r.*field;
}
}
ReservationLease::ReservationLease(std::shared_ptr<WindowBudgetState> s, WindowResources r, ReservationPurpose p,
                                   std::shared_ptr<const ReservationLease> returned)
    : state_(std::move(s)), resources_(std::move(r)), purpose_(p), return_(std::move(returned)) {}
ReservationLease::~ReservationLease() {
    if (charged_) {
        std::lock_guard lock(state_->mutex);
        charge(state_->reserved, resources_, false);
        if (return_ && purpose_ == ReservationPurpose::saved_maintenance) return_->return_in_use_ = false;
    }
    // return_ is released after the lock: its destructor can charge the same ledger.
}
WindowBudget::WindowBudget(WindowResources capacity, WindowResources baseline)
    : state_(std::make_shared<WindowBudgetState>()) {
    if (capacity.device_bytes.empty() || capacity.device_bytes.size() > 64 ||
        !valid(capacity,capacity.device_bytes.size()) || !valid(baseline,capacity.device_bytes.size()))
        throw std::invalid_argument("invalid window resource dimensions or pinned subset");
    state_->capacity = std::move(capacity); state_->baseline = std::move(baseline);
    state_->reserved.device_bytes.resize(state_->capacity.device_bytes.size());
}
ReservationResult WindowBudget::reserve(const WindowResources& resources, ReservationPurpose purpose,
                                        std::shared_ptr<const ReservationLease> returned,
                                        const WindowReturnObligation* obligation) {
    const bool return_related = purpose == ReservationPurpose::foreground_return || purpose == ReservationPurpose::saved_maintenance;
    if (return_related && (!obligation || obligation->identity.empty()))
        return {ReservationError::return_unavailable,-1,{}};
    if (!return_related && ((returned && purpose != ReservationPurpose::private_attachment) || obligation)) return {ReservationError::invalid_vector,-1,{}};
    if (purpose == ReservationPurpose::foreground_return &&
        (returned || !covers(resources,obligation->required))) return {ReservationError::return_unavailable,-1,{}};
    if ((purpose == ReservationPurpose::private_work || purpose == ReservationPurpose::saved_maintenance) && resources.shadows != 1)
        return {ReservationError::invalid_vector,-1,{}};
    if (purpose == ReservationPurpose::private_attachment && (resources.shadows != 0 || !returned))
        return {ReservationError::invalid_vector,-1,{}};
    switch (purpose) {
    case ReservationPurpose::foreground: case ReservationPurpose::foreground_return:
    case ReservationPurpose::private_work: case ReservationPurpose::saved_maintenance:
    case ReservationPurpose::private_attachment: break;
    default: return {ReservationError::invalid_vector,-1,{}};
    }
    std::shared_ptr<ReservationLease> lease;
    try {
        // Allocate before taking the ledger lock; failed allocation never changes admission.
        lease = std::shared_ptr<ReservationLease>(new ReservationLease(state_,resources,purpose,std::move(returned)));
        if (obligation) lease->obligation_ = *obligation;
    } catch (const std::bad_alloc&) { return {ReservationError::allocation_failed,-1,{}}; }
    std::lock_guard lock(state_->mutex);
    if (purpose == ReservationPurpose::saved_maintenance &&
        (!lease->return_ || lease->return_->state_ != state_ || !lease->return_->charged_ ||
         lease->return_->purpose_ != ReservationPurpose::foreground_return))
        return {ReservationError::return_unavailable,-1,{}};
    if (purpose == ReservationPurpose::saved_maintenance &&
        (lease->return_->return_in_use_ || lease->return_->obligation_ != *obligation ||
         !covers(lease->return_->resources_,obligation->required)))
        return {ReservationError::return_unavailable,-1,{}};
    if (purpose == ReservationPurpose::private_attachment &&
        (lease->return_->state_ != state_ || !lease->return_->charged_ ||
         (lease->return_->purpose_ != ReservationPurpose::private_work &&
          lease->return_->purpose_ != ReservationPurpose::saved_maintenance)))
        return {ReservationError::invalid_vector,-1,{}};
    auto result = check(*state_,resources);
    if (!result) return result;
    charge(state_->reserved,resources,true);
    lease->charged_ = true;
    if (purpose == ReservationPurpose::saved_maintenance) lease->return_->return_in_use_ = true;
    result.lease = std::move(lease);
    return result;
}
bool WindowBudget::observe_baseline(const WindowResources& baseline) {
    if (!valid(baseline,state_->capacity.device_bytes.size())) return false;
    WindowResources copy = baseline;
    std::lock_guard lock(state_->mutex);
    state_->baseline = std::move(copy);
    return true;
}
WindowBudgetSnapshot WindowBudget::snapshot() const {
    std::lock_guard lock(state_->mutex);
    WindowResources zero;
    zero.device_bytes.resize(state_->capacity.device_bytes.size());
    return {state_->capacity,state_->baseline,state_->reserved,!check(*state_,zero)};
}
} // namespace strata::core
