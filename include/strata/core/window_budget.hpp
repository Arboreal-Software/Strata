#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

// Host is total physical host memory, including pinned. Pinned is a second cap
// on the pinned subset, not an additional charge to physical memory.
struct WindowResources {
    std::vector<uint64_t> device_bytes;
    uint64_t host_bytes = 0, pinned_bytes = 0, disk_bytes = 0, io_slots = 0, shadows = 0;
    bool operator==(const WindowResources&) const = default;
};
enum class ReservationPurpose { foreground, foreground_return, private_work, saved_maintenance, private_attachment };
enum class ReservationError { none, invalid_vector, device_limit, host_limit, pinned_limit,
                              disk_limit, io_limit, shadow_limit, return_unavailable, allocation_failed };
struct WindowBudgetState;
// Canonical runtime identity includes engine/window generation and return cut.
// The runtime computes required from the destination allocator's actual layout.
struct WindowReturnObligation {
    std::string identity;
    WindowResources required;
    bool operator==(const WindowReturnObligation&) const = default;
};

class ReservationLease {
public:
    ~ReservationLease();
    ReservationLease(const ReservationLease&) = delete;
    ReservationLease& operator=(const ReservationLease&) = delete;
    const WindowResources& resources() const { return resources_; }
    ReservationPurpose purpose() const { return purpose_; }
private:
    friend class WindowBudget;
    ReservationLease(std::shared_ptr<WindowBudgetState>, WindowResources, ReservationPurpose,
                     std::shared_ptr<const ReservationLease> return_lease);
    std::shared_ptr<WindowBudgetState> state_;
    WindowResources resources_;
    ReservationPurpose purpose_;
    std::shared_ptr<const ReservationLease> return_;
    WindowReturnObligation obligation_;
    mutable bool return_in_use_ = false;
    bool charged_ = false;
};
struct ReservationResult {
    ReservationError error = ReservationError::none;
    int device = -1;
    std::shared_ptr<const ReservationLease> lease;
    explicit operator bool() const { return error == ReservationError::none; }
};
struct WindowBudgetSnapshot {
    WindowResources capacity, baseline, reserved;
    bool overcommitted = false;
};

class WindowBudget {
public:
    // Throws invalid_argument for malformed caps/baseline. The supplied baseline
    // includes weights, residency, live slots and safety headroom already held.
    WindowBudget(WindowResources capacity, WindowResources baseline);
    // private_attachment requires a same-ledger private_work/saved_maintenance
    // parent in foreground_return; it retains that parent and charges zero shadows.
    ReservationResult reserve(const WindowResources&, ReservationPurpose,
                              std::shared_ptr<const ReservationLease> foreground_return = {},
                              const WindowReturnObligation* obligation = nullptr);
    // Records actual external allocations, including growth beyond budget.
    // Existing leases stay charged. New work is refused while overcommitted.
    bool observe_baseline(const WindowResources&);
    WindowBudgetSnapshot snapshot() const;
private:
    std::shared_ptr<WindowBudgetState> state_;
};
} // namespace strata::core
