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
// Inventory preflight only: no resource vector, replacement control block or
// heap allocation to acquire this temporary host debit. Fixed source objects
// are paid separately by the verified-input setup baseline before construction.
class GgufHostWorkspace {
public:
    GgufHostWorkspace() = default;
    ~GgufHostWorkspace();
    GgufHostWorkspace(const GgufHostWorkspace&) = delete;
    GgufHostWorkspace& operator=(const GgufHostWorkspace&) = delete;
    GgufHostWorkspace(GgufHostWorkspace&&) noexcept;
    GgufHostWorkspace& operator=(GgufHostWorkspace&&) noexcept;
    void reset() noexcept;
    uint64_t bytes() const noexcept {return bytes_;}
private:
    friend class WindowBudget;
    std::shared_ptr<WindowBudgetState> state_;
    uint64_t bytes_=0,io_=0;
};

// Stack-only, single-use startup head transition. It never retains/reboxes a
// lease: its caller must keep the genuine exclusive owner alive until settlement.
class ReservationLease;
class HeadSpillCharge {
public:
    HeadSpillCharge() = default;
    ~HeadSpillCharge();
    HeadSpillCharge(const HeadSpillCharge&) = delete;
    HeadSpillCharge& operator=(const HeadSpillCharge&) = delete;
private:
    friend class WindowBudget;
    const ReservationLease* owner_ = nullptr;
    WindowBudgetState* state_ = nullptr;
    uint64_t bytes_ = 0, device_before_ = 0, host_before_ = 0, pinned_before_ = 0;
    int device_ = -1;
    enum class Phase { empty, peak, committed, rolled_back } phase_ = Phase::empty;
};
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
    // Authenticate the ownership group published by reserve(). Pointer identity
    // and use_count alone accept reboxed aliases with independent control blocks.
    // This is not exclusive consumption: callers needing a dedicated lease must
    // additionally require exclusive ownership before retaining it for storage.
    bool canonical_owner(const std::shared_ptr<const ReservationLease>& owner) const noexcept {
        return owner.get() == this && !owner_.owner_before(owner) && !owner.owner_before(owner_);
    }
private:
    friend class WindowBudget;
    ReservationLease(std::shared_ptr<WindowBudgetState>, WindowResources, ReservationPurpose,
                     std::shared_ptr<const ReservationLease> return_lease);
    std::shared_ptr<WindowBudgetState> state_;
    // Set once before publication. Weak ownership avoids a self-retaining cycle.
    std::weak_ptr<const ReservationLease> owner_;
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
    // Exclusive initialization only: reduce an unshared genuine foreground lease
    // without an uncharged interval for its still-live native allocations.
    bool shrink_foreground(const std::shared_ptr<const ReservationLease>&,WindowResources retained);
    WindowBudgetSnapshot snapshot() const;
    // Startup inventory only. Caller dedicates prepaid_setup as disjoint fixed
    // baseline before constructing scan/control objects; no nominal spare.
    bool gguf_setup_paid(uint64_t prepaid_setup) const noexcept;
    ReservationError begin_gguf_scan(uint64_t prepaid_setup,GgufHostWorkspace&) noexcept;
    ReservationError admit_gguf_workspace(uint64_t bytes,uint64_t prepaid_setup,GgufHostWorkspace&) noexcept;

    // Internal serialized startup only. No allocation, replacement control or
    // vector resize. fixed_host is separately prepaid by the SAME stage lease.
    bool admit_head_spill(const std::shared_ptr<const ReservationLease>&, int device,
                          uint64_t bytes, uint64_t fixed_host, HeadSpillCharge&) noexcept;
    // commit: old device free proved; rollback: new pinned free proved (or no
    // allocation occurred). A false result after native work requires failstop.
    bool settle_head_spill(const std::shared_ptr<const ReservationLease>&,
                           HeadSpillCharge&, bool commit) noexcept;
    static uint64_t head_spill_control_bytes() noexcept;
private:
    std::shared_ptr<WindowBudgetState> state_;
};
} // namespace strata::core
