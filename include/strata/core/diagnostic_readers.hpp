#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

namespace strata::core {

// Lifetime only: does not synchronize the object's mutable diagnostic fields.
// The registry must outlive its publishers and leases. Object ownership serializes
// publish/retire for the same object; retire must not run while holding its own lease.
// A full registry refuses publication (diagnostics remain best effort).
template<class T, std::size_t Capacity = 16>
class DiagnosticReaders {
    struct Slot {
        T* object = nullptr;                    // registry mutex
        bool published = false;                // registry mutex
        std::atomic<std::uint64_t> readers{0};
        std::mutex drain_mutex;
        std::condition_variable drained;
    };
public:
    class Lease {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
            : slot_(std::exchange(other.slot_, nullptr)), object_(std::exchange(other.object_, nullptr)) {}
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                reset();
                slot_ = std::exchange(other.slot_, nullptr);
                object_ = std::exchange(other.object_, nullptr);
            }
            return *this;
        }
        ~Lease() { reset(); }
        explicit operator bool() const noexcept { return object_ != nullptr; }
        T* operator->() const noexcept { return object_; }
        T* get() const noexcept { return object_; }
        void reset() noexcept {
            if (!slot_) return;
            // Pair the predicate update with the wait mutex: no lost wakeup.
            std::lock_guard<std::mutex> lock(slot_->drain_mutex);
            if (slot_->readers.fetch_sub(1, std::memory_order_acq_rel) == 1)
                slot_->drained.notify_all();
            slot_ = nullptr;
            object_ = nullptr;
        }
    private:
        friend class DiagnosticReaders;
        Lease(Slot* slot, T* object) : slot_(slot), object_(object) {}
        Slot* slot_ = nullptr;
        T* object_ = nullptr;
    };

    bool publish(T* object) {
        if (!object) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& slot : slots_) {
            if (slot.object == object) {
                if (!slot.published) return false;
                latest_ = &slot;
                return true;
            }
        }
        for (auto& slot : slots_) {
            if (!slot.object) {
                slot.object = object;
                slot.published = true;
                latest_ = &slot;
                return true;
            }
        }
        return false;
    }

    Lease acquire_latest() {
        std::lock_guard<std::mutex> lock(mutex_);
        return acquire_locked(latest_);
    }
    Lease acquire(std::size_t index) {
        std::lock_guard<std::mutex> lock(mutex_);
        return acquire_locked(index < Capacity ? &slots_[index] : nullptr);
    }

    void retire(T* object) noexcept {
        Slot* retiring = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& slot : slots_) {
                if (slot.object == object && slot.published) {
                    slot.published = false;
                    if (latest_ == &slot) latest_ = nullptr;
                    retiring = &slot;
                    break;
                }
            }
        }
        if (!retiring) return;
        {
            // No registry lock while draining; other slots remain usable.
            std::unique_lock<std::mutex> lock(retiring->drain_mutex);
            retiring->drained.wait(lock, [&] { return retiring->readers.load(std::memory_order_acquire) == 0; });
        }
        // The occupied slot cannot be reused until all readers have left.
        std::lock_guard<std::mutex> lock(mutex_);
        retiring->object = nullptr;
    }

private:
    Lease acquire_locked(Slot* slot) {
        if (!slot || !slot->published) return {};
        // Acquisition is serialized, releases only decrease the counter.
        if (slot->readers.load(std::memory_order_relaxed) == (std::numeric_limits<std::uint64_t>::max)()) return {};
        slot->readers.fetch_add(1, std::memory_order_acq_rel);
        return Lease(slot, slot->object);
    }
    std::mutex mutex_;
    std::array<Slot, Capacity> slots_{};
    Slot* latest_ = nullptr;
};

} // namespace strata::core
