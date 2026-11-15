#pragma once
#include "strata/core/native_batch_slot_binding.hpp"
#include <atomic>
namespace strata::program {class ServingRawSession;class OwnedImageRequestOperation;}
namespace strata::core {
// Conversation occupancy, distinct from physical allocation generation. Only the
// actual serving RAW owner can construct/revoke it or open its native work fence.
class NativeBatchSlotOccupancy final {
public:
    uint64_t incarnation()const noexcept{return incarnation_;}
    size_t slot()const noexcept{return token_.slot;}
    bool active()const noexcept{return active_.load(std::memory_order_acquire);}
private:
    friend class Verifier;
    friend class strata::program::ServingRawSession;
    NativeBatchSlotOccupancy()=default;
    // Exact predecessor data layout, not a guessed numerical ABI size. The
    // factory still uses new T plus its unchanged keeper-capturing deleter.
    struct PreviousOpaqueLayout {
        const uint64_t incarnation=0;
        NativeBatchSlotToken token{};
        int32_t pending_token=-1;
        int64_t pending_position=-1;
        std::shared_ptr<const void> runtime,metadata;
        mutable std::atomic<uint64_t> submitted_group{0};
        std::atomic<bool> active{true},running{false};
    };
    static void check_context_layout()noexcept;
    const uint64_t incarnation_=next_verifier_incarnation();
    NativeBatchSlotToken token_{};
    int32_t pending_token_=-1;
    int64_t pending_position_=-1;
    std::shared_ptr<const void> runtime_;
    // Replaces the formerly unused metadata handle: no allocation/layout delta.
    // Authoritative context is installed by the actual RAW owner before native
    // image mutation, retained through all groups and cleared only after reset.
    std::shared_ptr<const program::OwnedImageRequestOperation> image_context_;
    mutable std::atomic<uint64_t> submitted_group_{0};
    std::atomic<bool> active_{true},running_{false};
};
inline void NativeBatchSlotOccupancy::check_context_layout()noexcept {
    static_assert(sizeof(std::shared_ptr<const program::OwnedImageRequestOperation>)==sizeof(std::shared_ptr<const void>)&&
        alignof(std::shared_ptr<const program::OwnedImageRequestOperation>)==alignof(std::shared_ptr<const void>));
    static_assert(sizeof(NativeBatchSlotOccupancy)==sizeof(PreviousOpaqueLayout)&&
        alignof(NativeBatchSlotOccupancy)==alignof(PreviousOpaqueLayout),
        "image context must reuse the exact predecessor occupancy new request");
}
}
