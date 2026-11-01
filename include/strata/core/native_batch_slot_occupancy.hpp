#pragma once
#include "strata/core/native_batch_slot_binding.hpp"
#include <atomic>
namespace strata::program {class ServingRawSession;}
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
    const uint64_t incarnation_=next_verifier_incarnation();
    NativeBatchSlotToken token_{};
    int32_t pending_token_=-1;
    int64_t pending_position_=-1;
    std::shared_ptr<const void> runtime_,metadata_;
    mutable std::atomic<uint64_t> submitted_group_{0};
    std::atomic<bool> active_{true},running_{false};
};
}
