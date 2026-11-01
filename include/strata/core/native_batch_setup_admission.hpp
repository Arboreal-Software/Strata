#pragma once
#include "strata/core/window_budget.hpp"
#include <memory>
#include <array>
namespace strata::core {
class Verifier;
class NativeBatchSlotBindings;
// Owns the conservative setup reservation. Only Verifier setup may reduce it
// after checked exclusive native measurement; admission is not GPU completion.
class NativeBatchSetupAdmission final {
public:
    bool ready()const noexcept{return ready_;}
    static constexpr uint64_t metadata_bytes()noexcept{return sizeof(NativeBatchSetupAdmission)+16384;}
private:
    friend class Verifier;
    std::shared_ptr<const ReservationLease> credit_;
    std::shared_ptr<const NativeBatchSlotBindings> bindings_;
    std::array<Verifier*,8> stages_{};
    size_t count_=0;
    bool ready_=false;
};
}
