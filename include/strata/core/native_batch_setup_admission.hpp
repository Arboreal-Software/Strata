#pragma once
#include "strata/core/window_budget.hpp"
#include "strata/kernels/sampler.hpp"
#include <vector>
#include <cstddef>
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
    static constexpr uint64_t metadata_bytes() noexcept {
        // Existing eight-stage/two-slot setup bound. Fresh GNU vector assign
        // requests exactly two entries for each new per-slot directory; tails
        // stay empty because owned admission refuses penalties and logit bias.
        // Keep the inherited setup envelope separate from these new requests.
        return sizeof(NativeBatchSetupAdmission) + 16384 +
            8 * (2 * sizeof(strata::kernels::SamplerParams) + 2 * sizeof(float*) +
                 2 * sizeof(std::vector<int32_t>) + 3 * alignof(std::max_align_t));
    }
private:
    friend class Verifier;
    std::shared_ptr<const ReservationLease> credit_;
    std::shared_ptr<const NativeBatchSlotBindings> bindings_;
    std::array<Verifier*,8> stages_{};
    size_t count_=0;
    bool ready_=false;
};
}
