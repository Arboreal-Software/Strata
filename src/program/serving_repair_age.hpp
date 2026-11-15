#pragma once
#include "serving_repair_admission.hpp"
#include <chrono>

namespace strata::program {
// Domain is the concrete runtime incarnation, not a wall-clock or a thread id.
// Call at the causal import and again at a scheduling boundary; never at restoration.
inline core::RepairMonotonicBirth serving_repair_clock_now(uint64_t runtime_incarnation) noexcept {
    if(!runtime_incarnation)return {};
    const auto ticks=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if(ticks<0)return {};
    return {runtime_incarnation,uint64_t(ticks)};
}
// Used only when constructing the first root from the trusted import event.
// The default unknown birth stays unknown; reconstruction must never read a clock.
inline bool bind_initial_serving_repair_birth(core::RepairCausalRoot& root,
        core::RepairMonotonicBirth event_birth,uint64_t runtime_incarnation) noexcept {
    if((!event_birth.domain && event_birth.ticks_ns) ||
       (event_birth.domain && event_birth.domain!=runtime_incarnation))return false;
    if(root.monotonic_birth.domain || root.monotonic_birth.ticks_ns)
        return root.monotonic_birth==event_birth;
    root.monotonic_birth=event_birth;
    return true;
}
struct ServingRepairAgeObservation {
    RepairAgeInputs age;
    core::RepairMonotonicBirth observed_at;
    uint64_t ancestry_serial=0;
    bool valid=false;
};
// The trusted controller supplies its real published coverage and actual cuts.
// Unknown roots stay unknown: this never consults a cached RetainedRepairAge.
inline ServingRepairAgeObservation observe_serving_repair_age(
        const core::RepairSpatialCoverage& coverage,core::RepairMonotonicBirth now,
        uint64_t published_forwarded,uint64_t private_forwarded,uint64_t ancestry_serial) noexcept {
    ServingRepairAgeObservation out;
    if(!coverage.valid() || !ancestry_serial || private_forwarded>published_forwarded ||
       coverage.end()>published_forwarded)return out;
    out.age=make_repair_age_inputs(core::repair_ancestry_age(coverage.ancestry(),now),
                                  coverage.begin(),repair_token_debt(published_forwarded,private_forwarded));
    out.observed_at=now;out.ancestry_serial=ancestry_serial;out.valid=true;
    return out;
}
} // namespace strata::program
