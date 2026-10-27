#pragma once
#include <cmath>
#include <cstdint>

namespace strata::core {
// Partial producer observations, never a compatibility identity or generation fence.
// Call only with owner/reactor serialization against initialization, execution,
// reset, policy setters and destruction (including process-wide policy writers).
enum class NumericPolicyMode : uint8_t { unresolved, disabled, fixed, adaptive, invalid };
enum class PolicyApplicability : uint8_t { unknown, currently_inapplicable, conditional, callback_unverified };
struct RouteTailPolicyObservation {
    int rank = -1;
    NumericPolicyMode mode = NumericPolicyMode::unresolved;
    PolicyApplicability applicability = PolicyApplicability::callback_unverified;
    bool dynamic_assignment = false;
    bool incomplete_registry = true;
};
inline RouteTailPolicyObservation route_tail_observation(int cached_rank) noexcept {
    RouteTailPolicyObservation r;
    r.rank = cached_rank;
    r.mode = cached_rank == -1 ? NumericPolicyMode::unresolved :
        cached_rank == 0 ? NumericPolicyMode::disabled :
        cached_rank > 0 && cached_rank < 10 ? NumericPolicyMode::fixed : NumericPolicyMode::invalid;
    r.dynamic_assignment = r.mode == NumericPolicyMode::fixed;
    return r;
}
struct PrefillCpuShareObservation {
    NumericPolicyMode mode = NumericPolicyMode::unresolved;
    PolicyApplicability applicability = PolicyApplicability::unknown;
    bool explicit_resolved = false, default_armed = false;
    double cached_explicit = 0, effective_fraction = 0;
    bool maximum_resolved = false, explicit_maximum = false;
    int64_t cached_maximum = 0, effective_maximum = 0, armed_stream_limit = 0;
    bool instance_present = false, pool_present = false, source_present = false;
    bool peer_present = false, allocation_disabled = false;
    double measured_fraction = 0;
    bool measured_gate = false, measurement_pending = false;
    int measured_ratios = 0;
    bool dynamic_assignment = false;
    bool incomplete_registry = true;
};
// Pure classification of already observed values. No parsing, lazy resolution,
// environment access, allocation, or assertion of full layer/dispatch eligibility.
inline PrefillCpuShareObservation classify_cpu_share(PrefillCpuShareObservation r) noexcept {
    if (r.explicit_resolved) {
        const double v = r.cached_explicit == -2 ? (r.default_armed ? -1 : 0) : r.cached_explicit;
        r.effective_fraction = v;
        r.mode = !std::isfinite(v) ? NumericPolicyMode::invalid : v == -1 ? NumericPolicyMode::adaptive :
            v == 0 ? NumericPolicyMode::disabled : v > 0 && v <= 1 ? NumericPolicyMode::fixed : NumericPolicyMode::invalid;
    }
    r.dynamic_assignment = r.mode == NumericPolicyMode::fixed || r.mode == NumericPolicyMode::adaptive;
    if (r.instance_present) r.applicability = !r.pool_present || !r.source_present || r.peer_present ||
        r.allocation_disabled || r.mode == NumericPolicyMode::disabled ?
        PolicyApplicability::currently_inapplicable : PolicyApplicability::conditional;
    return r;
}
} // namespace strata::core
