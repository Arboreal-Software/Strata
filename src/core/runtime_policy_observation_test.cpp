#include "strata/core/runtime_policy_observation.hpp"
#include <cassert>
#include <limits>
using namespace strata::core;
int main() {
    auto tail = route_tail_observation(-1);
    assert(tail.mode == NumericPolicyMode::unresolved && tail.incomplete_registry);
    assert(route_tail_observation(0).mode == NumericPolicyMode::disabled);
    tail = route_tail_observation(7);
    assert(tail.dynamic_assignment && tail.applicability == PolicyApplicability::callback_unverified);
    assert(route_tail_observation(10).mode == NumericPolicyMode::invalid);
    PrefillCpuShareObservation r;
    assert(classify_cpu_share(r).mode == NumericPolicyMode::unresolved);
    r.explicit_resolved = true; r.cached_explicit = -2;
    assert(classify_cpu_share(r).mode == NumericPolicyMode::disabled);
    r.default_armed = true;
    assert(classify_cpu_share(r).mode == NumericPolicyMode::adaptive);
    r.cached_explicit = 0.4; r.instance_present = true; r.pool_present = true; r.source_present = true;
    auto fixed = classify_cpu_share(r);
    assert(fixed.mode == NumericPolicyMode::fixed && fixed.dynamic_assignment && fixed.incomplete_registry);
    assert(fixed.applicability == PolicyApplicability::conditional);
    r.allocation_disabled = true;
    assert(classify_cpu_share(r).applicability == PolicyApplicability::currently_inapplicable);
    r.cached_explicit = std::numeric_limits<double>::quiet_NaN();
    assert(classify_cpu_share(r).mode == NumericPolicyMode::invalid);
    r.cached_explicit = 2;
    assert(classify_cpu_share(r).mode == NumericPolicyMode::invalid);
}
