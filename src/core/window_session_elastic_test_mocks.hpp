#pragma once
// Include in exactly one mock-provider translation unit per standalone test.
// These fixtures model fixed KV only. Link the real layer/session implementation
// for elastic behavior; never manufacture successful elastic geometry or owners.
#include "strata/core/session.hpp"
#include <cstdlib>

namespace strata::kernels {
// Fixed-KV fixtures never construct a PLE reader. Keep terminal IO unsupported;
// real PLE closure has its own actual-reader tests, not a fake success here.
#ifndef STRATA_WINDOW_TEST_ACTUAL_PLE_CONTEXT
bool PleReadContext::close_for_retirement() noexcept { return false; }
#endif
}
namespace strata::core {
bool qsa_elastic_plan_matches(const ModelGeometry&, const QsaElasticPlan&) {
    return false;
}

bool session_private_plan(const ModelGeometry& g, int64_t cells, int64_t k,
                          int64_t lo, int64_t hi, SessionPrivatePlan& out,
                          const QsaElasticPlan* elastic) {
    if (elastic) return false;
    return session_private_plan(g, cells, k, lo, hi, out);
}

uint64_t session_init_private(const ModelGeometry& g, int64_t cells, int64_t k,
                              void* arena, SessionState& state, SessionInitUploads& uploads,
                              int64_t lo, int64_t hi, const QsaElasticPlan* elastic,
                              QsaElasticView* const* views, size_t count) {
    if (elastic || views || count) return 0;
    return session_init_private(g, cells, k, arena, state, uploads, lo, hi);
}

namespace window_test {
// Run before fixture setup. Refusals must not call legacy initialization, allocate
// session state or alter sentinels. Existing allocator tests exercise null-plan
// delegation through actual window_allocator.cpp, preserving their fault knobs.
inline void check_fixed_kv_elastic_mock_contract() {
    const auto check=[](bool value) { if (!value) std::abort(); };
    ModelGeometry g;
    QsaElasticPlan elastic;
    SessionPrivatePlan quote{17,19,23};
    check(!qsa_elastic_plan_matches(g,elastic));
    check(!session_private_plan(g,16,1,0,4,quote,&elastic));
    check(quote.device_bytes==17 && quote.host_bytes==19 && quote.pinned_bytes==23);
    SessionPrivatePlan expected,actual;
    check(session_private_plan(g,16,1,0,4,expected));
    check(session_private_plan(g,16,1,0,4,actual,nullptr));
    check(actual.device_bytes==expected.device_bytes && actual.host_bytes==expected.host_bytes &&
          actual.pinned_bytes==expected.pinned_bytes);
    SessionState state;state.layer_lo=29;state.layer_hi=31;
    SessionInitUploads uploads;
    QsaElasticView* view=nullptr;
    check(!session_init_private(g,16,1,nullptr,state,uploads,0,4,&elastic,nullptr,0));
    check(!session_init_private(g,16,1,nullptr,state,uploads,0,4,nullptr,&view,0));
    check(!session_init_private(g,16,1,nullptr,state,uploads,0,4,nullptr,nullptr,1));
    check(state.layer_lo==29 && state.layer_hi==31 && !state.qsa_states && uploads.qsa.empty());
}
} // namespace window_test
} // namespace strata::core
