#pragma once
#include <atomic>
#include <cstdint>
namespace strata::core {
// Process-local, nonreused identity; exhaustion permanently refuses new bindings.
inline uint64_t next_verifier_incarnation() noexcept {
    static std::atomic<uint64_t> next{1};
    auto value=next.load(std::memory_order_relaxed);
    while(value && !next.compare_exchange_weak(value,value==UINT64_MAX?0:value+1,
                                               std::memory_order_relaxed)) {}
    return value;
}
}
