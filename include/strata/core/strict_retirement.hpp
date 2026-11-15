#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::core {
// Strict worker retirement never unwinds past an uncertain device operation.
[[noreturn]] inline void retirement_failed(const char* operation) noexcept {
    std::fprintf(stderr, "strata: strict retirement failed at %s; worker restart required\n", operation);
    std::fflush(stderr);
    std::_Exit(1);
}
inline void retirement_check(bool strict, cudaError_t result, const char* operation) noexcept {
    if (strict && result != cudaSuccess) retirement_failed(operation);
}
inline cudaError_t retirement_result(bool strict, cudaError_t result, const char* operation) noexcept {
    retirement_check(strict, result, operation);
    return result;
}
class StrictRetirementDevice {
public:
    explicit StrictRetirementDevice(bool enabled, int device) : enabled_(enabled) {
        if (!enabled_) return;
        retirement_check(true, cudaGetDevice(&previous_), "get retirement device");
        if (device < 0) retirement_failed("missing retirement device");
        retirement_check(true, cudaSetDevice(device), "select retirement device");
        retirement_check(true, cudaDeviceSynchronize(), "drain retirement device");
    }
    void finish() noexcept {
        if (!enabled_) return;
        retirement_check(true, cudaSetDevice(previous_), "restore retirement device");
        enabled_ = false;
    }
    ~StrictRetirementDevice() { finish(); }
    StrictRetirementDevice(const StrictRetirementDevice&) = delete;
    StrictRetirementDevice& operator=(const StrictRetirementDevice&) = delete;
private:
    bool enabled_;
    int previous_ = -1;
};
} // namespace strata::core
