#pragma once
// Multi-GPU: make `device` current for a scope and restore the caller's device after it.  An object that owns CUDA
// streams, graphs and buffers on one device (a verify stage, the drafter) wraps its public calls in this, so the
// caller's thread may be on any device.  A negative device, or the device already current, does nothing.

#include <cuda_runtime.h>
#include "strata/core/strict_retirement.hpp"

namespace strata::core {

struct OnDevice {
    int previous = -1;
    // Checked mode is selection-only: never synchronize a device here (a
    // doorbell graph may require the caller to continue serving host work).
    explicit OnDevice(int device, bool checked = false) : checked_(checked) {
        if (checked_) {
            if (device < 0) retirement_failed("OnDevice invalid device");
            int cur = -1;
            retirement_check(true, cudaGetDevice(&cur), "OnDevice get device");
            if (cur != device) {
                retirement_check(true, cudaSetDevice(device), "OnDevice select device");
                previous = cur;
            }
            return;
        }
        int cur = 0;
        if (device >= 0 && cudaGetDevice(&cur) == cudaSuccess && cur != device && cudaSetDevice(device) == cudaSuccess)
            previous = cur;
    }
    ~OnDevice() {
        if (previous < 0) return;
        if (checked_) retirement_check(true, cudaSetDevice(previous), "OnDevice restore device");
        else cudaSetDevice(previous);
    }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
private:
    bool checked_ = false;
};

}  // namespace strata::core
