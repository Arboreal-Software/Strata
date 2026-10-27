#pragma once
#include <cuda_runtime.h>

namespace strata::core {
// Unlike legacy best-effort OnDevice, every setup error is sticky. Successful
// callers must explicitly finish before publishing their transaction result.
class WindowDeviceScope {
public:
    explicit WindowDeviceScope(int target) {
        status_ = cudaGetDevice(&previous_);
        if (status_ != cudaSuccess) return;
        if (target < 0) { valid_target_ = false; return; }
        if (previous_ != target) {
            restore_ = true;
            status_ = cudaSetDevice(target);
        }
    }
    ~WindowDeviceScope() { if (!finished_ && restore_) (void)cudaSetDevice(previous_); }
    WindowDeviceScope(const WindowDeviceScope&) = delete;
    WindowDeviceScope& operator=(const WindowDeviceScope&) = delete;
    bool ok() const { return valid_target_ && status_ == cudaSuccess; }
    bool finish() {
        if (!finished_ && restore_) {
            const auto restored = cudaSetDevice(previous_);
            if (status_ == cudaSuccess) status_ = restored;
        }
        finished_ = true;
        return ok();
    }
private:
    int previous_ = -1;
    bool restore_ = false, finished_ = false;
    bool valid_target_ = true;
    cudaError_t status_ = cudaSuccess;
};
} // namespace strata::core
