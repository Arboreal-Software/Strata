#pragma once
#include "strata/core/strict_retirement.hpp"
#include <algorithm>
#include <vector>
#include <string>

namespace strata::core {
// Event waits and allocation rollback must select the resource's device without
// draining unrelated executing kernels (which can be waiting on host producers).
class ExpertResourceDevice {
public:
    ExpertResourceDevice(bool strict, int device) : active_(strict) {
        if (!active_) return;
        retirement_check(true, cudaGetDevice(&previous_), "expert resource current device");
        if (device < 0) retirement_failed("missing expert resource device");
        retirement_check(true, cudaSetDevice(device), "expert resource device");
    }
    ~ExpertResourceDevice() {
        if (active_) retirement_check(true, cudaSetDevice(previous_), "expert resource restoration");
    }
private:
    bool active_;
    int previous_ = -1;
};
inline bool valid_retirement_topology(const std::vector<int>& devices, std::string& error) {
    if (devices.empty()) { error = "strict retirement needs the consuming devices"; return false; }
    for (size_t i = 0; i < devices.size(); ++i)
        if (devices[i] < 0 || std::find(devices.begin(), devices.begin() + i, devices[i]) != devices.begin() + i) {
            error = "invalid strict retirement device topology"; return false;
        }
    return true;
}
inline void retirement_capture_device(bool strict, const std::vector<int>& devices, int& device) {
    if (!strict) return;
    retirement_check(true, cudaGetDevice(&device), "expert allocation device");
    if (std::find(devices.begin(), devices.end(), device) == devices.end())
        retirement_failed("expert allocation device absent from consumer topology");
}
class ExpertRetirementScope {
public:
    ExpertRetirementScope(bool strict, int owner, const std::vector<int>& devices) : active_(strict && owner >= 0) {
        if (!active_) return;
        retirement_check(true, cudaGetDevice(&previous_), "expert retirement current device");
        for (int device : devices) {
            retirement_check(true, cudaSetDevice(device), "expert consumer selection");
            retirement_check(true, cudaDeviceSynchronize(), "expert consumer drain");
        }
        retirement_check(true, cudaSetDevice(owner), "expert owner selection");
    }
    ~ExpertRetirementScope() {
        if (active_) retirement_check(true, cudaSetDevice(previous_), "expert retirement device restoration");
    }
    ExpertRetirementScope(const ExpertRetirementScope&) = delete;
    ExpertRetirementScope& operator=(const ExpertRetirementScope&) = delete;
private:
    bool active_;
    int previous_ = -1;
};
} // namespace strata::core
