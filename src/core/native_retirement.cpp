#include "strata/core/native_retirement.hpp"
#include <cuda_runtime.h>

namespace strata::core {
bool NativeRetirementContext::drain_after_producers_closed() noexcept {
    if(phase_==Phase::drained)return true;
    if(phase_!=Phase::prepared)return false;
    phase_=Phase::draining;
    int previous=-1;
    if(cudaGetDevice(&previous)!=cudaSuccess){phase_=Phase::failed;return false;}
    bool success=true;
    for(size_t i=0;i<size_;++i){
        if(cudaSetDevice(devices_[i])!=cudaSuccess){success=false;break;}
        if(cudaDeviceSynchronize()!=cudaSuccess){success=false;break;}
    }
    const bool restored=cudaSetDevice(previous)==cudaSuccess;
    phase_=success&&restored?Phase::drained:Phase::failed;
    return phase_==Phase::drained;
}
} // namespace strata::core
