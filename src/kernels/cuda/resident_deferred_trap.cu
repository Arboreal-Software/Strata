// NEW qualification-only CUDA TU. Never compiled into default/OFF artifact.
#include <cuda_runtime.h>
#if !defined(STRATA_RESIDENT_FAULT_QUALIFICATION) || !defined(__linux__) || defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL) || defined(STRATA_USE_SYCL) || defined(STRATA_GPU_SYCL)
#error "Deferred trap is only the explicit Linux CUDA qualification target"
#endif
namespace strata::core::resident_deferred_fault {
namespace {
__global__ void private_stage_trap() { asm volatile("trap;"); }
}
// No destination buffer, allocation, request-controlled status, sync, process
// exit, reset or global arm. Caller owns the stream/context and retains owners.
cudaError_t launch_private_stage_trap(cudaStream_t stream) noexcept {
    private_stage_trap<<<1,1,0,stream>>>();
    return cudaGetLastError(); // actual launch status; NOT execution completion
}
static_assert(int(cudaErrorIllegalInstruction)==715 && int(cudaErrorLaunchFailure)==719,
              "expected-failure validator pins CUDA13 execution status values");
}
