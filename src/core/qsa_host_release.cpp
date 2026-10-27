#include "strata/core/layer.hpp"
#include "strata/core/pin_budget.hpp"
#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace strata::core {
void qsa_host_account_release(uint64_t bytes);
bool qsa_state_release_host(QsaState& state,std::string& error) {
    auto fail = [&](cudaError_t status) {
        error=std::string("QSA host release: ")+cudaGetErrorString(status);return false;
    };
    // Stop at the first release failure. Successful releases are nulled so an
    // explicitly recovered allocator can retry without double freeing anything.
    if(state.host_step) {
        const auto status=host_free_guarded(state.host_step);
        if(status!=cudaSuccess) return fail(status);
        state.host_step=nullptr;
    }
    if(state.host_pos) {
        const auto status=host_free_guarded(state.host_pos);
        if(status!=cudaSuccess) return fail(status);
        state.host_pos=nullptr;
    }
    if(!state.host_allocation) return true;
    if(state.host_allocation_kind==QsaState::HostAllocation::cuda_alloc) {
        const auto status=host_free_guarded(state.host_allocation);
        if(status!=cudaSuccess) return fail(status);
    } else {
#ifndef _WIN32
        if(state.host_allocation_kind==QsaState::HostAllocation::registered_mapping) {
            const auto status=cudaHostUnregister(state.host_allocation);
            if(status!=cudaSuccess) return fail(status);
            state.host_allocation_kind=QsaState::HostAllocation::unregistered_mapping;
        }
        if(state.host_allocation_kind!=QsaState::HostAllocation::unregistered_mapping) {
            error="QSA host release: unknown allocation owner";return false;
        }
        if(munmap(state.host_allocation,size_t(state.host_allocation_bytes))!=0) {
            error="QSA host release: unmapping failed";return false;
        }
#else
        error="QSA host release: unsupported allocation owner";return false;
#endif
    }
    qsa_host_account_release(state.host_allocation_bytes);
    state.host_allocation=nullptr;state.host_allocation_bytes=0;
    state.host_allocation_kind=QsaState::HostAllocation::none;
    state.host=strata::kernels::KvHostPools{};
    return true;
}
} // namespace strata::core
