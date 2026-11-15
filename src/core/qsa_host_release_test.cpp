#include "strata/core/layer.hpp"
#include "strata/core/pin_budget.hpp"
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>

static int free_calls=0,unregister_calls=0,fail_free=0;
static bool fail_unregister=false;
static uint64_t released_bytes=0;
extern "C" cudaError_t cudaFreeHost(void* p) {
    if(++free_calls==fail_free) return cudaErrorUnknown;
    std::free(p);return cudaSuccess;
}
extern "C" cudaError_t cudaHostUnregister(void*) {
    ++unregister_calls;return fail_unregister?cudaErrorUnknown:cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t) {return "injected release failure";}
namespace strata::core {void qsa_host_account_release(uint64_t bytes){released_bytes+=bytes;}}
static void check(bool b,const char* s){if(!b){std::fprintf(stderr,"FAIL: %s\n",s);std::exit(1);}}
int main(){
    using namespace strata::core;
    QsaState state;std::string error;
    state.host_step=static_cast<int32_t*>(std::malloc(64));
    state.host_pos=static_cast<int32_t*>(std::malloc(64));
    state.host_allocation=std::malloc(256);state.host_allocation_bytes=256;
    state.host_allocation_kind=QsaState::HostAllocation::cuda_alloc;
    auto* position=state.host_pos;auto* storage=state.host_allocation;
    fail_free=2;
    check(!qsa_state_release_host(state,error),"failed staging free reported");
    check(!state.host_step && state.host_pos==position && state.host_allocation==storage && released_bytes==0,
          "partial release retains remaining ownership and KV accounting");
    fail_free=0;
    check(qsa_state_release_host(state,error) && !state.host_pos && !state.host_allocation && released_bytes==256,
          "explicit recovery releases remaining original owners once");
    const int count=free_calls;
    check(qsa_state_release_host(state,error) && free_calls==count && released_bytes==256,"release is idempotent");
    state.host_allocation=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(state.host_allocation!=MAP_FAILED,"mapped fixture allocated");
    state.host_allocation_bytes=4096;state.host_allocation_kind=QsaState::HostAllocation::registered_mapping;
    fail_unregister=true;
    check(!qsa_state_release_host(state,error) && state.host_allocation && released_bytes==256,"unregister failure retains mapped ownership");
    fail_unregister=false;
    check(qsa_state_release_host(state,error) && !state.host_allocation && released_bytes==4352 && unregister_calls==2,
          "registered mapping unregistered then unmapped and uncharged");
    auto& pins=pin_detail::st();
    auto* pinned=std::malloc(128);
    {std::lock_guard lock(pins.mu);pins.sim_budget=1024;check(pin_detail::sim_charge(pins,pinned,128),"simulated pin charge");}
    fail_free=free_calls+1;
    check(host_free_guarded(pinned)!=cudaSuccess && pins.sim_used==128,"failed free preserves simulated pin accounting");
    fail_free=0;
    check(host_free_guarded(pinned)==cudaSuccess && pins.sim_used==0,"successful free removes pin accounting");
    std::puts("qsa_host_release_test passed");
}
