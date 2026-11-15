#include "strata/core/native_dense.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using strata::core::NativeDense;
namespace strata::core {
struct NativeDenseReleaseTestAccess {
    static void seed(NativeDense& dense,void* scratch,std::vector<void*> weights) {
        dense.scratch_=scratch;dense.weights_=std::move(weights);
        dense.weight_spans_valid_=true;
        for(auto* p:dense.weights_)dense.weight_spans_.emplace_back(p,32);dense.bytes_=200;dense.allocated_=400;
    }
};
}
static std::vector<void*> frees;
static void* failure=nullptr;
extern "C" cudaError_t cudaFree(void* p){frees.push_back(p);return p==failure?cudaErrorUnknown:cudaSuccess;}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected free error";}
namespace strata::kernels {void native_q8_0_packed_unregister(const void*){}}
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
int main() {
    int a,b,c;std::string error;
    {
        NativeDense dense;strata::core::NativeDenseReleaseTestAccess::seed(dense,&a,{&b,&c});
        check(dense.owns_weight_span(&b,32)&&!dense.owns_weight_span(&b,33)&&!dense.owns_weight_span(&a,1),"native allocation extent excludes scratch");
        check(dense.release_checked(error),"checked model release succeeds");
        check(!dense.owns_weight_span(&b,1),"retired native model proves no live span");
        check(frees==std::vector<void*>({&a,&b,&c})&&dense.allocated_bytes()==0,"all buffers freed and accounting cleared on success");
        check(dense.release_checked(error)&&frees.size()==3,"successful retirement is idempotent");
    }
    check(frees.size()==3,"destructor does not double free retired model");
    frees.clear();failure=&b;
    {
        NativeDense dense;strata::core::NativeDenseReleaseTestAccess::seed(dense,&a,{&b,&c});
        check(!dense.release_checked(error)&&frees==std::vector<void*>({&a,&b}),"partial failure stops before later buffers");
        check(dense.allocated_bytes()==400,"failed retirement retains conservative accounting");
        check(!dense.release_checked(error)&&frees.size()==2,"uncertain release is never retried");
    }
    check(frees.size()==2,"destructor cannot retry uncertain release");
    std::puts("native_dense_release_test passed");
}
