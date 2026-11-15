// Compile the real CUDA translation unit so the test exercises its private
// ownership registries and checked release functions without launching a kernel.
#include "../kernels/cuda/native_mmvq.cu"
#include <cstdio>
#include <cstdlib>
#include <map>
static std::map<void*,size_t> fixture_allocations;
static int fixture_frees=0,fixture_fail_free=0;
static void fixture_check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
cudaError_t native_head_test_free(void* p){++fixture_frees;if(fixture_frees==fixture_fail_free)return cudaErrorUnknown;fixture_check(fixture_allocations.erase(p)==1,"registry frees exactly its owned allocation");std::free(p);return cudaSuccess;}
static void* fixture_alloc(size_t n){void* p=std::malloc(n);fixture_allocations[p]=n;return p;}
static void fixture_register(const void* key){
 using namespace strata::kernels;
 // One Q6 block has 770 bytes after the registry's 256-byte plane alignment.
 auto* q6=static_cast<uint8_t*>(fixture_alloc(770));
 q6_packed_registry()[key]=Q6Packed{q6,reinterpret_cast<const int*>(q6),reinterpret_cast<const int*>(q6+256),reinterpret_cast<const int8_t*>(q6+512),reinterpret_cast<const half*>(q6+768),256,1};
 auto* q8=static_cast<int8_t*>(fixture_alloc(34));
 q8_sm60_owned()[key]=Q8HeadOwned{q8,34,false};
 q8_packed_registry()[key]=Q8Packed{q8,reinterpret_cast<const half*>(q8+32),32,1};
}
int main(){using namespace strata::kernels;int keys[3];
 for(int failure=0;failure<3;++failure){const void* key=&keys[failure];fixture_register(key);fixture_frees=0;fixture_fail_free=failure;
  fixture_check(native_head_packed_allocation_bytes(key)==804,"packed allocation accounting includes aligned Q6 and Q8 copies");
  const bool released=native_head_packed_release_checked(key);
  if(!failure){fixture_check(released&&fixture_allocations.empty()&&native_head_packed_allocation_bytes(key)==0,"successful release clears both registries");fixture_check(native_head_packed_release_checked(key)&&fixture_frees==2,"successful release idempotent");
   fixture_check(!q6_packed_registry().count(key)&&!q8_packed_registry().count(key),"freed entries cannot remain active under reused key");
   auto* replacement=static_cast<int8_t*>(fixture_alloc(1024*2048/32*34));native_q8_0_packed_register(key,replacement,1024,2048);
   fixture_check(q8_packed_registry().at(key).qs==replacement,"reused key resolves only to fresh allocation");native_q8_0_packed_unregister(key);cudaFree(replacement);
  }
  else{fixture_check(!released&&fixture_frees==failure,"release stops at uncertain allocation");const auto remaining=failure==1?804:34;
   fixture_check(native_head_packed_allocation_bytes(key)==size_t(remaining),"uncertain allocation remains counted");
   fixture_check(!native_head_packed_release_checked(key)&&fixture_frees==failure,"registry never retries uncertain free");
   bool rejected=false;
   try { if(failure==1) q6_packed_mmvq(key,nullptr,nullptr,256,1,1,nullptr); else q8_packed_mmvq(key,nullptr,nullptr,32,1,1,nullptr); }
   catch(const std::runtime_error&){rejected=true;}
   fixture_check(rejected,"failed allocation is unavailable to numeric lookup before launch");
   if(failure==2){
    fixture_check(!q6_packed_registry().count(key),"earlier successful free removes its active entry before later failure");
    rejected=false;try{native_q8_0_packed_register(key,key,1024,2048);}catch(const std::runtime_error&){rejected=true;}
    fixture_check(rejected,"failed key cannot be overwritten by address reuse");
    native_q8_0_packed_unregister(key);fixture_check(q8_packed_registry().at(key).release_failed,"unregister cannot erase failed key tombstone");
   }else{fixture_check(!q6_packed_registry().at(key).base||q6_packed_registry().at(key).release_failed,"failed Q6 retained only as unusable record");}
   native_q6_k_unpack(key);native_q8_0_sm60_release(key);
   // A legacy wrapper for the independent Q8 allocation may safely retire it
   // after a Q6 failure; the failed allocation itself must never be retried.
   fixture_check(native_head_packed_allocation_bytes(key)==size_t(failure==1?770:34),"legacy wrappers retain failed allocation");
   for(auto [p,n]:fixture_allocations)std::free(p);fixture_allocations.clear();q6_packed_registry().erase(key);q8_sm60_owned().erase(key);q8_packed_registry().erase(key);
  }
 }
 std::puts("native_head_registry_test passed");
}
