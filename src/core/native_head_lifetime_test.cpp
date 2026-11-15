#include "strata/core/native_head.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/geometry.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <unistd.h>
#include <sys/wait.h>
using namespace strata::core;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static int device=1,allocs=0,copies=0,frees=0,host_frees=0,fail_alloc=0,fail_copy=0,fail_free=0;
static bool fail_host=false,fail_alias=false,fail_sync=false,fail_host_free=false,pack_throw=false;
static bool skip_head_pack=false,integrated_device=false;
static int sync_calls=0,fail_sync_at=0;
static std::map<void*,size_t> live,host_live;
static std::map<const void*,void*> packed;
extern "C" cudaError_t cudaMalloc(void** p,size_t n){if(++allocs==fail_alloc){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){check(device==1,"device allocation released on owning device");if(++frees==fail_free)return cudaErrorUnknown;check(live.erase(p)==1,"one device allocation released once");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* d,const void* s,size_t n,cudaMemcpyKind){if(++copies==fail_copy)return cudaErrorUnknown;std::memcpy(d,s,n);return cudaSuccess;}
extern "C" cudaError_t cudaHostAlloc(void** p,size_t n,unsigned int){if(fail_host){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);host_live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFreeHost(void* p){++host_frees;if(fail_host_free)return cudaErrorUnknown;check(host_live.erase(p)==1,"one pinned allocation released once");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaHostGetDevicePointer(void** p,void* h,unsigned int){if(fail_alias){*p=nullptr;return cudaErrorUnknown;}*p=h;return cudaSuccess;}
extern "C" cudaError_t cudaGetDevice(int* p){*p=device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){device=d;return cudaSuccess;}
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p,int d){check(d==device,"spill inspects actual current device");*p={};p->integrated=integrated_device;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){++sync_calls;return (fail_sync||sync_calls==fail_sync_at)?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t cudaGetLastError(){return cudaSuccess;}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected";}
namespace strata::kernels {
bool native_mmvq_supported(int t) noexcept{return t==8;}
size_t native_mmvq_weight_bytes(int,int a,int b){return size_t(a)*b/32*34;}
size_t native_q8_1_bytes(int a,int b){return size_t(a)*b/32*36;}
bool native_q6_k_packed_enabled(){return false;}
bool native_q6_k_pack(const void*,int,int,const char*){return false;}
bool native_q8_0_sm60_pack(const void* key,int,int,const char*){if(skip_head_pack)return false;void* p=nullptr;if(cudaMalloc(&p,128)!=cudaSuccess)return false;packed[key]=p;if(pack_throw)throw std::bad_alloc();return true;}
size_t native_head_packed_allocation_bytes(const void* key){return packed.count(key)?128:0;}
bool native_head_packed_release_checked(const void* key){auto it=packed.find(key);if(it==packed.end())return true;if(cudaFree(it->second)!=cudaSuccess)return false;packed.erase(it);return true;}
bool embed_type_supported(int t)noexcept{return t==30;}
size_t iq_row_bytes(int,int64_t n)noexcept{return size_t(n)*2;}
}
template<class T>static void put(std::vector<uint8_t>& b,T v){auto n=b.size();b.resize(n+sizeof v);std::memcpy(b.data()+n,&v,sizeof v);}
static void str(std::vector<uint8_t>& b,const std::string& s){put<uint64_t>(b,s.size());b.insert(b.end(),s.begin(),s.end());}
static std::string fixture(){
 std::vector<uint8_t>b;put<uint32_t>(b,0x46554747);put<uint32_t>(b,3);put<uint64_t>(b,2);put<uint64_t>(b,7);
 str(b,"general.architecture");put<uint32_t>(b,8);str(b,"qwen4exp");const auto& g=kCompiledGeometry;
 for(const auto& [key,value]:std::vector<std::pair<std::string,int64_t>>{{"block_count",g.n_layers},{"embedding_length",g.n_embd},{"expert_count",g.n_expert},{"expert_used_count",g.n_expert_used},{"attention.head_count",g.n_head},{"attention.head_count_kv",g.n_head_kv}}){str(b,"qwen4exp."+key);put<uint32_t>(b,10);put<uint64_t>(b,value);}
 str(b,"output.weight");put<uint32_t>(b,2);put<uint64_t>(b,256);put<uint64_t>(b,32);put<uint32_t>(b,8);put<uint64_t>(b,0);
 str(b,"token_embd.weight");put<uint32_t>(b,2);put<uint64_t>(b,256);put<uint64_t>(b,32);put<uint32_t>(b,30);put<uint64_t>(b,8704);
 b.resize((b.size()+31)/32*32+8704+16384,0);
 const auto path="/tmp/strata-head-fixture-"+std::to_string(getpid())+".gguf";std::ofstream(path,std::ios::binary).write(reinterpret_cast<const char*>(b.data()),b.size());return path;
}
static void reset(){skip_head_pack=integrated_device=false;sync_calls=fail_sync_at=0;check(live.empty()&&host_live.empty()&&packed.empty(),"all normal fixture allocations released");allocs=copies=frees=host_frees=fail_alloc=fail_copy=fail_free=0;fail_host=fail_alias=fail_sync=fail_host_free=pack_throw=false;device=1;}
static void reclaim_quarantine(){for(auto [p,n]:live)std::free(p);live.clear();for(auto [p,n]:host_live)std::free(p);host_live.clear();packed.clear();}
int baseline_main(){const auto path=fixture();std::string error;
 for(int scenario=0;scenario<6;++scenario){reset();fail_host=scenario==1;fail_alloc=scenario==1?1:scenario==2?2:0;fail_copy=scenario==3?1:0;fail_sync=scenario==4;pack_throw=scenario==5;
  {NativeHead h;check(!h.ready_for_adoption(),"unloaded optional head not adoptable");const bool ok=h.load({path},256,32,error);
   check(ok==(scenario==0),"head load success/failure matches injection");check(h.ready_for_adoption()==ok,"head readiness tracks complete load");
   fail_sync=false;
   if(ok){check(h.allocated_bytes()==8704+288+128,"head allocation includes scratch and packed copy");const int before=allocs;check(!h.load({path},256,32,error)&&allocs==before&&h.ready_for_adoption(),"duplicate head load preserves ready model");
    device=0;h.unload();check(device==0&&!h.loaded()&&h.allocated_bytes()==0&&h.weight_bytes()==8704,"unload releases on owning device and preserves informational weight size");device=1;check(h.load({path},256,32,error),"real head reload after unload");}
  }check(live.empty()&&packed.empty(),"failed or successful head teardown releases owned allocations");
 }
 for(int failed_free=1;failed_free<=3;++failed_free){reset();auto h=std::make_unique<NativeHead>();check(h->load({path},256,32,error),"release failure head loads");fail_free=failed_free;
  check(!h->release_checked(error)&&!h->ready_for_adoption(),"head uncertain cleanup fences adoption");
  check(h->allocated_bytes()==uint64_t(failed_free==1?9120:failed_free==2?8992:8704),"partial retirement clears successful allocations while retaining uncertain bytes");int calls=frees;check(!h->release_checked(error)&&frees==calls,"head cleanup failure is sticky");h.reset();check(frees==calls,"destructor does not retry uncertain free");reclaim_quarantine();
 }
 for(int scenario=0;scenario<6;++scenario){reset();fail_host=scenario==1||scenario==3||scenario==4;fail_alias=scenario==2;fail_copy=scenario==3?1:0;fail_alloc=scenario==4?1:0;fail_sync=scenario==5;
  {NativeEmbed e;const bool ok=e.load({path},256,32,error);
#if defined(STRATA_USE_HIP)
   const bool expected=scenario<=2;
#else
   const bool expected=scenario<=1;
#endif
   check(ok==expected&&e.ready_for_adoption()==ok,"embedding readiness follows actual mapping/fallback completion");
   if(ok){check(e.device()==1,"embedding retains allocation device");const bool fallback=scenario==1||scenario==2;check(e.allocated_device_bytes()==(fallback?16384:0)&&e.allocated_host_bytes()==(fallback?0:16384)&&e.allocated_pinned_bytes()==e.allocated_host_bytes(),"embedding host/pinned/device accounting distinguishes fallback");
    check(!e.load({path},256,32,error)&&e.ready_for_adoption(),"duplicate embedding load preserves ready model");set_native_embed(&e);}
   fail_sync=false;device=0;
  }check(device==0&&live.empty()&&host_live.empty()&&!native_embed(),"embedding destructor releases on owner device and clears global alias");
 }
 for(bool fallback:{false,true}){reset();fail_host=fallback;auto e=std::make_unique<NativeEmbed>();check(e->load({path},256,32,error),"embedding retirement fixture loads");fail_free=fallback?1:0;fail_host_free=!fallback;
  check(!e->release_checked(error)&&!e->ready_for_adoption(),"embedding free failure fences adoption");const int calls=frees+host_frees;check(!e->release_checked(error),"embedding failure sticky");e.reset();check(frees+host_frees==calls,"embedding destructor does not retry uncertain release");reclaim_quarantine();
 }
#if defined(STRATA_USE_HIP)
 reset();fail_alias=true;fail_host_free=true;
 {auto e=std::make_unique<NativeEmbed>();check(!e->load({path},256,32,error)&&!e->ready_for_adoption(),"HIP fallback host retirement failure rejects model");
  check(e->allocated_device_bytes()==16384&&e->allocated_host_bytes()==16384&&e->allocated_pinned_bytes()==16384,"uncertain fallback retains and counts both allocations");
  const int before=frees+host_frees;e.reset();check(frees+host_frees==before,"HIP fallback failure does not retry uncertain pinned free");}
 reclaim_quarantine();
#endif
 reset();std::remove(path.c_str());std::puts("native_head_lifetime_test passed");return 0;}

int main() {
 baseline_main();
 const auto path=fixture();std::string error;
 // First device allocation failure is NOT complete allocation failure: this
 // independent positive executes the genuine mapped-host load fallback.
 reset();fail_alloc=1;skip_head_pack=true;
 {
  strata::GgufFile gguf(path);const auto* tensor=gguf.find("output.weight");
  NativeHead h;
  check(h.load({path},256,32,error)&&h.ready_for_adoption()&&h.loaded()&&h.on_host()&&h.device()==1,
        "ordinary first device failure loads ready mapped head");
  check(allocs==2&&live.size()==1&&host_live.size()==1&&packed.empty(),
        "fallback owns only device scratch and one plain mapped payload");
  check(h.weight_bytes()==8704&&h.allocated_bytes()==288&&h.allocated_host_bytes()==8704&&h.allocated_pinned_bytes()==8704,
        "ordinary load fallback accounts device scratch separately from host and pinned weight");
  check(tensor&&tensor->shape==std::vector<uint64_t>({256,32})&&
        std::memcmp(h.weights(),gguf.tensor_data(*tensor),8704)==0,
        "ordinary load fallback retains exact actual GGUF payload");
  device=0;h.unload();
  check(device==0&&!h.ready_for_adoption()&&!h.loaded()&&h.allocated_bytes()==0&&
        h.allocated_host_bytes()==0&&h.allocated_pinned_bytes()==0&&
        frees==1&&host_frees==1&&live.empty()&&host_live.empty()&&packed.empty(),
        "ordinary load fallback checked teardown uses owner device and restores caller");
 }
 reset();
 // Original load/retirement loops are unchanged. Only these new spill cases
 // disable the recording pack helper to represent a genuinely movable head.
 for(int fault=0;fault<5;++fault) {
  reset();skip_head_pack=true;NativeHead h;
  check(h.load({path},256,32,error),"ordinary movable head loaded");
  const auto* original=h.weights();const auto allocated=h.allocated_bytes();
  if(fault==1)fail_host=true;
  if(fault==2)fail_alias=true;
  if(fault==3)fail_copy=copies+1;
  if(fault==4)device=0;
  check(h.to_host(error)==(fault==0),"ordinary checked spill success/refusal");
  if(!fault) {
   check(h.on_host()&&h.allocated_bytes()==288&&h.allocated_host_bytes()==8704&&
         h.allocated_pinned_bytes()==8704,"ordinary final mapped accounting");
   for(size_t i=0;i<8704;++i)check(static_cast<const unsigned char*>(h.weights())[i]==0,"ordinary mapped head bytes unchanged");
   check(!h.to_host(error),"already mapped head does not spill again");
  } else check(!h.on_host()&&h.weights()==original&&h.allocated_bytes()==allocated&&h.loaded(),"ordinary rollback preserves entry state");
  device=1;
 }
 reset();skip_head_pack=true;integrated_device=true;
 {NativeHead h;check(h.load({path},256,32,error),"integrated recording head load");check(!h.to_host(error)&&host_live.empty(),"unified-memory spill remains refused");}
 for(int fault=0;fault<4;++fault) {
  const pid_t child=fork();check(child>=0,"fork ordinary uncertain spill");
  if(!child) {
   reset();skip_head_pack=true;NativeHead h;check(h.load({path},256,32,error),"uncertain ordinary head loaded");
   if(fault==0)fail_sync=true;
   if(fault==1)fail_free=frees+1;
   if(fault==2){fail_alias=true;fail_host_free=true;}
   if(fault==3){fail_copy=copies+1;fail_sync_at=sync_calls+2;}
   (void)h.to_host(error);std::_Exit(9);
  }
  int status=0;check(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==86,"uncertain ordinary spill failstops without returning/unwinding");
 }
 reset();std::remove(path.c_str());std::puts("native_head_checked_spill_test passed");
}
