#include "strata/core/window_model_owner.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static int device=1,copies=0,fail_copy=0,free_calls=0;
static bool fail_sync=false,fail_free=false,fail_malloc=false;
static std::set<void*> allocations;
extern "C" cudaError_t cudaGetDevice(int* p){*p=device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){device=d;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){return fail_sync?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t cudaHostAlloc(void** p,size_t n,unsigned int){*p=std::malloc(n);return *p?cudaSuccess:cudaErrorMemoryAllocation;}
extern "C" cudaError_t cudaFreeHost(void* p){std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMalloc(void** p,size_t n){if(fail_malloc)return cudaErrorMemoryAllocation;*p=std::malloc(n);if(!*p)return cudaErrorMemoryAllocation;allocations.insert(*p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* dst,const void* src,size_t n,cudaMemcpyKind){if(++copies==fail_copy)return cudaErrorUnknown;std::memcpy(dst,src,n);return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){check(device==1,"free on owning device");++free_calls;if(fail_free)return cudaErrorUnknown;allocations.erase(p);std::free(p);return cudaSuccess;}
namespace strata::platform {void advise_willneed(int,uint64_t,uint64_t){}}
namespace strata::core {
NativeDense::~NativeDense()=default;
NativeHead::~NativeHead()=default;
uint64_t NativeHead::allocated_bytes() const{return 0;}
bool NativeHead::release_checked(std::string&){return true;}
NativeEmbed::~NativeEmbed()=default;
bool NativeEmbed::release_checked(std::string&){return true;}
bool NativeDense::owns_weight_span(const void*,uint64_t)const{return false;}
bool NativeDense::release_checked(std::string&){return true;}
}
struct Pack {
    std::filesystem::path path;
    Pack(){char temp[]="/tmp/strata-hc-XXXXXX";check(mkdtemp(temp),"pack directory");path=temp;
        {std::ofstream f(path/"index.txt");f<<"# align 32 pool 160 tensors 3\n"
        "fixture.weight 0 2 0 32 0 32 8 1 0 0 0 0 0 0 0 0 0 0\n"
        "blk.0.hc_attn_down.weight 0 4 32 64 32 64 32 1 0 0 0 0 0 0 0 0 0 0\n"
        "blk.0.hc_ffn_down.weight 0 4 96 64 96 64 32 1 0 0 0 0 0 0 0 0 0 0\n";}
        {std::ofstream f(path/"dense.bin",std::ios::binary);float values[8]={1,2,3,4,5,6,7,8};f.write(reinterpret_cast<const char*>(values),32);
         uint16_t h[64];for(auto& x:h)x=0x3f80;f.write(reinterpret_cast<const char*>(h),sizeof h);}
    }
    ~Pack(){std::filesystem::remove_all(path);}
};
static const std::set<std::string> skip={"blk.0.hc_attn_down.weight","blk.0.hc_ffn_down.weight"};
static ModelStageResources make_model(const Pack& pack,WindowBudget& budget){
    ModelStageResources r;r.device=1;r.arena=std::malloc(32);r.arena_bytes=32;
    r.weights=std::make_unique<WeightTable>();r.native=std::make_unique<NativeDense>();std::string error;
    check(r.weights->load(pack.path.string(),r.arena,32,error,&skip),"canonical load with skipped HC");
    r.reservation=budget.reserve({{0,1056},0,0,0,0,0},ReservationPurpose::foreground).lease;
    check(bool(r.reservation),"model reservation");copies=0;return r;
}
int main(){
 const WindowResources cap{{4096,4096},0,0,0,0,0},zero{{0,0},0,0,0,0,0};
 for(int failure=0;failure<=7;++failure){
    Pack pack;WindowBudget budget(cap,zero);auto r=make_model(pack,budget);std::string error;uint64_t bytes=99;
    fail_copy=failure>=1&&failure<=4?failure:0;fail_sync=failure==5;fail_malloc=failure==6;
    if(failure==7)std::filesystem::resize_file(pack.path/"dense.bin",96); // second row truncated
    const bool ok=r.weights->load_hc_q8(pack.path.string(),bytes,error);
    fail_copy=0;fail_sync=false;fail_malloc=false;
    if(failure){
      check(!ok&&!r.weights->load_complete()&&bytes==0,"failed HC load revokes readiness and output");
      for(const auto& name:skip){const auto* w=r.weights->find(name);check(w&&!w->resident&&!w->data&&!w->q8_scale,"no partial HC publication");}
      std::shared_ptr<const OwnedModelStage> owner;
      check(!adopt_model_stage(r,owner,error)&&r.arena&&!owner,"partial HC model cannot be adopted");
      check(!r.weights->load_hc_q8(pack.path.string(),bytes,error),"failed HC attempt is not reused");
      r.weights.reset();check(allocations.empty(),"failed HC allocation retired by destructor");std::free(r.arena);
    }else{
      check(ok&&bytes==1024&&r.weights->auxiliary_bytes()==1024,"all auxiliary bytes accounted including padding");
      for(const auto& name:skip){const auto* w=r.weights->find(name);check(w&&w->resident&&w->q8_scale,"HC publication complete");
        check(static_cast<const int8_t*>(w->data)[0]==127&&w->q8_scale[0]==1.f/127.f,"quantized bytes preserved");}
      check(r.weights->load_hc_q8(pack.path.string(),bytes,error)&&allocations.size()==1,"successful repeat is allocation-free");
      r.device=0;std::shared_ptr<const OwnedModelStage> wrong_device;
      check(!adopt_model_stage(r,wrong_device,error),"auxiliary owner device mismatch refused");r.device=1;
      auto& refs=const_cast<std::map<std::string,WeightRef>&>(r.weights->all());auto& w=refs.at(*skip.begin());const float* saved=w.q8_scale;
      w.q8_scale=reinterpret_cast<const float*>(r.arena);std::shared_ptr<const OwnedModelStage> owner;
      check(!adopt_model_stage(r,owner,error),"external scale pointer refused");w.q8_scale=saved;
      r.reservation.reset();r.reservation=budget.reserve({{0,32},0,0,0,0,0},ReservationPurpose::foreground).lease;
      check(!adopt_model_stage(r,owner,error),"auxiliary undercharge refused");r.reservation.reset();
      r.reservation=budget.reserve({{0,1056},0,0,0,0,0},ReservationPurpose::foreground).lease;
      check(adopt_model_stage(r,owner,error),"complete HC model adopted");
      check(owner->owns_weight_span(saved,4,false),"owner retains auxiliary span");
      device=0;owner.reset();check(device==0&&allocations.empty()&&budget.snapshot().reserved==zero,"owner releases auxiliary allocation and charge on right device");device=1;
    }
 }
 // Isolate sticky uncertainty: freeing an HC allocation can fail without proof of release.
 const auto pid=fork();check(pid>=0,"failure isolation");if(pid==0){
    Pack pack;WindowBudget budget(cap,zero);auto r=make_model(pack,budget);std::string error;uint64_t bytes;
    check(r.weights->load_hc_q8(pack.path.string(),bytes,error),"quarantine fixture loads");
    std::shared_ptr<const OwnedModelStage> owner;check(adopt_model_stage(r,owner,error),"quarantine fixture adopts");
    const int before=free_calls;fail_free=true;owner.reset();
    check(free_calls==before+1&&model_stage_quarantine_count()==1&&allocations.size()==1,"failed auxiliary free quarantines without arena free or retry");
    check(budget.snapshot().reserved.device_bytes[1]==1056,"quarantine retains full charge");
    std::exit(0);
 }
 int status;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"quarantine case passed");
 std::puts("weights_hc_owner_test passed");
}
