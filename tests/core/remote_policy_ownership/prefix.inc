#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#define STRATA_REMOTE_OWNERSHIP_TEST 1
using cudaStream_t=void*;
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorNotReady=1,cudaMemcpyHostToDevice=2,cudaMemcpyDeviceToHost=3;
thread_local bool fail_allocations=false;
void* operator new(std::size_t n){if(fail_allocations)throw std::bad_alloc();if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
struct FakeCuda {
    int get_calls=0,set_calls=0,sync_calls=0,copy_calls=0,free_calls=0;
    int current=0,fail_set=0,fail_sync=0,fail_copy=0;
    bool fail_free=false,fail_destroy=false;
    bool get_fail=false,arm_alloc_on_copy_error=false,arm_alloc_on_sync_error=false;
    void (*during_sync)()=nullptr;
} gpu;
int cudaGetDevice(int* out){++gpu.get_calls;if(gpu.get_fail)return 7;*out=gpu.current;return 0;}
int cudaSetDevice(int v){++gpu.set_calls;if(gpu.fail_set==gpu.set_calls)return 7;gpu.current=v;return 0;}
int cudaStreamSynchronize(void*){++gpu.sync_calls;if(gpu.during_sync)gpu.during_sync();if(gpu.fail_sync==gpu.sync_calls){if(gpu.arm_alloc_on_sync_error)fail_allocations=true;return 7;}return 0;}
int cudaStreamQuery(void* p){return cudaStreamSynchronize(p);}
const char* cudaGetErrorString(int){return "fake error";}
int cudaMemcpyAsync(void*,const void*,size_t,int,void*){++gpu.copy_calls;if(gpu.fail_copy==gpu.copy_calls){if(gpu.arm_alloc_on_copy_error)fail_allocations=true;return 7;}return 0;}
int cudaFree(void*){++gpu.free_calls;return gpu.fail_free?7:0;}
int cudaFreeHost(void*){++gpu.free_calls;return gpu.fail_free?7:0;}
int cudaStreamDestroy(void*){++gpu.free_calls;return gpu.fail_destroy?7:0;}
namespace strata::kernels::cpu {
constexpr int H=4,FF=8,MAXT=8;
struct Format {bool had_gu=false,had_d=false;int had_block=0,gu_type=0,d_type=0,n_embd=4,n_ff=8;};
struct Layout {bool native=false;std::vector<Format> fmt=std::vector<Format>(1);int64_t blob_bytes(int64_t)const{return 8;}};
Layout& expert_layout(){static Layout layout;return layout;}
}
namespace strata::kernels {
struct NativeLayout{int had_block=0;const float* d_signs=nullptr;};
template<class...T>void hadamard_rows(T...){}
template<class...T>void quantize_q8_1_rows(T...){}
template<class...T>NativeLayout native_expert_layout(T...){return {};}
template<class...T>void native_expert_grouped(T...){}
template<class...T>void quantize_q8_0_scaled(T...){}
template<class...T>void moe_grouped_s2(T...){}
}
namespace strata::core {
struct NativeRuntimeAttachment { virtual ~NativeRuntimeAttachment()=default; };
class ExpertSource {};
class ExpertTierOpenObserver;
class ExpertCache {
public:
    int slots[2]={0,-1}; unsigned char data[8]{};
    int32_t slot_of(int64_t l,int64_t e)const{return l==0 && e>=0 && e<2?slots[e]:-1;}
    uint8_t* device_slot(int32_t s){return s==0?data:nullptr;}
    int64_t resident()const{return 1;}
    double gib()const{return 0;}
    bool set_strict_retirement(bool enabled,const std::vector<int>& devices,std::string&){return !enabled || !devices.empty();}
    void close(){}
};
class RemoteExperts;
class RemoteExpertOpt {
public:
    bool active()const{return false;}
    void prepare(const RemoteExperts&,void*)const{}
    static size_t metadata_bytes(){return 0;}
    bool reduce(RemoteExperts&,const void*,std::string&){return true;}
    void accumulate(const RemoteExperts&){}
};
}
