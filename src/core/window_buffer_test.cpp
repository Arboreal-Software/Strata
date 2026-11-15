#include "strata/core/window_buffer.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
static int device=9,drains=0,frees=0,host_frees=0,mode=0;
extern "C" cudaError_t cudaGetDevice(int* out){*out=device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){if(mode==3&&d==9)return cudaErrorUnknown;device=d;return cudaSuccess;}
extern "C" cudaError_t cudaMalloc(void** out,size_t bytes){if(mode==1){*out=nullptr;return cudaErrorMemoryAllocation;}*out=std::malloc(bytes);return cudaSuccess;}
extern "C" cudaError_t cudaHostAlloc(void** out,size_t bytes,unsigned flags){
    assert((flags&cudaHostAllocMapped)&&(flags&cudaHostAllocPortable));return cudaMalloc(out,bytes);
}
extern "C" cudaError_t cudaHostGetDevicePointer(void** out,void* host,unsigned){
    if(mode==5)return cudaErrorUnknown;
    *out=host;return cudaSuccess;
}
extern "C" cudaError_t cudaFreeHost(void* pointer){
    assert(device==0);++host_frees;if(mode==6)return cudaErrorUnknown;
    std::free(pointer);return cudaSuccess;
}
extern "C" cudaError_t cudaDeviceSynchronize(){
    std::thread observer([]{(void)window_buffer_quarantine_count();});observer.join();
    ++drains;return mode==2&&device==1?cudaErrorUnknown:cudaSuccess;
}
extern "C" cudaError_t cudaFree(void* pointer){
    assert(device==0);++frees;
    std::thread observer([]{(void)window_buffer_quarantine_count();});observer.join();
    if(mode==4)return cudaErrorUnknown;
    std::free(pointer);return cudaSuccess;
}
static void run(int failure) {
    WindowResources capacity{{1024,1024},8192,8192,0,0,1},baseline{{0,0},0,0,0,0,0};
    WindowBudget budget(capacity,baseline);
    WindowResources parent_charge{{16,16},0,0,0,0,1};
    auto parent=budget.reserve(parent_charge,ReservationPurpose::private_work);assert(parent);
    OwnedWindowBuffer buffer;std::string error;
    assert(!allocate_window_buffer(0,32,{0,0},budget,parent.lease,buffer,error));
    assert(!buffer.data()&&window_buffer_quarantine_count()==0);
    mode=failure==1||failure==3?failure:0;
    const bool allocated=allocate_window_buffer(0,32,{0,1},budget,parent.lease,buffer,error);
    if(failure==1||failure==3) {
        assert(!allocated&&!buffer.data()&&window_buffer_quarantine_count()==1);
        assert(budget.snapshot().reserved.device_bytes[0]==48);return;
    }
    assert(allocated&&buffer.bytes()==32&&device==9);
    auto copy=buffer;void* pointer=buffer.data();buffer={};assert(frees==0&&copy.data()==pointer);
    mode=failure;copy={};
    if(failure) {
        assert(window_buffer_quarantine_count()==1&&budget.snapshot().reserved.device_bytes[0]==48);
        assert(!allocate_window_buffer(0,8,{0},budget,parent.lease,buffer,error));
        assert(failure!=2||frees==0);
    } else {
        assert(drains==2&&frees==1&&device==9&&window_buffer_quarantine_count()==0);
        assert(budget.snapshot().reserved.device_bytes[0]==16);
    }
}
static void mapped(int failure) {
    WindowResources capacity{{1024,1024},8192,8192,0,0,1},baseline{{0,0},0,0,0,0,0};
    WindowBudget budget(capacity,baseline);
    WindowResources charge{{16,16},0,0,0,0,1};
    auto parent=budget.reserve(charge,ReservationPurpose::private_work);assert(parent);
    OwnedWindowBuffer buffer;std::string error;mode=failure==5||failure==1||failure==3?failure:0;
    const bool success=allocate_window_buffer(0,32,{0,1},budget,parent.lease,buffer,error,WindowBufferMemory::mapped_host);
    assert(budget.snapshot().reserved.device_bytes[0]==16);
    assert(budget.snapshot().reserved.pinned_bytes==32);
    if(failure==5||failure==1||failure==3) {
        assert(!success&&!buffer.data()&&window_buffer_quarantine_count()==1&&host_frees==0);return;
    }
    assert(success&&buffer.data()==buffer.device_data());
    for(size_t i=0;i<32;++i)assert(static_cast<unsigned char*>(buffer.data())[i]==0);
    mode=failure;buffer={};assert(frees==0);
    if(failure)assert(window_buffer_quarantine_count()==1&&budget.snapshot().reserved.pinned_bytes==32);
    else assert(drains==2&&host_frees==1&&budget.snapshot().reserved.pinned_bytes==0);
}
int main(){for(int scenario=0;scenario<11;++scenario){pid_t child=fork();assert(child>=0);if(!child){
    if(scenario<5)run(scenario);else {const int failures[]{0,1,2,3,5,6};mapped(failures[scenario-5]);}
    _Exit(0);
}int result=0;assert(waitpid(child,&result,0)==child);assert(WIFEXITED(result)&&WEXITSTATUS(result)==0);}return 0;}
