#pragma once
#include <cstddef>
#include <cstring>
using cudaError_t=int;using cudaStream_t=void*;using cudaEvent_t=void*;using cudaGraph_t=void*;using cudaGraphExec_t=void*;
inline constexpr int cudaSuccess=0,cudaErrorUnknown=1,cudaErrorInvalidValue=2;
inline constexpr unsigned cudaStreamNonBlocking=1;
enum cudaMemcpyKind{cudaMemcpyDefault,cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
namespace native_empty_test {inline int copies=0,fail_copy=0,syncs=0,fail_sync=0;}
inline cudaError_t cudaGetDevice(int* device){*device=0;return cudaSuccess;}
inline cudaError_t cudaSetDevice(int){return cudaSuccess;}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream,unsigned){*stream=reinterpret_cast<void*>(1);return cudaSuccess;}
inline cudaError_t cudaMemcpyAsync(void* destination,const void* source,size_t bytes,cudaMemcpyKind,cudaStream_t){
    if(++native_empty_test::copies==native_empty_test::fail_copy)return cudaErrorUnknown;
    if(bytes){std::memcpy(destination,source,bytes);}
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t){return ++native_empty_test::syncs==native_empty_test::fail_sync?cudaErrorUnknown:cudaSuccess;}
inline cudaError_t cudaDeviceSynchronize(){return ++native_empty_test::syncs==native_empty_test::fail_sync?cudaErrorUnknown:cudaSuccess;}
inline cudaError_t cudaGetLastError(){return cudaSuccess;}
inline const char* cudaGetErrorString(cudaError_t){return "recorded native transfer error";}

inline cudaError_t cudaMemsetAsync(void* destination,int value,size_t bytes,cudaStream_t){std::memset(destination,value,bytes);return cudaSuccess;}

inline constexpr unsigned cudaEventDisableTiming=2;
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event,unsigned){*event=reinterpret_cast<void*>(1);return cudaSuccess;}
inline cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t){return cudaSuccess;}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t,cudaEvent_t,unsigned){return cudaSuccess;}
