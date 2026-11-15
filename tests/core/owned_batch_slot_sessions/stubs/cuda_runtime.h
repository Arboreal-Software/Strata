#pragma once
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
namespace seam {
inline int device=0,zero_calls=0,zero_fail=0,allocation_calls=0,allocation_fail=0,partial_fail=0;
inline size_t native_used[2]={0,0};
inline int upload_calls=0,upload_fail_at=0;
inline int stream_fail=0,stream_calls=0;inline bool stream_fail_once=false;
inline int sessions_alive=0,sessions_destroyed=0;
inline std::vector<std::string> events;
// Extension-only bounded recording backend. It models copy ordering, not DMA.
inline bool image_mode=false;
struct Copy {void* dst=nullptr;const void* src=nullptr;size_t bytes=0;uintptr_t stream=0;int device=-1;uint64_t sequence=0;};
inline std::array<Copy,4096> copies{};
inline size_t copy_count=0;inline uint64_t copy_sequence=0;
inline int enqueue_calls=0,enqueue_fail=0,drain_fail=0;inline bool fail_queued_drain=false;
inline std::array<bool,2> uncertain{};
inline void (*host_lifetime)(bool)=nullptr;
inline int drain(uintptr_t stream,uint64_t through=UINT64_MAX,bool whole_device=false){
    if(device<0||device>1)std::abort();
    if(uncertain[size_t(device)])return 1;
    if(++stream_calls==drain_fail||(fail_queued_drain&&copy_count)){uncertain[size_t(device)]=true;return 1;}
    size_t keep=0;
    for(size_t i=0;i<copy_count;++i){auto op=copies[i];
        if(op.device==device&&(whole_device||op.stream==stream)&&op.sequence<=through){
            if(op.bytes&&(!op.dst||!op.src))std::abort();
            if(op.bytes)std::memmove(op.dst,op.src,op.bytes);
        }else copies[keep++]=op;
    }
    copy_count=keep;return 0;
}
inline bool pending(){return copy_count||uncertain[0]||uncertain[1];}

}
using cudaError_t=int;
inline constexpr int cudaSuccess=0;
inline int cudaGetDevice(int* out){*out=seam::device;return cudaSuccess;}
inline int cudaSetDevice(int device){if(seam::image_mode&&(device<0||device>1))return 1;seam::device=device;return cudaSuccess;}
inline int cudaDeviceSynchronize(){seam::events.push_back("drain"+std::to_string(seam::device));return seam::image_mode?seam::drain(0,UINT64_MAX,true):cudaSuccess;}
inline int cudaFree(void*){seam::events.push_back("free"+std::to_string(seam::device));return cudaSuccess;}

inline int cudaStreamSynchronize(int stream){if(seam::image_mode){if(seam::stream_fail&&stream==seam::stream_fail){seam::uncertain.at(size_t(seam::device))=true;return 1;}return seam::drain(uintptr_t(stream));}++seam::stream_calls;seam::events.push_back("stream"+std::to_string(stream));if(stream==seam::stream_fail){if(seam::stream_fail_once)seam::stream_fail=0;return 1;}return cudaSuccess;}

inline int cudaMemGetInfo(size_t* available,size_t* total){*total=10000;*available=*total-seam::native_used[seam::device];return cudaSuccess;}

inline int cudaGraphUpload(int,int){return ++seam::upload_calls==seam::upload_fail_at?1:cudaSuccess;}

// Events carry the real modeled device, stream and enqueue sequence. Event wait
// drains only that prefix, never unrelated streams/devices or a poisoned queue.
using cudaStream_t=void*;
struct RecordedCudaEvent {int device=-1;uintptr_t stream=0;uint64_t through=0;bool live=false,recorded=false;};
using cudaEvent_t=RecordedCudaEvent*;
inline constexpr int cudaErrorInvalidValue=1,cudaHostAllocPortable=1,cudaEventDisableTiming=2;
inline constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaMemcpyDefault=4;
namespace seam {inline std::array<RecordedCudaEvent,64> cuda_events{};}
inline const char* cudaGetErrorString(int status){return status?"recorded CUDA failure":"success";}
inline int cudaGetLastError(){return cudaSuccess;}
inline int cudaStreamSynchronize(cudaStream_t stream){
    if(!seam::image_mode)return cudaStreamSynchronize(int(uintptr_t(stream)));
    if(seam::stream_fail&&uintptr_t(stream)==uintptr_t(seam::stream_fail)){seam::uncertain.at(size_t(seam::device))=true;return 1;}
    return seam::drain(uintptr_t(stream));
}
inline int cudaMemcpyAsync(void* dst,const void* src,size_t bytes,int, cudaStream_t stream){
    if(!seam::image_mode)std::abort();
    if(++seam::enqueue_calls==seam::enqueue_fail)return 1;
    if(seam::device<0||seam::device>1||seam::uncertain.at(size_t(seam::device))||seam::copy_count==seam::copies.size()||
       (bytes&&(!dst||!src))||seam::copy_sequence==UINT64_MAX)return 1;
    seam::copies[seam::copy_count++]={dst,src,bytes,uintptr_t(stream),seam::device,++seam::copy_sequence};return cudaSuccess;
}
inline int cudaMemcpyAsync(void* dst,const void* src,size_t bytes,int kind,int stream){return cudaMemcpyAsync(dst,src,bytes,kind,reinterpret_cast<void*>(uintptr_t(stream)));}
inline int cudaHostAlloc(void** out,size_t bytes,unsigned){
    if(!seam::image_mode)return 1;
    if(seam::host_lifetime)seam::host_lifetime(false);
    *out=std::malloc(bytes);return *out?cudaSuccess:1;
}
inline int cudaFreeHost(void* pointer){
    if(!seam::image_mode||seam::pending())std::abort();
    if(seam::host_lifetime)seam::host_lifetime(true);
    std::free(pointer);return cudaSuccess;
}
inline int cudaEventCreateWithFlags(cudaEvent_t* out,unsigned){
    if(!seam::image_mode)return 1;
    for(auto& event:seam::cuda_events)if(!event.live){event={seam::device,0,0,true,false};*out=&event;return cudaSuccess;}
    return 1;
}
inline int cudaEventRecord(cudaEvent_t event,cudaStream_t stream){
    if(!event||!event->live||event->device!=seam::device||seam::uncertain.at(size_t(seam::device)))return 1;
    event->stream=uintptr_t(stream);event->through=seam::copy_sequence;event->recorded=true;return cudaSuccess;
}
inline int cudaEventSynchronize(cudaEvent_t event){
    if(!event||!event->live||!event->recorded||event->device!=seam::device)return 1;
    return seam::drain(event->stream,event->through);
}
inline int cudaEventDestroy(cudaEvent_t event){
    if(!event||!event->live||event->device!=seam::device||seam::pending())return 1;
    event->live=false;return cudaSuccess;
}
inline int cudaStreamWaitEvent(cudaStream_t,cudaEvent_t,unsigned){std::abort();}
