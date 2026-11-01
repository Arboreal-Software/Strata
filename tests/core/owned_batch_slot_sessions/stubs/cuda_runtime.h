#pragma once
#include <string>
#include <vector>
namespace seam {
inline int device=0,zero_calls=0,zero_fail=0,allocation_calls=0,allocation_fail=0,partial_fail=0;
inline size_t native_used[2]={0,0};
inline int upload_calls=0,upload_fail_at=0;
inline int stream_fail=0,stream_calls=0;inline bool stream_fail_once=false;
inline int sessions_alive=0,sessions_destroyed=0;
inline std::vector<std::string> events;
}
using cudaError_t=int;
inline constexpr int cudaSuccess=0;
inline int cudaGetDevice(int* out){*out=seam::device;return cudaSuccess;}
inline int cudaSetDevice(int device){seam::device=device;return cudaSuccess;}
inline int cudaDeviceSynchronize(){seam::events.push_back("drain"+std::to_string(seam::device));return cudaSuccess;}
inline int cudaFree(void*){seam::events.push_back("free"+std::to_string(seam::device));return cudaSuccess;}

inline int cudaStreamSynchronize(int stream){++seam::stream_calls;seam::events.push_back("stream"+std::to_string(stream));if(stream==seam::stream_fail){if(seam::stream_fail_once)seam::stream_fail=0;return 1;}return cudaSuccess;}

inline int cudaMemGetInfo(size_t* available,size_t* total){*total=10000;*available=*total-seam::native_used[seam::device];return cudaSuccess;}

inline int cudaGraphUpload(int,int){return ++seam::upload_calls==seam::upload_fail_at?1:cudaSuccess;}
