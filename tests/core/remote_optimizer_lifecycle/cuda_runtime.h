#pragma once
#include <cstddef>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaHostAllocPortable=1,cudaHostAllocMapped=2;
int cudaGetDevice(int*);int cudaGetDeviceCount(int*);int cudaSetDevice(int);
int cudaDeviceSynchronize();int cudaStreamSynchronize(void*);
int cudaHostAlloc(void**,size_t,unsigned);int cudaHostGetDevicePointer(void**,void*,int);
int cudaMalloc(void**,size_t);int cudaFree(void*);int cudaFreeHost(void*);
