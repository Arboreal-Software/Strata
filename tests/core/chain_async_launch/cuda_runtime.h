#pragma once
using cudaError_t=int;
using cudaStream_t=int;
using cudaGraphExec_t=int;
constexpr int cudaSuccess=0,cudaErrorNotReady=34;
cudaError_t cudaSetDevice(int);
cudaError_t cudaGraphLaunch(cudaGraphExec_t,cudaStream_t);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
const char* cudaGetErrorString(cudaError_t);
