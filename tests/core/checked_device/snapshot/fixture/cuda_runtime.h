#pragma once
using cudaError_t=int;
constexpr int cudaSuccess=0;
int cudaGetDevice(int*);
int cudaSetDevice(int);
int cudaDeviceSynchronize();
