#pragma once
// SOURCE PROPOSAL ONLY. Qualification preconditions; never clears CUDA errors.
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
#include "strata/core/strict_retirement.hpp"
#include "strata/core/resident_fault_receipt.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>
namespace strata::core::resident_fault_precondition {
static_assert(sizeof(std::size_t)==8,"qualified impossible-size request requires pinned64bit size_t");
struct Observed {cudaError_t sync_status;};
// expected_device is taken from the ACTUAL owned NativeSession stage; not argv.
// stream is the ACTUAL original copy stream, nullptr only for arena allocation.
inline Observed check(int fd,const char* site,int expected_device,cudaStream_t stream=nullptr){
    int actual_device=-1;const auto device_status=cudaGetDevice(&actual_device);
    const auto last_before=cudaPeekAtLastError(); // observe, NEVER clear and continue
    char line[768];int n=0;
    if(device_status!=cudaSuccess||actual_device!=expected_device||last_before!=cudaSuccess){
        n=std::snprintf(line,sizeof(line),
            "{\"site\":\"%s\",\"phase\":\"prerequisite\",\"expected_device\":%d,\"actual_device\":%d,\"get_device_status\":%d,\"last_error_before\":%d,\"sync_called\":0,\"pass\":0}\n",
            site,expected_device,actual_device,int(device_status),int(last_before));
        if(n<=0||size_t(n)>=sizeof(line)||!resident_fault_receipt::write_record(fd,line,size_t(n)))
            retirement_failed("resident fault prerequisite evidence failed");
        retirement_failed("resident fault prerequisite refused"); // NOT expected injected-error PASS
    }
    unsigned stream_flags=0;cudaError_t stream_status=cudaSuccess;
    if(stream)stream_status=cudaStreamGetFlags(stream,&stream_flags);
    if(stream&&stream_status!=cudaSuccess){
        n=std::snprintf(line,sizeof(line),
            "{\"site\":\"%s\",\"phase\":\"prerequisite\",\"expected_device\":%d,\"actual_device\":%d,\"get_device_status\":%d,\"last_error_before\":%d,\"stream_checked\":1,\"stream_status\":%d,\"sync_called\":0,\"pass\":0}\n",
            site,expected_device,actual_device,int(device_status),int(last_before),int(stream_status));
        if(n<=0||size_t(n)>=sizeof(line)||!resident_fault_receipt::write_record(fd,line,size_t(n)))
            retirement_failed("resident fault prerequisite evidence failed");
        retirement_failed("resident fault prerequisite refused");
    }
    const auto synced=cudaDeviceSynchronize(); // IMMEDIATE actual TARGET-device prerequisite
    const auto last_after=cudaPeekAtLastError();
    const bool pass=synced==cudaSuccess&&last_after==cudaSuccess;
    // Null stream is explicit NOT_CHECKED; don't invent a CUDA stream status.
    if(stream)n=std::snprintf(line,sizeof(line),
        "{\"site\":\"%s\",\"phase\":\"prerequisite\",\"expected_device\":%d,\"actual_device\":%d,\"get_device_status\":%d,\"last_error_before\":%d,\"stream_checked\":1,\"stream_status\":%d,\"stream_flags\":%u,\"sync_called\":1,\"sync_status\":%d,\"last_error_after\":%d,\"pass\":%u}\n",
        site,expected_device,actual_device,int(device_status),int(last_before),int(stream_status),stream_flags,int(synced),int(last_after),unsigned(pass));
    else n=std::snprintf(line,sizeof(line),
        "{\"site\":\"%s\",\"phase\":\"prerequisite\",\"expected_device\":%d,\"actual_device\":%d,\"get_device_status\":%d,\"last_error_before\":%d,\"stream_checked\":0,\"sync_called\":1,\"sync_status\":%d,\"last_error_after\":%d,\"pass\":%u}\n",
        site,expected_device,actual_device,int(device_status),int(last_before),int(synced),int(last_after),unsigned(pass));
    if(n<=0||size_t(n)>=sizeof(line)||!resident_fault_receipt::write_record(fd,line,size_t(n)))
        retirement_failed("resident fault prerequisite evidence failed");
    if(!pass)retirement_failed("resident fault prerequisite refused");
    return {synced}; // ACTUAL raw success; no CUDA call before the injected API
}
}
#endif
