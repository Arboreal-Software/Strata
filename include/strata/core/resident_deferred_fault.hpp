#pragma once
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
#include "strata/core/resident_fault_receipt.hpp"
#include "strata/core/strict_retirement.hpp"
#include "strata/core/resident_serving_fault.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <thread>
namespace strata::program::resident_repair_detail { class ResidentRepairController; }
namespace strata::core { class Verifier; }
namespace strata::core::resident_deferred_fault {
cudaError_t launch_private_stage_trap(cudaStream_t) noexcept;
enum class QueryApi { stream, event };
class Scope;
// One C++ inline entity shared across controller/core TUs, not TU-local arms.
inline thread_local Scope* current=nullptr;
void observe(const Verifier*,QueryApi,cudaStream_t,cudaError_t) noexcept;
class Scope final {
    friend class ::strata::program::resident_repair_detail::ResidentRepairController;
    friend void observe(const Verifier*,QueryApi,cudaStream_t,cudaError_t) noexcept;
    Scope()=default;
    ~Scope(){if(current==this)current=nullptr;}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
    void arm(int fd,const Verifier* verifier,cudaStream_t stream,uint64_t publication,uint64_t cut,
             uint64_t forwarded,uint64_t exact,uint64_t repaired,uint64_t foreground,unsigned attachments){
        if(current||fd<0||!verifier||!stream)retirement_failed("deferred trap scope refused");
        fd_=fd;verifier_=verifier;stream_=stream;publication_=publication;cut_=cut;forwarded_=forwarded;exact_=exact;
        repaired_=repaired;foreground_=foreground;attachments_=attachments;thread_=std::this_thread::get_id();current=this;armed_=true;
    }
    void arm_serving(const resident_serving_fault::Binding& binding,const resident_serving_fault::Snapshot& snapshot,
                     const Verifier* verifier,cudaStream_t stream){
        if(!binding.valid()||binding.site()!=resident_serving_fault::Site::private_replay)
            retirement_failed("serving deferred scope refused");
        arm(binding.fd(),verifier,stream,snapshot.publication,snapshot.live_cut,snapshot.forwarded,snapshot.exact,
            snapshot.repaired,snapshot.foreground,snapshot.attachments);
        serving_=binding;serving_snapshot_=snapshot;
    }
    void record(const char* phase,int status,int poisoned=-1,int unhealthy=-1)const noexcept {
        if(serving_.site()!=resident_serving_fault::Site::none){
            resident_serving_fault::event(serving_,serving_snapshot_,phase,phase,status,poisoned,unhealthy);return;
        }
        char line[1000];const int n=std::snprintf(line,sizeof(line),
          "{\"case\":\"private_stage1_trap\",\"phase\":\"%s\",\"device\":1,\"stage\":1,\"publication\":%llu,\"old_cut\":%llu,\"forwarded\":%llu,\"exact\":%llu,\"repaired\":%llu,\"foreground\":%llu,\"attachments\":%u,\"status\":%d,\"stream_bound\":1,\"blocks\":1,\"threads\":1,\"raw_poisoned\":%d,\"allocation_unhealthy\":%d}\n",
          phase,(unsigned long long)publication_,(unsigned long long)cut_,(unsigned long long)forwarded_,(unsigned long long)exact_,
          (unsigned long long)repaired_,(unsigned long long)foreground_,attachments_,status,poisoned,unhealthy);
        if(n<=0||size_t(n)>=sizeof(line)||!resident_fault_receipt::write_record(fd_,line,size_t(n)))
            retirement_failed("deferred trap durable evidence failed");
    }
    resident_serving_fault::Binding serving_{};resident_serving_fault::Snapshot serving_snapshot_{};
    int fd_=-1;const Verifier* verifier_=nullptr;cudaStream_t stream_=nullptr;std::thread::id thread_{};
    uint64_t publication_=0,cut_=0,forwarded_=0,exact_=0,repaired_=0,foreground_=0;unsigned attachments_=0;
    bool armed_=false,observed_=false;cudaError_t observed_status_=cudaSuccess;
};
inline void observe(const Verifier* verifier,QueryApi api,cudaStream_t stream,cudaError_t status) noexcept {
    auto* s=current;
    if(!s||s->observed_||std::this_thread::get_id()!=s->thread_||verifier!=s->verifier_||api!=QueryApi::stream||stream!=s->stream_||
       status==cudaSuccess||status==cudaErrorNotReady)return;
    s->observed_=true;s->observed_status_=status;
    s->record("cudaStreamQuery",int(status)); // ACTUAL native call result, unchanged
}
static_assert(int(cudaErrorIllegalInstruction)==715&&int(cudaErrorLaunchFailure)==719,"CUDA13 status binding");
}
#endif
