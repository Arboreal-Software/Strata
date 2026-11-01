#pragma once
// EXPERIMENT QUALIFICATION ONLY. No model-load ordinal or global persistent arm.
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
#include "strata/core/strict_retirement.hpp"
#include "strata/core/resident_fault_receipt.hpp"
#include "strata/core/resident_fault_precondition.hpp"
#include "strata/core/resident_serving_fault.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <thread>
namespace strata::program {class SuffixRawTransaction;namespace resident_repair_detail {class ResidentRepairController;}}
namespace strata::core {
class ResidentCopyFaultScope;
class ResidentCopyFault final {
    friend class ResidentCopyFaultScope;
    friend class ::strata::program::SuffixRawTransaction;
    friend class ::strata::program::resident_repair_detail::ResidentRepairController;
    int fd_=-1,expected_device_=-1;
    uint64_t publication_=0,target_cut_=0,old_cut_=0,old_forwarded_=0,old_exact_=0,foreground_=0;
    unsigned attachments_=0;
    std::thread::id controller_=std::this_thread::get_id();
    bool stage0_completed_=false,fired_=false;
    resident_serving_fault::Binding serving_{};
    resident_serving_fault::Snapshot serving_snapshot_{};
    inline static thread_local ResidentCopyFault* active_=nullptr;
    ResidentCopyFault(int fd,uint64_t publication,uint64_t cut,uint64_t old_cut,uint64_t forwarded,
        uint64_t exact,uint64_t foreground,unsigned attachments,int expected_device)noexcept
        :fd_(fd),expected_device_(expected_device),publication_(publication),target_cut_(cut),old_cut_(old_cut),old_forwarded_(forwarded),
         old_exact_(exact),foreground_(foreground),attachments_(attachments){}
    ResidentCopyFault(const resident_serving_fault::Binding& binding,const resident_serving_fault::Snapshot& snapshot,int device)noexcept
        :ResidentCopyFault(binding.fd(),snapshot.publication,snapshot.target_cut,snapshot.live_cut,snapshot.forwarded,
            snapshot.exact,snapshot.foreground,snapshot.attachments,device){
        if(!resident_serving_fault::copy_scope_valid(binding,snapshot)||device!=1)
            retirement_failed("serving copy fault actual scope refused");
        serving_=binding;serving_snapshot_=snapshot;
    }
public:
    ResidentCopyFault(const ResidentCopyFault&)=delete;
    ResidentCopyFault& operator=(const ResidentCopyFault&)=delete;
    static cudaError_t copy(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream) {
        auto* fault=active_;
        if(!fault)return cudaMemcpyAsync(dst,src,bytes,kind,stream);
        const bool serving=fault->serving_.site()!=resident_serving_fault::Site::none;
        if(fault->fired_||!fault->stage0_completed_||
           (serving?!resident_serving_fault::copy_scope_valid(fault->serving_,fault->serving_snapshot_):
               (fault->publication_!=2||fault->attachments_!=1||fault->foreground_!=4))||!fault->target_cut_||
           fault->controller_!=std::this_thread::get_id()||!bytes||bytes>(16ULL<<20)||
           !dst||!src||!stream||kind!=cudaMemcpyDefault||fault->expected_device_<0)
            retirement_failed("partial-copy fault reached invalid operation scope");
        // Value5 is inside the C++ non-fixed enum range0..7, but outside
        // CUDA13's named supported directions0..4. Do NOT cast -1 (C++ UB),
        // or select a valid wrong pointer direction (CUDA documents that UB).
        static_assert(int(cudaMemcpyHostToHost)==0&&int(cudaMemcpyHostToDevice)==1&&
            int(cudaMemcpyDeviceToHost)==2&&int(cudaMemcpyDeviceToDevice)==3&&int(cudaMemcpyDefault)==4);
        static_assert(int(cudaErrorInvalidValue)==1&&int(cudaErrorInvalidMemcpyDirection)==21,
            "qualification receipt parser requires pinned CUDA13 copy refusal values");
        const auto prerequisites=resident_fault_precondition::check(fault->fd_,serving?
            resident_serving_fault::name(fault->serving_.site()):"second_partial_stage1_copy",fault->expected_device_,stream);
        const auto pre_sync=prerequisites.sync_status; // raw observed status from immediate target-device drain
        const bool attempted=true; // prerequisite failures NEVER return/qualify injection
        fault->fired_=attempted;
        const auto actual=attempted?cudaMemcpyAsync(dst,src,bytes,static_cast<cudaMemcpyKind>(5),stream):pre_sync;
        if(serving){
            resident_serving_fault::copy_record(fault->serving_,fault->serving_snapshot_,bytes,int(actual));
            if(actual==cudaSuccess)retirement_failed("serving copy fault unexpectedly accepted invalid copy kind");
            return actual; // real native error reaches unchanged RAW poison; never synthesize a failure status
        }
        char line[512];
        const int count=std::snprintf(line,sizeof(line),
            "{\"site\":\"second_partial_stage1_copy\",\"stage\":1,\"stage0_completed\":1,\"publication_before\":%llu,\"target_cut\":%llu,\"old_cut\":%llu,\"old_forwarded\":%llu,\"old_exact\":%llu,\"foreground_before\":%llu,\"attachments_before\":%u,\"bytes\":%llu,\"cuda_call\":\"%s\",\"copy_kind\":5,\"injection_attempted\":%u,\"pre_sync_status\":%d,\"cuda_status\":%d}\n",
            (unsigned long long)fault->publication_,(unsigned long long)fault->target_cut_,
            (unsigned long long)fault->old_cut_,(unsigned long long)fault->old_forwarded_,
            (unsigned long long)fault->old_exact_,(unsigned long long)fault->foreground_,fault->attachments_,
            (unsigned long long)bytes,attempted?"cudaMemcpyAsync":"cudaDeviceSynchronize",unsigned(attempted),int(pre_sync),int(actual));
        if(count<=0||size_t(count)>=sizeof(line)||
           !resident_fault_receipt::write_record(fault->fd_,line,size_t(count)))
            retirement_failed("partial-copy fault durable evidence failed");
        if(actual==cudaSuccess)retirement_failed("partial-copy fault unexpectedly accepted invalid copy kind");
        return actual; // unchanged transfer→restore→RAW poison handles REAL error
    }
};
class ResidentCopyFaultScope final {
    friend class ::strata::program::SuffixRawTransaction;
    ResidentCopyFault* fault_;
    explicit ResidentCopyFaultScope(ResidentCopyFault* fault):fault_(fault){
        if(!fault_){if(ResidentCopyFault::active_)retirement_failed("partial-copy inactive nested restore scope");return;}
        if(ResidentCopyFault::active_||fault_->fired_||!fault_->stage0_completed_)
            retirement_failed("partial-copy nested/unready scope");
        ResidentCopyFault::active_=fault_;
    }
public:
    ResidentCopyFaultScope(const ResidentCopyFaultScope&)=delete;
    ResidentCopyFaultScope& operator=(const ResidentCopyFaultScope&)=delete;
    ~ResidentCopyFaultScope()noexcept {
        if(fault_){if(ResidentCopyFault::active_!=fault_)retirement_failed("partial-copy fault scope lifetime");ResidentCopyFault::active_=nullptr;}
    }
};
} // namespace strata::core
#endif
