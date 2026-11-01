#pragma once
// Three bounded disposable serving qualification sites. No public fault/status setter.
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
#include "strata/core/resident_fault_receipt.hpp"
#include "strata/core/strict_retirement.hpp"
#include "strata/core/window_hash.hpp"
#include <array>
#include <cstdio>
#include <thread>
namespace strata::program {class ServingSuffixFaultQualification;}
namespace strata::core::resident_serving_fault {
enum class Site { none, private_replay, second_partial_copy, final_exact_copy };
inline const char* name(Site site)noexcept {
    switch(site){case Site::private_replay:return "serving_private_replay_completion";
        case Site::second_partial_copy:return "serving_second_partial_stage1_copy";
        case Site::final_exact_copy:return "serving_final_exact_stage1_copy";default:return "none";}
}
class Binding final {
    friend class ::strata::program::ServingSuffixFaultQualification;
    Site site_=Site::none;uint64_t request_=0;int fd_=-1;
    std::thread::id controller_{};std::array<char,65> binary_{},source_{};
public:
    bool valid()const noexcept{return site_!=Site::none&&request_&&fd_>=0&&controller_==std::this_thread::get_id()&&resident_fault_receipt::valid_fd(fd_);}
    Site site()const noexcept{return site_;}uint64_t request()const noexcept{return request_;}int fd()const noexcept{return fd_;}
    const char* binary()const noexcept{return binary_.data();}const char* source()const noexcept{return source_.data();}
};
struct Snapshot {
    uint64_t publication=0,live_cut=0,accepted=0,forwarded=0,exact=0,watermark=0;
    uint64_t private_cut=0,repaired=0,target_cut=0,known=0,generated=0,foreground=0,foreground_at_attachment=0;
    uint64_t receipt_operation=0,receipt_publication=0,receipt_cut=0;
    int32_t receipt_token=-1;unsigned attachments=0;StateDigest history{};
};
inline bool copy_scope_valid(const Binding& b,const Snapshot& s)noexcept {
    if(!b.valid()||!s.generated||s.generated>64||s.known>UINT64_MAX-s.generated||s.known+s.generated!=s.foreground||
       !s.receipt_operation||(!s.receipt_publication||s.receipt_publication>s.publication)||s.receipt_cut>s.live_cut||s.receipt_token<0||
       !s.live_cut||s.target_cut<=s.live_cut||s.accepted!=s.forwarded+1||s.exact>=s.forwarded)return false;
    if(b.site()==Site::second_partial_copy)return s.publication==2&&s.attachments==1&&
        s.foreground>s.foreground_at_attachment&&s.exact<s.repaired&&s.repaired<s.forwarded&&
        s.exact%4==0&&s.repaired%4==0&&s.forwarded%4==0;
    if(b.site()==Site::final_exact_copy)return s.attachments>=1&&s.attachments<=2&&
        s.publication==uint64_t(s.attachments)+1&&s.repaired==s.forwarded;
    return false;
}
inline void binding_record(const Binding& b,const Snapshot& s)noexcept {
    std::array<char,65> history{};constexpr char digits[]="0123456789abcdef";
    for(size_t i=0;i<s.history.size();++i){history[2*i]=digits[s.history[i]>>4];history[2*i+1]=digits[s.history[i]&15];}
    char line[1024];const int n=std::snprintf(line,sizeof(line),
        "{\"site\":\"%s\",\"phase\":\"binding\",\"request\":%llu,\"publication\":%llu,\"old_cut\":%llu,\"accepted\":%llu,\"forwarded\":%llu,\"exact\":%llu,\"watermark\":%llu,\"private_cut\":%llu,\"repaired\":%llu,\"target_cut\":%llu,\"known\":%llu,\"generated\":%llu,\"foreground\":%llu,\"attachments\":%u,\"receipt_operation\":%llu,\"receipt_publication\":%llu,\"receipt_cut\":%llu,\"receipt_token\":%d,\"history\":\"%s\",\"binary\":\"%s\",\"source\":\"%s\"}\n",
        name(b.site()),(unsigned long long)b.request(),(unsigned long long)s.publication,(unsigned long long)s.live_cut,
        (unsigned long long)s.accepted,(unsigned long long)s.forwarded,(unsigned long long)s.exact,(unsigned long long)s.watermark,
        (unsigned long long)s.private_cut,(unsigned long long)s.repaired,(unsigned long long)s.target_cut,
        (unsigned long long)s.known,(unsigned long long)s.generated,(unsigned long long)s.foreground,s.attachments,
        (unsigned long long)s.receipt_operation,(unsigned long long)s.receipt_publication,(unsigned long long)s.receipt_cut,
        s.receipt_token,history.data(),b.binary(),b.source());
    if(!b.valid()||n<=0||size_t(n)>=sizeof(line)||!resident_fault_receipt::write_record(b.fd(),line,size_t(n)))
        retirement_failed("serving fault binding receipt failed");
}
inline void event(const Binding& b,const Snapshot& s,const char* phase,const char* api,int status,
                  int poisoned=-1,int unhealthy=-1,int unchanged=-1,int retained=-1)noexcept {
    char line[768];const int n=std::snprintf(line,sizeof(line),
        "{\"site\":\"%s\",\"phase\":\"%s\",\"request\":%llu,\"publication\":%llu,\"old_cut\":%llu,\"target_cut\":%llu,\"private_cut\":%llu,\"stage\":1,\"device\":1,\"api\":\"%s\",\"status\":%d,\"raw_poisoned\":%d,\"allocation_unhealthy\":%d,\"cut_unchanged\":%d,\"failure_owner_retained\":%d}\n",
        name(b.site()),phase,(unsigned long long)b.request(),(unsigned long long)s.publication,(unsigned long long)s.live_cut,
        (unsigned long long)s.target_cut,(unsigned long long)s.private_cut,api,status,poisoned,unhealthy,unchanged,retained);
    if(!b.valid()||n<=0||size_t(n)>=sizeof(line)||!resident_fault_receipt::write_record(b.fd(),line,size_t(n)))
        retirement_failed("serving fault event receipt failed");
}
inline void copy_record(const Binding& b,const Snapshot& s,size_t bytes,int status)noexcept {
    char line[768];const int n=std::snprintf(line,sizeof(line),
        "{\"site\":\"%s\",\"phase\":\"api_refusal\",\"request\":%llu,\"publication\":%llu,\"old_cut\":%llu,\"target_cut\":%llu,\"stage\":1,\"device\":1,\"stage0_completed\":1,\"bytes\":%llu,\"cuda_call\":\"cudaMemcpyAsync\",\"copy_kind\":5,\"injection_attempted\":1,\"pre_sync_status\":0,\"cuda_status\":%d}\n",
        name(b.site()),(unsigned long long)b.request(),(unsigned long long)s.publication,(unsigned long long)s.live_cut,
        (unsigned long long)s.target_cut,(unsigned long long)bytes,status);
    if(!b.valid()||n<=0||size_t(n)>=sizeof(line)||!resident_fault_receipt::write_record(b.fd(),line,size_t(n)))
        retirement_failed("serving copy fault receipt failed");
}
} // namespace strata::core::resident_serving_fault
#endif
