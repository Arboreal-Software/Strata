#pragma once
#include "serving_suffix_cancel_qualification.hpp"
#include "strata/core/resident_serving_fault.hpp"
#include <charconv>
#include <cstring>
#include <memory>
#include <string>
namespace strata::program {
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
#define STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE 1
class ServingSuffixFaultQualification final {
    // File closes after every borrower but BEFORE the real aggregate/global credit is released.
    std::shared_ptr<const void> credit_;
    core::resident_fault_receipt::File file_;
    core::resident_serving_fault::Binding binding_;
    uint64_t request_=0;core::resident_serving_fault::Site site_=core::resident_serving_fault::Site::none;
public:
    static constexpr uint64_t host_bytes()noexcept {
        return sizeof(ServingSuffixFaultQualification)+8*sizeof(core::resident_serving_fault::Binding)+
            8*sizeof(core::resident_serving_fault::Snapshot)+4096;
    }
    bool configure(const char* value,bool policy,std::string& error){
        if(!value)return true;
        if(!policy||request_||std::strlen(value)>64){error="serving faults need one supported fault-ON retained qualification request";return false;}
        const char* colon=std::strchr(value,':');uint64_t request=0;
        if(!colon||colon==value||value[0]=='0'){error="serving fault selector requires request:site";return false;}
        const auto parsed=std::from_chars(value,colon,request);
        if(parsed.ec!=std::errc{}||parsed.ptr!=colon||!request||request>65536){error="serving fault request must be canonical1..65536";return false;}
        using Site=core::resident_serving_fault::Site;Site site=Site::none;
        if(!std::strcmp(colon+1,"private_replay"))site=Site::private_replay;
        else if(!std::strcmp(colon+1,"second_partial_copy"))site=Site::second_partial_copy;
        else if(!std::strcmp(colon+1,"final_exact_copy"))site=Site::final_exact_copy;
        if(site==Site::none){error="unknown serving fault site";return false;}
        request_=request;site_=site;return true;
    }
    bool enabled()const noexcept{return request_!=0;}
    bool target(uint64_t request)const noexcept{return enabled()&&request==request_;}
    bool prepare(uint64_t request,std::shared_ptr<const void> credit,const std::array<char,65>& binary,
                 const std::array<char,65>& source,std::string& error){
        if(!target(request)||!credit||credit_||binding_.valid()){error="serving fault receipt/admission owner differs";return false;}
        credit_=std::move(credit); // the ORIGINAL qualification reservation, never a fabricated extra ticket
        if(!file_.open_new("/output/RESIDENT-FAULT.jsonl")){error="fresh durable serving fault receipt refused before mutation";return false;}
        binding_.site_=site_;binding_.request_=request;binding_.fd_=file_.get();
        binding_.controller_=std::this_thread::get_id();binding_.binary_=binary;binding_.source_=source;
        return binding_.valid();
    }
    const core::resident_serving_fault::Binding& binding()const noexcept{return binding_;}
};
#endif
} // namespace strata::program
