#pragma once
// Disposable serving qualification only. No wait/STOP witness code exists in fault-OFF builds.
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <thread>

#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(STRATA_WINDOW_STATE_AVAILABLE) && \
    defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && \
    !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
#define STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE 1
#endif

namespace strata::program {
class ServingSuffixCancelQualification final {
public:
    enum class Site { none, import, first_output, private_unit };
    struct Facts {
        uint64_t request=0,serial=0,cut=0,accepted=0,forwarded=0,exact_frontier=0,watermark=0;
        uint64_t known=0,generated=0,private_before=0,private_after=0;
        std::array<uint8_t,32> history{};
    };
    class Witness final {
        friend class ServingSuffixCancelQualification;
        uint64_t request_,sequence_;
        Witness(uint64_t request,uint64_t sequence):request_(request),sequence_(sequence){}
    public:
        uint64_t request()const noexcept{return request_;}
        uint64_t sequence()const noexcept{return sequence_;}
        bool abort(const std::atomic<uint64_t>& sequence,const std::atomic<bool>& input_error)const noexcept {
            return input_error.load()||sequence.load()!=sequence_;
        }
    };
    static constexpr bool available()noexcept {
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
        return true;
#else
        return false;
#endif
    }
    bool configure(const char* value,bool qualified_retained,std::string& error){
        if(!value)return true;
        if(!available()||!qualified_retained){error="retained cancellation selector requires supported fault-ON qualified retained serving";return false;}
        if(request_||std::strlen(value)>64){error="one bounded retained cancellation selector required";return false;}
        const char* colon=std::strchr(value,':');uint64_t request=0;
        if(!colon||colon==value||value[0]=='0'){error="cancel selector requires request:site";return false;}
        const auto parsed=std::from_chars(value,colon,request);
        if(parsed.ec!=std::errc{}||parsed.ptr!=colon||!request||request>65536){error="cancel request must be canonical 1..65536";return false;}
        Site site=Site::none;
        if(!std::strcmp(colon+1,"import"))site=Site::import;
        else if(!std::strcmp(colon+1,"first_output"))site=Site::first_output;
        else if(!std::strcmp(colon+1,"private_unit"))site=Site::private_unit;
        if(site==Site::none){error="cancel site must be import, first_output or private_unit";return false;}
        request_=request;site_=site;return true;
    }
    bool enabled()const noexcept{return request_!=0;}
    bool target(uint64_t request)const noexcept{return enabled()&&request==request_;}
    bool selected(uint64_t request,Site site)const noexcept{return target(request)&&site==site_;}
    bool complete()const noexcept{return completed_;}
    const Witness* witness(uint64_t request)const noexcept{return target(request)&&witness_?&*witness_:nullptr;}
    static const char* name(Site site)noexcept {
        switch(site){case Site::import:return "import";case Site::first_output:return "first_output";
            case Site::private_unit:return "private_unit";default:return "none";}
    }
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
    // Predicate is called after real work has drained. It does not certify work or replace a device join.
    bool wait(Site site,const Facts& facts,const std::atomic<bool>& stop,
              const std::atomic<uint64_t>& sequence,const std::atomic<bool>& input_error,std::string& error,
              std::chrono::milliseconds timeout=std::chrono::seconds(10)) {
        if(!selected(facts.request,site))return true;
        if(entered_||witness_||completed_||!facts.serial||!facts.cut||facts.accepted!=facts.forwarded+1||
           facts.exact_frontier>=facts.forwarded||stop.load()||input_error.load()||
           (site==Site::import&&facts.generated)||
           (site==Site::first_output&&facts.generated!=1)||
           (site==Site::private_unit&&(!facts.generated||facts.private_after<=facts.private_before||
                                      facts.private_after>=facts.forwarded))||
           timeout.count()<=0||timeout>std::chrono::seconds(10)) {
            error="cancel qualification site is stale, already stopped or lacks actual unresolved progress";return false;
        }
        entered_=true;observed_=facts;const uint64_t before=sequence.load();
        record("wait",before);
        const auto deadline=std::chrono::steady_clock::now()+timeout;
        for(;;){
            const uint64_t now=sequence.load();
            if(input_error.load()){error="cancel qualification stdin error";return false;}
            if(now!=before){
                if(now!=before+1||!now){error="cancel qualification needs exactly one new STOP";return false;}
                if(stop.load()){
                    witness_=Witness(facts.request,now);record("observed",now);return true;
                }
            }
            if(std::chrono::steady_clock::now()>=deadline){error="cancel qualification STOP handshake timed out";return false;}
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    bool final_history(const Facts& final,const std::atomic<bool>& stop,
                       const std::atomic<uint64_t>& sequence,const std::atomic<bool>& input_error,std::string& error)const {
        if(!witness(final.request)||completed_||!stop.load()||witness_->abort(sequence,input_error)||
           final.accepted!=observed_.accepted||final.forwarded!=observed_.forwarded||
           final.exact_frontier!=final.forwarded||final.watermark!=observed_.watermark||
           final.known!=observed_.known||final.generated!=observed_.generated||final.history!=observed_.history){
            error="cancel exact finish differs from witnessed accepted/emitted history";return false;
        }
        return true;
    }
    bool oracle_complete(uint64_t request,const std::atomic<uint64_t>& sequence,
                         const std::atomic<bool>& input_error,std::string& error){
        if(!witness(request)||completed_||witness_->abort(sequence,input_error)){
            error="cancel oracle completion lacks current STOP witness";return false;
        }
        completed_=true;record("oracle_complete",witness_->sequence());return true;
    }
#endif
private:
    uint64_t request_=0;Site site_=Site::none;bool entered_=false,completed_=false;
    Facts observed_{};std::optional<Witness> witness_;
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
    void record(const char* phase,uint64_t sequence)const noexcept {
        std::array<char,65> history{};constexpr char digits[]="0123456789abcdef";
        for(size_t i=0;i<observed_.history.size();++i){history[2*i]=digits[observed_.history[i]>>4];history[2*i+1]=digits[observed_.history[i]&15];}
        std::fprintf(stderr,"suffix_repair_cancel req=%llu phase=%s site=%s serial=%llu cut=%llu accepted=%llu forwarded=%llu exact_frontier=%llu watermark=%llu known_forwarded=%llu generated=%llu private_before=%llu private_after=%llu history=%s stop_sequence=%llu request_cancelled=%u\n",
            (unsigned long long)request_,phase,name(site_),(unsigned long long)observed_.serial,
            (unsigned long long)observed_.cut,(unsigned long long)observed_.accepted,(unsigned long long)observed_.forwarded,
            (unsigned long long)observed_.exact_frontier,(unsigned long long)observed_.watermark,
            (unsigned long long)observed_.known,(unsigned long long)observed_.generated,
            (unsigned long long)observed_.private_before,(unsigned long long)observed_.private_after,
            history.data(),(unsigned long long)sequence,unsigned(witness_.has_value()));std::fflush(stderr);
    }
#endif
};
} // namespace strata::program
