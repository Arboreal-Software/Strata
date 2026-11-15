// Executes the actual serving cancellation selector/witness helper in fault-ON and OFF builds.
// Atomic STOP publication is a CPU reader seam. This does not simulate native state or certify GPU cleanup.
#include "serving_suffix_cancel_qualification.hpp"
#include <cstdio>
#include <string>

using Cancel=strata::program::ServingSuffixCancelQualification;
namespace {
int failures=0;
void check(bool ok,const char* label){if(!ok){std::fprintf(stderr,"FAIL %s\n",label);++failures;}}
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
Cancel::Facts facts(Cancel::Site site){
    Cancel::Facts f;f.request=2;f.serial=2;f.cut=19;f.accepted=129;f.forwarded=128;f.exact_frontier=36;
    f.history[4]=0xa7;f.private_before=40;f.private_after=44;
    if(site!=Cancel::Site::import){f.generated=1;f.watermark=1;f.known=7;}
    return f;
}
void positive(Cancel::Site site,const char* selector){
    Cancel c;std::string error;
    check(c.configure(selector,true,error),"actual selector accepted");
    std::atomic<bool> stop{false},input_error{false};std::atomic<uint64_t> sequence{8};
    auto f=facts(site);
    // A different request/site cannot consume the selector or create a witness.
    auto other=f;other.request=1;
    check(c.wait(site,other,stop,sequence,input_error,error),"unselected request has no wait");
    check(!c.witness(1)&&!c.witness(2)&&!c.complete(),"unselected request cannot mint witness");
    std::thread reader([&]{std::this_thread::sleep_for(std::chrono::milliseconds(2));sequence.fetch_add(1);stop.store(true);});
    const bool observed=c.wait(site,f,stop,sequence,input_error,error,std::chrono::milliseconds(200));reader.join();
    check(observed&&c.witness(2)&&c.witness(2)->sequence()==9,"new actual reader sequence creates bound witness");
    check(!c.witness(1)&&stop.load(),"witness binds request and leaves STOP latched");
    auto final=f;final.serial=3;final.cut=55;final.exact_frontier=final.forwarded;
    check(c.final_history(final,stop,sequence,input_error,error),"final actual accepted history unchanged");
    auto corrupt=final;++corrupt.accepted;
    check(!c.final_history(corrupt,stop,sequence,input_error,error),"unread planned tail cannot replace accepted history");
    corrupt=final;corrupt.history[4]^=1;
    check(!c.final_history(corrupt,stop,sequence,input_error,error),"accepted/pending digest mutation refused");
    corrupt=final;++corrupt.generated;
    check(!c.final_history(corrupt,stop,sequence,input_error,error),"emitted receipt count mutation refused");
    corrupt=final;++corrupt.watermark;
    check(!c.final_history(corrupt,stop,sequence,input_error,error),"watermark mutation refused");
    check(!c.witness(2)->abort(sequence,input_error),"original STOP does not abort post-STOP oracle");
    sequence.fetch_add(1);check(c.witness(2)->abort(sequence,input_error),"new STOP aborts post-STOP oracle");
    check(!c.oracle_complete(2,sequence,input_error,error),"new STOP cannot report oracle completed");sequence.store(9);
    input_error.store(true);check(c.witness(2)->abort(sequence,input_error),"input error aborts post-STOP oracle");input_error.store(false);
    check(!c.oracle_complete(1,sequence,input_error,error),"different request cannot complete oracle");
    check(c.oracle_complete(2,sequence,input_error,error)&&c.complete(),"only witnessed oracle completion consumes selector");
    check(!c.oracle_complete(2,sequence,input_error,error),"duplicate oracle completion refused");
    check(!c.wait(site,f,stop,sequence,input_error,error),"site cannot run twice");
}
#endif
}
int main(){
    std::string error;
    {Cancel c;check(c.configure(nullptr,false,error)&&!c.enabled(),"absent selector stays inert");}
    for(const char* bad:{"","0:import","02:import","-2:import","2","2:","2:unknown","2:import:extra","65537:import","2:private_unit\n"}){
        Cancel c;check(!c.configure(bad,true,error),"malformed selector rejected");
    }
    {Cancel c;check(!c.configure("2:import",false,error),"unqualified or exact-first profile refused");}
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
    check(Cancel::available(),"fault-ON supported gate available");
    positive(Cancel::Site::import,"2:import");
    positive(Cancel::Site::first_output,"2:first_output");
    positive(Cancel::Site::private_unit,"2:private_unit");
    {
        Cancel c;check(c.configure("2:private_unit",true,error),"private selector configured");
        check(!c.configure("2:import",true,error),"duplicate selector configured refused");
        std::atomic<bool> stop{false},input_error{false};std::atomic<uint64_t> sequence{0};auto f=facts(Cancel::Site::private_unit);
        auto bad=f;bad.private_after=bad.private_before;
        check(!c.wait(Cancel::Site::private_unit,bad,stop,sequence,input_error,error),"fake completed private work refused");
        bad=f;bad.private_after=bad.forwarded;
        check(!c.wait(Cancel::Site::private_unit,bad,stop,sequence,input_error,error),"already caught-up private boundary refused");
        bad=f;bad.generated=0;
        check(!c.wait(Cancel::Site::private_unit,bad,stop,sequence,input_error,error),"private unit requires immutable output");
        stop.store(true);
        check(!c.wait(Cancel::Site::private_unit,f,stop,sequence,input_error,error),"preexisting STOP cannot qualify selected site");
    }
    {
        Cancel c;check(c.configure("2:import",true,error),"timeout selector configured");
        std::atomic<bool> stop{false},input_error{false};std::atomic<uint64_t> sequence{0};
        check(!c.wait(Cancel::Site::import,facts(Cancel::Site::import),stop,sequence,input_error,error,std::chrono::milliseconds(3)),
              "missing STOP times out and cannot mint witness");
        check(!c.witness(2)&&!c.complete()&&error.find("timed out")!=std::string::npos,"timeout explicitly unconsumed");
    }
    {
        Cancel c;check(c.configure("2:import",true,error),"input-error selector configured");
        std::atomic<bool> stop{false},input_error{false};std::atomic<uint64_t> sequence{0};
        std::thread reader([&]{std::this_thread::sleep_for(std::chrono::milliseconds(2));input_error.store(true);stop.store(true);});
        check(!c.wait(Cancel::Site::import,facts(Cancel::Site::import),stop,sequence,input_error,error,std::chrono::milliseconds(200)),
              "input failure setting stop flag is not a STOP receipt");reader.join();
        check(!c.witness(2),"input failure cannot mint witness");
    }
#else
    check(!Cancel::available(),"fault-OFF gate unavailable");
    for(const char* site:{"2:import","2:first_output","2:private_unit"}){
        Cancel c;check(!c.configure(site,true,error)&&!c.enabled(),"fault-OFF rejects every selector");
    }
#endif
    if(!failures)std::puts("PASS actual retained cancellation selector/witness helper; CPU-only, no native proof");
    return failures?1:0;
}
