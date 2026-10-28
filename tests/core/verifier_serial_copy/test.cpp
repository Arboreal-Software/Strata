#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
static bool allocation_failure=false;
void* operator new(std::size_t n) { if(allocation_failure) throw std::bad_alloc(); if(void* p=std::malloc(n?n:1)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
using cudaError_t=int;
constexpr int cudaSuccess=0;
static int status=0, primary_status=0, copies=0, primary=0;
int cudaStreamSynchronize(int stream) { if(stream==2) { ++copies; return status; } ++primary; return primary_status; }
const char* cudaGetErrorString(int code) { return code==77 ? "mock primary failure" : "mock copy failure"; }
void progress_at(const char*,int64_t) {}
struct Verifier {
    // Explicit gate-mark seam; actual gate/factory ordering has its own fixture.
    int completion_marks=0;
    void mark_completion_unhealthy()noexcept{++completion_marks;}
#include "completion-latch.inc"
    int copy_=2,cs_=1,T=1; bool copy_used_=true,commit_pending_=true,ok=true;
    int* h_plan_err_=nullptr;
    std::atomic<bool> terminal_copy_query_failed_{false};
#include "guard.inc"
    bool handoff_failed(std::string&) { return false; }
    bool ar_on() { return false; }
    bool finish_serial_copies(std::string&);
    bool terminal(std::string&); bool chain(std::string&);
};
#ifdef BEFORE
#include "before.inc"
#else
#include "fixed.inc"
#endif
static void check(bool b,const char* text) { if(!b) { std::fprintf(stderr,"FAIL: %s\n",text); std::exit(1); } }
int main(int argc,char**) {
    for(int index=0;index<2;++index) {
        const bool chain=(index==0)==(argc>1);
        auto run=[&](Verifier& v,std::string& e){return chain?v.chain(e):v.terminal(e);};
        std::string e;
        Verifier failed; status=9; copies=primary=0;
        check(!run(failed,e),"serial copy error must not report completion");
        check(failed.copy_used_ && failed.commit_pending_,"failed copy retains pending flags");
        check(failed.terminal_copy_query_failed_.load() && failed.completion_marks==1,"failure latched after completion invalidation");
        int calls=copies+primary; status=0;
        check(!run(failed,e) && copies+primary==calls,"retry must refuse without device calls");
        Verifier oom; status=9; e.clear(); e.shrink_to_fit(); allocation_failure=true;
        bool threw=false; try { run(oom,e); } catch(const std::bad_alloc&) { threw=true; }
        allocation_failure=false;
        check(threw && oom.terminal_copy_query_failed_.load() && oom.copy_used_ && oom.commit_pending_,"allocation failure retains terminal pending state");
        calls=copies+primary; status=0; check(!run(oom,e) && copies+primary==calls,"allocation failure cannot heal");
        Verifier success; status=0; copies=primary=0;
        check(run(success,e) && !success.copy_used_ && !success.commit_pending_ && copies==1,"successful copy completes exactly once");
        Verifier unused; unused.copy_used_=false; copies=primary=0;
        check(run(unused,e) && !unused.commit_pending_ && copies==0,"unused copy skips synchronization");
        Verifier pending; status=34;
        check(!run(pending,e) && pending.commit_pending_ && pending.copy_used_,"non-success pending status cannot certify sync");
    }
    // The extracted chain block checks the copy before downstream/primary
    // failures. A successful copy must not erase the earlier diagnostic; a copy
    // failure takes precedence and remains sticky even when another error exists.
    for(bool downstream_ok : {false,true})
        for(int primary_error : {0,77})
            for(int copy_error : {0,9}) {
                Verifier chain; chain.ok=downstream_ok;
                primary_status=primary_error; status=copy_error; copies=primary=0;
                std::string error=downstream_ok ? "unchanged" : "downstream failure";
                const bool success=chain.chain(error);
                const bool expected=downstream_ok && !primary_error && !copy_error;
                check(success==expected,"chain error matrix completion");
                check(primary==1 && copies==1,"chain always checks both streams once");
                check(chain.commit_pending_==!expected,"chain retains pending until complete");
                check(chain.copy_used_==(copy_error!=0),"copy flag follows copy completion only");
                check(chain.terminal_copy_query_failed_.load()==(copy_error!=0),"only copy error latches copy failure");
                if(copy_error) {
                    check(error.find("mock copy failure")!=std::string::npos,"copy diagnostic takes precedence");
                    primary_status=status=0;
                    check(!chain.chain(error) && primary==1 && copies==1,"combined error cannot heal or issue calls");
                } else if(!downstream_ok) {
                    check(error=="downstream failure","successful copy preserves downstream diagnostic");
                } else if(primary_error) {
                    check(error=="verify: mock primary failure","primary diagnostic preserved after copy success");
                } else check(error=="unchanged","successful chain leaves diagnostic untouched");
            }
    std::puts("PASS: terminal and chained serial copy completion cases");
}
