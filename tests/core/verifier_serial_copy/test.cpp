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
static int status=0, copies=0, primary=0;
int cudaStreamSynchronize(int stream) { if(stream==2) { ++copies; return status; } ++primary; return 0; }
const char* cudaGetErrorString(int) { return "mock copy failure"; }
void progress_at(const char*,int64_t) {}
struct Verifier {
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
        check(failed.terminal_copy_query_failed_.load(),"failure latched");
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
    std::puts("PASS: terminal and chained serial copy completion cases");
}
