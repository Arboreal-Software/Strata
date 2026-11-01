#include "serving_output_capacity.hpp"
#include "strata/core/window_budget.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>

static long fail_after=-1;
void* operator new(std::size_t n){if(fail_after==0)throw std::bad_alloc();if(fail_after>0)--fail_after;if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
static void require(bool b,const char* why){if(!b){std::fprintf(stderr,"FAIL: %s\n",why);std::abort();}}
using namespace strata;
static uint64_t reserved(core::WindowBudget& b){return b.snapshot().reserved.host_bytes;}
struct Receipt {
    uint64_t operation=0,cut=0;int32_t token=-1;
    core::WindowBudget* owner=nullptr;
    ~Receipt(){if(owner)require(reserved(*owner)>0,"receipt payload dies before final admission credit");}
};
using History=program::ServingReceiptHistory<Receipt>;
struct World {
    core::WindowBudget budget{{{0},16ULL<<20},{{0}}};
    bool admit(uint64_t bytes,std::shared_ptr<const void>& out){
        auto result=budget.reserve({{0},bytes},core::ReservationPurpose::foreground);
        if(!result)return false;out=std::move(result.lease);return true;
    }
};
int main(){
    using namespace program;
    require(serving_output_fits(40,64,512)&&serving_output_fits(40,65,512),"finite request exceeds old64");
    require(serving_output_fits(40,464,512)&&!serving_output_fits(40,465,512),"actual full budget/context/slack boundary");
    require(!serving_output_fits(40,0,512)&&!serving_output_fits(UINT64_MAX,1,512)&&
        !serving_output_fits(1,UINT64_MAX,UINT64_MAX),"zero/unbounded/overflow refused");
    uint64_t quote=7,short_quote=0,required=0;
    require(serving_output_history_quote(1,65536,short_quote)&&short_quote>(1ULL<<20)+128*65536,
        "short-input/large-output history paid");
    require(!serving_output_history_quote(UINT64_MAX,1,quote)&&quote==7,"history overflow leaves quote unchanged");
    require(serving_receipt_requirement(63,65,1,required)&&required==129,"cumulative outputs plus independent comparison slot");
    require(!serving_receipt_requirement(UINT64_MAX,1,0,required)&&!serving_receipt_requirement(0,0,0,required),"receipt bounds reject overflow/zero");
    World w;
    {
        History h;require(h.size()==64&&reserved(w.budget)==0,"old inline default no context allocation");
        History::Prepared first;
        auto admit=[&](auto bytes,auto& credit){return w.admit(bytes,credit);};
        require(h.prepare(65,admit,first)&&h.size()==64&&reserved(w.budget)>0,"prepared65 paid without publication");
        require(h.commit(first)&&h.size()==65,"allocation-free receipt publication");
        first={};
        h[64].operation=65;h[64].cut=77;h[64].token=9;h[64].owner=&w.budget;
        const auto* retry=&h[64];
        const auto before=reserved(w.budget);
        History::Prepared refused;
        require(!h.prepare(100,[](auto,auto&){return false;},refused)&&h.size()==65&&&h[64]==retry&&
            reserved(w.budget)==before,"ledger refusal preserves old state and exact charge");
        History::Prepared next,stale;
        require(h.prepare(129,admit,next)&&h.prepare(80,admit,stale),"independent preparations");
        require(h.commit(next)&&!h.commit(stale)&&h.size()==129,"stale preparation cannot replace published capacity");
        next={};stale={};
        require(&h[64]==retry&&retry->operation==65&&retry->cut==77,"historical receipt pointer/idempotent result stable across growth");
        for(size_t i=65;i<129;++i){h[i].operation=i+1;h[i].owner=&w.budget;}
        const auto charged=reserved(w.budget);
        History::Snapshot refused_snapshot;
        require(!h.snapshot(129,[](auto,auto&){return false;},refused_snapshot)&&reserved(w.budget)==charged,
            "observer admission refusal leaves history/charge unchanged");
        {
            History::Snapshot a,b,c;
            require(h.snapshot(129,admit,a)&&h.snapshot(129,admit,b)&&h.snapshot(129,admit,c),"each observer independently paid; no two-copy assumption");
            require(a.entries.size()==129&&a.entries[64].cut==77&&reserved(w.budget)>charged,"observer is immutable copied receipt history");
        }
        require(reserved(w.budget)==charged,"observer storage releases before own credit");
        uint64_t impossible=1;
        require(!h.growth_quote(UINT64_MAX,impossible),"unaddressable receipt capacity refused without allocation");
    }
    require(reserved(w.budget)==0,"all block/directory/control storage releases before final reservation");
    {
        History history;auto admit=[&](auto n,auto& c){return w.admit(n,c);};
        for(size_t capacity=65;capacity<=192;++capacity){
            History::Prepared one;
            require(history.prepare(capacity,admit,one)&&history.commit(one),"small cumulative continuations remain admitted");
        }
        require(reserved(w.budget)<65536,"replacement directory credits retire; small continuations have linear live charge");
    }
    require(reserved(w.budget)==0,"long cumulative history releases all directory generations");
    // Real allocator failures at every small preparation/snapshot allocation.
    bool reached_success=false;
    for(long failure=0;failure<30;++failure){
        World f;History h;History::Prepared prepared;auto admit=[&](auto n,auto& c){return f.admit(n,c);};
        fail_after=failure;const bool ok=h.prepare(129,admit,prepared);fail_after=-1;
        require(h.size()==64,"allocation failure sweep never publishes");
        if(ok){reached_success=true;require(h.commit(prepared),"successful swept preparation adopts");prepared={};}
        else require(!prepared.valid&&reserved(f.budget)==0,"failed preparation releases every partial allocation/credit");
        History::Snapshot snapshot;
        const auto prior=reserved(f.budget);
        fail_after=failure;const bool copied=h.snapshot(64,admit,snapshot);fail_after=-1;
        if(!copied)require(!snapshot.credit&&snapshot.entries.empty()&&reserved(f.budget)==prior,"failed observer copy releases temporary reservation");
    }
    require(reached_success,"allocation sweep includes successful admitted publication");
    std::puts("PASS finite output/cumulative stable receipt storage, ledger and allocation failure/lifetime contracts");
}
