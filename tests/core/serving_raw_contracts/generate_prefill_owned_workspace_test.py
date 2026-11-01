#!/usr/bin/env python3
"""Extract actual producer/driver/binding code; backend byte counter is a CPU fixture."""
import argparse
from pathlib import Path
import re
from test_serving_thread_bridge_source import body


def generate(root):
    cpp = (root / 'src/prefill/prefill.cpp').read_text()
    driver = (root / 'src/program/generate.cpp').read_text()
    header = (root / 'include/strata/prefill/prefill.hpp').read_text()
    def actual(source, marker):
        at = source.index(marker)
        block = body(source, marker)
        return source[at:source.index(block, at)] + block
    producer = actual(cpp, 'bool Prefill::admit_owned_workspace(')
    guard = actual(cpp, 'if(workspace_credit_&&')
    driver_body = body(driver, 'auto admit_prefill_workspace =')
    constants = re.search(r'constexpr int64_t N =[^;]+;', cpp)[0]
    threads = re.search(r'inline constexpr int kMaxStagerThreads =[^;]+;', header)[0]
    return r'''
#include "strata/prefill/owned_workspace.hpp"
#include "strata/core/window_budget.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
static void require(bool b,const char* why){if(!b){std::fprintf(stderr,"FAIL: %s\n",why);std::abort();}}
namespace strata::core {
struct ModelGeometry{int64_t n_embd=2560,hc=4,hc_lr=320,n_expert=16,n_layers=2;};
struct SessionState{int k=10;};
struct ExpertSource{bool transient(int64_t,int64_t)const{return false;}};
struct PinnedRing{static constexpr int kSlots=4;static constexpr size_t minimum_slot_bytes(){return 65536;}};
}
namespace strata::kernels{constexpr int kStepCount=32,PLE_N_HEADS=4;}
namespace strata::prefill {
using cudaEvent_t=void*;
''' + constants + '\n' + threads + r'''
static int64_t MAXBLOB(){return 4096;} // explicit backend byte-count fixture, no GPU allocation claim
struct Stager{struct Job{const uint8_t* src;size_t bytes;core::ExpertSource* from=nullptr;int32_t l=0,e=0;};};
struct Prefill {
    struct Impl{const core::ModelGeometry* g=nullptr;bool pp=false,help_pp=false;};
    using WorkspaceAdmitter=std::function<bool(const OwnedWorkspaceBytes&,std::shared_ptr<const void>&)>;
    bool strict_retirement_=true;void* cpu_pool_=nullptr;Prefill* helper_=nullptr;void* embd_rows=nullptr;
    std::function<void()> on_routes,on_taps;
    std::shared_ptr<const void> workspace_credit_;
    const core::ModelGeometry* workspace_geometry_=nullptr;const core::SessionState* workspace_session_=nullptr;
    const core::ExpertSource* workspace_source_=nullptr;int64_t workspace_chunk_=0;
    bool workspace_borrowed_=false,workspace_successor_=false;
    std::unique_ptr<Impl> impl_=std::make_unique<Impl>();Prefill* next_=nullptr;
    static uint64_t bytes_needed(const core::ModelGeometry&,const core::SessionState&,int64_t chunk,bool){return (16ULL<<20)+uint64_t(chunk)*4096;}
    static uint64_t bytes_needed_owned(const core::ModelGeometry& g,const core::SessionState& s,int64_t c,bool src){return bytes_needed(g,s,c,src);}
    bool admit_owned_workspace(const core::ModelGeometry&,const core::SessionState&,core::ExpertSource*,int64_t,bool,const WorkspaceAdmitter&,std::string&);
    bool init_binding(const core::ModelGeometry& g,core::SessionState& ss,core::ExpertSource* src,int64_t chunk,void* borrow,bool borrow_spill,std::string& err){
''' + guard + r'''
        return true;
    }
};
''' + producer + r'''
}
using namespace strata;
static bool actual_driver(prefill::Prefill& consumer,const core::ModelGeometry& g,const core::SessionState& session,
                          core::ExpertSource* source,int64_t chunk,bool borrowed,int device,bool enabled,
                          std::unique_ptr<core::WindowBudget>& model_budget,std::string& err){
    const bool owned_batch_allocation=enabled;struct Options{int64_t prefill_chunk;};Options o{chunk};
    auto invoke=[&](prefill::Prefill& consumer,const core::SessionState& session,core::ExpertSource* source,bool borrowed,int device)->bool
#define STRATA_WINDOW_STATE_AVAILABLE 1
''' + driver_body + r''';
    return invoke(consumer,session,source,borrowed,device);
}
static auto budget(uint64_t host,uint64_t device=512ULL<<20){return std::make_unique<core::WindowBudget>(core::WindowResources{{device,device},host,host},core::WindowResources{{0,0}});}
int main(){
    core::ModelGeometry g;core::SessionState a,b;core::ExpertSource source;std::string err;
    auto ledger=budget(512ULL<<20);
    std::weak_ptr<const void> late;
    {
        prefill::Prefill primary,secondary;primary.next_=&secondary;
        require(actual_driver(secondary,g,b,&source,1024,true,1,true,ledger,err),"actual secondary producer/ledger admission");
        require(actual_driver(primary,g,a,&source,1024,true,0,true,ledger,err),"actual successor producer/ledger admission");
        const auto paid=ledger->snapshot().reserved;
        require(paid.device_bytes==std::vector<uint64_t>({2ULL<<20,2ULL<<20})&&paid.host_bytes>0&&paid.pinned_bytes==paid.host_bytes,
            "actual callback pays standalone token pages and host/pinned workspace; arena already paid");
        late=primary.workspace_credit_;require(!late.expired(),"actual producer retains genuine reservation");
        require(primary.init_binding(g,a,&source,1024,reinterpret_cast<void*>(1),false,err),"actual init accepts identical owners/shape");
        require(!primary.init_binding(g,a,&source,512,reinterpret_cast<void*>(1),false,err),"actual init rejects unquoted chunk");
        core::SessionState other;
        require(!primary.init_binding(g,other,&source,1024,reinterpret_cast<void*>(1),false,err),"actual init rejects replacement session");
        require(!primary.init_binding(g,a,&source,1024,nullptr,false,err),"actual init rejects new own allocation under borrowed quote");
        primary.helper_=&secondary;
        require(!primary.init_binding(g,a,&source,1024,reinterpret_cast<void*>(1),false,err),"actual init rejects uncovered helper after quote");
        primary.helper_=nullptr;
        require(!actual_driver(primary,g,a,&source,1024,true,0,true,ledger,err),"actual producer refuses replacement before checked reset");
    }
    require(late.expired()&&ledger->snapshot().reserved.host_bytes==0,"actual producer owner retains then releases admission");
    {
        auto short_host=budget(1);prefill::Prefill p;
        require(!actual_driver(p,g,a,&source,1024,false,0,true,short_host,err)&&short_host->snapshot().reserved.host_bytes==0&&!p.workspace_credit_,
            "actual ledger refusal happens before init without partial charge");
    }
    {
        prefill::Prefill p;auto current=ledger->snapshot().reserved;
        require(actual_driver(p,g,a,&source,1024,false,0,false,ledger,err)&&!p.workspace_credit_&&ledger->snapshot().reserved==current,
            "actual default-off driver path unchanged");
        require(!actual_driver(p,g,a,&source,0,false,0,true,ledger,err),"actual zero shape refused");
        p.cpu_pool_=reinterpret_cast<void*>(1);
        require(!actual_driver(p,g,a,&source,1024,false,0,true,ledger,err),"actual uncovered CPU consumer refused");
    }
    std::puts("PASS extracted paid Prefill producer, actual ledger callback and init owner binding");
}
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--generate', type=Path, required=True)
    args = parser.parse_args()
    args.generate.parent.mkdir(parents=True, exist_ok=True)
    args.generate.write_text(generate(args.source_root))
