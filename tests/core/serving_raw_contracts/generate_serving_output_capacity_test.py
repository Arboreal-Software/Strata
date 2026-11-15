#!/usr/bin/env python3
"""Extract actual CUDA-free admission/preparation bodies; never compile generate.cpp."""
import argparse
from pathlib import Path
from test_serving_thread_bridge_source import body


def generate(root):
    source = (root / 'src/program/generate.cpp').read_text()
    def actual(marker):
        start = source.index(marker)
        block = body(source, marker)
        return source[start:source.index(block, start)] + block
    prepared = actual('struct PreparedContinuationRequest') + ';'
    prepare = actual('static bool prepare_continuation_request(')
    adopt = actual('static void adopt_continuation_request(')
    start = source.index('const int64_t n = (int64_t) ids.size();', source.index('ERR bad request:'))
    bgen = source[start:source.index('req_imgs.clear();', start)]
    later = source.index('sp.embd_rows = geni ? row_ptr.data() : nullptr;', start)
    later = source.index('\n', later) + 1
    generic_context = source[later:source.index('bool bad = false;', later)]
    command_guard = actual('if(owned_batch_allocation&&(admit_slot<0||req_temperature!=0||req_penalty_last_n!=0||req_bias_present||')
    receipt_source = (root / 'src/program/http_two_slot_receipt.hpp').read_text()
    key_start = receipt_source.index('inline bool http2_request_key(')
    key_body = body(receipt_source, 'inline bool http2_request_key(')
    request_key = receipt_source[key_start:receipt_source.index(key_body, key_start)] + key_body
    return r'''
#include "serving_output_capacity.hpp"
#include "strata/core/window_budget.hpp"
#include "strata/core/window_state.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
static void require(bool b,const char* why){if(!b){std::fprintf(stderr,"FAIL: %s\n",why);std::abort();}}
namespace strata::core { [[noreturn]] inline void retirement_failed(const char* why){require(false,why);std::abort();} }
namespace strata::program {
''' + request_key + r'''
struct SuffixExperimentDriver {static bool serving_qualification_enabled(){return false;}};
namespace resident_repair_detail {
struct ResidentRepairController {
    struct Receipt{uint64_t operation=0,publication=0,cut=0;int32_t token=-1;};
    using ReceiptHistory=ServingReceiptHistory<Receipt>;
    struct Publication{core::SnapshotDescriptor metadata;uint64_t serial=1;};
    std::shared_ptr<Publication> published_=std::make_shared<Publication>();
    ReceiptHistory receipts_;
    size_t receipt_count_=64;
    uint64_t foreground_count_=64,known_count_=0,next_cut_=1000;
    unsigned partial_count_=0;
};
}
struct ServingRawSession {
    core::WindowBudget ledger;
    explicit ServingRawSession(uint64_t host=64ULL<<20):ledger({{0},host},{{0}}){}
    bool request_open()const{return true;}
    uint64_t max_tokens()const{return 512;}
    struct Geometry{int64_t n_vocab=100;};
    Geometry geometry()const{return {};}
    struct Result{bool ok;std::string detail;explicit operator bool()const{return ok;}};
    Result admit_host(uint64_t n,std::shared_ptr<const void>& credit){
        auto result=ledger.reserve({{0},n},core::ReservationPurpose::foreground);
        if(!result)return {false,"host admission refused"};credit=std::move(result.lease);return {true,{}};
    }
};
struct Outcome {
    std::shared_ptr<const void> request_credit;
    std::vector<int32_t> request_tokens;
    core::SnapshotDescriptor published,cohort_publication;
    std::shared_ptr<resident_repair_detail::ResidentRepairController> controller;
    uint64_t next_operation=100,max_generated=1,cohort_capacity_bound=67;
    uint64_t request_receipt_base=0,request_known_base=0,request_watermark_base=0,generated=0,known_forwarded=0;
    std::chrono::steady_clock::time_point started;
    bool retired=false,yielded=true;
};
struct Actual {
''' + prepared + '\n' + prepare + '\n' + adopt + r'''
};
struct Options{int64_t max_context=512;};
static bool actual_bgen(bool owned_batch_allocation,const std::vector<int64_t>& ids,int64_t admit_max_new,int64_t context=512,int admit_slot=0){
    Options o;o.max_context=context;
    for(int once=0;once<1;++once){
#define STRATA_SERVING_SUFFIX_REPAIR_AVAILABLE 1
''' + bgen + r'''
        return true;
    }
    return false;
}
static bool actual_generic_context(bool owned_batch_allocation,int64_t n,int64_t max_new,int64_t admit_max_new,int64_t context,int admit_slot=0){
    Options o;o.max_context=context;
    for(int once=0;once<1;++once){
''' + generic_context + r'''
        return true;
    }
    return false;
}
static bool actual_owned_command(bool owned_batch_allocation,int admit_slot){
    const double req_temperature=0;const int req_penalty_last_n=0;const bool req_bias_present=false;
    const std::string req_http_request="000000000001";
    for(int once=0;once<1;++once){
''' + command_guard + r'''
        return true;
    }
    return false;
}
static void setup(Outcome& owner){
    owner.controller=std::make_shared<resident_repair_detail::ResidentRepairController>();
    auto& c=*owner.controller;auto& h=c.published_->metadata;
    h.key.engine_generation=1;h.key.window_generation=1;
    h.cut.id=100;h.cut.forwarded=65;h.cut.accepted=66;h.cut.output_watermark=64;
    h.accepted_tokens.assign(66,3);h.pending_tokens={3};
    owner.published=h;owner.request_tokens=h.accepted_tokens;
    for(size_t i=0;i<64;++i)c.receipts_[i]={i+1,1,i+2,3};
}
static void bgen_tests(){
    std::vector<int64_t> ids(40,3);
    require(actual_bgen(true,ids,65),"actual owned BGEN allows admitted65 output");
    require(actual_bgen(true,ids,472),"actual owned BGEN full budget fits exact logical context");
    require(!actual_bgen(true,ids,473),"actual owned BGEN rejects full budget, not merely GEN1 phase");
    require(!actual_bgen(true,ids,0)&&!actual_bgen(true,ids,INT64_MAX),"actual owned BGEN rejects unresolved or oversized budget");
    require(actual_bgen(false,ids,INT64_MAX),"default-off native admission fragment unchanged");
}
static void boundary_tests(){
    const std::vector<int64_t> ids(32765,3);
    require(actual_bgen(true,ids,3,32768),"actual early owned full-budget boundary32765/3 fits32768");
    require(actual_generic_context(true,32765,1,3,32768),"actual later owned GEN1 guard preserves full-budget boundary");
    require(!actual_bgen(true,ids,4,32768)&&!actual_generic_context(true,32765,1,4,32768),
        "both actual owned guards reject logical capacity+1");
    require(!actual_generic_context(false,32765,1,3,32768),"ordinary legacy counterexample retains8 slack");
    require(actual_generic_context(false,32759,1,3,32768)&&!actual_generic_context(false,32760,1,3,32768),
        "actual legacy GEN1 boundary unchanged");
}
static void plain_gen_tests(bool stale_only=false){
    const std::vector<int64_t> ordinary(40,3);
    for(int64_t previous:{int64_t(0),int64_t(1),int64_t(470),INT64_MAX}){
        if(stale_only&&previous!=470)continue;
        require(actual_bgen(true,ordinary,previous,512,-1),"plainGEN never consults absent/stale BGEN full budget");
        require(actual_generic_context(true,40,64,previous,512,-1)&&
            !actual_generic_context(true,40,480,previous,512,-1),"plainGEN context uses its current budget and legacy8");
        require(!actual_owned_command(true,-1),"actual earlier owned command guard retains pre-existing GEN refusal");
    }
    require(actual_owned_command(true,0)&&actual_owned_command(false,-1),"actual command guard preserves BGEN and ordinary legacy routes");
}
static void continuation_tests(){
    ServingRawSession session;Outcome owner;setup(owner);std::string error;
    std::vector<int64_t> incoming(66,3);
    const auto* historical=&owner.controller->receipts_[0];
    Actual::PreparedContinuationRequest prepared;
    require(Actual::prepare_continuation_request(session,owner,incoming,65,prepared,error),
        "actual production continuation admits cumulative129 outputs");
    require(owner.controller->receipts_.size()==64&&owner.request_tokens.size()==66,
        "actual production preparation leaves selected owner/capacity unchanged");
    require(session.ledger.snapshot().reserved.host_bytes>=(1ULL<<20)+128*(66+65),
        "actual production continuation pays full request+output history");
    Actual::adopt_continuation_request(owner,prepared);
    require(owner.controller->receipts_.size()==129&&&owner.controller->receipts_[0]==historical&&historical->operation==1,
        "actual adoption retains stable historical result");
    require(owner.cohort_capacity_bound==131&&owner.published.accepted_tokens.size()==66&&
        owner.published.accepted_tokens.capacity()>=131&&owner.max_generated==65&&owner.request_receipt_base==64,
        "actual adoption publishes paid history capacity without fabricating generated outputs");
    {
        ServingRawSession small(1024);Outcome old;setup(old);Actual::PreparedContinuationRequest refused;
        require(!Actual::prepare_continuation_request(small,old,incoming,65,refused,error)&&
            old.controller->receipts_.size()==64&&old.max_generated==1&&small.ledger.snapshot().reserved.host_bytes==0,
            "actual ledger refusal precedes displacement/publication");
    }
    {
        ServingRawSession s;Outcome old;setup(old);
        for(uint64_t bad:{uint64_t(0),uint64_t(447),UINT64_MAX}){
            Actual::PreparedContinuationRequest refused;
            require(!Actual::prepare_continuation_request(s,old,incoming,bad,refused,error)&&
                old.controller->receipts_.size()==64&&s.ledger.snapshot().reserved.host_bytes==0,
                "actual zero/context/overflow admission refuses before mutation");
        }
        old.next_operation=UINT64_MAX;Actual::PreparedContinuationRequest refused;
        require(!Actual::prepare_continuation_request(s,old,incoming,1,refused,error),"actual operation counter exhaustion remains refused");
    }
}
}
int main(int argc,char** argv){
    const std::string which=argc>1?argv[1]:"all";
    if(which=="all"||which=="bgen")strata::program::bgen_tests();
    if(which=="all"||which=="boundary")strata::program::boundary_tests();
    if(which=="all"||which=="plain-gen")strata::program::plain_gen_tests();
    if(which=="plain-gen-stale")strata::program::plain_gen_tests(true);
    if(which=="all"||which=="continuation")strata::program::continuation_tests();
    std::puts("PASS extracted production full-BGEN and retained-continuation admission/adoption");
}
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--generate', type=Path, required=True)
    args = parser.parse_args()
    args.generate.parent.mkdir(parents=True, exist_ok=True)
    args.generate.write_text(generate(args.source_root))
