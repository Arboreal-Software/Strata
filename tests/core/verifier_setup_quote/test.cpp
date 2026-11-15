#include "strata/core/verifier_setup_quote.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
using namespace strata::core;
static void check(bool yes,const char* why){if(!yes){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
static VerifierSetupDimensions small(){
    VerifierSetupDimensions d;d.n_layers=2;d.n_gdn_layers=1;d.n_qsa_layers=1;
    d.n_embd=32;d.hc=1;d.hc_lr=1;d.conv_channels=32;d.value_dim=32;d.v_heads=1;
    d.n_head=1;d.n_head_kv=1;d.idx_q_heads=1;d.idx_key_dim=1;
    d.n_ff=32;d.n_expert=1;d.n_vocab=4;d.max_cells=4;d.selection_cap=1;return d;
}
// Deliberately synthetic helper byte requests isolate arithmetic, not a claim
// that real native expert scratch for this geometry is one byte or runnable.
static VerifierSetupNativeRequests requests(){return {1,2080,1};}
static VerifierSetupQuoteLimits limits(){return {1ULL<<30,1ULL<<20};}
int main(){
    VerifierSetupQuote q;
    check(quote_verifier_explicit_setup(small(),requests(),{},limits(),q)==VerifierSetupQuoteStatus::ok,"small structural quote");
    // Independent hand-count:63 separately aligned regions,3364 units256.
    // Large fixed PLE snapshots alone account for3200 units. Real GPU/model
    // compatibility is intentionally NOT established by this numeric fixture.
    check(q.arena_bytes==861184&&q.arena_takes==63,"per-take alignment/fixed PLE/global scratch inventory");
    check(q.mapped_host_bytes==5592&&q.pinned_bytes==5592&&q.mapped_allocations==17,"seventeen mapped requests and same pinned subset");
    check(q.device_allocations==1&&q.created_streams==3&&q.created_events==3&&q.first_t1_graph_execs==2,"raw allocation and runtime-object counts");
    check(!VerifierSetupQuote::includes_runtime_or_graph_overhead,"never claims graph/driver admission");
    VerifierSetupLayout named;
    check(VerifierSetupLayout::arena_capacity==64&&VerifierSetupLayout::mapped_capacity==17,"fixed named table capacities");
    check(enumerate_verifier_explicit_setup(small(),requests(),{},limits(),named)==VerifierSetupQuoteStatus::ok,"actual shared enumeration");
    check(named.arena_count==63&&named.mapped_count==17,"optional nat_xr absent from active entries");
    const auto* tok=named.find(VerifierArenaField::tok);
    const auto* step=named.find(VerifierArenaField::step);
    const auto* xq=named.find(VerifierArenaField::xq);
    const auto* hist=named.find(VerifierArenaField::hist_snap);
    const auto* last=named.find(VerifierArenaField::ple_val);
    check(tok&&tok->requested_bytes==8&&tok->aligned_offset==0&&tok->aligned_bytes==256,"named first token staging offset");
    check(step&&step->aligned_offset==256&&step->requested_bytes==32,"named step starts at second256 unit");
    check(xq&&xq->aligned_offset==3584&&xq->requested_bytes==576&&xq->aligned_bytes==768,"named quantizer follows14small regions");
    check(hist&&hist->aligned_offset==41728&&hist->requested_bytes==737280,"named fixed PLE history placement");
    check(last&&last->aligned_offset==860928&&last->aligned_bytes==256,"named last end closes independent total");
    check(!named.find(VerifierArenaField::nat_xr),"no invented pointer for absent optional field");
    uint64_t end=0;
    for(uint32_t i=0;i<named.arena_count;++i){const auto& entry=named.arena[i];
        check(entry.aligned_offset==end&&entry.aligned_offset%256==0&&entry.aligned_bytes%256==0&&entry.requested_bytes<=entry.aligned_bytes,"nooverlap/hole and aligned span");
        for(uint32_t j=0;j<i;++j)check(named.arena[j].field!=entry.field,"named target IDs unique");
        end=entry.aligned_offset+entry.aligned_bytes;
    }
    check(end==861184&&end==named.quote.arena_bytes,"last named span is total shared quote");
    check(named.mapped[6].field==VerifierMappedField::nf&&named.mapped[6].requested_bytes==512&&named.mapped[16].field==VerifierMappedField::plan&&named.mapped[16].requested_bytes==1408,"mapped names retain source order and fixed capacities");
    VerifierSetupLayout unchanged;unchanged.arena_count=19;unchanged.arena[0].requested_bytes=77;
    auto excessive=requests();excessive.hit_scratch_bytes=std::numeric_limits<uint64_t>::max();
    check(enumerate_verifier_explicit_setup(small(),excessive,{},limits(),unchanged)==VerifierSetupQuoteStatus::overflow&&unchanged.arena_count==19&&unchanged.arena[0].requested_bytes==77,"failed enumeration cannot partially publish layout");
    VerifierSetupProfile flags;flags.had_gu_any=true;flags.qfuse_counters=true;flags.side_mixer_mode=2;flags.external_main_stream=true;
    VerifierSetupQuote extra;
    check(quote_verifier_explicit_setup(small(),requests(),flags,limits(),extra)==VerifierSetupQuoteStatus::ok,"explicit resolved branches");
    check(extra.arena_bytes==861440&&extra.arena_takes==64&&extra.extra_device_bytes==4&&extra.device_allocations==2,"optional rotated-input and qcnt requests");
    check(extra.created_streams==4&&extra.created_events==7,"borrowed main stream and side streams count separately");
    VerifierSetupLayout optional;
    check(enumerate_verifier_explicit_setup(small(),requests(),flags,limits(),optional)==VerifierSetupQuoteStatus::ok,"optional named enumeration");
    const auto* rotated=optional.find(VerifierArenaField::nat_xr);
    check(optional.arena_count==64&&rotated&&rotated->requested_bytes==256&&rotated->aligned_bytes==256,"optional nat_xr has one real named span");
    const auto* shifted=optional.find(VerifierArenaField::hist_snap);
    check(shifted&&shifted->aligned_offset==41984&&optional.quote.arena_bytes==861440,"optional span shifts following offsets and same total");
    auto global=small();global.n_layers=3;global.n_gdn_layers=2;VerifierSetupQuote larger;
    check(quote_verifier_explicit_setup(global,requests(),{},limits(),larger)==VerifierSetupQuoteStatus::ok&&larger.arena_bytes==861696,"GLOBAL GDN count retained per stage, not divided by local range");
    VerifierSetupQuote sentinel;sentinel.arena_bytes=77;
    auto bad=requests();bad.max_expert_blob_bytes=std::numeric_limits<uint64_t>::max();
    check(quote_verifier_explicit_setup(small(),bad,{},limits(),sentinel)==VerifierSetupQuoteStatus::overflow&&sentinel.arena_bytes==77,"blob multiplication overflow leaves output unchanged");
    bad=requests();bad.hit_scratch_bytes=std::numeric_limits<uint64_t>::max();
    check(quote_verifier_explicit_setup(small(),bad,{},limits(),sentinel)==VerifierSetupQuoteStatus::overflow&&sentinel.arena_bytes==77,"alignment rounding overflow");
    auto d=small();d.max_t=1;
    check(quote_verifier_explicit_setup(d,requests(),{},limits(),sentinel)==VerifierSetupQuoteStatus::invalid_dimensions,"executed T1 is not allocated capacity1");
    d=small();d.n_head=std::numeric_limits<uint64_t>::max();
    check(quote_verifier_explicit_setup(d,requests(),{},limits(),sentinel)==VerifierSetupQuoteStatus::invalid_dimensions,"int-cast bound before dimension arithmetic");
    d=small();d.n_embd=33;
    check(quote_verifier_explicit_setup(d,requests(),{},limits(),sentinel)==VerifierSetupQuoteStatus::invalid_dimensions,"quantizer block divisibility");
    flags={};flags.unsupported_policy_mask=1;
    check(quote_verifier_explicit_setup(small(),requests(),flags,limits(),sentinel)==VerifierSetupQuoteStatus::unsupported_profile,"unpriced optional policies refuse");
    check(quote_verifier_explicit_setup(small(),requests(),{},{861183,5592},sentinel)==VerifierSetupQuoteStatus::bound&&sentinel.arena_bytes==77,"exact arena limit refusal");
    check(quote_verifier_explicit_setup(small(),requests(),{},{861184,5591},sentinel)==VerifierSetupQuoteStatus::bound&&sentinel.arena_bytes==77,"exact host limit refusal");
    VerifierSerialHandoffQuote hand;
    check(quote_verifier_serial_handoffs(32,1,3,4160,hand)==VerifierSetupQuoteStatus::ok&&hand.bytes_per_edge==2080&&hand.mapped_host_bytes==4160&&hand.pinned_bytes==4160&&hand.mapped_allocations==2,"separate8row serial handoffs per edge");
    VerifierSerialHandoffQuote held;held.mapped_host_bytes=99;
    check(quote_verifier_serial_handoffs(32,1,3,4159,held)==VerifierSetupQuoteStatus::bound&&held.mapped_host_bytes==99,"handoff bound leaves output unchanged");
    check(quote_verifier_serial_handoffs(32,1,1,1,hand)==VerifierSetupQuoteStatus::ok&&hand.mapped_allocations==0&&hand.mapped_host_bytes==0,"single stage has no edge allocation");
    std::puts("PASS explicit verifier setup REQUEST quote (runtime/graphs NOT covered)");
}
