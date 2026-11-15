#pragma once
#include <cstdint>
#include <array>
#include <cstddef>
namespace strata::core {
// Pure requested-allocation DATA. No CUDA headers/calls, budget reservation,
// construction/completion authority, hidden-runtime/graph memory qualification.
struct VerifierSetupDimensions {
    uint64_t max_t=2,k=10,n_layers=0,n_gdn_layers=0,n_qsa_layers=0;
    uint64_t n_embd=0,hc=0,hc_lr=0,conv_channels=0,value_dim=0,v_heads=0;
    uint64_t n_head=0,head_dim=256,n_head_kv=0,idx_q_heads=0,idx_key_dim=0,idx_block=4;
    uint64_t n_ff=0,n_expert=0,n_vocab=0,max_cells=0,selection_cap=0;
};
struct VerifierSetupNativeRequests {
    // Actual trusted native helper results for storage T=2, not current T1.
    // Caller MUST bind these to real carve; this helper cannot authenticate them.
    uint64_t hit_scratch_bytes=0,argmax_scratch_bytes=0,max_expert_blob_bytes=0;
};
struct VerifierSetupProfile {
    bool had_gu_any=false,qfuse_counters=false,external_main_stream=false;
    uint8_t side_mixer_mode=0; // resolved actual0/1/2, not assumed env default.
    uint64_t unsupported_policy_mask=0; // taps/batch/trace/balance/route/adapt/etc.
};
struct VerifierSetupQuoteLimits {uint64_t max_device_bytes=0,max_host_bytes=0;};
struct VerifierSetupQuote {
    uint64_t arena_bytes=0,extra_device_bytes=0,mapped_host_bytes=0,pinned_bytes=0;
    uint32_t device_allocations=0,mapped_allocations=0,arena_takes=0;
    uint32_t created_streams=0,created_events=0,first_t1_graph_execs=0;
    // Sizes for these runtime objects are UNKNOWN and deliberately not priced.
    static constexpr bool includes_runtime_or_graph_overhead=false;
};
enum class VerifierSetupQuoteStatus {ok,invalid_dimensions,unsupported_profile,invalid_limits,overflow,bound};

// Stable actual carve targets. nat_xr is the ONLY optional arena entry. Zero
// requested bytes (e.g. global nG0) remain named entries with zero aligned span.
enum class VerifierArenaField : uint32_t {
    tok,step,pos,commit,ple,emb,R,mixed,bo,inj,inj2,lo,rs,xn,xq,sh_xq,xil,
    qkv_L,h_L,gate_L,beta_L,z,y,y_dummy,qfull,qcur,kcur,vcur,idx_raw_L,qidx,
    scores,sel,attn,attn32,attn_scratch,tail_snap,logits,w,ids,shared,parts,hit_out,
    hit_slot,hit_dst,hit_count,plan,staging,hit_xq,hit_xs,nat_xq,nat_xr,hit_scratch,
    head_mixed,head_inj,sh_bf16,sh_gate,sh_up,sh_g,head_logits,arg_scratch,one,
    hist_snap,ple_key,ple_val,count
};
enum class VerifierMappedField : uint32_t {tok,step,pos,commit,ple,out,nf,x,ids,w,
    seq,flag,flagA,flagB,plan_err,ymiss,plan,count};
struct VerifierArenaRequest {
    VerifierArenaField field{};
    uint64_t requested_bytes=0,aligned_offset=0,aligned_bytes=0;
};
struct VerifierMappedRequest {VerifierMappedField field{};uint64_t requested_bytes=0;};
struct VerifierSetupLayout {
    static constexpr std::size_t arena_capacity=static_cast<std::size_t>(VerifierArenaField::count);
    static constexpr std::size_t mapped_capacity=static_cast<std::size_t>(VerifierMappedField::count);
    std::array<VerifierArenaRequest,arena_capacity> arena{};
    std::array<VerifierMappedRequest,mapped_capacity> mapped{};
    uint32_t arena_count=0,mapped_count=0;
    VerifierSetupQuote quote;
    const VerifierArenaRequest* find(VerifierArenaField field) const noexcept {
        for(std::size_t i=0;i<arena_count&&i<arena.size();++i)if(arena[i].field==field)return &arena[i];
        return nullptr;
    }
};
// SINGLE checked enumeration is the source of both named layout and quote.
// No heap/CUDA. Caller must bind actual helper results/profile and use these
// offsets in real init; an independently duplicated old carve can still drift.
VerifierSetupQuoteStatus enumerate_verifier_explicit_setup(const VerifierSetupDimensions&,
    VerifierSetupNativeRequests,VerifierSetupProfile,VerifierSetupQuoteLimits,VerifierSetupLayout&) noexcept;

VerifierSetupQuoteStatus quote_verifier_explicit_setup(const VerifierSetupDimensions&,
    VerifierSetupNativeRequests,VerifierSetupProfile,VerifierSetupQuoteLimits,VerifierSetupQuote&) noexcept;
struct VerifierSerialHandoffQuote {
    uint64_t mapped_host_bytes=0,pinned_bytes=0,bytes_per_edge=0;
    uint32_t mapped_allocations=0;
};
// Separate gang-level edge request; current serial setup retains8 rows evenT2.
// Does not include overlap flags/counters, host vectors/owners or runtime costs.
VerifierSetupQuoteStatus quote_verifier_serial_handoffs(uint64_t n_embd,uint64_t hc,
    uint32_t stages,uint64_t max_host_bytes,VerifierSerialHandoffQuote&) noexcept;
} // namespace strata::core
