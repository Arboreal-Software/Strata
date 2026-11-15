#pragma once
#include "strata/core/window_native.hpp"
#include "strata/core/window_budget.hpp"
#include <span>
namespace strata::core {
struct NativeSuffixLimits {
    uint64_t max_tokens=0,max_payload_bytes=0,max_metadata_bytes=0;
};
struct NativeSuffixTarget {
    const WindowKey& key; // trusted call-scoped future selection identity
    std::span<const int32_t> accepted_tokens;
    uint64_t cut_id=0,first_unresolved_unix_ms=0; // strictly newer than BOTH operands
};
enum class NativeSuffixStatus { ok, exact_fallback, stale, budget_refused, allocation_failed };
enum class NativeSuffixReason {
    none, invalid_limits, stale_operand, invalid_target_identity, unsupported_geometry,
    unsupported_format, unsupported_flags, approximate_parent, invalid_operand,
    invalid_boundaries, changed_run, suffix_mismatch, payload_bound, metadata_bound, overflow
};
struct NativeSuffixResult {
    NativeSuffixStatus status=NativeSuffixStatus::exact_fallback;
    NativeSuffixReason reason=NativeSuffixReason::none;
    explicit operator bool() const noexcept{return status==NativeSuffixStatus::ok;}
};
struct NativeSuffixStats {
    uint64_t exact_prefix_tokens=0,reused_suffix_tokens=0,pending_tokens=0;
    uint64_t logical_host_charge=0;
};
class PreparedNativeSuffix {
public:
    explicit operator bool() const noexcept{return bool(bundle_);}
    const BundleHandle& bundle() const noexcept{return bundle_;}
    const NativeSuffixStats& stats() const noexcept{return stats_;}
private:
    friend NativeSuffixResult prepare_int8_native_suffix(const BundleHandle&,const WindowKey&,
        const BundleHandle&,const WindowKey&,const ModelGeometry&,const std::vector<NativeWindowStage>&,
        NativeSuffixTarget,NativeSuffixLimits,WindowBudget&,PreparedNativeSuffix&);
    BundleHandle bundle_;NativeSuffixStats stats_;
};
// Trusted stages/geometry come from actual runtime, never imported metadata.
// Donor: accepted N, exact forwarded F=N-1; prefix: exact accepted/forwarded B.
// Require aligned B,F; F-B>=4; one changed run before B, target[B,N)=donor[B,N).
// Only unrotated INT8 native format1, text/no steering/speculation/new-generation.
// Quote is checked logical introduced payload+metadata+freeze overlap, NOT admission
// or RSS. Unchanged out on refusal; original immutable captures retain own charges.
NativeSuffixResult quote_int8_native_suffix(const BundleHandle& donor,const WindowKey& expected_donor,
    const BundleHandle& prefix,const WindowKey& expected_prefix,const ModelGeometry&,
    const std::vector<NativeWindowStage>& trusted_stages,NativeSuffixTarget,NativeSuffixLimits,
    uint64_t& out_host_bytes) noexcept;
// Real WindowBudget reservation precedes introduced image/layout/metadata copies.
// Operands retained immutably during call; final frozen bundle retains charge.
// No GPU/runtime/cache publication. Out unchanged on EVERY failure. Old complete
// GDN/PLE/tail/spare/block endpoint stays old-conditioned and explicitly approximate.
// Driver must forward target[F] freshly for logits; no exact continuation claim.
NativeSuffixResult prepare_int8_native_suffix(const BundleHandle& donor,const WindowKey& expected_donor,
    const BundleHandle& prefix,const WindowKey& expected_prefix,const ModelGeometry&,
    const std::vector<NativeWindowStage>& trusted_stages,NativeSuffixTarget,NativeSuffixLimits,
    WindowBudget&,PreparedNativeSuffix& out);
} // namespace strata::core
