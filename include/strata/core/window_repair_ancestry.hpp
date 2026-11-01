#pragma once
#include "strata/core/window_hash.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace strata::core {
// DATA ONLY. None of these caller-provided fields authenticates a model state,
// executor completion, checkpoint, clock, publication or repair authorization.
struct RepairMonotonicBirth {
    uint64_t domain=0, ticks_ns=0; // domain0 means UNKNOWN (ticks MUST0);
    // nonzero domain with ticks0 is a valid clock-epoch value, not unknown.
    bool operator==(const RepairMonotonicBirth&) const = default;
};
struct RepairCausalRoot {
    // Opaque caller identity: NOT an encoded/verified owner/model/topology key.
    // Trusted controller must bind those independently; no digest authentication.
    StateDigest identity{}, original_history{};
    uint64_t engine_generation=0, window_generation=0, runtime_incarnation=0;
    uint64_t original_cut=0, original_begin=0, original_count=0;
    uint64_t first_unresolved_unix_ms=0;
    RepairMonotonicBirth monotonic_birth;
    bool operator==(const RepairCausalRoot&) const = default;
};
struct RepairAncestryLimits { uint32_t max_roots=0; uint64_t max_bytes=0; };
enum class RepairAncestryStatus { ok, invalid_limits, invalid_root, overflow,
    count_bound, byte_bound, identity_conflict, invalid_coverage, output_not_empty };
struct RepairAncestryAccess;
class RepairAncestry final {
public:
    static constexpr size_t capacity=64;
    RepairAncestry()=default;
    RepairAncestry(const RepairAncestry&)=default;
    std::span<const RepairCausalRoot> roots() const noexcept {return {roots_.data(),size_};}
    bool empty() const noexcept {return size_==0;}
    // Actual fixed retained inline storage, not a serialized-length/RSS quote.
    static size_t retained_bytes() noexcept;
private:
    friend struct RepairAncestryAccess;
    RepairAncestry& operator=(const RepairAncestry&)=default;
    std::array<RepairCausalRoot,capacity> roots_{};
    size_t size_=0;
};
// No heap/ledger activity. All outputs must be empty, remain unchanged on
// refusal. One additional fixed RepairAncestry workspace exists during union/
// construction; caller supplies/charges input+output+workspace separately.
// max_bytes must cover actual retained_bytes even for a one-root table.
RepairAncestryStatus make_repair_ancestry(std::span<const RepairCausalRoot>,
    RepairAncestryLimits,RepairAncestry&) noexcept;
RepairAncestryStatus union_repair_ancestry(const RepairAncestry&,const RepairAncestry&,
    RepairAncestryLimits,RepairAncestry&) noexcept;
class RepairSpatialCoverage final {
public:
    RepairSpatialCoverage()=default;
    RepairSpatialCoverage(const RepairSpatialCoverage&)=default;
    bool valid() const noexcept {return valid_;}
    uint64_t begin() const noexcept {return begin_;}
    uint64_t end() const noexcept {return end_;}
    const RepairAncestry& ancestry() const noexcept {return ancestry_;}
private:
    friend struct RepairAncestryAccess;
    RepairSpatialCoverage& operator=(const RepairSpatialCoverage&)=default;
    RepairAncestry ancestry_;
    uint64_t begin_=0,end_=0;
    bool valid_=false;
};
// Spatial range and GLOBAL causal roots are deliberately distinct. Even empty
// remaining coverage retains all roots: a donor endpoint can still depend on
// them. No root-deletion/exact-replacement operation exists in this helper.
RepairAncestryStatus make_repair_spatial_coverage(const RepairAncestry&,uint64_t begin,
    uint64_t end,RepairSpatialCoverage&) noexcept;
RepairAncestryStatus advance_repair_spatial_begin(const RepairSpatialCoverage&,
    uint64_t new_begin,RepairSpatialCoverage&) noexcept;
struct RepairAgeSummary {
    uint64_t oldest_unresolved_unix_ms=0;
    size_t roots=0,unknown_monotonic_anchors=0;
    // Present ONLY when every birth has same nonzero clock domain as now and
    // no birth lies in its future. Unix rollback never manufactures young age.
    std::optional<uint64_t> oldest_monotonic_age_ns;
};
RepairAgeSummary repair_ancestry_age(const RepairAncestry&,RepairMonotonicBirth now) noexcept;
} // namespace strata::core
