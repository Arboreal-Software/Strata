#include "strata/core/window_repair_ancestry.hpp"
#include <algorithm>
#include <limits>
#include <initializer_list>

namespace strata::core {
struct RepairAncestryAccess {
    static size_t& size(RepairAncestry& a) noexcept {return a.size_;}
    static auto& values(RepairAncestry& a) noexcept {return a.roots_;}
    static void publish(const RepairAncestry& a,RepairAncestry& out) noexcept {out=a;}
    static void coverage(const RepairAncestry& roots,uint64_t begin,uint64_t end,RepairSpatialCoverage& out) noexcept {
        out.ancestry_=roots;out.begin_=begin;out.end_=end;out.valid_=true;
    }
};
size_t RepairAncestry::retained_bytes() noexcept {return sizeof(RepairAncestry);}
namespace {
using S=RepairAncestryStatus;
bool present(const StateDigest& d) noexcept {for(auto b:d)if(b)return true;return false;}
S limits(RepairAncestryLimits l) noexcept {
    if(!l.max_roots || l.max_roots>RepairAncestry::capacity)return S::invalid_limits;
    return l.max_bytes<RepairAncestry::retained_bytes()?S::byte_bound:S::ok;
}
S valid(const RepairCausalRoot& r) noexcept {
    if(!present(r.identity)||!present(r.original_history)||!r.engine_generation||!r.window_generation||
       !r.runtime_incarnation||!r.original_cut||!r.original_count||!r.first_unresolved_unix_ms||
       (!r.monotonic_birth.domain&&r.monotonic_birth.ticks_ns))return S::invalid_root;
    if(r.original_begin>std::numeric_limits<uint64_t>::max()-r.original_count)return S::overflow;
    return S::ok;
}
S insert(RepairAncestry& a,const RepairCausalRoot& r,RepairAncestryLimits l) noexcept {
    auto status=valid(r);if(status!=S::ok)return status;
    auto& values=RepairAncestryAccess::values(a);auto& size=RepairAncestryAccess::size(a);
    size_t at=0;
    while(at<size && values[at].identity<r.identity)++at;
    if(at<size&&values[at].identity==r.identity)return values[at]==r?S::ok:S::identity_conflict;
    if(size>=l.max_roots || size>=RepairAncestry::capacity)return S::count_bound;
    for(size_t i=size;i>at;--i)values[i]=values[i-1];
    values[at]=r;++size;return S::ok;
}
}
RepairAncestryStatus make_repair_ancestry(std::span<const RepairCausalRoot> roots,
    RepairAncestryLimits bound,RepairAncestry& out) noexcept {
    if(!out.empty())return S::output_not_empty;
    auto status=limits(bound);if(status!=S::ok)return status;
    // Bound work BEFORE iterating caller input, including duplicate floods.
    if(roots.empty())return S::invalid_root;
    if(roots.size()>RepairAncestry::capacity)return S::count_bound;
    RepairAncestry candidate;
    for(const auto& r:roots){status=insert(candidate,r,bound);if(status!=S::ok)return status;}
    RepairAncestryAccess::publish(candidate,out);return S::ok;
}
RepairAncestryStatus union_repair_ancestry(const RepairAncestry& a,const RepairAncestry& b,
    RepairAncestryLimits bound,RepairAncestry& out) noexcept {
    if(!out.empty())return S::output_not_empty;
    auto status=limits(bound);if(status!=S::ok)return status;
    if(a.empty()||b.empty())return S::invalid_root;
    RepairAncestry candidate;
    for(auto* input:{&a,&b})for(const auto& r:input->roots()){
        status=insert(candidate,r,bound);if(status!=S::ok)return status;
    }
    RepairAncestryAccess::publish(candidate,out);return S::ok;
}
RepairAncestryStatus make_repair_spatial_coverage(const RepairAncestry& roots,uint64_t begin,
    uint64_t end,RepairSpatialCoverage& out) noexcept {
    if(out.valid())return S::output_not_empty;
    if(roots.empty()||begin>end)return S::invalid_coverage;
    RepairAncestryAccess::coverage(roots,begin,end,out);return S::ok;
}
RepairAncestryStatus advance_repair_spatial_begin(const RepairSpatialCoverage& old,uint64_t begin,
    RepairSpatialCoverage& out) noexcept {
    if(out.valid())return S::output_not_empty;
    if(!old.valid()||begin<old.begin()||begin>old.end())return S::invalid_coverage;
    RepairAncestryAccess::coverage(old.ancestry(),begin,old.end(),out);return S::ok;
}
RepairAgeSummary repair_ancestry_age(const RepairAncestry& a,RepairMonotonicBirth now) noexcept {
    RepairAgeSummary result;result.roots=a.roots().size();uint64_t age=0;
    for(const auto& r:a.roots()){
        if(!result.oldest_unresolved_unix_ms || r.first_unresolved_unix_ms<result.oldest_unresolved_unix_ms)
            result.oldest_unresolved_unix_ms=r.first_unresolved_unix_ms;
        if(!now.domain||!r.monotonic_birth.domain||now.domain!=r.monotonic_birth.domain||now.ticks_ns<r.monotonic_birth.ticks_ns)
            ++result.unknown_monotonic_anchors;
        else age=std::max(age,now.ticks_ns-r.monotonic_birth.ticks_ns);
    }
    if(result.roots && !result.unknown_monotonic_anchors)result.oldest_monotonic_age_ns=age;
    return result;
}
} // namespace strata::core
