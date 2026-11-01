#pragma once
#include "strata/core/window_suffix.hpp"
#include "strata/core/window_repair_ancestry.hpp"
namespace strata::core {
// DATA consistency only: root slot is not runtime, clock or checkpoint authority.
struct RepairRegionRootBinding { uint32_t region=0, root=0; };
struct NativeRepairSuffixTarget { const WindowKey& key; uint64_t cut_id=0; };
struct NativeRepairSuffixData {
 RepairSpatialCoverage coverage;
 std::vector<RepairRegionRootBinding> region_roots;
 NativeRepairSuffixData(const RepairSpatialCoverage& c):coverage(c) {}
};
// Semantic reattachment result = native bytes PLUS the paired full root closure.
// Retain this object/data through controller reattachment and later captures.
// A bare BundleHandle retains allocation lifetime, but cannot recover this root
// table semantically; reject export without a separately versioned root closure.
class PreparedNativeRepairSuffix {
public:
 // Native-byte transport ONLY; not a complete ancestry/age representation.
 const BundleHandle& bundle() const noexcept {return bundle_;}
 const NativeRepairSuffixData* data() const noexcept {return data_.get();}
 explicit operator bool() const noexcept {return bool(bundle_);}
private:
 friend NativeSuffixResult prepare_int8_native_repair_suffix(const BundleHandle&,const WindowKey&,
  const BundleHandle&,const WindowKey&,const ModelGeometry&,const std::vector<NativeWindowStage>&,
  const RepairSpatialCoverage&,std::span<const RepairRegionRootBinding>,NativeRepairSuffixTarget,
  NativeSuffixLimits,WindowBudget&,PreparedNativeRepairSuffix&);
 BundleHandle bundle_;
 std::shared_ptr<const NativeRepairSuffixData> data_;
};
// Current approximate donor [P,C); complete exact prefix R, P<R<C, aligned4.
// Prefix accepted/forwarded=R with no pending, actual tokens match donor[0,R).
// Same content revision, prefix append <= donor append; target preserves donor
// full history/pending/watermark and changes only selection (same or newer).
// Supplied coverage and
// bindings are DATA ONLY, supplied by controller; no authentication/publication.
NativeSuffixResult quote_int8_native_repair_suffix(const BundleHandle&,const WindowKey&,
 const BundleHandle&,const WindowKey&,const ModelGeometry&,const std::vector<NativeWindowStage>&,
 const RepairSpatialCoverage&,std::span<const RepairRegionRootBinding>,NativeRepairSuffixTarget,
 NativeSuffixLimits,uint64_t&) noexcept;
NativeSuffixResult prepare_int8_native_repair_suffix(const BundleHandle&,const WindowKey&,
 const BundleHandle&,const WindowKey&,const ModelGeometry&,const std::vector<NativeWindowStage>&,
 const RepairSpatialCoverage&,std::span<const RepairRegionRootBinding>,NativeRepairSuffixTarget,
 NativeSuffixLimits,WindowBudget&,PreparedNativeRepairSuffix&);
} // namespace strata::core
