// SOURCE ONLY, UNCOMPILED. Include outside anonymous namespace. Host data
// comparison only: caller supplies complete drained immutable RAW captures and
// the ACTUAL runtime-generated native layout for their common semantic cut.
#include "strata/core/window_state.hpp"
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
namespace strata::program::clean_recompute_qualification {
enum class Difference { invalid_input, invalid_repaired, invalid_clean, not_exact,
    accepted_tokens, pending_tokens, cut_semantics, continuation, execution_policy,
    native_layout, device_inventory, tensor_layout, payload_extent, payload_read,
    tensor_bytes, allocation_failed, unexpected_failure, control_identity, approximation_metadata };
struct Mismatch {
    Difference reason=Difference::invalid_input;
    uint32_t stage=0;
    int64_t layer=-1;
    core::StatePartKind kind=core::StatePartKind::qsa_k;
    uint64_t offset=0,left=0,right=0;
};
struct Report {
    bool completed=false,equal=false;
    uint64_t tensors_compared=0,bytes_compared=0,mismatches=0;
    std::array<Mismatch,16> first{};
    core::WindowStateError validation_error=core::WindowStateError::none;
    void note(Mismatch m)noexcept {
        if(mismatches<first.size())first[size_t(mismatches)]=m;
        ++mismatches;
    }
};
inline bool same_semantic_cut(const core::StateCut& a,const core::StateCut& b)noexcept {
    // Only monotonically minted cut ID differs. History/content/position and
    // output accounting are NOT ignored or inferred from one another.
    return a.accepted==b.accepted&&a.forwarded==b.forwarded&&
        a.exact_frontier==b.exact_frontier&&a.output_watermark==b.output_watermark&&a.history==b.history;
}
inline bool bounded(const core::SnapshotDescriptor& s,const core::WindowStateLayout& actual)noexcept {
    if(s.devices.empty()||s.devices.size()>8||s.devices.size()!=actual.stages.size()||
       !actual.max_tokens||!actual.max_payload_bytes||actual.layers.empty()||actual.layers.size()>4096||
       s.accepted_tokens.size()>actual.max_tokens||s.pending_tokens.size()>actual.max_tokens)return false;
    uint64_t tensors=0,bytes=0;
    for(const auto& d:s.devices){
        if(d.tensors.size()>65536-tensors)return false;tensors+=d.tensors.size();
        for(const auto& t:d.tensors){
            if(t.payload.size()>actual.max_payload_bytes-bytes)return false;bytes+=t.payload.size();
        }
    }
    return tensors!=0;
}
namespace detail {
enum class Purpose { clean_exact, preserved_selection };
inline bool same_physical_layout(const core::WindowStateLayout& a,const core::WindowStateLayout& b)noexcept {
    // Topology binds actual Session addresses, not portable physical tensor meaning.
    if(a.model_profile!=b.model_profile||a.native_layout!=b.native_layout||a.max_tokens!=b.max_tokens||
       a.max_payload_bytes!=b.max_payload_bytes||a.steering_enabled!=b.steering_enabled||a.layers!=b.layers||
       a.stages.size()!=b.stages.size())return false;
    for(size_t i=0;i<a.stages.size();++i){const auto& x=a.stages[i];const auto& y=b.stages[i];
        if(x.id!=y.id||x.layer_begin!=y.layer_begin||x.layer_end!=y.layer_end||x.ple!=y.ple||x.tensors!=y.tensors)return false;}
    return true;
}
inline bool compare_impl(const core::BundleHandle& repaired,const core::BundleHandle& clean,
                    const core::WindowStateLayout& actual,const core::WindowStateLayout& clean_actual,Report& report,Purpose purpose)noexcept {
    report={};
    if(!repaired||!clean||repaired.get()==clean.get()){
        // Same immutable object is not an independently captured recomputation.
        report.note({Difference::invalid_input});return false;
    }
    const auto& a=*repaired.get();const auto& b=*clean.get();
    if(!bounded(a,actual)||!bounded(b,clean_actual)){
        report.note({Difference::invalid_input});return false;
    }
    try {
        // Existing actual validator checks complete native inventory, exact
        // tensor bytes/shape, device cuts and supported continuation. Own key
        // is deliberate: cross-run control identifiers are not state payload.
        const auto av=core::validate_window_state(a,a.key,actual);
        if(!av){report.validation_error=av.error;report.note({Difference::invalid_repaired});return false;}
        const auto bv=core::validate_window_state(b,b.key,clean_actual);
        if(!bv){report.validation_error=bv.error;report.note({Difference::invalid_clean});return false;}
        if(!same_physical_layout(actual,clean_actual)){report.note({Difference::native_layout});return false;}
        if(purpose==Purpose::clean_exact){
            if(!a.approximation.empty()||!b.approximation.empty()||
               a.cut.exact_frontier!=a.cut.forwarded||b.cut.exact_frontier!=b.cut.forwarded){
                report.note({Difference::not_exact});return false;
            }
        }else{
            // Preservation compares the SAME selected publication, not a clean
            // recomputation. Every identity/cut/approximation field is strict.
            if(a.key!=b.key||a.cut!=b.cut)report.note({Difference::control_identity});
            if(a.approximation.size()!=b.approximation.size())report.note({Difference::approximation_metadata});
            else for(size_t i=0;i<a.approximation.size();++i){const auto& x=a.approximation[i];const auto& y=b.approximation[i];
                if(x.begin!=y.begin||x.end!=y.end||x.first_unresolved_unix_ms!=y.first_unresolved_unix_ms||x.ancestor!=y.ancestor)
                    report.note({Difference::approximation_metadata});
            }
        }
        // In clean_exact ONLY, identity/key generations, session/thread/window IDs, revision/append/
        // selection/provenance epochs and cut IDs are deliberately not compared
        // ACROSS captures. Existing validator checks each own key validity;
        // model_profile/topology/native schema MUST match actual on BOTH sides.
        if(a.native_layout!=b.native_layout)report.note({Difference::native_layout});
        if(a.accepted_tokens!=b.accepted_tokens)report.note({Difference::accepted_tokens});
        if(a.pending_tokens!=b.pending_tokens)report.note({Difference::pending_tokens});
        if(!same_semantic_cut(a.cut,b.cut))report.note({Difference::cut_semantics});
        if(a.continuation.mode!=b.continuation.mode||a.continuation.adapter!=b.continuation.adapter||
           a.continuation.payload!=b.continuation.payload)report.note({Difference::continuation});
        if(a.speculation_disabled!=b.speculation_disabled||a.steering_enabled!=b.steering_enabled)
            report.note({Difference::execution_policy});
        std::array<uint8_t,4096> left{},right{}; // fixed8KiB comparison scratch
        for(size_t i=0;i<a.devices.size();++i){const auto& x=a.devices[i];const auto& y=b.devices[i];
            if(x.id!=y.id||x.layer_begin!=y.layer_begin||x.layer_end!=y.layer_end||
               !same_semantic_cut(x.cut,y.cut)||x.tensors.size()!=y.tensors.size()){
                report.note({Difference::device_inventory,x.id});continue;
            }
            for(size_t j=0;j<x.tensors.size();++j){const auto& p=x.tensors[j];const auto& q=y.tensors[j];
                Mismatch m;m.stage=x.id;m.layer=p.layout.layer;m.kind=p.layout.kind;
                if(p.layout!=q.layout){m.reason=Difference::tensor_layout;report.note(m);continue;}
                if(p.payload.size()!=q.payload.size()||p.payload.size()!=p.layout.bytes){
                    m.reason=Difference::payload_extent;report.note(m);continue;
                }
                bool differs=false;
                for(size_t at=0;at<p.payload.size();){const size_t n=std::min(left.size(),p.payload.size()-at);
                    if(!p.payload.read(left.data(),at,n)||!q.payload.read(right.data(),at,n)){
                        m.reason=Difference::payload_read;m.offset=at;report.note(m);return false;
                    }
                    if(!differs&&std::memcmp(left.data(),right.data(),n)!=0){
                        size_t k=0;while(k<n&&left[k]==right[k])++k;
                        m.reason=Difference::tensor_bytes;m.offset=at+k;m.left=left[k];m.right=right[k];report.note(m);differs=true;
                    }
                    report.bytes_compared+=n;at+=n;
                }
                ++report.tensors_compared;
            }
        }
        report.completed=true;report.equal=report.mismatches==0;return report.equal;
    }catch(const std::bad_alloc&){report.note({Difference::allocation_failed});return false;}
    catch(...){report.note({Difference::unexpected_failure});return false;}
}
} // namespace detail
// Existing clean recomputation API remains EXACT ONLY.
inline bool compare(const core::BundleHandle& repaired,const core::BundleHandle& clean,
                    const core::WindowStateLayout& actual,Report& report)noexcept {
    return detail::compare_impl(repaired,clean,actual,actual,report,detail::Purpose::clean_exact);
}
// Explicit physical bridge only: trusted callers independently derive each actual owner's layout.
inline bool compare_physical_bridge(const core::BundleHandle& source,const core::BundleHandle& destination,
                    const core::WindowStateLayout& source_actual,const core::WindowStateLayout& destination_actual,
                    Report& report)noexcept {
    return detail::compare_impl(source,destination,source_actual,destination_actual,report,detail::Purpose::clean_exact);
}
// Separate preservation API: valid approximate lineage is allowed only when
// BOTH captures have exactly the same selected key/cut/region metadata.
inline bool compare_preserved_selection(const core::BundleHandle& before,const core::BundleHandle& after,
                    const core::WindowStateLayout& actual,Report& report)noexcept {
    return detail::compare_impl(before,after,actual,actual,report,detail::Purpose::preserved_selection);
}
} // namespace strata::program::clean_recompute_qualification
