#include "strata/core/window_edit.hpp"

#include <algorithm>
#include <new>

namespace strata::core {
namespace {
WindowStateResult fail(WindowStateError e,const char* message) { return {e,message}; }
bool same_lineage(const WindowKey& a,const WindowKey& b) {
    return a.engine_generation == b.engine_generation && a.window_generation == b.window_generation &&
           a.window_id == b.window_id && a.session_id == b.session_id && a.thread_id == b.thread_id &&
           a.revision == b.revision && a.provenance_epoch == b.provenance_epoch &&
           a.model_profile == b.model_profile && a.topology == b.topology &&
           a.append_epoch <= b.append_epoch && a.selection_epoch <= b.selection_epoch;
}
}
WindowStateResult plan_exact_window_edit(const BundleHandle& source,const WindowKey& expected,
                                        const WindowKey& target,const std::vector<int32_t>& ids,
                                        uint64_t forwarded,uint64_t max_tokens,
                                        const std::vector<BundleHandle>& checkpoints,ExactWindowEdit& out) {
    if (!source || source.get()->key != expected)
        return fail(WindowStateError::stale_generation,"source bundle identity no longer matches selection");
    const auto& parent = *source.get();
    if (target.engine_generation != expected.engine_generation || !target.window_generation ||
        target.window_id.empty() || target.session_id.empty() || target.thread_id.empty() ||
        (target.window_id == expected.window_id && target.window_generation != expected.window_generation))
        return fail(WindowStateError::stale_generation,"target identity must name this window or an explicit new fork");
    if (target.window_id == expected.window_id &&
        (target.revision <= expected.revision || target.session_id != expected.session_id || target.thread_id != expected.thread_id))
        return fail(WindowStateError::stale_revision,"an edit must advance the selected window revision");
    if (target.model_profile != expected.model_profile || target.topology != expected.topology ||
        target.provenance_epoch != expected.provenance_epoch)
        return fail(WindowStateError::incompatible_layout,"edit reuse requires identical model/profile/topology/provenance");
    if (ids.size() > max_tokens) return fail(WindowStateError::limit_exceeded,"edited history exceeds token limit");
    if (forwarded > ids.size() || std::any_of(ids.begin(),ids.end(),[](int32_t t){return t<0;}))
        return fail(WindowStateError::invalid_cut,"invalid edited forwarded history");
    try {
        ExactWindowEdit plan;
        plan.target = target; plan.accepted_tokens = ids; plan.target_history = window_history_digest(ids);
        plan.replay_end = forwarded; plan.output_watermark = parent.cut.output_watermark;
        plan.pending_tokens.assign(ids.begin()+forwarded,ids.end());
        const uint64_t bound = std::min(parent.cut.exact_frontier,forwarded);
        while (plan.unchanged_exact_prefix < bound &&
               parent.accepted_tokens[size_t(plan.unchanged_exact_prefix)] == ids[size_t(plan.unchanged_exact_prefix)])
            ++plan.unchanged_exact_prefix;
        plan.gap_end = plan.unchanged_exact_prefix;
        auto consider = [&](const BundleHandle& handle) {
            if (!handle) return;
            const auto& cp = *handle.get();
            const uint64_t c = cp.cut.forwarded;
            if (!same_lineage(cp.key,expected) || cp.steering_enabled != parent.steering_enabled ||
                cp.native_layout != parent.native_layout || cp.cut.exact_frontier != c ||
                c > plan.unchanged_exact_prefix || c <= plan.replay_begin ||
                !std::equal(cp.accepted_tokens.begin(),cp.accepted_tokens.begin()+c,ids.begin())) return;
            plan.replay_begin = c; plan.checkpoint = handle;
        };
        consider(source);
        for (const auto& checkpoint : checkpoints) consider(checkpoint);
        out = std::move(plan);
        return {};
    } catch (const std::bad_alloc&) {
        return fail(WindowStateError::limit_exceeded,"exact edit planning allocation failed");
    }
}
} // namespace strata::core
