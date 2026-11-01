#pragma once

namespace strata::core::native_serial_serving {

// Requested --spec 0 remains execution intent. Native Verifier storage still needs capacity >=2, even though
// run() supports T=1. This resolves only native serial serving, including the feature-free cold reference.
template<class Policy>
bool eligible(bool native_pack,const Policy& policy) noexcept {
    return native_pack&&policy.serve&&policy.spec==0&&!policy.native_preset.empty()&&!policy.keep_canonical&&
        policy.prefill_chunk>0&&policy.mtp.empty()&&policy.draft.empty()&&policy.draft_model.empty()&&
        policy.spec_oracle.empty()&&policy.spec_follow.empty()&&policy.suffix_draft==0&&policy.lookup_chain==0&&
        policy.pipeline_windows==0&&policy.batch==0&&!policy.batch_mtp;
}

inline int verifier_capacity(int requested_spec,bool resolved_serial) noexcept {
    return resolved_serial&&requested_spec==0?2:requested_spec;
}

inline int execution_window(int requested_spec,bool resolved_serial) noexcept {
    return resolved_serial&&requested_spec==0?1:requested_spec;
}

} // namespace strata::core::native_serial_serving
