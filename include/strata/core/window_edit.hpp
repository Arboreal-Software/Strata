#pragma once

#include "strata/core/window_state.hpp"

namespace strata::core {
struct ExactWindowEdit {
    WindowKey target;
    StateDigest target_history{};
    uint64_t unchanged_exact_prefix = 0;
    uint64_t replay_begin = 0, replay_end = 0;
    uint64_t gap_end = 0; // [replay_begin,gap_end) restores the unchanged checkpoint gap
    uint64_t output_watermark = 0;
    BundleHandle checkpoint; // empty means initialize clean state at token zero
    std::vector<int32_t> accepted_tokens, pending_tokens;
};

// Pure exact planning. No state is installed, no F is advanced, and no emitted
// effects are replayed. The executor must teacher-force [replay_begin,replay_end)
// through every model layer and certify a new all-stage cut before publication.
WindowStateResult plan_exact_window_edit(const BundleHandle& source, const WindowKey& expected_source,
                                        const WindowKey& target, const std::vector<int32_t>& target_tokens,
                                        uint64_t target_forwarded, uint64_t max_tokens,
                                        const std::vector<BundleHandle>& checkpoints, ExactWindowEdit& out);
} // namespace strata::core
