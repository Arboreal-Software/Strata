#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/window_state.hpp"

namespace strata::core {

// Resolved from live SessionState by the caller, never from an imported image.
// Native formats match conversation_snapshot.cpp: 0 FP16, 1 INT8, 2 rotated
// Q4_0, 3 K8V4; +16 distinguishes storage-rotated FP16/INT8.
struct NativeWindowStage {
    int64_t layer_begin = 0, layer_end = 0;
    std::vector<int> qsa_formats;
    bool ple = false;
};

WindowStateResult native_window_layout(const ModelGeometry&, const std::vector<NativeWindowStage>&,
                                      const WindowKey&, const StateCut&, uint64_t max_tokens,
                                      uint64_t max_payload_bytes, WindowStateLayout& out, bool steering_enabled = true);

struct NativePleTokens {
    std::array<int32_t, 2> previous{-1, -1};
    int32_t current = -1;
};

// Convert an already drained, text-only, non-speculative native capture. This
// never touches a running session. The accepted/pending cut and actual PLE token
// metadata must be supplied by the execution owner, not inferred from KV bytes.
// Historical checkpoint caches are rebuildable acceleration data, not the live
// causal endpoint. The adapter intentionally does not persist those caches.
WindowStateResult describe_native_window(const SavedConversation&, const ModelGeometry&,
                                        const WindowStateLayout&, const SnapshotDescriptor& metadata,
                                        const std::vector<NativePleTokens>&,
                                        SnapshotDescriptor& out);

// Rebuild a private native host image for the existing restore validator. Does
// not install it or mutate a SessionState. The execution owner must restore into
// private device destinations and atomically publish only after all stages drain.
WindowStateResult materialize_native_window(const SnapshotDescriptor&, const WindowKey& expected,
                                           const ModelGeometry&, const std::vector<NativeWindowStage>&,
                                           uint64_t max_tokens, uint64_t max_payload_bytes,
                                           bool steering_enabled, SavedConversation& out,
                                           std::vector<NativePleTokens>& tokens);

} // namespace strata::core
