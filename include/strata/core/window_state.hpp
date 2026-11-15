// Complete-window metadata. No GPU calls and no changes to the legacy cache path.
#pragma once

#include "strata/core/conversation_buffer.hpp"
#include "strata/core/window_hash.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct WindowKey {
    uint64_t engine_generation = 0, window_generation = 0;
    std::string session_id, thread_id, window_id;
    uint64_t revision = 0, append_epoch = 0, selection_epoch = 0, provenance_epoch = 0;
    StateDigest model_profile{}, topology{};
    bool operator==(const WindowKey&) const = default;
};

struct StateCut {
    uint64_t id = 0;
    uint64_t accepted = 0, forwarded = 0, exact_frontier = 0;
    uint64_t output_watermark = 0;
    StateDigest history{};
    bool operator==(const StateCut&) const = default;
};

enum class StatePartKind {
    qsa_k, qsa_v, qsa_k_scale, qsa_v_scale, qsa_pooled,
    qsa_tail, qsa_spare, qsa_block_position,
    gdn_recurrence_and_convolution, ple_history, ple_previous_tokens, ple_current_token,
};

// The runtime adapter supplies this exact native layout, including quantization
// block geometry, page padding and valid cells. Import metadata cannot supply its
// own expected layout. Logical device IDs are stage IDs, not raw CUDA pointers.
struct TensorLayout {
    StatePartKind kind = StatePartKind::qsa_k;
    int64_t layer = -1;
    uint32_t device_part = 0;
    std::string dtype, native_layout;
    std::vector<uint64_t> shape, strides;
    uint64_t logical_begin = 0, valid_elements = 0, bytes = 0;
    bool operator==(const TensorLayout&) const = default;
};

struct TensorPart {
    TensorLayout layout;
    ConversationBuffer payload;
};

struct DevicePart {
    uint32_t id = 0;
    int64_t layer_begin = 0, layer_end = 0;
    StateCut cut;
    std::vector<TensorPart> tensors;
};

enum class ContinuationMode { new_generation_policy, exact_continuation };
struct ContinuationState {
    ContinuationMode mode = ContinuationMode::new_generation_policy;
    // No opaque parser/RNG state is silently accepted. Exact continuation will
    // require a separately implemented, versioned continuation adapter.
    std::string adapter;
    std::vector<uint8_t> payload;
};

struct ApproximationRegion {
    uint64_t begin = 0, end = 0;
    uint64_t first_unresolved_unix_ms = 0;
    StateDigest ancestor{};
};

struct SnapshotDescriptor {
    WindowKey key;
    StateCut cut;
    std::string native_layout;
    std::vector<int32_t> accepted_tokens;
    std::vector<int32_t> pending_tokens; // exactly accepted_tokens[C:L]
    std::vector<DevicePart> devices;
    std::vector<ApproximationRegion> approximation;
    ContinuationState continuation;
    // Initial adapter only accepts fully drained, disabled speculation. Enabling
    // a drafter requires a new explicit schema and adapter, not dropping its state.
    bool speculation_disabled = true;
    bool steering_enabled = true;
};

struct WindowStateLayout {
    std::string native_layout;
    StateDigest model_profile{}, topology{};
    enum class LayerKind { qsa, gdn };
    std::vector<LayerKind> layers; // global execution-layer inventory from the model
    struct Stage {
        uint32_t id = 0;
        int64_t layer_begin = 0, layer_end = 0;
        bool ple = false;
        std::vector<TensorLayout> tensors;
    };
    std::vector<Stage> stages;
    uint64_t max_tokens = 0, max_payload_bytes = 0;
    bool steering_enabled = true;
};

enum class WindowStateError {
    none, stale_generation, stale_revision, stale_append, stale_selection,
    invalid_cut, missing_state_part, incompatible_layout, unsupported_continuation,
    unsupported_speculation, limit_exceeded,
};

struct WindowStateResult {
    WindowStateError error = WindowStateError::none;
    std::string detail;
    explicit operator bool() const { return error == WindowStateError::none; }
};

WindowStateResult validate_window_state(const SnapshotDescriptor&, const WindowKey& expected,
                                       const WindowStateLayout&);

// Owns a deep, immutable copy. Source buffers may be reused only after P02's
// asynchronous cut/drain contract; this host type does not certify GPU completion.
// No mutable constructor/accessor is exposed. Failed freeze leaves `out` intact.
class BundleHandle {
public:
    const SnapshotDescriptor* get() const { return state_.get(); }
    explicit operator bool() const { return bool(state_); }
    static WindowStateResult freeze(const SnapshotDescriptor&, const WindowKey&,
                                    const WindowStateLayout&, BundleHandle& out,
                                    std::shared_ptr<const void> allocation_owner = {});
private:
    std::shared_ptr<const SnapshotDescriptor> state_;
};

} // namespace strata::core
