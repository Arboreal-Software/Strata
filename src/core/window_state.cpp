#include "strata/core/window_state.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <set>
#include <tuple>

namespace strata::core {
namespace {
WindowStateResult fail(WindowStateError code, const char* detail) { return {code, detail}; }
bool present(const StateDigest& d) {
    return std::any_of(d.begin(), d.end(), [](uint8_t x) { return x != 0; });
}
bool valid_cut(const StateCut& c) {
    return c.id != 0 && c.exact_frontier <= c.forwarded && c.forwarded <= c.accepted && present(c.history);
}
auto part_id(const TensorLayout& p) { return std::make_tuple(p.device_part, p.layer, p.kind); }
bool known_kind(StatePartKind k) {
    switch (k) {
    case StatePartKind::qsa_k: case StatePartKind::qsa_v:
    case StatePartKind::qsa_k_scale: case StatePartKind::qsa_v_scale:
    case StatePartKind::qsa_pooled: case StatePartKind::qsa_tail:
    case StatePartKind::qsa_spare: case StatePartKind::qsa_block_position:
    case StatePartKind::gdn_recurrence_and_convolution:
    case StatePartKind::ple_history: case StatePartKind::ple_previous_tokens:
    case StatePartKind::ple_current_token: return true;
    }
    return false;
}
}

WindowStateResult validate_window_state(const SnapshotDescriptor& s, const WindowKey& expected,
                                       const WindowStateLayout& layout) {
    const auto& k = s.key;
    if (!k.engine_generation || !k.window_generation || k.session_id.empty() || k.thread_id.empty() ||
        k.window_id.empty() || k.engine_generation != expected.engine_generation ||
        k.window_generation != expected.window_generation || k.window_id != expected.window_id ||
        k.session_id != expected.session_id || k.thread_id != expected.thread_id)
        return fail(WindowStateError::stale_generation, "window identity or generation differs");
    if (k.revision != expected.revision || k.provenance_epoch != expected.provenance_epoch)
        return fail(WindowStateError::stale_revision, "revision or provenance differs");
    if (k.append_epoch != expected.append_epoch)
        return fail(WindowStateError::stale_append, "accepted-history epoch differs");
    if (k.selection_epoch != expected.selection_epoch)
        return fail(WindowStateError::stale_selection, "selection epoch differs");
    if (!present(k.model_profile) || !present(k.topology) || k.model_profile != expected.model_profile ||
        k.topology != expected.topology || k.model_profile != layout.model_profile || k.topology != layout.topology ||
        s.native_layout.empty() || s.native_layout != layout.native_layout || s.steering_enabled != layout.steering_enabled)
        return fail(WindowStateError::incompatible_layout, "model, topology or native layout differs");
    if (!valid_cut(s.cut) || s.accepted_tokens.size() != s.cut.accepted ||
        s.pending_tokens.size() != s.cut.accepted - s.cut.forwarded)
        return fail(WindowStateError::invalid_cut, "invalid accepted/forwarded/exact/pending cut");
    if (s.cut.accepted > layout.max_tokens)
        return fail(WindowStateError::limit_exceeded, "token limit exceeded");
    if (std::any_of(s.accepted_tokens.begin(), s.accepted_tokens.end(), [](int32_t t) { return t < 0; }) ||
        !std::equal(s.pending_tokens.begin(), s.pending_tokens.end(),
                    s.accepted_tokens.begin() + static_cast<size_t>(s.cut.forwarded)))
        return fail(WindowStateError::invalid_cut, "pending tokens do not match accepted history");
    if (s.cut.history != window_history_digest(s.accepted_tokens))
        return fail(WindowStateError::invalid_cut, "accepted-history digest differs");
    if (s.continuation.mode != ContinuationMode::new_generation_policy ||
        !s.continuation.adapter.empty() || !s.continuation.payload.empty())
        return fail(WindowStateError::unsupported_continuation, "exact continuation adapter is unavailable");
    if (!s.speculation_disabled)
        return fail(WindowStateError::unsupported_speculation, "speculative state adapter is unavailable");
    uint64_t frontier = s.cut.exact_frontier;
    for (const auto& region : s.approximation) {
        if (region.begin != frontier || region.end <= region.begin || region.end > s.cut.forwarded ||
            !region.first_unresolved_unix_ms || !present(region.ancestor))
            return fail(WindowStateError::invalid_cut, "approximate suffix lacks contiguous age/ancestry");
        frontier = region.end;
    }
    if (frontier != s.cut.forwarded)
        return fail(WindowStateError::invalid_cut, "approximate frontier lacks provenance");
    if (layout.stages.empty() || layout.layers.empty() || s.devices.size() != layout.stages.size())
        return fail(WindowStateError::missing_state_part, "device stage is missing");
    std::set<uint32_t> devices;
    std::set<std::tuple<uint32_t, int64_t, StatePartKind>> seen;
    uint64_t bytes = 0;
    int64_t layer_end = 0;
    for (size_t i = 0; i < layout.stages.size(); ++i) {
        const auto& wanted = layout.stages[i];
        const auto& stage = s.devices[i];
        if (!devices.insert(wanted.id).second || wanted.layer_begin != layer_end ||
            wanted.layer_end <= wanted.layer_begin || wanted.tensors.empty())
            return fail(WindowStateError::incompatible_layout, "invalid expected stage partition");
        layer_end = wanted.layer_end;
        if (stage.id != wanted.id || stage.layer_begin != wanted.layer_begin || stage.layer_end != wanted.layer_end)
            return fail(WindowStateError::incompatible_layout, "device layer partition differs");
        if (stage.cut != s.cut)
            return fail(WindowStateError::invalid_cut, "device parts disagree on common cut");
        if (stage.tensors.size() != wanted.tensors.size())
            return fail(WindowStateError::missing_state_part, "native tensor inventory differs");
        for (size_t j = 0; j < wanted.tensors.size(); ++j) {
            const auto& spec = wanted.tensors[j];
            const auto& part = stage.tensors[j];
            if (!known_kind(spec.kind) || spec.device_part != stage.id || spec.layer < -1 ||
                (spec.layer >= 0 && (spec.layer < stage.layer_begin || spec.layer >= stage.layer_end)) ||
                spec.dtype.empty() || spec.native_layout.empty() || spec.shape.empty() ||
                spec.shape.size() != spec.strides.size() || !seen.insert(part_id(spec)).second)
                return fail(WindowStateError::incompatible_layout, "invalid or duplicate native tensor layout");
            if (part.layout != spec)
                return fail(WindowStateError::incompatible_layout, "tensor meaning or valid extent differs");
            if (part.payload.size() != spec.bytes)
                return fail(WindowStateError::missing_state_part, "native tensor payload length differs");
            if (bytes > layout.max_payload_bytes || spec.bytes > layout.max_payload_bytes - bytes)
                return fail(WindowStateError::limit_exceeded, "payload budget exceeded");
            bytes += spec.bytes;
        }
        if (static_cast<uint64_t>(wanted.layer_end) > layout.layers.size())
            return fail(WindowStateError::incompatible_layout, "stage exceeds model layers");
        auto has = [&](int64_t layer, StatePartKind kind) {
            return seen.count(std::make_tuple(stage.id, layer, kind)) != 0;
        };
        for (int64_t layer = wanted.layer_begin; layer < wanted.layer_end; ++layer) {
            switch (layout.layers[static_cast<size_t>(layer)]) {
            case WindowStateLayout::LayerKind::qsa:
                for (auto kind : {StatePartKind::qsa_k, StatePartKind::qsa_v, StatePartKind::qsa_pooled,
                                  StatePartKind::qsa_tail, StatePartKind::qsa_spare, StatePartKind::qsa_block_position})
                    if (!has(layer, kind))
                        return fail(WindowStateError::missing_state_part, "QSA state inventory incomplete");
                break;
            case WindowStateLayout::LayerKind::gdn:
                if (!has(layer, StatePartKind::gdn_recurrence_and_convolution))
                    return fail(WindowStateError::missing_state_part, "GDN state inventory incomplete");
                break;
            default:
                return fail(WindowStateError::incompatible_layout, "unknown model layer kind");
            }
        }
        if (wanted.ple)
            for (auto kind : {StatePartKind::ple_history, StatePartKind::ple_previous_tokens, StatePartKind::ple_current_token})
                if (!has(-1, kind))
                    return fail(WindowStateError::missing_state_part, "PLE state inventory incomplete");
    }
    if (static_cast<uint64_t>(layer_end) != layout.layers.size())
        return fail(WindowStateError::missing_state_part, "stage partition omits model layers");
    return {};
}

WindowStateResult BundleHandle::freeze(const SnapshotDescriptor& s, const WindowKey& expected,
                                      const WindowStateLayout& layout, BundleHandle& out,
                                      std::shared_ptr<const void> allocation_owner) {
    try {
        auto result = validate_window_state(s, expected, layout);
        if (!result) return result;
        // Copy even when the source is an rvalue: an old mutable pointer into a
        // moved buffer must never mutate a published immutable bundle.
        struct OwnedSnapshot {
            // Payload must be destroyed before its memory reservation.
            std::shared_ptr<const void> owner;
            SnapshotDescriptor snapshot;
        };
        auto owned = std::make_shared<const OwnedSnapshot>(OwnedSnapshot{std::move(allocation_owner), s});
        out.state_ = std::shared_ptr<const SnapshotDescriptor>(owned, &owned->snapshot);
    } catch (const std::bad_alloc&) {
        return fail(WindowStateError::limit_exceeded, "immutable bundle allocation failed");
    }
    return {};
}
} // namespace strata::core
