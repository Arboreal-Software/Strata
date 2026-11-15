#pragma once
// The SERVING RAW metadata contract, in one place: what a capture/restore descriptor must state to be accepted,
// and how the certificate for the state a SERVING SESSION ACTUALLY STANDS ON is derived from the request that
// produced it.
//
// These rules are exactly what `SuffixRawTransaction` enforces
// (`src/program/suffix_raw_runtime.cpp`: bounded extent :64-72, capture metadata :76-91, the re-check at :366-370).
// They are defined here ONCE and called by the transaction and by the serving path, so the CPU gate
// (`tests/core/serving_raw_contracts`) exercises the same predicate the server runs instead of a parallel
// test-only copy or a hand-copied simulation.
//
// Deliberate consequences, stated so nobody "fixes" them by weakening the RAW checks:
//  - an approximate state MUST describe its gap: contiguous regions from `cut.exact_frontier` to
//    `cut.forwarded`, each with a real age anchor and the digest of the state whose suffix was reused. A
//    descriptor that leaves the gap undeclared is refused, and so is one that relabels an approximate cut as
//    exact by pushing `exact_frontier` to `forwarded`;
//  - the cut's history digest is DERIVED from the accepted token run, never supplied: a call site cannot pair a
//    position with a token run it did not actually consume;
//  - the accepted run ends at the ONE pending token the device state was about to consume when one is certified,
//    exactly as the qualified donor capture requires.
#include "strata/core/serving_raw_contracts.hpp"
#include "strata/core/window_hash.hpp"
#include "strata/core/window_state.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace strata::core::serving_raw {

/// The extent bound the transaction applies before it will capture or restore anything.
inline bool bounded_capture_metadata(const SnapshotDescriptor& s, uint64_t max_tokens) noexcept {
    return s.key.session_id.size() <= 4096 && s.key.thread_id.size() <= 4096 &&
        s.key.window_id.size() <= 4096 && s.native_layout.size() <= 4096 &&
        s.accepted_tokens.size() <= max_tokens && s.pending_tokens.size() <= max_tokens &&
        s.approximation.size() <= 1 && s.continuation.adapter.empty() &&
        s.continuation.payload.empty() && s.speculation_disabled &&
        s.continuation.mode == ContinuationMode::new_generation_policy;
}

/// The capture-side metadata predicate: the accepted run and its digest, the pending tail, and contiguous
/// approximation coverage up to the forwarded position.
inline bool capture_metadata_acceptable(const SnapshotDescriptor& s) noexcept {
    if (!s.cut.id || s.cut.exact_frontier > s.cut.forwarded || s.cut.forwarded > s.cut.accepted ||
        s.accepted_tokens.size() != s.cut.accepted ||
        s.pending_tokens.size() != s.cut.accepted - s.cut.forwarded ||
        window_history_digest(s.accepted_tokens) != s.cut.history ||
        !std::equal(s.pending_tokens.begin(), s.pending_tokens.end(),
                    s.accepted_tokens.begin() + s.cut.forwarded) ||
        std::any_of(s.accepted_tokens.begin(), s.accepted_tokens.end(),
                    [](int32_t token) { return token < 0; }))
        return false;
    uint64_t frontier = s.cut.exact_frontier;
    for (const auto& region : s.approximation) {
        if (region.begin != frontier || region.end <= region.begin || region.end > s.cut.forwarded ||
            !region.first_unresolved_unix_ms || !digest_present(region.ancestor))
            return false;
        frontier = region.end;
    }
    return frontier == s.cut.forwarded;
}

/// The honest provenance of an unresolved approximation carried by a live session state: the real wall-clock
/// anchor of its first unresolved observation, and the digest of the state whose suffix was reused.
struct ApproximationProvenance {
    uint64_t first_unresolved_unix_ms = 0;
    StateDigest ancestor{};
};

/// What one serving request actually left on the device, as its own completion point observes it.
/// `consumed` is the token run whose rows the session actually forwarded; `pending` is the genuine pending
/// token the state was about to consume, negative when nothing certifies one. `state_exact` is provenance, not
/// a wish: it is true only when the request started from a zeroed session, or from a state already certified
/// exact, and read every further row itself.
struct TrackedStateFacts {
    uint64_t cut_id = 0;
    std::span<const int32_t> consumed{};
    int32_t pending = -1;
    bool state_exact = false;
    uint64_t inherited_frontier = 0;
    ApproximationProvenance approximation{};
    /// True only for a session this request explicitly reset to zero: it stands on NO conversation run, which is
    /// a truthful position to certify (and one no retained activation may use).  It is never a way to certify a
    /// state whose history the request cannot name.
    bool certify_empty = false;
};

enum class TrackedStateRefusal : uint8_t {
    none, no_cut_id, empty_history, negative_token, history_past_capacity,
    unresolvable_approximation, approximation_unanchored, approximation_ancestor_missing,
    empty_needs_exactness
};

inline const char* tracked_state_reason(TrackedStateRefusal why) noexcept {
    switch (why) {
        case TrackedStateRefusal::none: return "certified";
        case TrackedStateRefusal::no_cut_id: return "no cut identity for this completion";
        case TrackedStateRefusal::empty_history: return "the session consumed no history to certify";
        case TrackedStateRefusal::negative_token: return "the consumed run carries a negative token";
        case TrackedStateRefusal::history_past_capacity: return "the accepted run exceeds the session capacity";
        case TrackedStateRefusal::unresolvable_approximation:
            return "an inexact state cannot describe its gap at this cut";
        case TrackedStateRefusal::approximation_unanchored:
            return "the inherited approximation has no real age anchor";
        case TrackedStateRefusal::approximation_ancestor_missing:
            return "the inherited approximation names no ancestor state";
        case TrackedStateRefusal::empty_needs_exactness:
            return "only a reset session can certify that it stands on no conversation run";
    }
    return "unknown";
}

/// The certificate of the state a request produced: the cut to drain at, the accepted run whose digest that cut
/// records, and the approximation regions (empty for an exact state) that describe it honestly.
struct TrackedState {
    StateCut cut{};
    std::vector<int32_t> accepted{};
    std::vector<ApproximationRegion> approximation{};

    bool valid() const noexcept { return cut.id != 0 && !accepted.empty(); }
    void clear() noexcept { *this = TrackedState{}; }
};

/// Derives the certificate from the actual facts of one completion.  Nothing here invents a position, a digest,
/// an age anchor or an ancestor: an inexact state with no inherited provenance is REFUSED, and the caller must
/// then leave the session untracked rather than certify it.
inline TrackedStateRefusal track_state_certificate(const TrackedStateFacts& facts, uint64_t max_tokens,
                                                   TrackedState& out) {
    out.clear();
    if (!facts.cut_id) return TrackedStateRefusal::no_cut_id;
    if (facts.certify_empty && !facts.state_exact) return TrackedStateRefusal::empty_needs_exactness;
    if (facts.consumed.empty() && !facts.certify_empty) return TrackedStateRefusal::empty_history;
    if (std::any_of(facts.consumed.begin(), facts.consumed.end(),
                    [](int32_t token) { return token < 0; }))
        return TrackedStateRefusal::negative_token;
    const uint64_t forwarded = facts.certify_empty ? 0 : facts.consumed.size();
    const uint64_t accepted_count = forwarded + (facts.pending >= 0 ? 1 : 0);
    if (accepted_count > max_tokens) return TrackedStateRefusal::history_past_capacity;
    std::vector<ApproximationRegion> regions;
    uint64_t exact_frontier = forwarded;
    if (!facts.state_exact) {
        // The gap the state carries runs from the frontier it INHERITED to the position it now stands on. It can
        // only be described with the anchor and ancestor that were recorded when the approximation was
        // introduced; a frontier already at (or past) the current position would claim the gap away.
        if (facts.inherited_frontier >= forwarded) return TrackedStateRefusal::unresolvable_approximation;
        if (!facts.approximation.first_unresolved_unix_ms)
            return TrackedStateRefusal::approximation_unanchored;
        if (!digest_present(facts.approximation.ancestor))
            return TrackedStateRefusal::approximation_ancestor_missing;
        exact_frontier = facts.inherited_frontier;
        regions.push_back(ApproximationRegion{exact_frontier, forwarded,
                                              facts.approximation.first_unresolved_unix_ms,
                                              facts.approximation.ancestor});
    }
    TrackedState state;
    state.accepted.assign(facts.consumed.begin(), facts.consumed.end());
    if (facts.pending >= 0) state.accepted.push_back(facts.pending);
    state.cut = StateCut{facts.cut_id, accepted_count, forwarded, exact_frontier, 0,
                         window_history_digest(state.accepted)};
    state.approximation = std::move(regions);
    out = std::move(state);
    return TrackedStateRefusal::none;
}

/// The retained-activation guard's own predicate: the accepted run on record must name the run the LIVE
/// transaction's current cut certifies.  `recorded` is what the tracking site drained, `live` is what the
/// transaction reports now; a stale certificate (an earlier conversation's cut kept across another request's
/// mutation) cannot match, and neither can an accepted run that was not the one that produced the state.
inline bool accepted_run_certifies_cut(const StateCut& recorded, const StateCut& live,
                                       const std::vector<int32_t>& accepted) noexcept {
    return recorded == live && !accepted.empty() && window_history_digest(accepted) == live.history;
}

/// The capture descriptor for a tracked state: the same shape the qualified driver builds (accepted is the whole
/// run, pending is the part past the forwarded position), plus whatever approximation coverage the certificate
/// carries.  The key comes from the owning session's own binding, never from imported metadata.
inline SnapshotDescriptor tracked_metadata(const WindowKey& key, const TrackedState& state) {
    SnapshotDescriptor value;
    value.key = key;
    value.cut = state.cut;
    value.accepted_tokens = state.accepted;
    if (state.cut.forwarded <= state.accepted.size())
        value.pending_tokens.assign(state.accepted.begin() + static_cast<long long>(state.cut.forwarded),
                                    state.accepted.end());
    value.approximation = state.approximation;
    value.steering_enabled = false;
    value.speculation_disabled = true;
    return value;
}

}   // namespace strata::core::serving_raw
