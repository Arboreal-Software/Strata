#pragma once
// P12 suffix reuse: the ADMISSION decision for one serving request.  Pure - it reads numbers and reports a
// verdict.  It grants no reuse, restores nothing, certifies nothing and touches no engine state.  The serving
// request path (src/program/generate.cpp) is the only caller, and the CPU gate
// (tests/core/suffix_import_policy) exercises THIS function, so what the test proves is what serving decides.
//
// What the feature is.  The session already holds the state of an older run  old = A + B + C.  The new request
// is  A + B' + C + D  with |B| == |B'|, C unchanged at IDENTICAL token positions, and D this request's new
// tokens.  Ordinary prefix reuse stops at A and re-reads B' + C.  The suffix cache additionally keeps the
// state for C - which was computed under B - and reads only D.  That state is therefore APPROXIMATE: equal
// token ids for C do not make its K/V or recurrent/indexer/PLE state exact under B'.  It can never certify an
// exact reuse, and the conversation it produces must park as approximate so ConversationCache::best() refuses
// it.  Everything outside the admitted shape takes EXACT replay, with the reason logged.
//
// Deliberately NOT here: any exactness shortcut, any relaxation of the resident INT8 shape, any reuse of an
// approximate entry as if it were exact, and any concurrency/batch/drafter/vision support.  Each of those is
// separate work with its own qualification.
#include "strata/core/suffix_edit_span.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::core {

/// The facts the caller measured about this process and request.  Every field is a refusal when false (except
/// `exact_reuse_available` and `donor_approximate`, which refuse when true), so the serving path cannot quietly
/// forget a precondition: a default-constructed shape refuses everything.
struct SuffixImportShape {
    bool enabled = false;               // started with --suffix-cache
    bool prompt_cache_on = false;       // the conversation checkpoint machinery is live
    bool state_resumable = false;       // the session holds a complete, untainted endpoint for stored_ids
    bool cvec_matches = false;          // the request's cross-turn setting equals the state's
    bool resident_int8 = false;         // owned, fully mapped, unrotated native INT8 pools: the qualified plan
    // The reused state must be reachable only through an OWNED private session (the serving process was started
    // with --owned-session-lifetime), because the attach/repair integration works on a private RAW transaction
    // over that allocation.  Without it there is no transaction owner, so the feature refuses rather than
    // approximating over a shared allocation it cannot fence.
    bool session_owned = false;
    bool single_session = false;        // one session: no batch slots, no concurrent windows
    bool speculation_off = false;       // no drafter / MTP / lookup chain: its ring is separate state
    bool images_absent = false;         // neither this request nor the stored state carries pictures
    bool exact_reuse_available = false; // an exact prefix reuse exists, and exact always wins
    bool donor_approximate = false;     // the stored state is itself an approximate import
    int64_t page_size = 0;              // the REAL cache page: strata::kernels::qsa_real_shapes().page_size
};

struct SuffixImportDecision {
    bool proceed = false;
    const char* reason = nullptr;       // a literal, never allocated: safe on the request path
    SuffixEditSpan span{};              // the located A / B->B' / C, in the stored run's coordinates
    int64_t frontier = 0;               // prompt tokens the imported state already covers (== |stored_ids|)
    int64_t fresh = 0;                  // prompt tokens this request still reads (>= 1 by construction)
    int64_t reused_suffix = 0;          // |C|: stored positions kept without reading them again
    int64_t exact_prefix = 0;           // |A|: what ordinary prefix reuse would have stopped at
    bool certifies_exact = false;       // ALWAYS false: this path never certifies exactness
    bool remains_approximate = true;    // what this request parks must carry approximate provenance
};

/// The two serving admission modes, defined ONCE (the request path calls these, the CPU gate tests these, so a
/// test cannot drift away from the server and the shortcut can never be switched on by the repaired path):
///  - the REPAIRED admission is the scoped feature: `--suffix-cache` alone, every other guard unchanged;
///  - the metadata SHORTCUT (keep all old state, relabel ids, read only D) additionally requires its explicit
///    experiment flag and stays measurement-only.
struct SuffixCacheFlags {
    bool suffix_cache = false;                 // --suffix-cache
    bool unrepaired_experiment = false;        // --suffix-cache-unrepaired-experiment
};
inline bool repaired_admission_enabled(SuffixCacheFlags flags) noexcept { return flags.suffix_cache; }
inline bool shortcut_admission_enabled(SuffixCacheFlags flags) noexcept {
    return flags.suffix_cache && flags.unrepaired_experiment;
}

/// Decide whether one request may continue approximately from the state the session already holds.
/// `stored_ids` are the tokens that state describes (its own ids, NOT the prompt's); `prompt_ids` is this
/// request's whole prompt, which must be strictly longer (the last prompt token always starts the next window).
template <class StoredTokens, class PromptTokens>
SuffixImportDecision suffix_import_plan(const StoredTokens& stored_ids, const PromptTokens& prompt_ids,
                                        const SuffixImportShape& shape) {
    SuffixImportDecision d;
    // Order is deliberate: process-level facts first (they are the cheapest and the ones an operator can fix),
    // then the state facts, then the request shape, then the token-level classification.
    if (!shape.enabled) { d.reason = "not enabled"; return d; }
    if (!shape.prompt_cache_on) { d.reason = "conversation checkpointing is off"; return d; }
    if (!shape.single_session) { d.reason = "batch slots or concurrent sessions are outside this milestone"; return d; }
    if (!shape.speculation_off) { d.reason = "speculative decoding keeps separate draft state"; return d; }
    if (!shape.images_absent) { d.reason = "pictures in the request or the stored state"; return d; }
    if (!shape.resident_int8) {
        d.reason = "K/V is not owned fully resident unrotated native INT8";   // the only qualified shape
        return d;
    }
    if (!shape.session_owned) {
        d.reason = "the serving session is not an owned private session (no RAW transaction owner)";
        return d;
    }
    if (!shape.state_resumable) { d.reason = "the stored state is not resumable"; return d; }
    if (!shape.cvec_matches) { d.reason = "this request changes the cross-turn setting"; return d; }
    if (shape.exact_reuse_available) { d.reason = "an exact prefix reuse is available"; return d; }
    if (shape.donor_approximate) {
        // one import per conversation: an approximate chain compounds divergence without an exact rebuild
        d.reason = "the stored state is already an approximate import";
        return d;
    }
    if (stored_ids.empty()) { d.reason = "the session holds no tokens"; return d; }
    if (shape.page_size <= 0) { d.reason = "invalid cache page size"; return d; }

    d.span = suffix_edit_span_grown(stored_ids, prompt_ids, shape.page_size);
    if (!d.span.eligible) { d.reason = d.span.reason; return d; }

    d.proceed = true;
    d.frontier = (int64_t)stored_ids.size();
    d.fresh = (int64_t)prompt_ids.size() - d.frontier;
    d.reused_suffix = d.span.suffix;
    d.exact_prefix = d.span.prefix;
    return d;
}

}   // namespace strata::core
