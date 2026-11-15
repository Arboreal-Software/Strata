#pragma once
// P12 suffix-reuse primitive: locate the edit between an old token run and a new one, and say what may be reused.
//
// The feature is NOT prefix reuse. With old = A + B + C and new = A + B' + C (|B| == |B'|, C at the SAME token
// positions), ordinary prefix caching can only save A. The suffix-cache milestone additionally imports eligible
// state for C, which was computed under B - so C's state is APPROXIMATE until the repair/exactness protocol says
// otherwise. Equal token ids for C do not make its cached KV or recurrent state exact under B'.
//
// This header only classifies the request: it grants nothing, restores nothing, and certifies nothing. The caller
// must still (a) obtain complete old endpoint state, (b) import through the private RAW/repair machinery,
// (c) label the result approximate, and (d) fall back to exact replay whenever eligible == false.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

struct SuffixEditSpan {
    // True when the bounded P12 case applies: equal lengths, exactly one changed run, an unambiguous unchanged
    // suffix at identical positions, and both cut points four-token aligned. Anything else must take exact replay.
    bool eligible = false;
    // Why not eligible (nullptr when eligible). Kept as literals so the serving report cannot allocate on a hot path.
    const char* reason = nullptr;
    int64_t prefix = 0;      // |A|: identical tokens before the changed run
    int64_t edit_from = 0;   // first differing position
    int64_t edit_to = 0;     // one past the last differing position (|B'| == |B| so this equals old edit end)
    int64_t suffix = 0;      // |C|: positions >= edit_to that are identical in both runs
    int64_t length = 0;      // |old| == |new| when the bounded case applies
    bool same_length = false;
    bool aligned = false;    // both edit boundaries are on four-token page boundaries
};

namespace suffix_edit_detail {
/// A read-only view of the first `n` elements of a longer run, so one classifier serves both the same-length
/// edit and the serving case (the stored conversation followed by this request's new turn).
template <class Tokens>
struct Prefix {
    const Tokens& v;
    int64_t n;
    size_t size() const { return (size_t)n; }
    decltype(v[0]) operator[](size_t i) const { return v[i]; }
};

/// The shared classification. Requires a.size() == b.size() (callers slice), and fills everything except
/// `same_length`. Maximality of the common prefix/suffix is what makes the changed run unambiguous: any
/// alternative split would have to place a differing token inside a region that matches at every offset tested.
template <class A, class B>
void classify(const A& old_ids, const B& new_ids, int64_t page, SuffixEditSpan& s) {
    s.length = (int64_t)old_ids.size();
    if (page <= 0) { s.reason = "invalid cache page size"; return; }
    const int64_t n = (int64_t)old_ids.size();
    int64_t l = 0;
    while (l < n && old_ids[(size_t)l] == new_ids[(size_t)l]) ++l;             // |A| plus any shared head of B
    if (l == n) { s.prefix = n; s.suffix = 0; s.reason = "no edit"; return; }
    int64_t t = 0;
    while (t < n - l && old_ids[(size_t)(n - 1 - t)] == new_ids[(size_t)(n - 1 - t)]) ++t;   // |C|
    s.prefix = l;
    s.suffix = t;
    s.edit_from = l;
    s.edit_to = n - t;
    if (s.edit_to <= s.edit_from) { s.reason = "empty changed run"; return; }
    s.aligned = (s.edit_from % page) == 0 && (s.edit_to % page) == 0;
    if (!s.aligned) { s.reason = "edit cuts are not page aligned"; return; }
    if (s.suffix == 0) { s.reason = "no unchanged suffix at identical positions"; return; }
    s.eligible = true;
}
}   // namespace suffix_edit_detail

/// Classify old vs new. `page` is the cache page size (4 for the qualified native INT8 window).  The two runs may
/// use different token types (a stored conversation's int32 ids against a request's int64 ids).
template <class A, class B>
SuffixEditSpan suffix_edit_span(const A& old_ids, const B& new_ids, int64_t page = 4) {
    SuffixEditSpan s;
    s.same_length = old_ids.size() == new_ids.size();
    if (!s.same_length) {
        s.length = (int64_t)old_ids.size();
        s.reason = "lengths differ (bounded same-length edit only)";
        return s;
    }
    suffix_edit_detail::classify(old_ids, new_ids, page, s);
    return s;
}

/// The SHAPE THE SERVING PATH ACTUALLY SEES: the stored run `old` (the conversation the session holds, whose
/// state is already on the device) followed by a NEW request, so `new_ids` is longer.  The stored run is
/// classified against the prompt's first |old| tokens exactly as above; the extra tokens after it are the fresh
/// prompt this request must still read, and they are what makes the last-prompt-token invariant hold.
/// An empty tail (new.size() <= old.size()) is refused here: serving must forward at least the last prompt
/// token, so a run with nothing to read is not a serving request.  This grants nothing: it locates the edit.
template <class StoredTokens, class PromptTokens>
SuffixEditSpan suffix_edit_span_grown(const StoredTokens& stored_ids, const PromptTokens& prompt_ids,
                                      int64_t page = 4) {
    SuffixEditSpan s;
    s.same_length = false;
    s.length = (int64_t)stored_ids.size();
    if (prompt_ids.size() <= stored_ids.size()) {
        s.reason = "no prompt tokens after the stored run";
        return s;
    }
    suffix_edit_detail::classify(stored_ids, suffix_edit_detail::Prefix<PromptTokens>{prompt_ids, (int64_t)stored_ids.size()},
                                 page, s);
    return s;
}

}   // namespace strata::core
