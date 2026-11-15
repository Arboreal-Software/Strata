// CPU gate for the suffix-reuse edit classifier (P12). It proves only classification: which token ranges may be
// considered for an APPROXIMATE suffix import, and when a request must take exact replay. It grants no reuse,
// touches no state, and qualifies nothing about the engine.
#include "strata/core/suffix_edit_span.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++failures; }
}
}   // namespace

int main() {
    // A(4) + B(4) + C(8): the bounded P12 case. C's ids are identical, and that is exactly why the state must
    // still be treated as approximate - it was computed under B.
    std::vector<int32_t> old_tokens, new_tokens;
    for (int32_t i = 0; i < 4; ++i) { old_tokens.push_back(100 + i); new_tokens.push_back(100 + i); }
    for (int32_t i = 0; i < 4; ++i) { old_tokens.push_back(200 + i); new_tokens.push_back(900 + i); }
    for (int32_t i = 0; i < 8; ++i) { old_tokens.push_back(300 + i); new_tokens.push_back(300 + i); }
    auto s = suffix_edit_span(old_tokens, new_tokens, 4);
    check(s.eligible, "aligned same-length edit with an unchanged suffix is eligible");
    check(s.prefix == 4 && s.suffix == 8, "prefix A=4, suffix C=8");
    check(s.edit_from == 4 && s.edit_to == 8, "changed run is exactly B -> B'");
    check(s.same_length && s.aligned && s.reason == nullptr, "shape flags");

    // B and B' share their first token: the maximal prefix runs past the page boundary, so the cut is not aligned.
    auto partial = new_tokens; partial[4] = 200;   // B' begins with B's first token
    auto p = suffix_edit_span(old_tokens, partial, 4);
    check(!p.eligible, "an edit whose changed run starts mid-page is refused rather than approximated anyway");
    check(p.edit_from == 5, "the maximal prefix, not the intended edit, decides the cut");

    // No edit at all: the exact path already handles this; an approximate import would be a downgrade.
    auto none = suffix_edit_span(old_tokens, old_tokens, 4);
    check(!none.eligible && none.prefix == 16, "identical prompts are refused as 'no edit'");

    // Suffix must exist at identical positions.
    std::vector<int32_t> tail_changed = new_tokens;
    tail_changed[15] = 999;
    auto t = suffix_edit_span(old_tokens, tail_changed, 4);
    check(!t.eligible && t.suffix == 0, "an edit reaching the end leaves no unchanged suffix");

    // Two separate edits: only the tail after the LAST difference is a suffix; the span between them is recomputed.
    std::vector<int32_t> two = new_tokens;
    two[6] = 777;                 // inside B' region already changed
    auto d = suffix_edit_span(old_tokens, two, 4);
    check(d.eligible && d.prefix == 4 && d.suffix == 8 && d.edit_to == 8,
          "second edit stays inside the changed run; suffix is still only C");
    std::vector<int32_t> three = new_tokens;
    three[10] = 555;              // an edit inside C
    auto e = suffix_edit_span(old_tokens, three, 4);
    check(!e.eligible || e.suffix < 8, "an edit inside C shrinks the reusable suffix (never keeps all of C)");
    check(e.suffix == 5, "suffix is only what survives past the last difference");

    // Unequal lengths are outside the bounded milestone.
    std::vector<int32_t> longer = new_tokens;
    longer.push_back(1);
    auto u = suffix_edit_span(old_tokens, longer, 4);
    check(!u.eligible && !u.same_length, "length change is refused (exact replay)");

    // A different page size changes the alignment contract; the classifier follows the caller's real shape.
    // Same edit, but under a 8-token page the cut at 4 is mid-page: it must be refused, not silently accepted.
    // Same edit, but under an 8-token page the cut at 4 is mid-page: it must be refused, not silently accepted.
    auto s8 = suffix_edit_span(old_tokens, new_tokens, 8);
    check(!s8.eligible && !s8.aligned && s8.prefix == 4 && s8.suffix == 8,
          "page-8 shape refuses a cut at 4 while still reporting the located spans");
    auto misaligned = suffix_edit_span(old_tokens, partial, 2);
    check(misaligned.aligned == (misaligned.edit_from % 2 == 0 && misaligned.edit_to % 2 == 0),
          "alignment is computed against the caller's page, not a hardcoded 4");

    if (failures) { std::printf("suffix edit span gate FAILED: %d checks\n", failures); return 1; }
    std::printf("suffix edit span gate PASS (classification only; no import, no certificate)\n");
    return 0;
}
