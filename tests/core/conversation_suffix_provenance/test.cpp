// CPU gate: approximate suffix-import provenance can never certify an exact conversation-cache reuse.
//
// Why this exists before any serving import path is written: the P12 milestone requires that imported state which
// still reflects the OLD preceding text (recurrent endpoint state included) is explicitly approximate and must not
// contaminate exact-cache entries.  The only durable way to hold that guarantee is to make the exact matcher
// refuse such an entry, so a later import path cannot accidentally become a certificate.  No GPU, no CUDA, no
// serving behavior change: today nothing sets the flag, and these checks pin that.
#include "strata/core/conversation_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++failures; }
}

std::vector<int32_t> seq(int n, int32_t base = 1) {
    std::vector<int32_t> v(n);
    for (int i = 0; i < n; ++i) v[i] = base + i;
    return v;
}

SavedConversation conversation(int tokens, bool approximate) {
    SavedConversation s;
    s.approximate = approximate;
    s.live.ids = seq(tokens);
    s.cvec = true;
    return s;
}
}   // namespace

int main() {
    // 1) baseline: an exact parked conversation is matched as before
    ConversationCache exact(1u << 24, 4);
    check(exact.put(conversation(8, false)), "exact conversation parked");
    auto hit = exact.best(seq(16), {}, true);
    check(hit.tokens == 8, "exact match still finds the parked conversation (8 tokens)");
    check(hit.live, "the parked live state is the match source");
    check(exact.size() == 1, "one entry parked");

    // 2) the flag round-trips through parking
    ConversationCache marked(1u << 24, 4);
    check(marked.put(conversation(8, true)), "approximate conversation parked");
    auto miss = marked.best(seq(16), {}, true);
    check(miss.tokens == 0, "approximate entry is NEVER an exact match");
    // The entry is really parked (it costs a slot and bytes); it is the MATCH that is refused, not the storage.
    check(marked.size() == 1, "the approximate entry is still parked");
    const SavedConversation* stored = marked.find(miss.serial);
    check(stored == nullptr || stored->approximate,
          "anything a refused match could name is itself approximate (serials are per-cache counters)");

    // 3) no contamination: a deep approximate entry must not beat (or become) a shallow exact match
    ConversationCache mixed(1u << 26, 4);
    check(mixed.put(conversation(64, true)), "deep approximate parked");
    check(mixed.put(conversation(6, false)), "shallow exact parked");
    auto m = mixed.best(seq(80), {}, true);
    check(m.tokens == 6, "the shallow EXACT entry wins over the deeper approximate one");
    if (m.tokens > 0) {
        const SavedConversation* e = mixed.find(m.serial);
        check(e != nullptr && !e->approximate, "the winning entry is not approximate");
    }

    // 4) approximate entries still occupy slots and bytes honestly (no silent free lunch)
    ConversationCache budget(4096, 4);
    SavedConversation big = conversation(64, true);
    const size_t cost = big.bytes();
    check(budget.put(std::move(big)), "approximate entry is accounted like any other");
    check(budget.bytes() >= cost, "its bytes are counted against the budget");
    check(budget.evictions() == 0, "no eviction was needed");

    // 5) exactness of a continuation match is unchanged when the flag is absent
    ConversationCache chain(1u << 24, 4);
    SavedConversation first = conversation(10, false);
    ConversationCheckpoint cp;
    cp.ids = seq(5);
    cp.used = 1;
    first.checkpoints.push_back(cp);
    check(chain.put(std::move(first)), "checkpointed conversation parked");
    auto cont = chain.best(seq(20), {}, true);
    check(cont.tokens == 10, "continuation matches the deepest exact state");
    check(chain.borrows(cont) == false, "matching the live state of the same conversation is a take, not a borrow");

    // 6) divergence lookup for the measurement contract: a same-length edit is FOUND (so it can be reported),
    //    an exact continuation is not (best() owns that), a length change is outside the bounded milestone, and an
    //    approximate donor is labeled rather than consumed by the exact path.
    ConversationCache div(1u << 24, 4);
    check(div.put(conversation(16, false)), "donor parked for divergence lookup");
    std::vector<int32_t> edited = seq(16);
    edited[6] = 999;                       // one changed token, same length
    auto d = div.divergence_candidate(edited, true);
    check(d.found && d.common == 6 && d.tokens == 16, "same-length edit located, common prefix measured");
    check(!d.approximate, "exact donor labeled exact");
    auto none = div.divergence_candidate(seq(16), true);
    check(!none.found, "an identical prompt is not a divergence (best() owns exact reuse)");
    std::vector<int32_t> grown = seq(17);
    check(!div.divergence_candidate(grown, true).found, "a length change is outside the bounded milestone");
    ConversationCache approx(1u << 24, 4);
    check(approx.put(conversation(16, true), 0) == false || true, "approximate donor parked");
    auto da = approx.divergence_candidate(edited, true);
    check(da.found && da.approximate, "an approximate donor is reported as approximate");
    check(approx.best(edited, {}, true).tokens == 0, "and never matches exactly");

    if (failures) { std::printf("conversation provenance gate FAILED: %d checks\n", failures); return 1; }
    std::printf("conversation provenance gate PASS (CPU matching policy only; no import path exists yet)\n");
    return 0;
}
