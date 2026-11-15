// CPU gate for the P12 serving suffix-reuse ADMISSION decision (include/strata/core/suffix_import_policy.hpp).
// It proves only the verdict: which requests may continue from an APPROXIMATE imported state, and which must take
// exact replay (with the reason).  It touches no engine state, restores nothing, certifies nothing, and needs no
// GPU, model or runtime - the serving path calls this same function.
#include "strata/core/suffix_import_policy.hpp"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::core;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++failures; }
}

// stored = A(4) + B(4) + C(8); prompt = A(4) + B'(4) + C(8) + D(4).  The serving shape: the session already
// holds A+B+C, the client re-sent it with one edited run and then asked a new question (D).
std::vector<int32_t> stored_run() {
    std::vector<int32_t> v;
    for (int32_t i = 0; i < 4; ++i) v.push_back(100 + i);       // A
    for (int32_t i = 0; i < 4; ++i) v.push_back(200 + i);       // B
    for (int32_t i = 0; i < 8; ++i) v.push_back(300 + i);       // C
    return v;
}
std::vector<int32_t> edited_prompt() {
    std::vector<int32_t> v;
    for (int32_t i = 0; i < 4; ++i) v.push_back(100 + i);       // A again
    for (int32_t i = 0; i < 4; ++i) v.push_back(900 + i);       // B'
    for (int32_t i = 0; i < 8; ++i) v.push_back(300 + i);       // C, identical positions
    for (int32_t i = 0; i < 4; ++i) v.push_back(400 + i);       // D: this request's new tokens
    return v;
}
SuffixImportShape admitted_shape() {
    SuffixImportShape s;
    s.enabled = s.prompt_cache_on = s.state_resumable = s.cvec_matches = true;
    s.resident_int8 = s.single_session = s.speculation_off = s.images_absent = s.session_owned = true;
    s.page_size = 4;
    return s;
}
}   // namespace

int main() {
    const auto stored = stored_run(), prompt = edited_prompt();

    // 1. the admitted case, and what it reports
    auto ok = suffix_import_plan(stored, prompt, admitted_shape());
    check(ok.proceed && ok.reason == nullptr, "the bounded serving shape is admitted");
    check(ok.frontier == 16 && ok.fresh == 4, "imported state covers 16 stored tokens; 4 prompt tokens are read");
    check(ok.reused_suffix == 8 && ok.exact_prefix == 4, "8 suffix tokens are reused that prefix reuse (only |A|=4) would re-read");
    check(ok.span.edit_from == 4 && ok.span.edit_to == 8, "the changed run B -> B' is located at 4..8");
    check(!ok.certifies_exact, "an approximate import never certifies exactness");
    check(ok.remains_approximate, "what this request produces must park as approximate");
    check(ok.span.prefix == 4 && ok.span.suffix == 8, "the span is reported in the stored run's coordinates");

    // 2. every unsupported fact refuses on its own, even with a perfect token shape
    struct Flip { const char* what; void (*mutate)(SuffixImportShape&); };
    const Flip flips[] = {
        {"enabled", [](SuffixImportShape& s) { s.enabled = false; }},
        {"prompt_cache_on", [](SuffixImportShape& s) { s.prompt_cache_on = false; }},
        {"state_resumable", [](SuffixImportShape& s) { s.state_resumable = false; }},
        {"cvec_matches", [](SuffixImportShape& s) { s.cvec_matches = false; }},
        {"resident_int8", [](SuffixImportShape& s) { s.resident_int8 = false; }},
        {"session_owned (the owned private session / RAW transaction owner)",
         [](SuffixImportShape& s) { s.session_owned = false; }},
        {"single_session", [](SuffixImportShape& s) { s.single_session = false; }},
        {"speculation_off", [](SuffixImportShape& s) { s.speculation_off = false; }},
        {"images_absent", [](SuffixImportShape& s) { s.images_absent = false; }},
        {"exact_reuse_available (refuses when TRUE)", [](SuffixImportShape& s) { s.exact_reuse_available = true; }},
        {"donor_approximate (refuses when TRUE)", [](SuffixImportShape& s) { s.donor_approximate = true; }},
    };
    for (const auto& f : flips) {
        auto sh = admitted_shape();
        f.mutate(sh);
        auto d = suffix_import_plan(stored, prompt, sh);
        check(!d.proceed && d.reason != nullptr, std::string("refuses when ") + f.what);
        check(d.frontier == 0 && d.fresh == 0, std::string("a refusal grants no frontier: ") + f.what);
    }
    // a default-constructed shape must refuse, so a caller that forgets a fact cannot import
    check(!suffix_import_plan(stored, prompt, SuffixImportShape{}).proceed, "a default shape refuses everything");

    // 3. exact always wins: an available exact reuse is a refusal, not a preference
    {
        auto sh = admitted_shape();
        sh.exact_reuse_available = true;
        check(suffix_import_plan(stored, prompt, sh).reason ==
                  std::string("an exact prefix reuse is available"),
              "the exact-reuse reason is the literal the log prints");
    }
    // 4. no approximate chains: one import per conversation
    {
        auto sh = admitted_shape();
        sh.donor_approximate = true;
        auto d = suffix_import_plan(stored, prompt, sh);
        check(!d.proceed && d.reason == std::string("the stored state is already an approximate import"),
              "an approximate donor is refused (bounded divergence)");
    }

    // 5. the request must have something to read: the last prompt token always starts the next window
    {
        auto same = stored;                      // same length as the stored run
        auto d = suffix_import_plan(stored, same, admitted_shape());
        check(!d.proceed && d.reason == std::string("no prompt tokens after the stored run"),
              "a same-length prompt is not a serving request here");
        auto shorter = stored; shorter.pop_back();
        check(!suffix_import_plan(stored, shorter, admitted_shape()).proceed, "a shorter prompt is refused");
    }

    // 6. the token-level refusals still apply through the grown classifier
    {
        auto mid = prompt; mid[4] = 200;         // B' begins with B's first token: maximal prefix runs mid-page
        auto d = suffix_import_plan(stored, mid, admitted_shape());
        check(!d.proceed && d.reason == std::string("edit cuts are not page aligned"),
              "a mid-page cut is refused rather than approximated anyway");
        check(d.span.edit_from == 5, "the maximal prefix, not the intended edit, decides the cut");

        auto tail = prompt; tail[15] = 999;      // an edit reaching the end of the stored run: no unchanged C
        auto t = suffix_import_plan(stored, tail, admitted_shape());
        check(!t.proceed && t.reason == std::string("no unchanged suffix at identical positions"),
              "an edit that reaches the stored end leaves no suffix");

        auto inside = prompt; inside[11] = 555;  // an edit at the end of C's first page: the cut lands on 12
        auto e = suffix_import_plan(stored, inside, admitted_shape());
        check(e.proceed && e.reused_suffix == 4, "an edit inside C shrinks the reused suffix (never keeps all of C)");
        check(e.span.edit_to == 12 && e.span.edit_from == 4, "the changed run extends to the LAST difference");
        auto inside2 = prompt; inside2[10] = 555;    // the same kind of edit, but mid-page
        auto e2 = suffix_import_plan(stored, inside2, admitted_shape());
        check(!e2.proceed && e2.reason == std::string("edit cuts are not page aligned"),
              "a mid-page edit inside C is refused, not half-reused");

        auto noedit = prompt;                    // stored is an exact prefix of the prompt: ordinary prefix reuse
        noedit[4] = 200; noedit[5] = 201; noedit[6] = 202; noedit[7] = 203;
        auto ne = suffix_import_plan(stored, noedit, admitted_shape());
        check(!ne.proceed && ne.reason == std::string("no edit"),
              "an unchanged stored run is ordinary prefix reuse, not suffix reuse");

        auto p8 = admitted_shape(); p8.page_size = 8;
        auto s8 = suffix_import_plan(stored, prompt, p8);
        check(!s8.proceed && !s8.span.aligned && s8.span.prefix == 4,
              "an 8-token page refuses a cut at 4 instead of accepting it");

        auto bad_page = admitted_shape(); bad_page.page_size = 0;
        check(!suffix_import_plan(stored, prompt, bad_page).proceed, "a zero page size is refused");
        check(!suffix_import_plan(std::vector<int32_t>{}, prompt, admitted_shape()).proceed,
              "an empty stored run is refused");
    }

    // 7. the strict same-length primitive was NOT loosened by sharing the classifier
    {
        std::vector<int32_t> same_stored = stored, same_new = edited_prompt();
        same_new.resize(same_new.size() - 4);          // drop D: exactly the 0550Z reporting shape
        auto strict = suffix_edit_span(same_stored, same_new, 4);
        check(strict.eligible && strict.same_length, "the same-length classifier still works");
        auto grown = suffix_edit_span(same_stored, edited_prompt(), 4);
        check(!grown.eligible && !grown.same_length, "the same-length classifier still refuses a grown prompt");
        check(suffix_edit_span_grown(same_stored, edited_prompt(), 4).eligible,
              "the grown classifier accepts what the serving path sees");
        check(!suffix_edit_span_grown(same_stored, same_new, 4).eligible,
              "the grown classifier refuses what is not grown");
    }

    // 7. The two serving admission MODES, decided by the same helpers the request path calls: the repaired path is
    //    what --suffix-cache means, and the metadata shortcut additionally needs its experiment flag.  A repaired
    //    admission must produce a real, non-empty span, and every guard still refuses.
    {
        check(repaired_admission_enabled(SuffixCacheFlags{true, false}),
              "--suffix-cache alone admits the REPAIRED decision (the scoped feature)");
        check(!shortcut_admission_enabled(SuffixCacheFlags{true, false}),
              "the repaired mode can never activate the metadata shortcut");
        check(shortcut_admission_enabled(SuffixCacheFlags{true, true}),
              "the shortcut exists only behind its explicit experiment flag");
        check(!repaired_admission_enabled(SuffixCacheFlags{false, false}),
              "without the feature flag nothing is admitted at all");

        SuffixImportShape repaired = admitted_shape();
        repaired.enabled = repaired_admission_enabled(SuffixCacheFlags{true, false});
        const auto d = suffix_import_plan(stored, prompt, repaired);
        check(d.proceed && d.span.eligible && d.span.edit_from < d.span.edit_to && d.span.suffix > 0,
              "supported repaired admission obtains a non-empty eligible span");
        check(d.span.prefix == 4 && d.span.edit_from == 4 && d.span.edit_to == 8 && d.reused_suffix == 8,
              "the located A / B->B' / C boundaries are the request's own positions");
        check(!d.certifies_exact && d.remains_approximate,
              "admission alone never certifies exactness: repair decides that later");

        SuffixImportShape unflagged = admitted_shape();
        unflagged.enabled = repaired_admission_enabled(SuffixCacheFlags{false, false});
        const auto refused = suffix_import_plan(stored, prompt, unflagged);
        check(!refused.proceed && refused.reason != nullptr && std::string_view(refused.reason) == "not enabled",
              "the unflagged baseline refuses before any fence could be opened");

        const char* guards[] = {"prompt_cache_on", "state_resumable", "cvec_matches", "resident_int8",
                                "single_session", "speculation_off", "images_absent", "session_owned"};
        auto clear = [](SuffixImportShape& sh, const char* name) {
            if (!std::string_view(name).compare("prompt_cache_on")) sh.prompt_cache_on = false;
            else if (!std::string_view(name).compare("state_resumable")) sh.state_resumable = false;
            else if (!std::string_view(name).compare("cvec_matches")) sh.cvec_matches = false;
            else if (!std::string_view(name).compare("resident_int8")) sh.resident_int8 = false;
            else if (!std::string_view(name).compare("single_session")) sh.single_session = false;
            else if (!std::string_view(name).compare("speculation_off")) sh.speculation_off = false;
            else if (!std::string_view(name).compare("images_absent")) sh.images_absent = false;
            else sh.session_owned = false;
        };
        for (const char* guard : guards) {
            SuffixImportShape sh = admitted_shape();
            sh.enabled = repaired_admission_enabled(SuffixCacheFlags{true, false});
            clear(sh, guard);
            const auto one = suffix_import_plan(stored, prompt, sh);
            check(!one.proceed && one.reason != nullptr,
                  std::string("repaired mode refuses without ") + guard);
        }
        SuffixImportShape exact_wins = admitted_shape();
        exact_wins.enabled = repaired_admission_enabled(SuffixCacheFlags{true, false});
        exact_wins.exact_reuse_available = true;
        check(!suffix_import_plan(stored, prompt, exact_wins).proceed,
              "an available exact reuse always wins over the repaired path");
        SuffixImportShape chained = admitted_shape();
        chained.enabled = repaired_admission_enabled(SuffixCacheFlags{true, false});
        chained.donor_approximate = true;
        check(!suffix_import_plan(stored, prompt, chained).proceed,
              "an approximate donor is never imported again (no chaining)");
    }

    if (failures) { std::printf("suffix import policy gate FAILED: %d checks\n", failures); return 1; }
    std::printf("suffix import policy gate PASS (admission only: no import, no restore, no certificate)\n");
    return 0;
}
