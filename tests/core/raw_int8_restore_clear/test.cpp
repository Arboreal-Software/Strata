// CPU gate for the repeated-restore future-page defect.
//
// What is production code here: raw_int8_restore_clear_plan(), raw_int8_restore_clear_preflight() and
// apply_raw_int8_restore_clear_plan() - the exact planner, the exact all-stage preflight ordering and the exact
// dispatch closure the CUDA restore calls.  Only the executor is a host seam (cudaMemsetAsync -> host write), as
// it must be for a CPU test.
//
// What is NOT proven here, and stays with the GPU run: that cudaMemsetAsync lands on the right device/stream,
// that the checked stream drain orders before the restore copy on a real 3090, and that the two-cycle native
// 136/137/138 comparison now passes on live state.  Nothing in this file makes that claim.
//
// The layout math is written from layer.hpp's documented [page][kv_head][page_size][head_dim] pool and from
// conversation_snapshot.cpp's serialized extents, independently of the planner, so an off-by-one in either side
// shows up as a mismatch rather than agreeing with itself.
#include "strata/core/raw_int8_restore_clear.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++failures; }
}

constexpr int64_t kPage = 4, kHeads = 2, kDim = 256, kGroup = 64;
constexpr uint64_t kCodeRow = uint64_t(kHeads) * kDim;              // 512 bytes per cell, per array
constexpr uint64_t kScaleRow = uint64_t(kHeads) * (kDim / kGroup) * 2;   // 16 bytes per cell, per scale array

// The documented blocked layout, derived here and not from the planner.
size_t row_of(int64_t pos, int64_t head) {
    return size_t((pos / kPage * kHeads + head) * kPage + pos % kPage);
}
size_t code_byte(int64_t pos, int64_t head) { return row_of(pos, head) * size_t(kDim); }
size_t scale_unit(int64_t pos, int64_t head) { return row_of(pos, head) * size_t(kDim / kGroup); }

int64_t pages_for(int64_t cells) { return (cells + kPage - 1) / kPage; }

struct Pool {   // one QSA layer's INT8 KV arrays, host-allocated at the production capacity
    int64_t max_cells;
    std::vector<int8_t> k, v;
    std::vector<uint16_t> ks, vs;
    std::vector<int32_t> page_table;
    QsaState st;

    explicit Pool(int64_t cells) : max_cells(cells) {
        const uint64_t rows = uint64_t(pages_for(cells)) * kHeads * kPage;    // layer.cpp: slots*n_head_kv*page
        k.assign(size_t(rows * kDim), 0);
        v.assign(size_t(rows * kDim), 0);
        ks.assign(size_t(rows * (kDim / kGroup)), 0);
        vs.assign(size_t(rows * (kDim / kGroup)), 0);
        page_table.assign(size_t(pages_for(cells)), 0);
        st.kv_int8 = true; st.kv_q4 = false; st.kv_hybrid = false; st.kv_rot = false;
        st.kv_mode = 0; st.kv_elastic = -1; st.owned_elastic = nullptr;
        st.max_cells = cells; st.n_pages = pages_for(cells); st.n_slots = st.n_pages;
        st.k_q = k.data(); st.v_q = v.data(); st.k_scale = ks.data(); st.v_scale = vs.data();
        st.page_table = page_table.data();
    }
    size_t code_bytes() const { return k.size(); }
    size_t scale_bytes() const { return ks.size(); }
};

ModelGeometry geometry() {
    ModelGeometry g;
    g.n_head_kv = kHeads;
    g.head_dim = kDim;
    return g;
}

// model of kv_append writing one position's rows, both heads, codes and scales
void write_position(Pool& p, int64_t pos, int seed) {
    for (int64_t h = 0; h < kHeads; ++h) {
        std::memset(p.k.data() + code_byte(pos, h), uint8_t(1 + seed * 7), size_t(kDim));
        std::memset(p.v.data() + code_byte(pos, h), uint8_t(2 + seed * 11), size_t(kDim));
        for (int64_t c = 0; c < kDim / kGroup; ++c) {
            p.ks[scale_unit(pos, h) + size_t(c)] = uint16_t(300 + seed);
            p.vs[scale_unit(pos, h) + size_t(c)] = uint16_t(400 + seed);
        }
    }
}

struct Serialized {   // a snapshot image's K/V part, sized by conversation_snapshot.cpp's extents
    int64_t cells = 0;
    std::vector<uint8_t> k, v, ks, vs;
};

Serialized capture(const Pool& p, int64_t upto) {
    Serialized s;
    s.cells = (upto + kPage - 1) / kPage * kPage;                  // conversation_snapshot.cpp layout().cells
    const size_t data = size_t(s.cells * kHeads * kDim);           // int8: per = head_dim
    const size_t scales = size_t(s.cells * kHeads * (kDim / kGroup) * 2);
    s.k.resize(data); s.v.resize(data); s.ks.resize(scales); s.vs.resize(scales);
    std::memcpy(s.k.data(), p.k.data(), data);
    std::memcpy(s.v.data(), p.v.data(), data);
    std::memcpy(s.ks.data(), p.ks.data(), scales);
    std::memcpy(s.vs.data(), p.vs.data(), scales);
    return s;
}

RawInt8RestoreClearStage stage_of(const Serialized& s, int64_t forwarded) {
    RawInt8RestoreClearStage st;
    st.forwarded = uint64_t(forwarded);
    RawInt8RestoreClearExtent e;
    e.format = int(strata::kernels::kKvInt8);
    e.cells = s.cells; e.heads = kHeads; e.head_dim = kDim; e.page_size = kPage;
    e.k = s.k.size(); e.v = s.v.size(); e.k_scale = s.ks.size(); e.v_scale = s.vs.size();
    st.layers.push_back(e);
    return st;
}

struct ClearLog {
    int calls = 0;
    std::vector<std::pair<size_t, size_t>> ranges;   // offset,bytes for every accepted span
    int fail_at = -1;                                 // injected executor failure
};

// The restore sequence: plan/preflight, clear the omitted tail (production dispatch), then copy the serialized
// prefix (the untouched conversation_kv_restore path).  `with_clear` exists only to prove this test can fail.
bool restore_into(Pool& p, const Serialized& s, int64_t forwarded, ClearLog& log, bool with_clear = true) {
    const auto stages = std::vector<RawInt8RestoreClearStage>{stage_of(s, forwarded)};
    std::vector<std::vector<RawInt8RestoreClearPlan>> plans;
    const char* reason = nullptr;
    if (!raw_int8_restore_clear_preflight(stages,
            [&](size_t, size_t) -> const QsaState& { return p.st; }, geometry(), kPage, plans, reason))
        return false;
    if (plans.size() != 1 || plans[0].size() != 1) return false;
    if (with_clear) {
        const bool ok = apply_raw_int8_restore_clear_plan(plans[0][0], [&](void* data, size_t off, size_t bytes) {
            if (log.fail_at == log.calls) { ++log.calls; return false; }
            ++log.calls;
            log.ranges.push_back({off, bytes});
            std::memset(static_cast<uint8_t*>(data) + off, 0, bytes);
            return true;
        });
        if (!ok) return false;
    }
    std::memcpy(p.k.data(), s.k.data(), s.k.size());
    std::memcpy(p.v.data(), s.v.data(), s.v.size());
    std::memcpy(p.ks.data(), s.ks.data(), s.ks.size());
    std::memcpy(p.vs.data(), s.vs.data(), s.vs.size());
    return true;
}

// ---- scenarios -------------------------------------------------------------

void test_incident_layout() {
    check(row_of(137, 0) == 273, "blocked layout: position137/head0 is row273");
    check(code_byte(137, 0) == 69888, "blocked layout: position137/head0 code byte is 69888 (measured)");
    check(scale_unit(137, 0) * 2 == 2184, "blocked layout: position137/head0 scale byte is 2184 (measured)");
    check(kCodeRow == 512 && kScaleRow == 16, "per-cell row sizes match the snapshot formulas");
}

void test_repeated_restore_cycle() {
    Pool clean(224), dirty(224);
    Serialized src = capture(clean, 136);          // cycle boundary: exactly 136 cells, no partial page
    // previous cycle's continuation left real bytes at 128..137 and garbage beyond
    for (int64_t pos = 128; pos <= 137; ++pos) write_position(dirty, pos, int(pos));
    for (size_t i = 0; i < dirty.k.size(); ++i) if (dirty.k[i] == 0) dirty.k[i] = int8_t(0x5a);
    for (size_t i = 0; i < dirty.ks.size(); ++i) if (dirty.ks[i] == 0) dirty.ks[i] = 0xbeef;

    ClearLog lc, ld;
    check(restore_into(clean, src, 136, lc), "clean restore at 136 accepted");
    check(restore_into(dirty, src, 136, ld), "dirty restore at 136 accepted");
    write_position(clean, 136, 136);
    write_position(dirty, 136, 136);

    // the next cycle advances and captures at a logical extent that rounds into the unused page
    Serialized a = capture(clean, 137), b = capture(dirty, 137);
    check(a.cells == 140, "capture at logical137 serializes 140 cells");
    check(a.k == b.k && a.v == b.v && a.ks == b.ks && a.vs == b.vs,
          "repeated restore: capture at 137 matches the clean reference byte-for-byte");
    check(ld.ranges.size() == 4, "all four spans (K,V,K-scale,V-scale) dispatched");
    check(ld.ranges[0].first == 69632 && ld.ranges[2].first == 2176,
          "clear starts at 136 cells: code 69632, scale 2176");
    check(ld.ranges[0].first + ld.ranges[0].second == 224 * kCodeRow &&
          ld.ranges[2].first + ld.ranges[2].second == 224 * kScaleRow,
          "clear ends exactly at the actual allocated capacity");

    // the gate must be able to fail: without the clear the stale rows survive, at the measured offsets
    Pool stale(224);
    for (int64_t pos = 128; pos <= 137; ++pos) write_position(stale, pos, int(pos));
    ClearLog ls;
    check(restore_into(stale, src, 136, ls, false), "no-clear control restore accepted");
    write_position(stale, 136, 136);
    Serialized c = capture(stale, 137);
    check(!(a.k == c.k && a.ks == c.ks), "CONTROL: without the tail clear the 137 capture still differs (test bites)");
    size_t first = c.k.size();
    for (size_t i = 0; i < c.k.size(); ++i) if (a.k[i] != c.k[i]) { first = i; break; }
    check(first == 69888, "CONTROL: first surviving K difference is exactly the measured offset 69888");
    size_t first_s = c.ks.size();
    for (size_t i = 0; i < c.ks.size(); ++i) if (a.ks[i] != c.ks[i]) { first_s = i; break; }
    check(first_s == 2184, "CONTROL: first surviving scale difference is exactly the measured offset 2184");
}

void test_encoded_partial_page_is_preserved() {
    Pool src_pool(224);
    for (int64_t pos = 0; pos <= 139; ++pos) write_position(src_pool, pos, int(pos));
    Serialized src = capture(src_pool, 137);          // 140 serialized cells, 137..139 are ENCODED bytes
    Pool dst(224);
    for (int64_t pos = 130; pos < 224; ++pos) write_position(dst, pos, int(pos) + 7);
    ClearLog log;
    check(restore_into(dst, src, 137, log), "restore of a 137-extent image accepted");
    bool tail_zero = true;
    for (int64_t pos = 140; pos < 224; ++pos)
        for (int64_t h = 0; h < kHeads; ++h) {
            for (size_t i = 0; i < size_t(kDim); ++i)
                if (dst.k[code_byte(pos, h) + i] || dst.v[code_byte(pos, h) + i]) tail_zero = false;
            for (size_t i = 0; i < size_t(kDim / kGroup); ++i)
                if (dst.ks[scale_unit(pos, h) + i] || dst.vs[scale_unit(pos, h) + i]) tail_zero = false;
        }
    check(tail_zero, "restore at 137 zeroes only beyond the 140 serialized cells");
    bool encoded_kept = true;
    for (int64_t pos = 137; pos <= 139; ++pos)
        for (int64_t h = 0; h < kHeads; ++h) {
            // the serialized image is a byte-exact copy of the destination prefix, so the same blocked
            // offsets address both sides
            for (size_t i = 0; i < size_t(kDim); ++i)
                if (dst.k[code_byte(pos, h) + i] != int8_t(src.k[code_byte(pos, h) + i])) encoded_kept = false;
            uint16_t want = 0, want_v = 0;
            std::memcpy(&want, src.ks.data() + scale_unit(pos, h) * 2, 2);
            std::memcpy(&want_v, src.vs.data() + scale_unit(pos, h) * 2, 2);
            if (dst.ks[scale_unit(pos, h)] != want || dst.vs[scale_unit(pos, h)] != want_v) encoded_kept = false;
        }
    check(encoded_kept, "encoded partial-page bytes 137..139 survive the restore untouched");
    check(log.ranges[0].first == 140 * kCodeRow, "clear boundary for a 140-cell image is 140 cells, not 137");
}

void test_both_heads_and_scales() {
    Pool p(224);
    for (int64_t pos = 136; pos < 224; ++pos) write_position(p, pos, int(pos));
    Serialized src = capture(Pool(224), 136);
    ClearLog log;
    check(restore_into(p, src, 136, log), "restore at 136 accepted");
    check(p.k[code_byte(137, 0)] == 0 && p.k[code_byte(137, 1)] == 0, "head0 and head1 rows both cleared");
    check(p.ks[scale_unit(137, 0)] == 0 && p.vs[scale_unit(139, 1)] == 0, "both scale arrays cleared at both heads");
    check(p.k[code_byte(135, 0)] == 0 && p.k[code_byte(0, 1)] == 0, "no code inside the serialized prefix was touched");
}

void test_boundaries() {
    Pool p(224);
    Serialized empty = capture(Pool(224), 0);
    ClearLog log;
    check(restore_into(p, empty, 0, log), "restore at 0 accepted");
    check(log.ranges.size() == 4 && log.ranges[0].first == 0 && log.ranges[0].second == 224 * kCodeRow,
          "extent 0 clears the whole capacity");

    Pool full(224);
    Serialized whole = capture(full, 224);
    ClearLog log2;
    check(restore_into(full, whole, 224, log2), "restore at full capacity accepted");
    check(log2.ranges.empty(), "full-capacity restore dispatches no clear at all (zero-length spans are no-ops)");

    Pool rounding(224);
    RawInt8RestoreClearPlan plan;
    check(raw_int8_restore_clear_plan(rounding.st, geometry(), 138, kPage, plan) && plan.serialized_cells == 140,
          "extent 138 rounds to 140 serialized cells, matching the snapshot");
    check(plan.capacity_cells == 224 && plan.capacity_cells == uint64_t(rounding.st.n_pages) * kPage,
          "capacity is exactly n_pages*page of the actual allocation");
}

void expect_reject(const std::string& what, const std::function<void(QsaState&, ModelGeometry&, Pool&)>& mutate,
                   uint64_t forwarded = 136, int64_t page = kPage) {
    Pool p(224);
    ModelGeometry g = geometry();
    QsaState st = p.st;
    mutate(st, g, p);
    RawInt8RestoreClearPlan plan;
    check(!raw_int8_restore_clear_plan(st, g, forwarded, page, plan), "reject: " + what);
}

void test_refusals() {
    expect_reject("rotated INT8", [](QsaState& s, ModelGeometry&, Pool&) { s.kv_rot = true; });
    expect_reject("Q4_0", [](QsaState& s, ModelGeometry&, Pool&) { s.kv_q4 = true; s.kv_int8 = false; });
    expect_reject("K8V4 hybrid", [](QsaState& s, ModelGeometry&, Pool&) { s.kv_hybrid = true; s.kv_int8 = false; });
    expect_reject("streamed mode1", [](QsaState& s, ModelGeometry&, Pool&) { s.kv_mode = 1; });
    expect_reject("ring mode2", [](QsaState& s, ModelGeometry&, Pool&) { s.kv_mode = 2; });
    expect_reject("elastic VMM pool", [](QsaState& s, ModelGeometry&, Pool&) { s.kv_elastic = 0; });
    expect_reject("borrowed elastic owner", [](QsaState& s, ModelGeometry&, Pool&) { s.owned_elastic = reinterpret_cast<QsaElasticView*>(1); });
    expect_reject("slots != pages", [](QsaState& s, ModelGeometry&, Pool&) { s.n_slots = s.n_pages - 1; });
    expect_reject("n_pages below capacity", [](QsaState& s, ModelGeometry&, Pool&) { s.n_pages -= 1; });
    expect_reject("n_pages above capacity", [](QsaState& s, ModelGeometry&, Pool&) { s.n_pages += 1; });
    expect_reject("extent beyond max_cells", [](QsaState&, ModelGeometry&, Pool&) {}, 225);
    expect_reject("null K codes", [](QsaState& s, ModelGeometry&, Pool&) { s.k_q = nullptr; });
    expect_reject("null V codes", [](QsaState& s, ModelGeometry&, Pool&) { s.v_q = nullptr; });
    expect_reject("null K scales", [](QsaState& s, ModelGeometry&, Pool&) { s.k_scale = nullptr; });
    expect_reject("null page table", [](QsaState& s, ModelGeometry&, Pool&) { s.page_table = nullptr; });
    expect_reject("aliased K/V pools", [](QsaState& s, ModelGeometry&, Pool&) { s.v_q = s.k_q; });
    expect_reject("aliased scale pools", [](QsaState& s, ModelGeometry&, Pool&) { s.v_scale = s.k_scale; });
    expect_reject("head_dim not a group multiple", [](QsaState&, ModelGeometry& g, Pool&) { g.head_dim = 100; });
    expect_reject("no kv heads", [](QsaState&, ModelGeometry& g, Pool&) { g.n_head_kv = 0; });
    expect_reject("negative page size", [](QsaState&, ModelGeometry&, Pool&) {}, 136, -4);
    expect_reject("zero page size", [](QsaState&, ModelGeometry&, Pool&) {}, 136, 0);
    expect_reject("zero max_cells", [](QsaState& s, ModelGeometry&, Pool&) { s.max_cells = 0; });
    // checked overflow: n_slots/n_pages consistent with max_cells, but the byte counts do not fit
    expect_reject("byte-count overflow", [](QsaState& s, ModelGeometry&, Pool&) {
        s.max_cells = int64_t(1) << 55; s.n_pages = s.n_slots = pages_for(s.max_cells); });
    // a page argument that disagrees with the state's own capacity must be refused, not silently retargeted
    Pool p(224);
    RawInt8RestoreClearPlan plan;
    check(!raw_int8_restore_clear_plan(p.st, geometry(), 136, 8, plan), "reject: page8 vs page4 capacity");
}

void test_preflight_refuses_late_stage_before_any_mutation() {
    Pool good(224), bad(224);
    bad.st.kv_rot = true;
    for (int64_t pos = 136; pos < 224; ++pos) write_position(good, pos, int(pos));
    std::vector<int8_t> before(good.k.begin(), good.k.end());
    Serialized src = capture(Pool(224), 136);
    std::vector<RawInt8RestoreClearStage> stages{stage_of(src, 136), stage_of(src, 136)};
    std::vector<std::vector<RawInt8RestoreClearPlan>> plans;
    const char* reason = nullptr;
    const bool ok = raw_int8_restore_clear_preflight(stages,
        [&](size_t i, size_t) -> const QsaState& { return i ? bad.st : good.st; }, geometry(), kPage, plans, reason);
    check(!ok, "preflight refuses when only the later stage is unsupported");
    check(reason != nullptr, "preflight reports a reason");
    check(plans.empty(), "preflight leaves no partial plan set");
    check(before.size() == good.k.size() && std::equal(before.begin(), before.end(), good.k.begin()),
          "stage0 destination is untouched when a later stage is refused (pure preflight, no writes)");
}

void test_failed_clear_stops_and_is_detectable() {
    Pool p(224);
    for (int64_t pos = 136; pos < 224; ++pos) write_position(p, pos, int(pos));
    Serialized src = capture(Pool(224), 136);
    ClearLog log;
    log.fail_at = 1;                      // second span fails
    check(!restore_into(p, src, 136, log), "failed executor fails the restore");
    check(log.calls == 2, "dispatch stops at the first failing span");
    check(p.k[code_byte(137, 0)] == 0, "first span was already cleared when the second failed");
    check(p.ks[scale_unit(137, 0)] != 0, "later spans were not dispatched (caller must poison, not continue)");
}

void test_plan_identity_recheck() {
    Pool p(224);
    RawInt8RestoreClearPlan a, b;
    check(raw_int8_restore_clear_plan(p.st, geometry(), 136, kPage, a), "plan for identity recheck");
    b = a;
    check(a == b, "identical plans compare equal");
    b.spans[0].offset += 1;
    check(a != b, "an offset change is detected");
    b = a; b.spans[0].bytes -= kCodeRow;
    check(a != b, "a length change is detected");
    b = a; b.spans[2].data = nullptr;
    check(a != b, "a pointer change is detected");
}

void test_multi_token_prefill_across_page() {
    for (int64_t upto : {139, 140, 141}) {
        Pool clean(224), dirty(224);
        Serialized src = capture(clean, 136);
        for (int64_t pos = 128; pos <= 137; ++pos) write_position(dirty, pos, int(pos));
        ClearLog l1, l2;
        check(restore_into(clean, src, 136, l1), "clean restore (multi-token case)");
        check(restore_into(dirty, src, 136, l2), "dirty restore (multi-token case)");
        for (int64_t pos = 136; pos <= 139; ++pos) { write_position(clean, pos, int(pos)); write_position(dirty, pos, int(pos)); }
        Serialized a = capture(clean, upto), b = capture(dirty, upto);
        check(a.k == b.k && a.v == b.v && a.ks == b.ks && a.vs == b.vs,
              "multi-token prefill across the page boundary matches clean at extent " + std::to_string(upto));
    }
}

}   // namespace

int main() {
    test_incident_layout();
    test_repeated_restore_cycle();
    test_encoded_partial_page_is_preserved();
    test_both_heads_and_scales();
    test_boundaries();
    test_refusals();
    test_preflight_refuses_late_stage_before_any_mutation();
    test_failed_clear_stops_and_is_detectable();
    test_plan_identity_recheck();
    test_multi_token_prefill_across_page();
    if (failures) { std::printf("RAW INT8 restore clear gate FAILED: %d checks\n", failures); return 1; }
    std::printf("RAW INT8 restore clear gate PASS (CPU planner+dispatch only; no CUDA behavior proven)\n");
    return 0;
}
