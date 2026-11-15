#pragma once
// DATA range plan, all-stage preflight, and the dispatch closure ONLY.  This is not admission, stream-join,
// residency or completion authority: the caller owns device selection, the private stream, the checked drain
// and poisoning.
//
// WHY a page-aligned split is the right boundary.  The INT8 KV pools are laid out blocked as
// [page][kv_head][page_size][head_dim] (layer.hpp), so one page covers n_head_kv*page_size consecutive rows of
// every array, and a snapshot of `upto` tokens serializes ceil(upto/page)*page cells.  conversation_kv_save /
// conversation_kv_restore (conversation_snapshot.cpp) move exactly the `cells`-derived prefix of K, V and both
// scale arrays and nothing beyond it, while layer.cpp sized those arrays for `slots` pages.  So a split at a
// page-aligned cell count separates "bytes the snapshot owns" from "destination bytes the snapshot omits" for
// any page size, and the omitted tail is precisely capacity the destination physically has.
//
// The omitted tail must be canonical because it is READ: attention's partial page includes rows for positions
// that were never computed, and a capture at a logical extent rounds them into the serialized image.  A
// previous cycle left real bytes in those rows, which is what made the second repair cycle fail native137 while
// argmax and full continuation logits stayed identical.  Clearing the live destination tail fixes the state;
// masking comparator fields, sanitizing at capture time, clearing serialized (possibly encoded) bytes, or
// zeroing the model is a false fix.
#include "strata/core/geometry.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace strata::core {

struct RawInt8RestoreClearSpan {
    void* data = nullptr;
    size_t offset = 0, bytes = 0;
    bool operator==(const RawInt8RestoreClearSpan&) const = default;
};

struct RawInt8RestoreClearPlan {
    // K, V, K-scale, V-scale ONLY.  No pooled/tail/dead/block-pos/GDN/PLE/spare/page-table range: those are
    // restored by their own paths or are proven overwritten before any read, and zeroing them indiscriminately
    // destroys live indexer and recurrence state.
    std::array<RawInt8RestoreClearSpan, 4> spans{};
    uint64_t serialized_cells = 0, capacity_cells = 0;
    bool operator==(const RawInt8RestoreClearPlan&) const = default;
};

/// One layer's serialized extents, taken from the image that the restore will actually write.  `forwarded` and
/// these sizes are the copy's own inputs; binding the clear to them is what makes "clear starts exactly where
/// the copy ends" true by construction instead of by agreement between two counters.
struct RawInt8RestoreClearExtent {
    int format = -1;
    int64_t cells = -1, heads = -1, head_dim = -1, page_size = -1;
    size_t k = 0, v = 0, k_scale = 0, v_scale = 0;
    bool operator==(const RawInt8RestoreClearExtent&) const = default;
};

struct RawInt8RestoreClearStage {
    uint64_t forwarded = 0;
    std::vector<RawInt8RestoreClearExtent> layers;
};

/// `page_size`: the page the snapshot layout used, i.e. strata::kernels::qsa_real_shapes().page_size - the same
/// value the allocator used.  Passing it (rather than hardcoding 4) keeps the plan pinned to the real shape.
/// `forwarded`: the extent the restore will use, i.e. image.live.ids.size(), the same value
/// conversation_snapshot_restore() hands to layout().
/// Qualification is the fixed resident CUDA INT8 shape (mode 0, unrotated, group 64).  Anything else is refused
/// here and must be separately proven - streamed, ring/draft, sliding/rebased, remapped, shared, elastic-VMM,
/// Q4_0, K8V4, rotated INT8, HIP and SYCL must not be absorbed by relaxing this gate.
inline bool raw_int8_restore_clear_plan(const QsaState& st, const ModelGeometry& g, uint64_t forwarded,
                                        int64_t page_size, RawInt8RestoreClearPlan& out) noexcept {
    constexpr uint64_t group = uint64_t(strata::kernels::KV_Q8_GROUP);   // the snapshot's own scale group
    if (page_size <= 0 ||
        // exclusively owned, physically mapped, fully resident pools only
        st.kv_mode != 0 || st.kv_elastic != -1 || st.owned_elastic != nullptr ||
        // unrotated plain INT8: Q4_0/K8V4/rotated bytes mean nothing to another format
        !st.kv_int8 || st.kv_q4 || st.kv_hybrid || st.kv_rot ||
        st.max_cells <= 0 || st.n_pages <= 0 || st.n_slots != st.n_pages ||
        g.n_head_kv <= 0 || g.head_dim <= 0 || g.head_dim % int64_t(group) ||
        forwarded > uint64_t(st.max_cells) ||
        !st.k_q || !st.v_q || !st.k_scale || !st.v_scale || !st.page_table)
        return false;
    // mode 0 IS the identity page map with every page resident and no host copy (layer.hpp).  The table is
    // DEVICE memory: it is checked for presence only, because reading it from the host would itself be a bug.
    // The four destination arrays must be distinct; an aliased pair means the restore is already broken.
    const void* p[4] = {st.k_q, st.v_q, st.k_scale, st.v_scale};
    for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) if (p[a] == p[b]) return false;

    const uint64_t page = uint64_t(page_size);
    const uint64_t expected_pages = (uint64_t(st.max_cells) + page - 1) / page;   // actual allocation, from slots
    if (uint64_t(st.n_pages) != expected_pages) return false;

    RawInt8RestoreClearPlan plan;
    plan.serialized_cells = (forwarded + page - 1) / page * page;
    plan.capacity_cells = expected_pages * page;
    if (plan.serialized_cells > plan.capacity_cells) return false;

    auto product = [](uint64_t a, uint64_t b, uint64_t& value) noexcept {
        if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
        value = a * b;
        return value <= SIZE_MAX;
    };
    uint64_t code_row = 0, scale_row = 0, code_start = 0, code_end = 0, scale_start = 0, scale_end = 0;
    if (!product(uint64_t(g.n_head_kv), uint64_t(g.head_dim), code_row) ||
        !product(uint64_t(g.n_head_kv), uint64_t(g.head_dim / int64_t(group)) * sizeof(uint16_t), scale_row) ||
        !product(plan.serialized_cells, code_row, code_start) || !product(plan.capacity_cells, code_row, code_end) ||
        !product(plan.serialized_cells, scale_row, scale_start) || !product(plan.capacity_cells, scale_row, scale_end))
        return false;
    plan.spans = {{{st.k_q, size_t(code_start), size_t(code_end - code_start)},
                   {st.v_q, size_t(code_start), size_t(code_end - code_start)},
                   {st.k_scale, size_t(scale_start), size_t(scale_end - scale_start)},
                   {st.v_scale, size_t(scale_start), size_t(scale_end - scale_start)}}};
    out = plan;
    return true;   // bytes [0, serialized_cells*row) are never inside a span: every serialized byte survives
}

/// Validate EVERY stage and layer before any device mutation, so a later stage cannot be refused after an
/// earlier one was already cleared.  Pure: writes nothing, only plans.  `state_of` reaches the live destination
/// state; the extents come from the immutable images.  `reason` points at a literal (no allocation).
inline bool raw_int8_restore_clear_preflight(const std::vector<RawInt8RestoreClearStage>& stages,
                                             const std::function<const QsaState&(size_t, size_t)>& state_of,
                                             const ModelGeometry& g, int64_t page_size,
                                             std::vector<std::vector<RawInt8RestoreClearPlan>>& out,
                                             const char*& reason) noexcept {
    out.clear();
    reason = nullptr;
    std::vector<std::vector<RawInt8RestoreClearPlan>> plans;
    try {
        plans.resize(stages.size());
        for (size_t i = 0; i < stages.size(); ++i) {
            plans[i].resize(stages[i].layers.size());
            for (size_t j = 0; j < stages[i].layers.size(); ++j) {
                const auto& kv = stages[i].layers[j];
                if (kv.format != int(strata::kernels::kKvInt8) || kv.heads != g.n_head_kv ||
                    kv.head_dim != g.head_dim || kv.page_size != page_size) {
                    reason = "serialized K/V descriptor differs from the owned native INT8 shape";
                    return false;
                }
                if (!raw_int8_restore_clear_plan(state_of(i, j), g, stages[i].forwarded, page_size, plans[i][j])) {
                    reason = "RAW suffix canonicalization requires owned resident unrotated native INT8 capacity";
                    return false;
                }
                const auto& plan = plans[i][j];
                // the clear boundary must be exactly where this image's copy ends, for all four arrays
                if (kv.cells != int64_t(plan.serialized_cells) || kv.k != plan.spans[0].offset ||
                    kv.v != plan.spans[1].offset || kv.k_scale != plan.spans[2].offset ||
                    kv.v_scale != plan.spans[3].offset) {
                    reason = "RAW suffix clear boundary differs from the serialized image extents";
                    return false;
                }
            }
        }
    } catch (...) {
        reason = "RAW suffix preflight allocation failed";
        return false;
    }
    out = std::move(plans);
    return true;
}

/// The production dispatch closure.  `clear(data, offset, bytes)` runs on the caller's device/stream and must
/// return false on any submission or completion failure; the first failure stops the dispatch (the caller then
/// poisons and retains owners).  A zero-length span is a genuine no-op: a restore at full capacity clears
/// nothing.  CPU tests inject a host executor here and exercise this same function.
template <class Clear>
bool apply_raw_int8_restore_clear_plan(const RawInt8RestoreClearPlan& plan, Clear&& clear) {
    for (const auto& span : plan.spans)
        if (span.bytes && !clear(span.data, span.offset, span.bytes)) return false;
    return true;
}

}   // namespace strata::core
