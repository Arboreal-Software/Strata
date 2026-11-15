#pragma once
#include <cstdint>
// Exact pure shape declaration/helper from include/strata/kernels/qsa.hpp.
// This shadow has no kernels, allocator, eligibility or ownership implementation.
namespace strata::kernels {
struct QsaShapes {
    int64_t n_head = 0;     ///< query heads (24).  n_head must be a multiple of n_head_kv
    int64_t n_head_kv = 0;  ///< key/value heads (2).  The q -> kv map is `q / (n_head/n_head_kv)`, NOT `q % kv`
    int64_t head_dim = 0;   ///< 256, and must be a multiple of 4
    int64_t n_rot = 0;      ///< 64 rotated dims of head_dim (and of idx_dim), NEOX pairs
    int64_t idx_n_head = 0; ///< indexer query heads (4)
    int64_t idx_dim = 0;    ///< indexer key width (128)
    int64_t idx_block = 0;  ///< r = 4: the indexer pools one key per 4 cells
    int64_t idx_top_k = 0;  ///< 2048 cells of selection budget
    int64_t page_size = 0;  ///< KV page granule in CELLS; 1 is a legal unpaged configuration
};

/// The real artifact's geometry, so no caller transcribes it by hand.
inline QsaShapes qsa_real_shapes() {
    QsaShapes s;
    s.n_head = 24; s.n_head_kv = 2; s.head_dim = 256; s.n_rot = 64;
    s.idx_n_head = 4; s.idx_dim = 128; s.idx_block = 4; s.idx_top_k = 2048;
    // one page = one indexer block (4 cells): the granule KV streaming keeps resident (kv_stream.hpp). The readers
    // resolve a row per cell anyway, so the page size costs them nothing (measured: identical output and speed).
    s.page_size = 4;
    return s;
}

}
