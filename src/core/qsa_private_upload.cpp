#include "strata/core/layer.hpp"
#include <cuda_runtime.h>
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/native_rope.hpp"
#include <limits>
#include <new>

namespace strata::core {
bool qsa_upload_private_initial_state(QsaState& state, int n_rot, bool owns_rope, QsaInitUploads& uploads) {
    if (uploads.started) return false;
    uploads.started = true;
    if (state.max_cells <= 0 || state.max_cells > std::numeric_limits<int>::max() ||
        state.n_pages <= 0 || state.n_pages > std::numeric_limits<int32_t>::max() ||
        !state.page_table || n_rot <= 0 || n_rot % 2 || state.kv_mode < 0 || state.kv_mode > 2 ||
        (owns_rope && (!state.cos_tab || !state.sin_tab))) {
        uploads.error = "invalid private QSA upload layout"; return false;
    }
    auto check = [&](cudaError_t status) {
        if (status == cudaSuccess) return true;
        uploads.error = std::string("private QSA upload: ") + cudaGetErrorString(status);
        return false;
    };
    // Allocate/fill all host sources before any copy is issued. Caller retains
    // this entire object on every error, including errors from an earlier launch.
    try {
        if (owns_rope) {
            const uint64_t count = uint64_t(state.max_cells) * uint64_t(n_rot / 2);
            if (count > uploads.rope_cos.max_size()) {
                uploads.error = "private QSA RoPE upload too large"; return false;
            }
            uploads.rope_cos.resize(size_t(count)); uploads.rope_sin.resize(size_t(count));
            strata::kernels::build_rope_table(n_rot,strata::kernels::rope_scaling(),int(state.max_cells),
                uploads.rope_cos.data(),uploads.rope_sin.data());
        }
        if (state.kv_mode == 0) {
            uploads.page_table.resize(size_t(state.n_pages));
            for (int64_t i=0;i<state.n_pages;++i) uploads.page_table[size_t(i)] = int32_t(i);
        }
    } catch (const std::bad_alloc&) { uploads.error = "private QSA upload allocation failed"; return false; }
    if (owns_rope) {
        if (!check(cudaMemcpy(state.cos_tab,uploads.rope_cos.data(),uploads.rope_cos.size()*sizeof(float),cudaMemcpyHostToDevice)) ||
            !check(cudaMemcpy(state.sin_tab,uploads.rope_sin.data(),uploads.rope_sin.size()*sizeof(float),cudaMemcpyHostToDevice))) return false;
    }
    if (state.kv_mode == 0) {
        if (!check(cudaMemcpy(state.page_table,uploads.page_table.data(),uploads.page_table.size()*sizeof(int32_t),cudaMemcpyHostToDevice))) return false;
    } else {
        if (state.kv_mode == 1) strata::kernels::kv_stream_reset(state.map,nullptr);
        else strata::kernels::kv_ring_table(state.page_table,state.n_pages,state.n_slots,nullptr);
        if (!check(cudaGetLastError())) return false;
    }
    if (!check(cudaDeviceSynchronize())) return false;
    uploads.drained = true;
    // Own the private table without replacing the device-wide lookup used by
    // selected execution. The scheduler binds that lookup only at a drained
    // execution switch; private preparation cannot publish global bindings.
    if (owns_rope) state.owns_rope = true;
    return true;
}
} // namespace strata::core
