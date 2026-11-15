#pragma once
#include "strata/core/layer.hpp"
#include "strata/core/window_budget.hpp"
#include "strata/core/window_ple_weights.hpp"
#include "strata/core/window_model_owner.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/kernels/native_mmvq.hpp"

namespace strata::core {
// Internal, quiescent private-stage binding. The caller validates model identity,
// device placement and baseline accounting of the shared table/weight owner.
// The owner must own immutable weights, never this destination session itself.
// Native key uses scratch already allocated in destination.w by the private
// session factory. Source mutable scratch/precomputed projections are rejected.
bool bind_private_ple(PleRun& destination,
    std::shared_ptr<strata::kernels::PleTable> table,
    const strata::kernels::PleWeights& weights,
    const strata::kernels::PleConsts& constants,
    std::shared_ptr<const void> weight_owner, WindowBudget& budget,
    std::shared_ptr<const ReservationLease> session_reservation,
    std::string& error);
// Production metadata adapter. The owner must retain the WeightTable's device
// arena AND optional NativeDense weight allocations, not merely the metadata map.
inline bool bind_private_ple_model(PleRun& destination,
    std::shared_ptr<strata::kernels::PleTable> table, const WeightTable& model,
    bool native_key, const strata::kernels::PleConsts& constants,
    std::shared_ptr<const void> model_owner, WindowBudget& budget,
    std::shared_ptr<const ReservationLease> session_reservation, std::string& error) {
    strata::kernels::PleWeights weights;
    if(!private_ple_weights(model,native_key,weights,error))return false;
    return bind_private_ple(destination,std::move(table),weights,constants,
        std::move(model_owner),budget,std::move(session_reservation),error);
}
// Factory-owned stage/model adapter: retains the actual model resource owner in
// the PLE context, and ties the reader lease to the native session reservation.
inline bool bind_owned_private_ple(const NativeSessionAllocation& allocation, size_t stage_index,
    std::shared_ptr<const OwnedModelStage> model, std::shared_ptr<strata::kernels::PleTable> table,
    bool native_key, const strata::kernels::PleConsts& constants, WindowBudget& budget,
    std::string& error) {
    if(!allocation.healthy() || stage_index>=allocation.stages().size() || !model || !model->healthy()) {
        error="private PLE requires healthy owned model and session";return false;
    }
    const auto& stage=allocation.stages()[stage_index];
    if(!stage.session || !stage.owner || stage.device!=model->device()) {
        error="private PLE model/session device mismatch";return false;
    }
    strata::kernels::PleWeights weights;
    if(!private_ple_weights(model->weights(),native_key,weights,error))return false;
    using namespace strata::kernels;
    const uint64_t elements=uint64_t(NG_N_EMBD)*NG_HC_DIM;
    const auto key_bytes=native_key?native_mmvq_weight_bytes(weights.key_native_type,NG_N_EMBD,NG_HC_DIM):
        (weights.key_bf16?elements*2:elements/4+elements/64*sizeof(float));
    const void* key=native_key?weights.key_native_data:(weights.key_bf16?
        static_cast<const void*>(weights.key_bf16):static_cast<const void*>(weights.key_codes));
    if(!model->owns_weight_span(key,key_bytes,native_key)||
       !model->owns_weight_span(weights.value_bf16,uint64_t(NG_N_EMBD)*NG_N_EMBD*2,false)||
       !model->owns_weight_span(weights.norm_key,uint64_t(NG_HC_DIM)*4,false)||
       !model->owns_weight_span(weights.norm_query,uint64_t(NG_HC_DIM)*4,false)||
       !model->owns_weight_span(weights.norm_conv,uint64_t(NG_HC_DIM)*4,false)||
       !model->owns_weight_span(weights.conv1d_f16,uint64_t(PLE_CONV_KERNEL)*NG_HC_DIM*2,false)) {
        error="private PLE weight span is outside its retained model allocation";return false;
    }
    return bind_private_ple(stage.session->ple,std::move(table),weights,
        constants,std::move(model),budget,allocation.reservation(),error);
}
// Conservative per-context admission headroom, excluding shared cache/slab and
// weights already charged to the baseline. Held even while the context is idle.
inline constexpr uint64_t private_ple_host_headroom = 1024 * 1024;
inline constexpr uint64_t private_ple_io_slots = strata::kernels::PLE_N_HEADS;
} // namespace strata::core
