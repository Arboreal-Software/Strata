#pragma once
#include "strata/core/native_batch_slot_binding.hpp"
#include <array>
#include "strata/core/native_batch_slot_occupancy.hpp"
namespace strata::core {
// Native execution/commit evidence only; never a RAW cut or conversation certificate.
class NativeBatchGroupCompletion final {
public:
    static constexpr size_t max_rows=64,max_stages=8;
    static constexpr uint64_t metadata_bytes()noexcept{return sizeof(NativeBatchGroupCompletion)+16384;}
    uint64_t operation()const noexcept{return operation_;}
    size_t rows()const noexcept{return rows_;}
    size_t stages()const noexcept{return stages_;}
    uint64_t occupancy_incarnation(size_t slot)const noexcept{
        return slot<occupancies_.size()&&occupancies_[slot]?occupancies_[slot]->incarnation():0;
    }
    int32_t input(size_t row)const noexcept{return row<rows_?tokens_[row]:-1;}
    int32_t output(size_t row)const noexcept{return row<rows_?outputs_[row]:-1;}
    int slot(size_t row)const noexcept{return row<rows_?row_slots_[row]:-1;}
    int64_t position(size_t row)const noexcept{return row<rows_?positions_[row]:-1;}
    int accepted(size_t slot)const noexcept{return slot<2?accepted_[slot]:0;}
private:
    friend class Verifier;
    friend class strata::program::ServingRawSession;
    NativeBatchGroupCompletion()=default;
    std::array<std::shared_ptr<const void>,max_stages> setup_owners_{};
    std::array<std::shared_ptr<const NativeBatchSlotOccupancy>,2> occupancies_{};
    mutable std::array<std::atomic<bool>,2> consumed_{};
    uint64_t operation_=0;
    size_t rows_=0,stages_=0;
    std::array<int,max_rows> row_slots_{};
    std::array<int64_t,max_rows> positions_{};
    std::array<int32_t,max_rows> tokens_{},outputs_{};
    std::array<int,2> accepted_{};
    std::array<bool,max_rows> nonfinite_{};
    std::array<NativeBatchSlotInspection,max_stages*2> bindings_{};
};
}
