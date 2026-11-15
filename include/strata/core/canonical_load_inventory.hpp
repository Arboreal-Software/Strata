#pragma once
#include "strata/core/window_budget.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace strata::core {
class VerifiedDriverBindings;
struct CanonicalInventoryLimits {uint64_t max_records=0,max_string_bytes=0;};
enum class CanonicalPlaneConversion {copy,bf16_high_bits,f32_to_f16,f16_to_f32};
struct CanonicalPlaneRecord {
    uint64_t source_offset=0,source_bytes=0,runtime_offset=0,runtime_bytes=0;
    CanonicalPlaneConversion conversion=CanonicalPlaneConversion::copy;
};
struct CanonicalTensorLoadRecord {
    std::string name;
    int source_index_kind=0,normalized_kind=0,file_id=0;
    bool raw16=false,chosen=false;
    uint64_t source_offset=0,source_bytes=0,index_destination_offset=0,runtime_offset=0,runtime_bytes=0;
    int64_t ne0=0,ne1=0;
    int code_bits=0,code_bias=0,group_elems=0,activation_kind=0;
    int index_codebook=0,index_has_offset=0,index_scales_fp16=0;
    uint64_t index_codes_bytes=0,index_scales_bytes=0,index_offsets_bytes=0;
    // Exactly the segments executed. Skipped rows have no executed planes;
    // their original index plane fields above are retained without inference.
    std::array<CanonicalPlaneRecord,3> planes{};
    uint32_t plane_count=0;
};
enum class CanonicalHcConversion {bf16_raw16_to_q8_host,bf16_in_f32_to_q8_host};
struct CanonicalHcLoadRecord {
    uint64_t canonical_row=0;
    CanonicalHcConversion conversion=CanonicalHcConversion::bf16_in_f32_to_q8_host;
    uint64_t codes_offset=0,codes_bytes=0,scales_offset=0,scales_bytes=0;
};
struct CanonicalLoadInventory {
    // These are retained authorities, not verification performed by this loader.
    std::shared_ptr<const ReservationLease> metadata_reservation;
    std::shared_ptr<const VerifiedDriverBindings> retained_input_bindings;
    CanonicalInventoryLimits limits;
    std::string expected_pack_loader_path;
    std::vector<CanonicalTensorLoadRecord> records; // Actual parser order, including skipped rows.
    std::vector<CanonicalHcLoadRecord> hc_records; // Actual conversion/publication order.
    uint64_t index_pool_bytes=0,canonical_arena_bytes=0,hc_allocation_bytes=0;
    int index_alignment=0;
    bool hc_completed=false;
    CanonicalLoadInventory* quarantined_next=nullptr;
};
inline bool canonical_inventory_host_bound(const CanonicalInventoryLimits& l,uint64_t& out,std::string& error) {
    if(!l.max_records||l.max_records>65536||!l.max_string_bytes||l.max_string_bytes>4096){error="canonical inventory bounds outside supported limits";return false;}
    const uint64_t per=sizeof(CanonicalTensorLoadRecord)+sizeof(CanonicalHcLoadRecord)+l.max_string_bytes+256;
    const uint64_t base=4096+sizeof(CanonicalLoadInventory)+l.max_string_bytes;
    if(l.max_records>(UINT64_MAX-base)/per){error="canonical inventory metadata overflow";return false;}
    out=base+l.max_records*per;return true;
}
}
