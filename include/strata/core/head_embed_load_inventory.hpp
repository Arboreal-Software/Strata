#pragma once
#include "strata/core/native_load_inventory.hpp"
namespace strata::core {
enum class HeadEmbedRole { head, embedding };
enum class HeadPackedForm { none, q6_planes_256, q8_sm60_qs_d };
enum class EmbedPlacement { none, mapped_portable, device_pin_fallback, device_alias_fallback };
struct HeadEmbedLoadInventory {
    std::shared_ptr<const ReservationLease> metadata_reservation;
    std::shared_ptr<const void> retained_input_bindings; // retention, not verification
    NativeInventoryLimits limits;
    std::vector<std::string> sources;
    NativeTensorLoadRecord tensor;
    HeadEmbedRole role=HeadEmbedRole::head;
    HeadPackedForm packed=HeadPackedForm::none;
    EmbedPlacement placement=EmbedPlacement::none;
    uint64_t generation=0,row_bytes=0,main_bytes=0,scratch_bytes=0,helper_bytes=0;
    int device=-1;
    bool embedding_override=false;
    HeadEmbedLoadInventory* quarantined_next=nullptr;
};
inline bool head_embed_inventory_host_bound(const NativeInventoryLimits& limits,uint64_t& out,std::string& error) {
    uint64_t n=0;
    if(limits.max_records!=1||limits.max_rank<2) {error="head/embed inventory requires one record and rank two capacity";return false;}
    if(!native_inventory_host_bound(limits,n,error))return false;
    if(n>UINT64_MAX-sizeof(HeadEmbedLoadInventory)) {error="head/embed inventory bound overflow";return false;}
    out=n+sizeof(HeadEmbedLoadInventory);return true;
}
} // namespace strata::core
