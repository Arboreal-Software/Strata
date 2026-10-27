#pragma once
#include "strata/core/window_budget.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace strata::core {
struct NativeInventoryLimits {uint64_t max_sources=0,max_records=0,max_rank=0,max_string_bytes=0;};
enum class NativeLoadSource {gguf_argument,unresolved_canonical};
enum class NativeLoadLayout {gguf_native,hc_q8_0};
enum class NativeLoadConversion {identity,f32_to_q8_0_host,bf16_to_q8_0_host,bf16_to_q8_0_device};
struct NativeCanonicalDependency {
    int64_t file_id=0,weight_kind=0,code_bits=0,code_bias=0,group_elems=0,activation_kind=0;
    uint64_t source_offset=0,source_bytes=0,codes_bytes=0,scales_bytes=0,offsets_bytes=0;
    bool source_scales_fp16=false,has_offset=false,codebook_iq4nl=false;
};
struct NativeTensorLoadRecord {
    std::string name;
    NativeLoadSource source=NativeLoadSource::gguf_argument;
    uint64_t source_ordinal=0,source_offset=0,source_bytes=0;
    int64_t source_type=-1;
    std::vector<uint64_t> source_shape; // GGUF ne0-first; canonical dependency ne0,ne1
    NativeLoadLayout runtime_layout=NativeLoadLayout::gguf_native;
    NativeLoadConversion conversion=NativeLoadConversion::identity;
    uint64_t runtime_bytes=0,q8_packed_helper_bytes=0;
    NativeCanonicalDependency canonical;
};
struct NativeLoadInventory {
    // Retention is not provenance verification: trusted collector must join
    // sources/ordinals to actual typed VerifiedInputLeases before composition.
    std::shared_ptr<const ReservationLease> metadata_reservation;
    std::shared_ptr<const void> retained_input_bindings;
    NativeInventoryLimits limits;
    std::vector<std::string> sources;
    std::vector<NativeTensorLoadRecord> records;
    int64_t layer_lo=0,layer_hi=-1,global_layer_lo=-1,global_layer_hi=-1;
    bool include_ple_key=false,has_unresolved_canonical=false;
    NativeLoadInventory* quarantined_next=nullptr; // internal allocation-free retention
};
// Conservative reserved metadata envelope, not exact physical allocator/RSS.
// Output unchanged on refusal. Administrative maxima bound all vector counts.
inline bool native_inventory_host_bound(const NativeInventoryLimits& l,uint64_t& out,std::string& error) {
    if(!l.max_sources||l.max_sources>1024||!l.max_records||l.max_records>65536||
       !l.max_rank||l.max_rank>8||!l.max_string_bytes||l.max_string_bytes>4096) {
        error="native inventory bounds outside supported limits";return false;
    }
    auto add=[](uint64_t a,uint64_t b,uint64_t& n){if(b>UINT64_MAX-a)return false;n=a+b;return true;};
    auto mul=[](uint64_t a,uint64_t b,uint64_t& n){if(a&&b>UINT64_MAX/a)return false;n=a*b;return true;};
    uint64_t sources=0,records=0,total=0;
    if(!mul(l.max_sources,sizeof(std::string)+l.max_string_bytes+128,sources)||
       !mul(l.max_records,sizeof(NativeTensorLoadRecord)+l.max_string_bytes+128+l.max_rank*sizeof(uint64_t),records)||
       !add(4096+sizeof(NativeLoadInventory),sources,total)||!add(total,records,total)) {
        error="native inventory metadata overflow";return false;
    }
    out=total;return true;
}
} // namespace strata::core
