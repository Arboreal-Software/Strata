#pragma once
#include "strata/core/window_budget.hpp"
#include <cstdint>
#include <climits>
#include <memory>
#include <new>
#include <string>
#include <vector>
namespace strata::core {
struct NativeInventoryLimits {
    uint64_t max_sources=0,max_records=0,max_rank=0,max_string_bytes=0;
    // Exact maximum applicable converted payload, preflighted from these sources
    // and this stage's canonical table before the foreground reservation.
    uint64_t max_bf16_q8_bytes=0,bf16_control_bytes=0;
};
// One serial transform has one explicit operator-new payload and this explicit control.
// The ordinary threaded converter uses the identical block worker below; the
// inventoried path creates no thread pool/task or vector-capacity peak.
struct NativeBf16Q8Delete {
    void operator()(uint8_t* p) const noexcept {::operator delete(p);}
};
struct NativeBf16Q8Control {
    std::unique_ptr<uint8_t,NativeBf16Q8Delete> storage;
    const uint16_t* source=nullptr;
    uint8_t* output=nullptr;
    uint8_t* block_output=nullptr;
    uint64_t block=0,end=0;
    float values[32],amax=0,d=0,id=0;
    uint32_t bits=0;
    uint16_t scale=0;
    int column=0;
};
enum class NativeLoadSource {gguf_argument,unresolved_canonical};
enum class NativeLoadLayout {gguf_native,hc_q8_0,transformed_q8_0};
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
// Pure production classifier shared with the profile collector. Failure leaves
// the caller's prior runtime type unchanged; no allocation or source authority
// is manufactured here. The collector still checks the actual verified range.
inline bool native_load_runtime_type(const NativeTensorLoadRecord& r,int64_t& out) noexcept {
    if(!r.runtime_bytes || uint32_t(r.conversion)>uint32_t(NativeLoadConversion::bf16_to_q8_0_device))return false;
    if(r.runtime_layout==NativeLoadLayout::transformed_q8_0) {
        if(r.source!=NativeLoadSource::gguf_argument||r.source_type!=30 ||
           r.conversion!=NativeLoadConversion::bf16_to_q8_0_host||r.source_shape.size()!=2 ||
           !r.source_shape[0]||!r.source_shape[1]||r.source_shape[0]%32 ||
           r.source_shape[0]>INT_MAX||r.source_shape[1]>INT_MAX||r.runtime_bytes>SIZE_MAX ||
           r.source_shape[0]>UINT64_MAX/r.source_shape[1])return false;
        const uint64_t n=r.source_shape[0]*r.source_shape[1];
        if(n>UINT64_MAX/2 || n/32>UINT64_MAX/34 || r.source_bytes!=n*2 ||
           r.runtime_bytes!=n/32*34 || r.source_offset>UINT64_MAX-r.source_bytes ||
           r.q8_packed_helper_bytes)return false;
        out=8;return true;
    }
    // Existing layouts retain their original classification/extent predicate.
    if(uint32_t(r.runtime_layout)>uint32_t(NativeLoadLayout::hc_q8_0))return false;
    out=r.source_type;return true;
}
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
// Reserved metadata plus explicit serial-conversion requested-size envelope.
// This does not certify allocator rounding, physical stack frames or RSS.
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
    if(l.max_bf16_q8_bytes) {
        // No second payload, new task, or private library control block. The
        // actual source-frame/control terms are provided by the real loader;
        // allocator rounding and physical stack frames are not RSS claims.
        if(l.max_bf16_q8_bytes>SIZE_MAX ||
           !add(total,l.max_bf16_q8_bytes,total) ||
           !l.bf16_control_bytes || !add(total,l.bf16_control_bytes,total)) {
            error="native inventory BF16 conversion peak overflow";return false;
        }
    }
    out=total;return true;
}
} // namespace strata::core
