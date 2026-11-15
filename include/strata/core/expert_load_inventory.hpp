#pragma once
#include "strata/core/window_budget.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>
namespace strata::core {class ExpertLayoutCapture;class VerifiedDriverBindings;}
namespace strata::kernels::cpu {
bool expert_layout_load(const std::string&,int64_t,int64_t,std::string&,strata::core::ExpertLayoutCapture*);
uint64_t expert_layout_generation() noexcept;
uint64_t hadamard_generation() noexcept;
}
namespace strata::core {
struct ExpertInventoryLimits {
    uint64_t max_layers=0,max_string_bytes=0,max_line_bytes=0,max_metadata_bytes=0;
    uint64_t max_sign_vectors=0,max_sign_elements=0;
};
struct ExpertMetadataClaims {
    std::string expected_pack_path;
    // Caller derives these from authenticated membership/absence claims.
    bool native_layout_present=false,hadamard_present=false;
};
struct ExpertLayerLoadRecord {
    uint64_t source_order=0,layer=0,packed_offset=0,blob_bytes=0;
    strata::kernels::cpu::NativeFmt format; // Sign pointers always null; values below own exact signs.
    std::array<uint64_t,3> gguf_offsets{};
    std::array<std::string,3> shard_names;
    bool has_gguf_offsets=false;
};
struct ExpertSignRecord {uint64_t width=0;std::vector<int8_t> values;};
struct ExpertCanonicalLayout {
    uint64_t n_embd=0,n_ff=0,group_elements=0,scale_bytes=0;
    int code_bits=0,code_bias=0;
    // Actual packed planes: gate/up codes, down codes, gate/up scales, down scales.
    std::array<uint64_t,4> offsets{},bytes{};
};
struct ExpertLayoutInventory {
    std::shared_ptr<const ReservationLease> metadata_reservation;
    std::shared_ptr<const VerifiedDriverBindings> retained_input_bindings;
    ExpertInventoryLimits limits;
    ExpertMetadataClaims claims;
    std::vector<ExpertLayerLoadRecord> layers; // Actual parsed source order, not a later scan.
    std::vector<ExpertSignRecord> signs; // Exact successfully parsed +/-1 values.
    std::vector<uint8_t> hadamard_roles;
    int layout_version=0,hadamard_block=0;
    int64_t requested_layers=0,requested_experts=0,effective_experts=0;
    uint64_t total_bytes=0,max_blob_bytes=0,layout_generation=0,hadamard_generation=0;
    bool native=false;
    ExpertCanonicalLayout canonical; // Populated only for the actual canonical fallback.
};
inline bool expert_inventory_host_bound(const ExpertInventoryLimits& l,uint64_t& out,std::string& error){
    if(!l.max_layers||l.max_layers>65536||!l.max_string_bytes||l.max_string_bytes>4096||
       !l.max_line_bytes||l.max_line_bytes>(16ull<<20)||!l.max_metadata_bytes||l.max_metadata_bytes>(64ull<<20)||
       !l.max_sign_vectors||l.max_sign_vectors>1024||!l.max_sign_elements||l.max_sign_elements>(4ull<<20)){
        error="expert inventory bounds outside supported limits";return false;
    }
    // Covers receipt, parser/layout temporaries and bounded string/stream copies.
    const uint64_t layer=sizeof(ExpertLayerLoadRecord)+3*l.max_string_bytes+sizeof(strata::kernels::cpu::NativeFmt)*3+1024;
    const uint64_t fixed=65536+sizeof(ExpertLayoutInventory)+l.max_string_bytes+8*l.max_line_bytes;
    const uint64_t signs=l.max_sign_elements*16+l.max_sign_vectors*256;
    if(l.max_layers>(UINT64_MAX-fixed-signs)/layer){error="expert inventory envelope overflow";return false;}
    out=fixed+signs+l.max_layers*layer;return true;
}
class ExpertLayoutCapture {
public:
    static bool create(const ExpertInventoryLimits&,std::shared_ptr<const ReservationLease>,
        std::shared_ptr<const VerifiedDriverBindings>,const ExpertMetadataClaims&,
        std::unique_ptr<ExpertLayoutCapture>& out,std::string& error);
    // Borrow invalidated by a new capture attempt/destruction. A legacy mutation
    // invalidates this accessor too. This is NOT concurrent execution authority.
    const ExpertLayoutInventory* completed_inventory()const noexcept;
private:
    std::unique_ptr<ExpertLayoutInventory> inventory_;
    bool complete_=false;
    friend bool strata::kernels::cpu::expert_layout_load(const std::string&,int64_t,int64_t,std::string&,ExpertLayoutCapture*);
};
}
