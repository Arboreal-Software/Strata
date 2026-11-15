#pragma once
#include "strata/core/window_budget.hpp"
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class VerifiedDriverBindings;
struct ExpertProfileLimits { uint64_t max_pairs=0,max_path_bytes=0; };
// Caller derives this claim from the retained verified profile lease. The reader
// checks its actual opened descriptor; path equality alone is not authority.
struct ExpertProfileClaims {
    std::string expected_profile_path;
    uint64_t device=0,inode=0,bytes=0;
};
enum class ExpertProfileOrigin : uint8_t { file, verified_disabled };
struct ExpertProfileInventory {
    std::shared_ptr<const ReservationLease> metadata_reservation;
    std::shared_ptr<const VerifiedDriverBindings> retained_input_bindings;
    ExpertProfileLimits limits;
    ExpertProfileClaims claims;
    std::array<uint32_t,5> header{}; // Actual version, geometry, requested slots, pair count.
    std::vector<std::pair<int32_t,int32_t>> ranked; // Actual order, including duplicates.
    uint64_t device=0,inode=0,bytes=0,consumed_bytes=0;
    ExpertProfileOrigin origin=ExpertProfileOrigin::file;
    // Only verified_disabled uses these trusted driver geometry arguments.
    // File captures continue to expose geometry from their actual file header.
    uint64_t disabled_layers=0,disabled_experts=0;
    uint64_t layers()const noexcept{return origin==ExpertProfileOrigin::file?header[1]:disabled_layers;}
    uint64_t experts()const noexcept{return origin==ExpertProfileOrigin::file?header[2]:disabled_experts;}
    // This is the file request, NOT the caller's effective post-trim policy.
};
inline bool expert_profile_inventory_host_bound(const ExpertProfileLimits& l,uint64_t& out,std::string& error) {
    if(!l.max_pairs||l.max_pairs>(1ull<<24)||!l.max_path_bytes||l.max_path_bytes>4096) {
        error="profile inventory limits outside supported bounds";return false;
    }
    // Receipt + staged returned ranking + raw uint16 pairs, allocator/stdio and
    // bounded diagnostic/path copies. The lease is acquired before any growth.
    out=65536+sizeof(ExpertProfileInventory)+8*l.max_path_bytes+l.max_pairs*32;
    return true;
}
class ExpertProfileCapture {
public:
    // Defined with verified bindings implementation; file-only users do not gain
    // a mandatory link dependency. Requires a distinct foreground metadata lease.
    static bool create_disabled(uint64_t layers,uint64_t experts,
        std::shared_ptr<const ReservationLease>,std::shared_ptr<const VerifiedDriverBindings>,
        std::unique_ptr<ExpertProfileCapture>&,std::string&);
    static bool create(const ExpertProfileLimits& limits,std::shared_ptr<const ReservationLease> lease,
        std::shared_ptr<const VerifiedDriverBindings> bindings,const ExpertProfileClaims& claims,
        std::unique_ptr<ExpertProfileCapture>& out,std::string& error) {
        uint64_t required=0;
        if(!expert_profile_inventory_host_bound(limits,required,error))return false;
        if(!lease||lease->purpose()!=ReservationPurpose::foreground||lease->resources().host_bytes<required||
           !bindings||claims.expected_profile_path.empty()||claims.expected_profile_path.size()>limits.max_path_bytes) {
            error="profile capture requires admitted metadata and retained input claims";return false;
        }
        try {
            auto c=std::make_unique<ExpertProfileCapture>();
            c->inventory_.metadata_reservation=std::move(lease);
            c->inventory_.retained_input_bindings=std::move(bindings);
            c->inventory_.limits=limits;c->inventory_.claims=claims;
            c->inventory_.ranked.reserve(size_t(limits.max_pairs));
            out=std::move(c);return true;
        } catch(const std::bad_alloc&) {error.clear();return false;}
    }
    // Borrow expires at the next read attempt or capture destruction. Caller
    // retains this owner through final-policy provenance publication/use.
    uint64_t attempt_generation()const noexcept{return attempt_generation_;}
    const ExpertProfileInventory* completed_inventory()const noexcept {
        if(!complete_)return nullptr;
        if(inventory_.origin==ExpertProfileOrigin::verified_disabled &&
           (attempt_generation_!=1||!disabled_claim_current_||!inventory_.retained_input_bindings||
            !disabled_claim_current_(*inventory_.retained_input_bindings)))return nullptr;
        return &inventory_;
    }
private:
    ExpertProfileInventory inventory_;
    bool (*disabled_claim_current_)(const VerifiedDriverBindings&) noexcept=nullptr;
    bool complete_=false;
    uint64_t attempt_generation_=0;
    friend bool read_expert_profile(const std::string&,int64_t,int64_t,
        std::vector<std::pair<int32_t,int32_t>>&,int64_t&,std::string&,ExpertProfileCapture*);
};
inline uint64_t expert_profile_disabled_host_bound() noexcept {
    return 4096+sizeof(ExpertProfileCapture)+sizeof(ExpertProfileInventory);
}
inline bool expert_profile_disabled_host_bound(uint64_t layers,uint64_t experts,uint64_t& out,std::string& error) {
    if(!layers||!experts||layers>uint64_t(INT32_MAX)||experts>uint64_t(INT32_MAX)||layers>uint64_t(INT64_MAX)/experts){
        error="disabled profile geometry outside supported representation";return false;
    }
    out=expert_profile_disabled_host_bound();return true;
}
}
