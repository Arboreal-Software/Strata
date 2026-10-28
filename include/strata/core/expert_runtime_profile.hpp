#pragma once
#include "strata/core/expert_load_inventory.hpp"
#include "strata/core/expert_source_observer.hpp"
#include "strata/core/verified_inputs.hpp"
namespace strata::core {
struct NativeRuntimeAttachment;
class FileExpertSource;
class ArenaExpertSource;
struct ExpertProfileSourceView {
    const FileExpertSource* file=nullptr;
    const ArenaExpertSource* arena=nullptr;
    std::shared_ptr<const VerifiedDriverBindings> bindings;
};
// Trusted runtime implementation must return its own retained source and actual
// input authority, never a caller-supplied receipt or unrelated owner. All calls
// require serialized startup with source close/global layout mutation excluded.
class ExpertRuntimeProfileBinding {
public:
    virtual ~ExpertRuntimeProfileBinding()=default;
    virtual ExpertProfileSourceView expert_profile_source() const noexcept=0;
};
struct ExpertRuntimeProfileLimits {
    uint64_t max_sources=0,max_layers=0,max_string_bytes=0,max_host_bytes=0;
};
// Historical successful load evidence. transfer/mapped_file_ordinal do not
// describe the current read mode: FileExpertSource may later drop mappings.
struct ExpertRuntimeProfileSpan {
    uint64_t layer=0,role=0,artifact=0,source_offset=0,source_stride=0;
    uint64_t bytes_per_expert=0,experts=0,destination_offset=0,destination_stride=0;
    uint64_t mapped_file_ordinal=UINT64_MAX;
    ExpertSourceTransfer transfer=ExpertSourceTransfer::mapped_packed;
};
class ExpertRuntimeProfile {
public:
    ExpertRuntimeProfile();
    ~ExpertRuntimeProfile();
    ExpertRuntimeProfile(const ExpertRuntimeProfile&)=delete;
    ExpertRuntimeProfile& operator=(const ExpertRuntimeProfile&)=delete;
    bool partial() const noexcept{return true;}
    // Borrowed views require serialization with source/layout changes for their
    // entire use, not merely while this check runs. No mutation lock is acquired.
    // Checks source/capture/layout authority, NOT current mapping/read mode.
    bool current() const noexcept;
    const std::vector<ProfileArtifact>* artifacts() const noexcept;
    const std::vector<ExpertRuntimeProfileSpan>* spans() const noexcept;
    // Exact original parsed layouts/sign vectors; accessor refuses stale state.
    const ExpertLayoutInventory* layout() const noexcept;
private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
    friend bool collect_expert_runtime_profile(std::shared_ptr<const NativeRuntimeAttachment>,
        const ExpertRuntimeProfileLimits&,WindowBudget,ExpertRuntimeProfile&,std::string&);
};
// Output MUST live outside retained runtime (otherwise a reference cycle).
// Failure leaves output unchanged. No policy readiness, complete model profile
// or execution permission is produced. No model payload reads/hashes/GPU calls.
bool collect_expert_runtime_profile(std::shared_ptr<const NativeRuntimeAttachment>,
    const ExpertRuntimeProfileLimits&,WindowBudget,ExpertRuntimeProfile&,std::string&);
}
