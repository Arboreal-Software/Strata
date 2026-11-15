#pragma once
#include "strata/core/window_model_owner.hpp"
#include "strata/core/verified_input_manifest.hpp"
namespace strata::core {
struct LoaderProfileLimits {
    uint64_t max_stages=0,max_artifacts=0,max_tensors=0,max_rank=0,max_string_bytes=0,max_host_bytes=0;
};
// Deliberately partial: these copied artifacts/tensors cover only the actual
// canonical/native/head/embed loaders. They do not cover experts, table PLE,
// runtime policy, engine/frontend, or establish a complete WindowProfileIdentity.
// Retains actual loader owners and genuine input leases through profile joining.
class WindowLoaderProfile {
public:
    const std::vector<ProfileArtifact>& artifacts() const noexcept{return artifacts_;}
    const std::vector<ProfileTensor>& tensors() const noexcept{return tensors_;}
    // These indices describe index metadata, never loaded execution tensors.
    const std::vector<uint64_t>& metadata_only_tensor_indices() const noexcept{return metadata_only_;}
    bool partial() const noexcept{return true;}
    explicit operator bool() const noexcept{return bool(gang_);}
private:
    std::shared_ptr<const ReservationLease> reservation_;
    std::shared_ptr<const OwnedModelGang> gang_;
    std::shared_ptr<const VerifiedDriverBindings> bindings_;
    std::vector<ProfileArtifact> artifacts_;
    std::vector<ProfileTensor> tensors_;
    std::vector<uint64_t> metadata_only_;
    friend bool collect_window_loader_profile(std::shared_ptr<const OwnedModelGang>,
        std::shared_ptr<const VerifiedDriverBindings>,const LoaderProfileLimits&,
        WindowBudget,WindowLoaderProfile&,std::string&);
};
// Trusted startup/reactor call AFTER adoption, with all loader/policy mutation
// and retirement excluded. No files/GPU/environment reads. Never takes a raw
// inventory plus an unrelated retaining-owner claim. Output unchanged on error;
// old output destruction occurs after validation with no registry lock held.
bool collect_window_loader_profile(std::shared_ptr<const OwnedModelGang>,
    std::shared_ptr<const VerifiedDriverBindings>,const LoaderProfileLimits&,
    WindowBudget,WindowLoaderProfile&,std::string&);
}
