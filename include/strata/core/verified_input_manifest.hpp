#pragma once
#include "strata/core/verified_inputs.hpp"
#include <map>
#include <array>
namespace strata::core {
struct VerifiedManifestParseLimits {
    uint64_t max_bytes=0,max_records=0,max_line_bytes=0,max_field_bytes=0;
};
struct VerifiedDriverOverrides {
    // Only explicitly requested original paths; keys are the BIND role names.
    // Group roles list sources in exact shard order. Unknown keys are refused.
    std::map<std::string,std::vector<std::string>> paths;
};
class ParsedVerifiedInputs;
class ExpectedRunningExecutable;
class VerifiedDriverBindings {
public:
    VerifiedDriverBindings()=default;
    VerifiedDriverBindings(const VerifiedDriverBindings&)=delete;
    VerifiedDriverBindings& operator=(const VerifiedDriverBindings&)=delete;
    VerifiedDriverBindings(VerifiedDriverBindings&&) noexcept=default;
    VerifiedDriverBindings& operator=(VerifiedDriverBindings&&) noexcept=default;
    ProfileFrontend frontend=ProfileFrontend::raw_token_ids; // TSV-v1 initial capability
    std::string pack_path() const{return pack.loader_path();}
    std::vector<std::string> native_paths() const;
    std::vector<std::string> dense_paths() const;
    std::vector<std::string> head_paths() const;
    // All path strings are backed by these actual retained leases. Keep this
    // object until loader maps/workers retire; release before bundle retirement.
    VerifiedInputLease pack,native_primary,head_primary,embedding,ple,expert_profile;
    // Actual immutable pack files for joining executed loader source offsets.
    // Payload slots follow WeightTable file IDs: dense, embd, experts, extra.
    // An undeclared/unused file has an empty slot; never substitute the directory.
    VerifiedInputLease pack_index;
    std::array<VerifiedInputLease,4> pack_payloads;
    VerifiedInputLease native_expert_layout,hadamard;
    // Authenticated manifest membership, not an inference from a failed open.
    // Actual parser validity and effective runtime state remain loader evidence.
    bool native_expert_layout_absent=true,hadamard_absent=true;
    std::vector<VerifiedInputLease> native,dense,head;
    bool embedding_from_native=false,ple_disabled=false,head_disabled=false,expert_profile_disabled=true;
    // Admits bounded header IO; verifies prepaid reader allowance before real profile load.
    VerifiedArtifactResult preflight_expert_profile(WindowBudget,const VerifiedArtifactOptions& = {});
    StateDigest manifest_digest{};
    bool confirms_expert_profile_disabled() const noexcept;
private:
    friend class ParsedVerifiedInputs;
    friend class ExpertProfileCapture;
    bool expert_profile_disabled_minted_=false;
    std::shared_ptr<const ParsedVerifiedInputs> plan_;
    std::shared_ptr<const ReservationLease> expert_profile_workspace_;
};
// Bounded strict TSV-v1/v2 parsing; expected full manifest SHA binds exact local
// configuration, not signer authentication. No snapshot writes during read.
class ParsedVerifiedInputs : public std::enable_shared_from_this<ParsedVerifiedInputs> {
public:
    ParsedVerifiedInputs(const ParsedVerifiedInputs&)=delete;
    ParsedVerifiedInputs& operator=(const ParsedVerifiedInputs&)=delete;
    static VerifiedArtifactResult read(const std::string& path,const StateDigest& expected,
        const VerifiedManifestParseLimits&,WindowBudget,std::shared_ptr<const ParsedVerifiedInputs>& out,
        const VerifiedArtifactOptions& = {});
    const VerifiedInputManifest& manifest() const{return manifest_;}
    const VerifiedInputLimits& limits() const{return limits_;}
    const StateDigest& digest() const{return digest_;}
    // Only schema3 MAIN_IMAGE mints this private expectation. Legacy schema or
    // metadata admission/allocation refusal returns null; no borrowed fake claim.
    bool requires_running_executable_observation() const noexcept {return main_image_present_;}
    std::shared_ptr<const ExpectedRunningExecutable> expected_running_executable(WindowBudget) const noexcept;
    // Call BEFORE bundle preparation. No original-path fallback or unlisted
    // caller override. Failure leaves bundle/output binding untouched.
    VerifiedArtifactResult validate_overrides(const VerifiedDriverOverrides&) const;
    VerifiedArtifactResult prepare_bundle(const std::string& private_directory,WindowBudget,
        const VerifiedDriverOverrides&,std::shared_ptr<const VerifiedInputBundle>& out,
        const VerifiedArtifactOptions& = {}) const;
    VerifiedArtifactResult bind(const std::shared_ptr<const VerifiedInputBundle>&,
        const VerifiedDriverOverrides&,VerifiedDriverBindings& out) const;
private:
    friend class ExpertProfileCapture;
    ParsedVerifiedInputs()=default;
    bool main_image_present_=false;
    uint64_t main_image_bytes_=0;
    StateDigest main_image_digest_{};
    VerifiedInputManifest manifest_;
    VerifiedInputLimits limits_;
    StateDigest digest_{};
    std::map<std::string,std::vector<std::string>> groups_;
    std::map<std::string,std::string> binds_,original_by_alias_;
    std::shared_ptr<const ReservationLease> parse_owner_,profile_reader_owner_;
};
} // namespace strata::core
