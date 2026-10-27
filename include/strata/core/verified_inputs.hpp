#pragma once
#include "strata/core/verified_artifact.hpp"
#include <vector>
namespace strata::core {
struct VerifiedInputFile {
    std::string source; // normalized absolute path, trusted parent directories
    uint64_t expected_bytes=0;
    StateDigest expected_digest{};
    std::vector<std::string> aliases; // relative to private loader view
};
struct VerifiedPackClosure {
    std::string pack_directory; // relative directory with index.txt
    std::string primary_gguf; // required for native_experts.txt GGUF references
};
struct VerifiedInputManifest {
    std::vector<VerifiedInputFile> files;
    std::vector<std::vector<std::string>> gguf_groups; // explicit ordered aliases
    std::vector<VerifiedPackClosure> packs;
};
struct VerifiedInputLimits {
    uint64_t max_files=0,max_aliases=0,max_path_bytes=0,max_depth=0,max_metadata_bytes=0;
    uint64_t view_host_bytes=0,view_disk_bytes=0;
    VerifiedArtifactLimits artifact;
};
struct VerifiedInputGroup;
class VerifiedInputLease {
public:
    explicit operator bool() const{return bool(group_);}
    std::string loader_path() const;
    int readonly_fd() const;
    uint64_t size() const;
    StateDigest content_digest() const;
    ProfileArtifact profile_artifact(ProfileRole,std::string name) const;
private:
    friend class VerifiedInputBundle;
    std::shared_ptr<const VerifiedInputGroup> group_;
    size_t index_=0;
    std::string alias_;
};
// Every lease retains the actual whole group: snapshots, directory and outer
// reservations. No naked snapshot-owner extraction. Trusted same-process only.
enum class VerifiedInputRetirement {retired,busy_leases,budget_refused,busy_retiring};
class VerifiedInputBundle {
public:
    VerifiedInputBundle(const VerifiedInputBundle&)=delete;
    VerifiedInputBundle& operator=(const VerifiedInputBundle&)=delete;
    static VerifiedArtifactResult prepare(const VerifiedInputManifest&,const std::string& private_directory,
        const VerifiedInputLimits&,WindowBudget,std::shared_ptr<const VerifiedInputBundle>& out,
        const VerifiedArtifactOptions& = {});
    size_t alias_count() const; // immutable membership count, no owner exposure
    bool resolve(const std::string& alias,VerifiedInputLease& out) const;
    bool resolve_directory(const std::string& relative,VerifiedInputLease& out) const;
    // Close/join all loaders/producers and drop their leases first. Seals new
    // resolution; normal IO-capacity refusal preserves resources for retry.
    // Success must be explicitly retired before last bundle owner is dropped.
    VerifiedInputRetirement retire_checked() const;
    bool retired() const;
private:
    VerifiedInputBundle()=default;
    std::shared_ptr<VerifiedInputGroup> group_;
};
} // namespace strata::core
