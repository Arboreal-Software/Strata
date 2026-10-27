#pragma once
#include "strata/core/window_budget.hpp"
#include "strata/core/window_hash.hpp"
#include "strata/core/window_profile.hpp"
#include <functional>
#include <memory>
#include <string>
namespace strata::core {
struct VerifiedArtifactLimits {
    uint64_t max_file_bytes=0,block_bytes=0,metadata_host_bytes=0,snapshot_disk_overhead_bytes=0;
    bool allow_full_copy=false;
};
struct VerifiedArtifactOptions {
    // Synchronous non-reentrant callbacks; no borrowed descriptor access.
    std::function<bool()> cancelled;
    std::function<void(uint64_t,uint64_t)> progress;
    std::function<int(const char*)> fault;
};
enum class VerifiedArtifactError {none,unsupported,invalid,budget_refused,cancelled,io,hash_mismatch};
struct VerifiedArtifactResult {
    VerifiedArtifactError error=VerifiedArtifactError::none;
    int system_error=0;
    std::string detail;
    explicit operator bool() const{return error==VerifiedArtifactError::none;}
};
enum class ArtifactSnapshotMethod {reflink,full_copy};
// Private source-independent snapshot. API-level immutability in a trusted
// process, NOT kernel seals or protection against same-UID/root chmod/reopens.
// Consumer must retain this owner until ALL loader FDs/mappings/producers retire.
class VerifiedArtifact {
public:
    static VerifiedArtifactResult snapshot(const std::string& source,const std::string& private_directory,
        const StateDigest& expected,const VerifiedArtifactLimits&,WindowBudget,
        std::shared_ptr<const VerifiedArtifact>& out,const VerifiedArtifactOptions& = {});
    ~VerifiedArtifact();
    VerifiedArtifact(const VerifiedArtifact&)=delete;
    VerifiedArtifact& operator=(const VerifiedArtifact&)=delete;
    uint64_t size() const{return bytes_;}
    const StateDigest& content_digest() const{return content_;}
    const StateDigest& verification_digest() const{return verification_;}
    ArtifactSnapshotMethod snapshot_method() const{return method_;}
    // Borrowed, not caller-owned. Never close/chmod/write/link/reopen writable.
    int readonly_fd() const{return fd_;}
    // Same-process Linux bridge only; no shard-name inference, exec inheritance
    // or portable URI. Keep shared owner while ANY use of this string survives.
    std::string loader_path() const;
private:
    VerifiedArtifact()=default;
    int fd_=-1;
    uint64_t bytes_=0;
    StateDigest content_{},verification_{};
    ArtifactSnapshotMethod method_=ArtifactSnapshotMethod::full_copy;
    std::shared_ptr<const ReservationLease> retained_;
};
// Builds attestation fields from the actual snapshot owner. Loader consumption
// of this owner is still the caller's responsibility. Throws on null owner.
ProfileArtifact verified_profile_artifact(ProfileRole,std::string name,std::shared_ptr<const VerifiedArtifact>);
} // namespace strata::core
