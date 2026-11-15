#pragma once
#include "strata/core/window_state.hpp"
#include "strata/core/window_budget.hpp"
#include <functional>
#include <memory>
#include <string>

namespace strata::core {
// Caller-provided policy bounds; no production quota is inferred by this API.
// max_host_bytes caps the sum of conservative retained and transient reservation
// envelopes for this store (including returned decoded bundles). It is not an
// exact allocator/RSS measurement. Every filesystem operation requires IO and
// workspace admission before its first filesystem effect, including recovery.
// The standalone full-digest utility has no store and uses fixed 64KiB workspace;
// its caller must separately admit that IO/workspace before invoking it.
struct WindowDiskLimits {
    uint64_t max_file_bytes=0, max_host_bytes=0, max_string_bytes=0, max_tokens=0;
    uint64_t max_tensors=0, max_regions=0, max_entries=0, io_block_bytes=0;
    uint32_t max_stages=0, max_rank=0;
};
struct WindowDiskOperation {
    uint64_t engine_generation=0, window_generation=0;
    StateDigest nonce{};
};
enum class WindowDiskPhase { unknown, not_published, published_durability_unknown, durable_artifact, catalog_durability_unknown, catalog_linked, retirement_durability_unknown, retired_artifact };
enum class WindowDiskError { none, invalid, budget_refused, cancelled, conflict, io, unsupported };
struct WindowDiskReceipt {
    WindowDiskPhase phase=WindowDiskPhase::not_published;
    std::string operation, artifact, catalog;
    StateDigest fingerprint{};
    uint64_t catalog_revision=0;
};
struct WindowDiskResult {
    WindowDiskError error=WindowDiskError::none;
    std::string detail;
    WindowDiskReceipt receipt;
    int system_error=0; // errno for filesystem/injected IO failures
    explicit operator bool() const { return error==WindowDiskError::none; }
};
struct WindowDiskOptions {
    // Synchronous callbacks: no re-entry into the store. Cancellation never rolls
    // back a published artifact or claims that a post-publication error had no effect.
    std::function<bool()> cancelled;
    // Tests/embedding: return errno to fail before a step; a callback may _exit
    // to simulate process death. Steps are documented in the task receipt.
    std::function<int(const char*)> fault;
};
struct WindowDiskHostUsage {uint64_t reserved_bytes=0, limit_bytes=0;};
struct WindowDiskState;
class WindowDiskArtifact {
public:
    explicit operator bool() const { return bool(owner_); }
    const std::string& id() const { return id_; }
private:
    friend class WindowDiskStore;
    std::string id_;
    std::shared_ptr<const void> owner_;
};
// POSIX cache-v2 namespace, single manager per configured root (process lock).
// Root is trusted caller configuration, already created; IDs are lowercase hex
// digests, never filenames/paths supplied by a client. Holding an artifact keeps
// the manager lock and disk accounting alive and prevents its deletion.
class WindowDiskStore {
public:
    WindowDiskHostUsage host_usage() const;
    static WindowDiskResult open(const std::string& root,const WindowDiskLimits&,WindowBudget,
                                 std::shared_ptr<WindowDiskStore>& out);
    WindowDiskResult save(const WindowDiskOperation&,const BundleHandle&,const WindowKey&,
                         const WindowStateLayout&,const StateDigest& catalog,uint64_t expected_revision,
                         WindowDiskArtifact& out,const WindowDiskOptions& = {});
    WindowDiskResult query(const WindowDiskOperation&,const WindowDiskOptions& = {});
    WindowDiskResult catalog(const StateDigest&,WindowDiskReceipt& out);
    WindowDiskResult acquire(const std::string& artifact,WindowDiskArtifact& out);
    WindowDiskResult load(const WindowDiskArtifact&,const WindowKey&,const WindowStateLayout&,
                         BundleHandle& out,const WindowDiskOptions& = {});
    // Refuses leased artifacts and artifacts referenced by ANY retained catalog
    // record. Catalog compaction/GC is a separate future transaction, not unlink-on-save.
    WindowDiskResult erase(const std::string& artifact,const WindowDiskOptions& = {});
private:
    explicit WindowDiskStore(std::shared_ptr<WindowDiskState> state):state_(std::move(state)){}
    std::shared_ptr<WindowDiskState> state_;
};
// Full file contents, not legacy first/last sampling. Caller composes model roles,
// engine configuration and these digests into trusted WindowKey::model_profile.
WindowDiskResult window_disk_full_digest(const std::string& path,uint64_t max_bytes,StateDigest& out,const WindowDiskOptions& = {});
} // namespace strata::core
