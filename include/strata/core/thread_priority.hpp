#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace strata::core {

struct ThreadKey {
    uint64_t engine_generation = 0, generation = 0;
    std::string id;
    bool operator==(const ThreadKey&) const = default;
};
struct PriorityLimits {
    int32_t min_base, max_base, max_activity_boost;
    uint64_t max_activity_ttl_ms;
    size_t max_threads, max_activity_leases, max_id_bytes;
};
struct ActivityPriority {
    std::string lease_id;
    uint64_t version = 0, expires_at_ms = 0;
    int32_t boost = 0;
    bool active = false;
    uint64_t incarnation = 0; // zero denotes the legacy lease API
};
struct ActivityKey { ThreadKey thread; std::string lease_id; uint64_t incarnation=0; };
struct PriorityLifecycle { uint64_t retired_generation_floor=0, next_thread_generation=0, next_actor_incarnation=0; };
struct ThreadPrioritySnapshot {
    ThreadKey key;
    int32_t base = 0;
    int64_t local_effective = 0; // dependency inheritance is applied by the work graph
    uint64_t base_version = 0, accepted_revision = 0, applied_revision = 0, applied_cut = 0;
    bool closed = false;
    std::vector<ActivityPriority> activity;
};
enum class PriorityError { none, unknown_generation, stale_version, invalid_priority, invalid_time,
                           limit_exceeded, closed, invalid_acknowledgement };
struct PriorityResult {
    PriorityError error = PriorityError::none;
    ThreadPrioritySnapshot snapshot;
    explicit operator bool() const { return error == PriorityError::none; }
};

// Control-plane registry only: no GPU preemption, work-class promotion or final
// selection policy is implied. Successful mutation responses are fully built
// before the requested operation commits. Allocation failure leaves that requested
// mutation (including version/mint counters) unapplied. Advancing monotonic time
// and expiring leases happens independently BEFORE validation/response allocation;
// those expiry effects are NOT rolled back on a later failure. Error snapshots
// may allocate. This is not durable reply delivery or wire-command idempotency.
// One bounded response snapshot plus a pending actor/map node may transiently
// overlap live registry metadata; no whole-registry clone is performed.
// The engine's ordered control path supplies time
// in monotonic milliseconds and acknowledges revisions at admitted unit cuts.
class ThreadPriorities {
public:
    ThreadPriorities(uint64_t engine_generation, PriorityLimits);
    PriorityResult register_thread(const ThreadKey&, int32_t base, uint64_t now_ms);
    // Managed thread generations are minted here, independently of external DAG/window generations.
    PriorityResult create_thread(const std::string& id, int32_t base, uint64_t now_ms);
    // Retirement advances a global floor: delayed new creates at/below it are refused.
    // Existing live records remain valid. Use create_thread for race-free new generations.
    PriorityResult retire_closed_thread(const ThreadKey&, uint64_t expected_revision, uint64_t now_ms);
    // A zero next-generation/incarnation means permanent counter exhaustion.
    PriorityLifecycle lifecycle() const;
    // Opt into reusable actor slots. Managed APIs require the complete incarnation handle.
    // Registration is revision-fenced; retirement requires an inactive matching lease.
    PriorityResult register_activity(const ThreadKey&, const std::string& id, uint64_t expected_revision, uint64_t now_ms);
    PriorityResult set_activity(const ActivityKey&, uint64_t expected_lease_version, int32_t boost, uint64_t ttl_ms, uint64_t now_ms);
    PriorityResult disconnect(const ActivityKey&, uint64_t expected_lease_version, uint64_t now_ms);
    PriorityResult retire_activity(const ActivityKey&, uint64_t expected_lease_version, uint64_t now_ms);
    PriorityResult set_base(const ThreadKey&, uint64_t expected_base_version, int32_t base, uint64_t now_ms);
    PriorityResult set_activity(const ThreadKey&, const std::string& lease_id, uint64_t expected_lease_version,
                                int32_t boost, uint64_t ttl_ms, uint64_t now_ms);
    PriorityResult disconnect(const ThreadKey&, const std::string& lease_id, uint64_t expected_lease_version, uint64_t now_ms);
    PriorityResult snapshot(const ThreadKey&, uint64_t now_ms);
    PriorityResult acknowledge(const ThreadKey&, uint64_t revision, uint64_t safe_cut, uint64_t now_ms);
    PriorityResult close(const ThreadKey&, uint64_t expected_revision, uint64_t now_ms);
    // Called by the control timer even with no new requests. Expiry cannot rely
    // on another user action arriving to undo a boost.
    bool advance_time(uint64_t now_ms);
    // Allocation-free timer hint across ALL registered actors, including threads
    // without dependency operations. May wake early after refresh/disconnect,
    // never later than the earliest active expiry. UINT64_MAX is a valid value;
    // nullopt means no pending timer. On wake call advance_time(now), then read
    // again. This accessor neither advances time nor acknowledges a revision.
    std::optional<uint64_t> next_activity_expiry_ms() const;
private:
    friend struct ThreadPriorityTestAccess;
    struct Record {
        ThreadPrioritySnapshot view;
        int32_t registered_base = 0;
        std::map<std::string,ActivityPriority> leases;
        bool managed_actors = false;
    };
    bool advance_locked(uint64_t);
    PriorityResult register_locked(const ThreadKey&, int32_t base);
    PriorityResult set_activity_locked(Record&, const std::string&, uint64_t version, int32_t boost, uint64_t ttl, uint64_t now);
    PriorityResult disconnect_locked(Record&, const std::string&, uint64_t version, uint64_t now);
    PriorityResult result(const Record&, PriorityError = PriorityError::none) const;
    Record* find(const ThreadKey&);
    bool valid_id(const std::string&) const;
    bool bump(Record&);
    uint64_t engine_generation_, now_ms_ = 0;
    uint64_t next_expiry_ms_ = UINT64_MAX;
    bool expiry_pending_ = false;
    PriorityLimits limits_;
    std::map<std::string,Record> threads_;
    uint64_t retired_generation_floor_=0, highest_thread_generation_=0, next_actor_incarnation_=1;
    mutable std::mutex mutex_;
};
} // namespace strata::core
