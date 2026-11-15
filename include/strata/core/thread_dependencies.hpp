#pragma once
#include "strata/core/thread_priority.hpp"
#include <optional>
#include <set>

namespace strata::core {
struct OperationKey {
    uint64_t engine_generation=0, serial=0, result_revision=0;
    bool operator==(const OperationKey&) const = default;
};
enum class DependencyWorkClass { foreground, live_repair, saved_maintenance };
enum class DependencyState { pending, complete, failed, cancelled, dependency_failed };
enum class DependencyError { none, stale_version, unknown_generation, invalid_argument,
    limit_exceeded, cycle, terminal, invalid_time, stale_priority, allocation_failed };
struct DependencyLimits { size_t max_operations, max_edges; };
struct DependencyChange { OperationKey consumer, prerequisite; bool add=true; };
struct DependencyOperation {
    OperationKey key;
    ThreadPrioritySnapshot thread;
    DependencyWorkClass work_class=DependencyWorkClass::foreground;
    DependencyState state=DependencyState::pending;
    int64_t effective_priority=0;
    OperationKey priority_source;
    ThreadKey priority_source_thread;
    bool dependencies_satisfied=false; // NOT resource/work-class/executor eligibility
    std::vector<OperationKey> prerequisites;
};
struct DependencySnapshot { uint64_t revision=0; std::vector<DependencyOperation> operations; };
struct DependencyResult {
    DependencyError error=DependencyError::none;
    uint64_t revision=0;
    OperationKey key;
    explicit operator bool() const { return error==DependencyError::none; }
};

// Bounded local operation/result graph, not the external context DAG. The ordered
// control path owns this object and ThreadPriorities; refresh on timer expiry and
// before future execution units. Registry mutations and refresh must be serialized
// by that path to obtain a common control cut. Snapshots preserve accepted/applied
// priority revisions; accepting graph changes is not an executor acknowledgement.
//
// Operation serials are minted, never reused, and refuse exhaustion. Terminal
// records can be retired without tombstones. A completion is a trusted reactor
// report, not proof of GPU/IO drain; runtime lifetime leases remain independent.
class ThreadDependencies {
public:
    ThreadDependencies(uint64_t engine_generation, DependencyLimits, ThreadPriorities&);
    DependencyResult register_operation(const ThreadKey&, uint64_t result_revision,
        DependencyWorkClass, uint64_t expected_graph_revision, uint64_t now_ms);
    DependencyResult update_dependencies(const std::vector<DependencyChange>&, uint64_t expected_graph_revision);
    DependencyResult finish(const OperationKey&, DependencyState outcome, uint64_t expected_graph_revision);
    DependencyResult retire(const OperationKey&, uint64_t expected_graph_revision);
    DependencyResult refresh_priorities(uint64_t now_ms);
    DependencySnapshot snapshot() const;
private:
    friend struct ThreadDependenciesTestAccess;
    struct Node {
        OperationKey key;
        ThreadPrioritySnapshot thread;
        DependencyWorkClass work_class;
        DependencyState state=DependencyState::pending;
        std::set<uint64_t> prerequisites;
    };
    using Nodes=std::map<uint64_t,Node>;
    bool matches(const Nodes&,const OperationKey&) const;
    static bool order(const Nodes&,std::vector<uint64_t>&);
    static void propagate_failures(Nodes&);
    DependencyResult result(DependencyError error=DependencyError::none,OperationKey key={}) const;
    bool can_change(uint64_t expected) const;
    uint64_t engine_,revision_=0,next_serial_=1;
    DependencyLimits limits_;
    ThreadPriorities& priorities_;
    Nodes nodes_;
    mutable std::mutex mutex_;
};
} // namespace strata::core
