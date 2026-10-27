#pragma once
#include "strata/core/window_lease.hpp"
#include "strata/core/window_native.hpp"
#include <mutex>

namespace strata::core {
class Verifier;
struct SessionState;
struct NativeCaptureStage {
    int device = -1;
    const SessionState* session = nullptr;
    Verifier* verifier = nullptr;
};
enum class NativeCaptureStatus { captured, pending, invalid, budget_refused, unhealthy };
struct NativeCaptureStorage;
struct NativeCaptureResult;
class NativeCaptureFailure {
public:
    // Only after affected device contexts are torn down and host/IO producers
    // joined. An error code or a later successful query is not teardown evidence.
    void retire_after_teardown();
private:
    friend NativeCaptureResult capture_native_window(WindowRead, const ModelGeometry&,
        const std::vector<NativeCaptureStage>&, const SnapshotDescriptor&, uint64_t, uint64_t,
        WindowBudget&, BundleHandle&);
    std::shared_ptr<NativeCaptureStorage> storage_;
    std::mutex mutex_;
};
struct NativeCaptureResult {
    NativeCaptureStatus status = NativeCaptureStatus::invalid;
    std::string detail;
    std::shared_ptr<NativeCaptureFailure> failure{};
    explicit operator bool() const { return status == NativeCaptureStatus::captured; }
};

// Synchronous RAW capture into private host storage. The read hold prevents
// execution resume for the entire call. All supplied sessions/verifiers must be
// owned by that hold's allocation owner. The scheduler supplies metadata and must
// have joined host producers before acquiring the hold. No live state is written.
// A successful bundle owns its peak reservation until its final reference dies.
// No speculation, multimodal state or exact-continuation state is supported yet.
NativeCaptureResult capture_native_window(WindowRead read, const ModelGeometry&,
                                          const std::vector<NativeCaptureStage>&,
                                          const SnapshotDescriptor& metadata,
                                          uint64_t max_tokens, uint64_t max_payload_bytes,
                                          WindowBudget&, BundleHandle& out);
} // namespace strata::core
