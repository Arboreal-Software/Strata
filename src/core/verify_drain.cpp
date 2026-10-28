#include "strata/core/verify.hpp"
#include "strata/core/window_device.hpp"

namespace strata::core {
bool Verifier::wait_commit(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
#if defined(STRATA_GPU_METAL)
    (void) err;
    return true;
#endif
    if (!commit_failure_.empty()) { err = commit_failure_; return false; }
    if (commit_pending_) {
        WindowDeviceScope on_device(device_);
        if (!on_device.ok()) { err = commit_failure_ = "verify: commit device selection failed"; return false; }
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = commit_failure_ = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
        if (!on_device.finish()) { err = commit_failure_ = "verify: commit caller-device restoration failed"; return false; }
        commit_pending_ = false;
    }
    return next_ == nullptr || next_->wait_commit(err);
}

Verifier::SnapshotDrain Verifier::poll_snapshot_drain(std::string& err) {
    if (refuse_copy_query_failure(err)) return SnapshotDrain::unhealthy;
    if (batch_copy_failed_) { err = "verify batch: previous copy-stream failure; reconstruct verifier"; return SnapshotDrain::unhealthy; }
#if defined(STRATA_GPU_METAL)
    err = "verify: snapshot drain is not implemented for Metal";
    return SnapshotDrain::unsupported;
#else
    if (!snapshot_failure_.empty()) { err = snapshot_failure_; return SnapshotDrain::unhealthy; }
    if (!commit_failure_.empty()) {
        err = snapshot_failure_ = commit_failure_;
        return SnapshotDrain::unhealthy;
    }
    if (released_.load()) {
        err = snapshot_failure_ = "verify: released GPU waits invalidate snapshot state; restart required";
        return SnapshotDrain::unhealthy;
    }
    if (!cs_) { err = "verify: snapshot drain before initialization"; return SnapshotDrain::unsupported; }
    WindowDeviceScope on_device(device_);
    if (!on_device.ok()) {
        err = snapshot_failure_ = "verify: snapshot device selection failed";
        return SnapshotDrain::unhealthy;
    }
    bool pending = false;
    auto observe = [&](cudaError_t status) {
        if (status == cudaErrorNotReady) { pending = true; return true; }
        if (status == cudaSuccess) return true;
        err = snapshot_failure_ = std::string("verify: snapshot drain: ") + cudaGetErrorString(status);
        return false;
    };
    // Query all owned work, even if an earlier stream is pending: a later error
    // must not be hidden behind a stream that cannot finish after that error.
    for (cudaStream_t stream : {cs_, sh_cs_, copy_, side_, side2_, df_side_[0], df_side_[1]})
        if (stream && !observe(cudaStreamQuery(stream))) return SnapshotDrain::unhealthy;
    if (commit_pending_) {
        const cudaError_t status = cudaEventQuery(commit_done_);
        if (!observe(status)) { commit_failure_ = snapshot_failure_; return SnapshotDrain::unhealthy; }
        if (status == cudaSuccess) commit_pending_ = false;
    }
    if (!pending && handoff_failed(err)) { snapshot_failure_ = err; return SnapshotDrain::unhealthy; }
    if (next_) {
        const SnapshotDrain next = next_->poll_snapshot_drain(err);
        if (next == SnapshotDrain::unhealthy) { snapshot_failure_ = err; return next; }
        if (next == SnapshotDrain::unsupported) return next;
        pending = pending || next == SnapshotDrain::pending;
    }
    if (!on_device.finish()) {
        err = snapshot_failure_ = "verify: snapshot caller-device restoration failed";
        return SnapshotDrain::unhealthy;
    }
    if (pending) return SnapshotDrain::pending;
    copy_used_ = false;
    return SnapshotDrain::drained;
#endif
}


} // namespace strata::core
