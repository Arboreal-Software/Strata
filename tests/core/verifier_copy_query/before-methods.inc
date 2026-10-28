bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = base + t;
    if (pre_launch_ && !pre_launch_(cs_, err)) return false;
    if (!stage_batch(rows, S, base, tokens, pos, err)) return false;
    cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(rows, S, base)], cs_);
    if (le == cudaSuccess && !batch_self_commits(rows, S))   // (else the window has kept its rows itself)
        le = cudaGraphLaunch(commit_bm_[bkey(rows, S, base)], cs_);   // right behind it: every row is kept
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    ring_now();
    b_running_.store(true, std::memory_order_release);   // last: an external server reads the rest after it
    return true;
}
int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    // set_external_serve: another thread serves the doorbells (batch_serve) and this one only checks for a window that
    // will not finish, as below: its graph ended (or failed) with a layer still to ring, or no layer rang for 20 s.
    // The stream query can wait for the driver's lock while a read's call holds it; the doorbells go on meanwhile.
    // The served count is read before the query and the ring count after it: a layer served in between rang first.
    if (ext_serve_.load(std::memory_order_acquire) && !ar_on()) {
        const int64_t k = b_k_.load(std::memory_order_acquire);
        if (k < b_steps_) {
            const auto now = Clock::now();
            if (now - last_ring() <= std::chrono::milliseconds(2) || now - ext_checked_ < std::chrono::milliseconds(1))
                return 0;
            ext_checked_ = now;   // (at most one query a millisecond: pump calls this in a loop)
            const OnDevice on_device(device_);
            const cudaError_t q = cudaStreamQuery(cs_);
            if (q != cudaErrorNotReady && *(volatile uint32_t*) h_seq_ < (uint32_t) (k + 1)) {
                err = "verify batch: layer " + std::to_string(lb_ + k) + " never rang (" +
                      (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                b_running_ = false;
                return -1;
            }
            if (now - last_ring() > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(lb_ + k) +
                      released_note(release_gpu_waits(5000));
                b_running_ = false;
                return -1;
            }
            return 0;
        }
    }
    const OnDevice on_device(device_);
    const int S = last_t_;
    // the layers that have rung so far (batch_serve); with one still to ring: did the graph finish without it, or
    // has it waited too long?
    if (!batch_serve(pool, user)) {
        const auto now = Clock::now();
        if (now - last_ring() > std::chrono::milliseconds(2)) {
            const cudaError_t q = cudaStreamQuery(cs_);
            if (q != cudaErrorNotReady && *(volatile uint32_t*) h_seq_ < (uint32_t) (b_k_ + 1)) {
                err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (" +
                      (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                b_running_ = false;
                return -1;
            }
            if (now - last_ring() > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_) +
                      released_note(release_gpu_waits(5000));
                b_running_ = false;
                return -1;
            }
        }
        return 0;
    }
    const cudaError_t q = cudaStreamQuery(cs_);
    if (q == cudaErrorNotReady) return 0;
    if (q != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(q); b_running_ = false; return -1; }
    const cudaError_t qc = cudaStreamQuery(copy_);   // no host function of this window may raise flag B in the next
    if (qc == cudaErrorNotReady) return 0;
    if (prof_on_) collect_profile();
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
        note_nonfinite(S, last_pos_b_);
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}
Verifier::SnapshotDrain Verifier::poll_snapshot_drain(std::string& err) {
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
