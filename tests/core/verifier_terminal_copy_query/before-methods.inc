bool Verifier::done(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
    if (!fl_active_ || fl_k_ < fl_total_) return false;
    const OnDevice on_device(device_);
    const cudaError_t q = cudaEventQuery(ev_done_);
    if (q == cudaErrorNotReady) {
        const double now = now_ms();
        if (now - fl_flush_ms_ > 2.0) { fl_flush_ms_ = now; (void) cudaStreamQuery(cs_); }
        if (now - fl_since_ms_ > 20000.0) {   // every layer served, and the graph still runs: #267 as above
            err = "verify: the window never finished" + released_note(release_gpu_waits(5000));
        }
        return false;
    }
    if (q != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(q); return false; }
    if (copy_used_) {   // no host function of this window may raise flag B in the next one
        if (cudaStreamQuery(copy_) == cudaErrorNotReady) return false;
        copy_used_ = false;
    }
    return true;
}
bool Verifier::pl_finish(int32_t* out, std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    fl_active_ = false;
    if (fl_prof_) accumulate_profile(prof_pin_);
    ++windows;
    trace_ev("DONE (pipelined)", -1, -1, (int64_t) (now_ms() - fl_launch_ms_));
    if (le_ < g.n_layers) return true;   // an earlier stage: the hand-off is written
    const int T = fl_T_;
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr)) {   // run()'s host-side sampling, Philox(seed, pos0 + t)
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) last_pos0_;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = "verify: the head sampling failed";
            return false;
        }
    }
    if (out != nullptr)
        for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    progress_beat();
    return true;
}
bool Verifier::prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
    if (fl_active_) { err = "verify: a window is in flight on this verifier"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (pl_ple_rows_.empty()) { err = "verify: pipelined window not prepared (capture_all)"; return false; }
    const Clock::time_point t0 = Clock::now();
    pl_stage(T, tokens, pos0, ple_prev);
    pl_prestaged_ = true;
    ms_host += ms_since(t0);
    return true;
}
bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}
bool Verifier::commit_slots(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
    std::vector<int> keep(slots_.size(), 0);
    for (int t = 0; t < last_t_; ++t) ++keep[(size_t) last_rows_[t]];
    return commit_slot_prefixes(keep.data(), err);
}
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
std::string Verifier::route_stats_report() {
    if (terminal_copy_query_failed_.load(std::memory_order_acquire)) return std::string();
    if (!rs_on_ || rs_dev_ == nullptr || g_ == nullptr) return std::string();
    const int64_t L = g_->n_layers;
    std::vector<unsigned long long> h((size_t) 9 * (size_t) L * 8);
    cudaMemcpy(h.data(), rs_dev_, h.size() * 8, cudaMemcpyDeviceToHost);
    std::string out;
    char b[160];
    for (int T = 1; T <= 8; ++T) {
        if (rs_win_[T] == 0) continue;
        double row[8] = {};
        for (int64_t l = 0; l < L; ++l)
            for (int r = 0; r < 8; ++r) row[r] += (double) h[((size_t) T * (size_t) L + (size_t) l) * 8 + (size_t) r];
        const double denom = (double) rs_win_[T] * (double) L;
        double sum = 0;
        for (int r = 0; r < T; ++r) sum += row[r] / denom;
        std::snprintf(b, sizeof b, "T%d (n=%lld): distinct experts per layer %.2f; new per row:", T, (long long) rs_win_[T], sum);
        out += b;
        for (int r = 0; r < T; ++r) {
            std::snprintf(b, sizeof b, " %.2f", row[r] / denom);
            out += b;
        }
        out += "\n";
    }
    cudaMemset(rs_dev_, 0, h.size() * 8);
    std::memset(rs_win_, 0, sizeof rs_win_);
    return out;
}
