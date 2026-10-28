bool Verifier::finish_serial_copies(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
    if (!copy_used_) return true;
    // No host function of this window may raise flag B in the next one.
    const cudaError_t status = cudaStreamSynchronize(copy_);
    if (status != cudaSuccess) {
        // Preserve outstanding-work flags even if diagnostic allocation throws.
        // Cleanup remains callable; a successful later sync cannot heal this state.
        latch_copy_query_failure();
        err = std::string("verify: serial copy-stream synchronization failed: ") + cudaGetErrorString(status);
        return false;
    }
    copy_used_ = false;
    return true;
}

bool Verifier::terminal(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    if (!finish_serial_copies(err)) return false;
    commit_pending_ = false;
    return true;
}
bool Verifier::chain(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (!finish_serial_copies(err)) return false;
        if (!ok) return false;
        if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
        if (handoff_failed(err)) return false;   // a middle stage of three or more: its own hand-in
        if (ar_on() && h_plan_err_ != nullptr && *(volatile uint32_t*) h_plan_err_ != 0) {   // #871, as below
            *(volatile uint32_t*) h_plan_err_ = 0;
            err = "verify: the all-resident plan met an expert that is not in VRAM (the residency table changed during the window)";
            return false;
        }
        commit_pending_ = false;
    return true;
}
