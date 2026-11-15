bool Verifier::terminal(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
    commit_pending_ = false;
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    if (copy_used_) {
        cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
        copy_used_ = false;
    }
    return true;
}
bool Verifier::chain(std::string& err) {
    if (refuse_copy_query_failure(err)) return false;
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (copy_used_) {
            cudaStreamSynchronize(copy_);
            copy_used_ = false;
        }
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
