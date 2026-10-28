    if (chain) {
        // The overlapped split: no host sync between the stages.  The next stage's GPU starts on this stage's flag;
        // this thread goes straight to serving the next stage's layers, and syncs this stage once the chain is done.
        if (!nl.join(err)) return false;
        nl.handed = true;
        ++windows;
        const bool ok = next_->run(T, tokens, pos0, pool, next_user_, out, err);
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
        // Host service and the downstream stage are done: join this stage's
        // launcher before certifying completion. A pending status() is not a
        // successful launch, and destructor cleanup must not release waits on
        // a successful chained window.
        const cudaError_t launch_result = launch_o ? launch_o->finish() : cudaSuccess;
        if (launch_result != cudaSuccess) {
            err = std::string("verify: launch: ") + cudaGetErrorString(launch_result);
            return false;
        }
#endif
        const OnDevice back(device_);
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
        if (prof_on_ && G == 1) collect_profile();
        return true;
    }
