    void mark_completion_unhealthy()noexcept {
        if(completion_gate_)completion_gate_->fail_stage(completion_stage_);
    }
    void latch_copy_query_failure()noexcept {
        mark_completion_unhealthy();
        terminal_copy_query_failed_.store(true,std::memory_order_release);
    }
