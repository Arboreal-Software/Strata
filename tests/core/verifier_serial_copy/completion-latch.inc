    void latch_copy_query_failure()noexcept {
        mark_completion_unhealthy();
        terminal_copy_query_failed_.store(true,std::memory_order_release);
    }
