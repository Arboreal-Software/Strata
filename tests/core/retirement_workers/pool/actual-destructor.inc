ExpertPool::~ExpertPool() {
    g_pool_readers.retire(this);   // drain readers before member mutation and worker joins
    if (g_numa_owner == this) {
        g_numa_plan = NumaSplitPlan{};
        g_numa_owner = nullptr;
    }
    if (checked_retirement_) {
        if (!stop_after_dispatch_callers_joined()) std::terminate();
    } else {
        stop_.store(true, std::memory_order_release);
        // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
        publish();
        for (auto& t : threads_) t.join();
    }
}
