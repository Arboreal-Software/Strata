ExpertPool::DispatchGuard::DispatchGuard(ExpertPool& p) : pool(p) {
    auto expected = DispatchState::idle;
    if (!pool.dispatch_state_.compare_exchange_strong(expected, DispatchState::active,
            std::memory_order_acq_rel))
        throw std::logic_error("expert pool dispatch is active, stopped, or failed");
}

ExpertPool::DispatchGuard::~DispatchGuard() {
    pool.dispatch_state_.store(complete ? DispatchState::idle : DispatchState::failed,
                               std::memory_order_release);
}

bool ExpertPool::stop_after_dispatch_callers_joined() noexcept {
    if (!checked_retirement_) return false;
    auto expected = DispatchState::idle;
    if (!dispatch_state_.compare_exchange_strong(expected, DispatchState::sealed,
            std::memory_order_acq_rel))
        return expected == DispatchState::joined;
    // Never set stop while a dispatch still needs workers to complete its phases.
    try {
        stop_.store(true, std::memory_order_release);
        publish();
        for (auto& t : threads_) if (t.joinable()) t.join();
    } catch (...) {
        dispatch_state_.store(DispatchState::failed, std::memory_order_release);
        return false;
    }
    dispatch_state_.store(DispatchState::joined, std::memory_order_release);
    return true;
}
