    const bool checked_retirement_; // Fixed before worker creation; default dispatch remains untracked.
    enum class DispatchState { idle, active, sealed, joined, failed };
    std::atomic<DispatchState> dispatch_state_{DispatchState::idle};
    // Failure is sticky: unwinding a published batch does not prove its borrowers completed.
    struct DispatchGuard {
        ExpertPool& pool;
        bool complete = false;
        explicit DispatchGuard(ExpertPool& p);
        ~DispatchGuard();
    };
    void run_impl(ExpertJob* jobs, int n);
    void run_split_impl(ExpertJob* jobs, int n);
    void run_split_multi_impl(ExpertJobMulti* jobs, int n);
    void run_split_multi_native_impl(const NativeFmt& f, ExpertJobMulti* jobs, int n);
