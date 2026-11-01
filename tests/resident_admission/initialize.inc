SuffixRawResult SuffixRawTransaction::initialize() {
#if !defined(__linux__) || !defined(__GLIBCXX__) || defined(STRATA_GPU_METAL)
    return invalid("initial standalone RAW helper requires Linux/libstdc++ native sessions");
#else
    if(controller_!=std::this_thread::get_id() || phase_!=Phase::idle || setup_credit_ || !max_tokens_ || !max_bytes_ || !host_limit_ ||
       max_tokens_>SIZE_MAX || max_bytes_>SIZE_MAX || geometry_.n_layers<=0 || geometry_.n_layers>4096) return invalid("invalid setup limits/phase");
    if(!identity_.engine_generation || !identity_.window_generation || identity_.session_id.empty() ||
       identity_.thread_id.empty() || identity_.window_id.empty() || identity_.session_id.size()>4096 ||
       identity_.thread_id.size()>4096 || identity_.window_id.size()>4096 ||
       !present(identity_.model_profile) || !present(identity_.topology))return invalid("missing fixed actual runtime identity");
    const auto& parent=allocation_.reservation();
    if(!parent || !parent->canonical_owner(parent) ||
       (parent->purpose()!=ReservationPurpose::private_work &&
        parent->purpose()!=ReservationPurpose::saved_maintenance))
        return invalid("runtime parent must be genuine private_work/saved_maintenance");
    // Setup baseline admission precedes layout/validation/container metadata.
    // It is intentionally retained for the entire controller lifetime.
    uint64_t tokens=0,layers=0,bytes=8ULL<<20;
    if(!product(max_tokens_,128,tokens) || !product(uint64_t(geometry_.n_layers),65536,layers) ||
       !add(bytes,tokens) || !add(bytes,layers)) return invalid("setup quote overflow");
    try {
        // Bounded bootstrap metadata is included in the setup quote, analogous
        // to WindowBudget's precharge lease/control-block transients.
        if(bytes>host_limit_) return {SuffixRawStatus::budget_refused,"H cannot cover setup metadata",{}};
        host_cap_=std::make_shared<HostCap>(host_limit_);
        auto admitted=admit_host(bytes,true,setup_credit_);if(!admitted)return admitted;
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error)) return {SuffixRawStatus::invalid,std::move(error),{}};
        initial_config_=std::move(config);initialized_=true;
        return {SuffixRawStatus::ok,{},{}};
    } catch(const std::bad_alloc&) {return {SuffixRawStatus::budget_refused,"setup allocation refused",{}};}
#endif
}
