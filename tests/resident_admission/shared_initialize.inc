SuffixRawResult SuffixRawTransaction::initialize_resident_shared_host(const SuffixRawTransaction& live) {
#if !defined(__linux__) || !defined(__GLIBCXX__) || defined(STRATA_GPU_METAL)
    return invalid("resident shared RAW setup requires Linux/libstdc++ native sessions");
#else
    if(this==&live||controller_!=std::this_thread::get_id()||controller_!=live.controller_||
       phase_!=Phase::idle||setup_credit_||initialized_||host_cap_||!live.initialized_||
       live.phase_!=Phase::drained||!live.host_cap_||&budget_!=&live.budget_||
       host_limit_!=live.host_limit_||!max_tokens_||max_tokens_>live.max_tokens_||
       !max_bytes_||max_bytes_>live.max_bytes_||steering_enabled_||live.steering_enabled_||
       !live.allocation_.healthy()||!identity_matches(live.identity_)||
       geometry_.n_layers<=0||geometry_.n_layers>4096)
        return invalid("resident RAW setup requires actual same-ledger drained host-cap owner");
    const auto& parent=allocation_.reservation();
    if(!parent||!parent->canonical_owner(parent)||!allocation_.healthy()||
       (parent->purpose()!=ReservationPurpose::private_work&&parent->purpose()!=ReservationPurpose::saved_maintenance))
        return invalid("resident RAW parent is not genuine private allocation");
    uint64_t tokens=0,layers=0,bytes=8ULL<<20;
    if(!product(max_tokens_,128,tokens)||!product(uint64_t(geometry_.n_layers),65536,layers)||
       !add(bytes,tokens)||!add(bytes,layers))return invalid("resident RAW setup quote overflow");
    try {
        host_cap_=live.host_cap_; // SAME actual H, never a per-runtime second cap
        auto admitted=admit_host(bytes,true,setup_credit_);if(!admitted){host_cap_.reset();return admitted;}
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error))return {SuffixRawStatus::invalid,std::move(error),{}};
        initial_config_=std::move(config);initialized_=true;return {SuffixRawStatus::ok,{},{}};
    }catch(const std::bad_alloc&){if(!setup_credit_)host_cap_.reset();return {SuffixRawStatus::budget_refused,"resident RAW shared setup allocation refused",{}};}
#endif
}

