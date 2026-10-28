bool Verifier::prepare_serial_completion_gate(Verifier& root,
    std::shared_ptr<const ReservationLease> metadata,
    std::shared_ptr<VerifierCompletionGate>& output,std::string& error) {
    // Bind before init can publish a stage to the watchdog registry. Caller
    // excludes topology mutation; allocation failure leaves stages untouched.
    if(output||!metadata||metadata.use_count()!=1||
       metadata->purpose()!=ReservationPurpose::foreground)return false;
    const auto& charge=metadata->resources();
    if(charge.host_bytes<VerifierCompletionGate::metadata_bytes()||charge.pinned_bytes||
       charge.disk_bytes||charge.io_slots||charge.shadows)return false;
    for(uint64_t bytes:charge.device_bytes)if(bytes)return false;
    std::array<Verifier*,VerifierCompletionGate::max_stages> stages{};
    std::size_t count=0;
    for(Verifier* p=&root;p;p=p->next_) {
        if(count==stages.size()||p->retirement_init_started_||!p->strict_retirement_||p->completion_gate_||
           p->released_.load(std::memory_order_acquire)||
           p->terminal_copy_query_failed_.load(std::memory_order_acquire))return false;
        for(std::size_t i=0;i<count;++i)if(stages[i]==p)return false;
        stages[count++]=p;
    }
    try {
        auto gate=std::shared_ptr<VerifierCompletionGate>(new VerifierCompletionGate);
        gate->metadata_=std::move(metadata);
        gate->count_=count;
        for(std::size_t i=0;i<count;++i)gate->identities_[i]=stages[i];
        for(std::size_t i=0;i<count;++i) {
            stages[i]->completion_stage_=i;stages[i]->completion_gate_=gate;
        }
        output=std::move(gate);error.clear();return true;
    }catch(const std::bad_alloc&){return false;}
}

