SuffixRawResult SuffixRawTransaction::admit_resident_setup(const WindowResources& resources,
    std::shared_ptr<const void>& out) {
    if(out || controller_!=std::this_thread::get_id() || !initialized_ ||
       phase_!=Phase::drained || !allocation_.healthy() || !allocation_.reservation() ||
       resources.device_bytes.size()!=allocation_.reservation()->resources().device_bytes.size() ||
       !resources.host_bytes || resources.pinned_bytes>resources.host_bytes ||
       resources.disk_bytes || resources.io_slots || resources.shadows)
        return invalid("resident setup requires drained owner and bounded setup resources");
    try {
        // Aggregate first, global second; neither allocation payload nor CUDA
        // work exists here. Failure drops the temporary aggregate claim.
        std::shared_ptr<HostCredit> credit;
        auto admitted=admit_host(resources.host_bytes,false,credit);
        if(!admitted)return admitted;
        auto reserved=budget_.reserve(resources,ReservationPurpose::private_attachment,
                                      allocation_.reservation());
        if(!reserved)return {SuffixRawStatus::budget_refused,"resident global setup admission refused",{}};
        credit->global=std::move(reserved.lease);
        out=std::move(credit);
        return {SuffixRawStatus::ok,{},{}};
    } catch(const std::bad_alloc&) {
        return {SuffixRawStatus::budget_refused,"resident setup admission allocation refused",{}};
    }
}
SuffixRawResult SuffixRawTransaction::admit_completion_gate_metadata(
    std::shared_ptr<const ReservationLease>& metadata,
    std::shared_ptr<const void>& aggregate_credit) {
    if(metadata || aggregate_credit || controller_!=std::this_thread::get_id() ||
       !initialized_ || phase_!=Phase::drained || !allocation_.healthy() ||
       !allocation_.reservation())
        return invalid("completion metadata requires actual drained resident controller");
    try {
        const auto bytes=VerifierCompletionGate::metadata_bytes();
        std::shared_ptr<HostCredit> credit;
        auto admitted=admit_host(bytes,false,credit);
        if(!admitted)return admitted;
        WindowResources resources;
        resources.device_bytes.resize(allocation_.reservation()->resources().device_bytes.size());
        resources.host_bytes=bytes;
        auto reserved=budget_.reserve(resources,ReservationPurpose::foreground);
        if(!reserved)return {SuffixRawStatus::budget_refused,"completion metadata global admission refused",{}};
        // Keep the canonical lease exclusive for prepare_serial_completion_gate.
        // The aggregate claim is a separate retained lifetime owner, not a second
        // global charge or an alias that defeats canonical-owner validation.
        metadata=std::move(reserved.lease);aggregate_credit=std::move(credit);
        return {SuffixRawStatus::ok,{},{}};
    } catch(const std::bad_alloc&) {
        return {SuffixRawStatus::budget_refused,"completion metadata allocation refused",{}};
    }
}
