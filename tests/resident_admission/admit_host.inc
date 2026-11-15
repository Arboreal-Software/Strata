SuffixRawResult SuffixRawTransaction::admit_host(uint64_t bytes,bool global_charge,std::shared_ptr<HostCredit>& out) {
    if(!host_cap_ || !bytes || bytes>SIZE_MAX)return invalid("invalid host cap claim");
    auto credit=std::make_shared<HostCredit>();
    // Atomics permit final bundle/claim destruction off the controller thread;
    // admissions themselves remain private and single-controller-thread only.
    unsigned slots=host_cap_->claims.load();
    do {
        if(slots>=host_cap_->claim_limit)return {SuffixRawStatus::budget_refused,"fixed host claim slots exhausted",{}};
    } while(!host_cap_->claims.compare_exchange_weak(slots,slots+1));
    uint64_t live=host_cap_->live.load();
    do {
        if(live>host_cap_->limit || bytes>host_cap_->limit-live) {
            host_cap_->claims.fetch_sub(1);
            return {SuffixRawStatus::budget_refused,"aggregate suffix H exhausted",{}};
        }
    } while(!host_cap_->live.compare_exchange_weak(live,live+bytes));
    credit->cap=host_cap_;credit->bytes=bytes;
    if(global_charge) {
        WindowResources r;r.device_bytes.resize(allocation_.reservation()->resources().device_bytes.size());r.host_bytes=bytes;
        auto reserved=budget_.reserve(r,ReservationPurpose::private_attachment,allocation_.reservation());
        if(!reserved)return {SuffixRawStatus::budget_refused,"actual global host attachment refused (including wrong ledger)",{}};
        credit->global=std::move(reserved.lease);
    }
    out=std::move(credit);return {SuffixRawStatus::ok,{},{}};
}
