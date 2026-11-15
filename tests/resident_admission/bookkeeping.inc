struct SuffixRawTransaction::HostCap {
    explicit HostCap(uint64_t n):limit(n) {}
    const uint64_t limit;
    unsigned claim_limit=16; // fixed main-only metadata lane; genuine two-slot sharing derives48
    std::atomic<uint64_t> live{0};
    // Fixed admission bound; no growing claim registry or unpriced directory.
    std::atomic<unsigned> claims{0};
};
bool SuffixRawTransaction::observe_host_usage(HostUsage& out) const noexcept {
    if(controller_!=std::this_thread::get_id()||phase_!=Phase::drained||pending_||!initialized_||!host_cap_)return false;
    // All claim producers must already be synchronously joined by this owner.
    // Atomic observations alone are NOT a producer-close or completion proof.
    HostUsage value{host_cap_->limit,host_cap_->live.load(std::memory_order_acquire),
                    host_cap_->claims.load(std::memory_order_acquire)};
    if(value.live>value.limit)return false;
    out=value;return true;
}
struct SuffixRawTransaction::HostCredit {
    std::shared_ptr<HostCap> cap;
    std::shared_ptr<const ReservationLease> global;
    uint64_t bytes=0;
    ~HostCredit() {
        // Relevant payloads must already be destroyed by their owning wrapper.
        global.reset();
        if(cap) {
            if(cap->live.fetch_sub(bytes)<bytes || cap->claims.fetch_sub(1)==0) std::terminate();
        }
    }
};
