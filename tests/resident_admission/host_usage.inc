    struct HostUsage { uint64_t limit=0,live=0;unsigned claims=0; };
    bool observe_host_usage(HostUsage&) const noexcept;
