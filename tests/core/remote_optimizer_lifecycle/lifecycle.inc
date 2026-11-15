bool RemoteExpertOpt::set_strict_retirement(bool enabled, const std::vector<int>& consumers,
                                          std::string& error) {
    if (lifecycle_started_) { error = "remote optimizer lifecycle already started"; return false; }
    if (enabled) {
        if (consumers.empty()) { error = "remote optimizer requires complete consumer topology"; return false; }
        for (size_t i = 0; i < consumers.size(); ++i) {
            if (consumers[i] < 0 || std::find(consumers.begin(), consumers.begin() + i,
                                            consumers[i]) != consumers.begin() + i) {
                error = "invalid remote optimizer consumer topology"; return false;
            }
        }
    }
    // Strong allocation boundary: no helper or device effects before commit.
    std::vector<int> topology(enabled ? consumers : std::vector<int>{});
    std::vector<Peer> peers;
    if (enabled) peers.reserve(3);
    retirement_consumers_.swap(topology);
    peers_.swap(peers);
    strict_retirement_ = enabled;
    return true;
}

bool RemoteExpertOpt::attach_checked(RemoteExperts& remote, std::string& error) {
    if (!strict_retirement_) { error = "checked attach requires strict retirement"; return false; }
    lifecycle_started_ = true;
    if (init_attempted_ || retirement_closed_ || peers_.size() == 3 || remote.remote_opt_ ||
        std::any_of(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; })) {
        error = "remote optimizer attach refused"; return false;
    }
    // Capacity was prepaid before any attach. Publish the borrowed backpointer
    // only after the owning record exists.
    peers_.push_back({&remote});
    remote.remote_opt_ = this;
    return true;
}

void RemoteExpertOpt::attach(RemoteExperts& remote) {
    lifecycle_started_ = true;
    if (strict_retirement_) {
        std::string error;
        if (!attach_checked(remote, error)) retirement_failed("remote optimizer attach");
        return;
    }
    remote.remote_opt_ = this;
    peers_.push_back({&remote});
}

bool RemoteExpertOpt::init(std::string& err) {
    lifecycle_started_ = true;
    if (strict_retirement_) {
        if (init_attempted_ || retirement_closed_) { err = "remote optimizer init refused"; return false; }
        init_attempted_ = true;
        int previous = -1, count = 0;
        retirement_check(true, cudaGetDevice(&previous), "optimizer init get device");
        retirement_check(true, cudaGetDeviceCount(&count), "optimizer init device count");
        const auto covered = [&](int device) {
            return device >= 0 && device < count &&
                std::find(retirement_consumers_.begin(), retirement_consumers_.end(), device) != retirement_consumers_.end();
        };
        if (!covered(previous) || std::any_of(retirement_consumers_.begin(), retirement_consumers_.end(),
                                             [&](int d) { return d >= count; })) {
            err = "remote optimizer allocation device missing from topology"; return false;
        }
        for (const auto& p : peers_) {
            if (!covered(p.remote->device_) || p.remote->remote_opt_ != this) {
                err = "remote optimizer helper missing from topology"; return false;
            }
        }
        for (size_t i = 0; i < peers_.size(); ++i)
            peer_allocation_devices_[i] = peers_[i].remote->device_;
        allocation_device_ = previous;
        struct Restore {
            int device;
            ~Restore() { retirement_check(true, cudaSetDevice(device), "optimizer init restore device"); }
        } restore{previous};
        void* mapped = nullptr;
        if (!check(cudaHostAlloc((void**) &h_sum_, MAXT * (H + K) * sizeof(float),
                                 cudaHostAllocPortable | cudaHostAllocMapped), err) ||
            !check(cudaHostGetDevicePointer(&mapped, h_sum_, 0), err)) return false;
        m_sum_ = (float*) mapped;
        h_mask_ = reinterpret_cast<int32_t*>(h_sum_ + MAXT * H);
        m_mask_ = reinterpret_cast<int32_t*>(m_sum_ + MAXT * H);
        for (size_t i = 0; i < peers_.size(); ++i) {
            retirement_check(true, cudaSetDevice(peer_allocation_devices_[i]), "optimizer init select helper");
            if (!check(cudaMalloc((void**) &peers_[i].sum, MAXT * H * sizeof(float)), err)) return false;
        }
        return true;
    }
    void* mapped = nullptr;
    if (!check(cudaHostAlloc((void**) &h_sum_, MAXT * (H + K) * sizeof(float),
                             cudaHostAllocPortable | cudaHostAllocMapped), err) ||
        !check(cudaHostGetDevicePointer(&mapped, h_sum_, 0), err)) return false;
    m_sum_ = (float*) mapped;
    h_mask_ = reinterpret_cast<int32_t*>(h_sum_ + MAXT * H);
    m_mask_ = reinterpret_cast<int32_t*>(m_sum_ + MAXT * H);
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (!check(cudaMalloc((void**) &p.sum, MAXT * H * sizeof(float)), err)) return false;
    }
    return true;
}

bool RemoteExpertOpt::close_after_callers_joined() noexcept {
    if (!strict_retirement_) return false;
    if (retirement_closed_) return true;
    lifecycle_started_ = true;
    const bool allocated = h_sum_ || std::any_of(peers_.begin(), peers_.end(),
                                                [](const Peer& p) { return p.sum != nullptr; });
    for (size_t i = 0; i < peers_.size(); ++i) {
        if (peers_[i].remote->remote_opt_ != this ||
            (allocated && peers_[i].remote->device_ != peer_allocation_devices_[i]))
            retirement_failed("optimizer close helper identity/device");
    }
    if (allocated) {
        int previous = -1;
        retirement_check(true, cudaGetDevice(&previous), "optimizer close get device");
        // Gang barrier: no allocation or borrowed target is released until ALL
        // consumers (including primary mapped readers) have completed.
        for (int device : retirement_consumers_) {
            retirement_check(true, cudaSetDevice(device), "optimizer close select consumer");
            retirement_check(true, cudaDeviceSynchronize(), "optimizer close drain consumer");
        }
        for (const auto& p : peers_) {
            retirement_check(true, cudaSetDevice(p.remote->device_), "optimizer close select stream device");
            if (p.remote->stream_)
                retirement_check(true, cudaStreamSynchronize(p.remote->stream_), "optimizer close helper stream");
        }
        for (size_t i = 0; i < peers_.size(); ++i) {
            if (!peers_[i].sum) continue;
            retirement_check(true, cudaSetDevice(peer_allocation_devices_[i]), "optimizer close select allocation device");
            retirement_check(true, cudaFree(peers_[i].sum), "optimizer close free helper sum");
            peers_[i].sum = nullptr;
        }
        if (h_sum_) {
            retirement_check(true, cudaSetDevice(allocation_device_), "optimizer close select mapped device");
            retirement_check(true, cudaFreeHost(h_sum_), "optimizer close free mapped sum");
            h_sum_ = nullptr;
        }
        retirement_check(true, cudaSetDevice(previous), "optimizer close restore device");
    }
    for (auto& p : peers_) {
        if (p.remote->remote_opt_ != this) retirement_failed("optimizer close helper identity");
        p.remote->remote_opt_ = nullptr;
    }
    m_sum_ = nullptr; h_mask_ = m_mask_ = nullptr; weights_ = nullptr;
    retirement_closed_ = true;
    return true;
}

RemoteExpertOpt::~RemoteExpertOpt() {
    if (strict_retirement_) {
        if (!close_after_callers_joined()) retirement_failed("remote optimizer close");
        return;
    }
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (p.remote->stream_) cudaStreamSynchronize(p.remote->stream_);
        if (p.sum) cudaFree(p.sum);
        p.remote->remote_opt_ = nullptr;
    }
    if (h_sum_) cudaFreeHost(h_sum_);
}

