    std::mutex background_close_mu_;
    std::atomic<bool> background_sealed_{false};
    std::atomic<bool> background_close_failed_{false};
    bool background_joined_ = false; // background_close_mu_
