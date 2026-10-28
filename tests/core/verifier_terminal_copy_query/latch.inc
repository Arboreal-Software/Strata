    std::atomic<bool> terminal_copy_query_failed_{false};
    bool refuse_copy_query_failure(std::string& err) const {
        if (!terminal_copy_query_failed_.load(std::memory_order_acquire)) return false;
        err = "verify: previous terminal copy-stream query failure; reconstruct verifier";
        return true;
    }

