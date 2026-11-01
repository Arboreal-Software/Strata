    bool set_strict_retirement(bool enabled, const std::vector<int>&, std::string& error) {
        if (!enabled) return true;
        error = "strict remote expert retirement is not supported by the SYCL engine";
        return false;
    }
