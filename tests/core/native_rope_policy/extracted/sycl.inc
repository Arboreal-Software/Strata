bool rope_table_enabled() {
    static const bool on = [] {
        const char* e = fake_getenv("STRATA_ROPE_TABLE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}
void native_rope_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_rope_enabled() { return enabled.load(std::memory_order_relaxed); }
bool native_norm_rope_usable(int head_dim, int n_rot) {
    static const bool on = [] {
        const char* off = fake_getenv("STRATA_NO_NORM_ROPE");
        if (off != nullptr && off[0] != '\0' && off[0] != '0') return false;
#if defined(__HIPCC__)
        const char* v = fake_getenv("STRATA_NORM_ROPE");   // AMD: opt in until rope_parity check 6 passes there
        return v != nullptr && v[0] != '\0' && v[0] != '0';
#else
        return true;
#endif
    }();
    return on && (head_dim == 128 || head_dim == 256) && n_rot == 64;
}
NativeRopePolicyObservation native_rope_policy_observation(int head_dim, int n_rot) {
    return {true, head_dim, n_rot, native_rope_enabled(), rope_table_enabled(),
            native_norm_rope_usable(head_dim, n_rot)};
}
