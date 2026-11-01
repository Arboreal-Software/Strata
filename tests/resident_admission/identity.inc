bool SuffixRawTransaction::identity_matches(const WindowKey& key) const {
    return key.engine_generation==identity_.engine_generation && key.window_generation==identity_.window_generation &&
        key.session_id==identity_.session_id && key.thread_id==identity_.thread_id && key.window_id==identity_.window_id &&
        key.model_profile==identity_.model_profile && key.topology==identity_.topology;
}
