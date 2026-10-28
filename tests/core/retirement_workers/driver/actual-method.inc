bool DriverWindowRuntime::close_background_after_callers_joined() noexcept {
    unsigned expected=0;
    if(!background_retirement.compare_exchange_strong(expected,1,std::memory_order_acq_rel))
        return expected==2;
    bool closed=false;
    try {
        // No new predictions may reach the swap source or dispatch pool once
        // their workers are stopped. Keep every buffer and callback target alive.
        closed=(!lookahead||lookahead->close_for_retirement())&&
               (!foresight||foresight->close_for_retirement())&&
               (!pool||pool->stop_after_dispatch_callers_joined())&&
               (!source||source->stop_background_after_callers_joined());
        if(closed)for(const auto& stage:sessions.stages()){
            if(stage.session&&stage.session->ple.read_context&&
               !stage.session->ple.read_context->close_for_retirement()){
                closed=false;break;
            }
        }
    } catch(...) {closed=false;}
    background_retirement.store(closed?2u:3u,std::memory_order_release);
    return closed;
}
