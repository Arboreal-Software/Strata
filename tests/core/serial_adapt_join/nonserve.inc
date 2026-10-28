            std::thread adapt_thr;
            bool adapt_ok = true;
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
            // This serial verifier run has finished; no host-pumped adapt fence
            // exists in the non-serving path. Outcome outlives this guard.
            strata::core::JoinedThreadScope adapt_join(o.owned_session_lifetime,adapt_thr);
#endif
            if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_iv(rounds)) == 0)
                adapt_thr = std::thread([&] {
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
                    if(o.owned_session_lifetime) {
                        try { strata::platform::name_this_thread("strata-adapt"); strata::aux_cpus::pin_current_thread(); adapt_ok=adapt(); }
                        catch(...) { adapt_ok=false; }
                        return;
                    }
#endif
                    strata::platform::name_this_thread("strata-adapt"); strata::aux_cpus::pin_current_thread(); adapt_ok=adapt();
                });
