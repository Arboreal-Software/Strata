                std::thread adapt_thr;   // the adaptive tier beside the commit and the draft (as in generate)
                bool adapt_ok = true;
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
                // ver.run finished before this serial scope. A two-window fence
                // would need caller-driven progress and must never use this join.
                if(o.owned_session_lifetime && adapt_fence) {
                    std::fputs("strata: serial adaptation retains a pipeline fence; worker restart required\n",stderr);
                    std::_Exit(1);
                }
                strata::core::JoinedThreadScope adapt_join(o.owned_session_lifetime,adapt_thr);
#endif
                // (--adapt-async 1: the asynchronous tier above instead, ticked before the window)
                if (!drive.d.usage.empty() && !ajob && ((rounds + 1) % o.adapt_iv(rounds)) == 0)
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
