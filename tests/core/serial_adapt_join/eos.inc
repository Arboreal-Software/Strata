            if (eos) {
                if (adapt_thr.joinable()) adapt_thr.join();
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
                if(o.owned_session_lifetime && !adapt_ok) {
                    std::fputs("strata generate: an adaptive refill failed before EOS\n",stderr);
                    return 1;
                }
#endif
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
