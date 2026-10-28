        bool pipe_completion_uncertain = false;
        auto pump = [&](bool may_start) -> bool {
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
            if(o.owned_session_lifetime) {
                if(pipe_completion_uncertain)return false;
                // This rejection precedes any submitted-work effect. Keep
                // pump(false) available to finish previously admitted groups.
                if(may_start && in_shared->input_error.load())return false;
                // Set before entering: exceptions and false results both make
                // completion uncertain. Teardown must not retry their effects.
                pipe_completion_uncertain = true;
                const bool ok = pump_step(may_start);
                if(ok)pipe_completion_uncertain = false;
                return ok;
            }
#endif
            return pump_step(may_start);
        };
        auto pipe_drain = [&]() -> bool {
            while (pipe_inflight())
                if (!pump(false)) return false;
            return true;
        };
