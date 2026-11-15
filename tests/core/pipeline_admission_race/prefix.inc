            // A terminal input failure forbids new units; draining an already
            // submitted pipeline group remains required and permitted.
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
            // Owned admission is checked once by pump before uncertainty starts.
            // Rechecking here could turn a clean racing rejection into an
            // uncertain submission and prevent draining already admitted groups.
            if(!o.owned_session_lifetime)
#endif
            if(may_start && in_shared->input_error.load())return false;
