                if (!ver.run(T, window.data(), p, win_pool_fn, win_pool_user, outv.data(), err) || drive.d.failed) {
                    std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                    return 1;
                }
                if (aflipped) apply_pending(true);   // 0.1.43: every layer's copies were waited for in the window; the bookkeeping now
                // Diagnostics (as in generate): STRATA_DUMP_FIRST_LOGITS=<path> writes the logits of each request's first
                // window to <path>.<n>, so the prompt paths (a peer card's share, a split) can be compared by KL
                if (first_window) {
                    static const char* fl = std::getenv("STRATA_DUMP_FIRST_LOGITS");
                    static int fl_n = 0;
                    if (fl != nullptr) {
                        std::vector<float> row((size_t) ver.vocab());
                        const std::string path = std::string(fl) + "." + std::to_string(fl_n++);
                        std::FILE* f = ver.copy_logits(merged ? T - 1 : 0, row.data()) ? std::fopen(path.c_str(), "wb") : nullptr;
                        if (f == nullptr || std::fwrite(row.data(), sizeof(float), row.size(), f) != row.size())
                            std::fprintf(stderr, "strata serve: STRATA_DUMP_FIRST_LOGITS: cannot write %s\n", path.c_str());
                        if (f) std::fclose(f);
                    }
                }
                int a = merged ? T - 1 : 0;   // a merged window commits every row, as the prompt windows do
                while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
                if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
                if (chain_n > 0) { ++chain_windows; chain_drafts += chain_n; chain_ok += std::max(0, a - (T_mtp - 1)); }
                if (!from_sfx && !first_window)
                    for (int d = 0; d < T_mtp - 1 && d < 8 && d <= a; ++d) { ++dp_tested[d]; dp_ok[d] += d < a; }
                if (strata::core::MtpDrafter::top2_env() && !from_sfx && !first_window && a < T_mtp - 1 && a < 8) {
                    ++t2_rej[a];
                    if (mtp.top2(a) == outv[(size_t) a]) ++t2_hit[a];
                }
                const Clock::time_point tw1 = Clock::now();
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
