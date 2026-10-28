            if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)) {
                // The graph of a window size is captured the first time it is used, and the draft source took VRAM: when the capture runs
                // out of memory the window gets one row narrower (the widths below it are already captured) instead of the run failing.
                if (T > 2 && !first_window && err.find("out of memory") != std::string::npos) {
                    win_cap = T - 1;
                    --window_hist[(size_t) T];
                    std::fprintf(stderr, "strata generate: not enough VRAM left for a %d-token verify window: windows are %d tokens at most from here\n", T, win_cap);
                    err.clear();
                    goto retry_round;
                }
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            if (drive.d.failed) {
                std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                             (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                             drive.d.fail ? drive.d.fail : "(no message)");
                return 1;
            }
            if (hashes != nullptr) {   // FNV-1a over the window's final residual rows (a test path: a plain copy)
                const size_t n_floats = (size_t) T * (size_t) (g.hc * g.n_embd);
                hash_rows.resize(n_floats);
                cudaMemcpy(hash_rows.data(), ver.final_R_all(), n_floats * sizeof(float), cudaMemcpyDeviceToHost);
                uint64_t h = 1469598103934665603ull;
                const auto* bytes = (const uint8_t*) hash_rows.data();
                for (size_t i = 0; i < n_floats * sizeof(float); ++i) h = (h ^ bytes[i]) * 1099511628211ull;
                std::fprintf(hashes, "%lld %lld %d %016llx", (long long) rounds, (long long) p, T, (unsigned long long) h);
                for (int i = 0; i < T; ++i) std::fprintf(hashes, " %d", (int) outv[(size_t) i]);
                std::fprintf(hashes, "\n");
            }
            if (!follow.empty()) {   // the continuation stands in for the argmax
                const size_t k0 = produced.size();
                for (int i = 0; i < T && k0 + (size_t) i < follow.size(); ++i) {
                    fdiff[(size_t) i] = outv[(size_t) i] != (int32_t) follow[k0 + (size_t) i];
                    outv[(size_t) i] = (int32_t) follow[k0 + (size_t) i];
                }
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                // Diagnostics: the first window runs the prompt's last token over the state the prompt path left
                // (keys, values, recurrent state), so its logits carry whatever that path did. A native pack never
                // runs the per-token loop --dump-logits reads; this is where prompt paths can be compared by output.
                if (const char* fl = std::getenv("STRATA_DUMP_FIRST_LOGITS")) {
                    std::vector<float> row((size_t) ver.vocab());
                    std::FILE* f = ver.copy_logits(0, row.data()) ? std::fopen(fl, "wb") : nullptr;
                    if (f == nullptr || std::fwrite(row.data(), sizeof(float), row.size(), f) != row.size())
                        std::fprintf(stderr, "strata generate: STRATA_DUMP_FIRST_LOGITS: cannot write %s\n", fl);
                    if (f) std::fclose(f);
                }
            }
            // plan v0.3 P6: the adaptive tier's host work (ranking, copy submission) runs on its own thread while the
            // GPU commits and drafts; it touches only the residency tables, which nothing reads until the next window
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
