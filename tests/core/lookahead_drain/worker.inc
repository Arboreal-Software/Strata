void RouterLookahead::run() try {
    { std::lock_guard<std::mutex> lock(mu_); worker_id_=std::this_thread::get_id(); }
    std::vector<float> logits((size_t) (8 * n_expert_));
    std::vector<int32_t> order((size_t) n_expert_);
    std::vector<int64_t> want;
    std::vector<float> wconf;   // 149-prefetch: per wanted expert, its best rank's confidence (warm_scored)
    for (;;) {
        int64_t layer, nt;
        const int32_t* host_res;
        ForesightSwap* fs;
        uint64_t sgen;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || pending_; });
            if (quit_) return;
            pending_ = false;
            busy_ = true;
            layer = layer_;
            nt = n_tok_;
            host_res = has_residency_ ? residency_snapshot_.data() : nullptr;
            fs = fs_;
            sgen = sub_gen_;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t layer0 = layer;
        const int depth = src_ != nullptr && !src_->read_ahead_open() ? 1 : depth_;   // 149: the gate closed
        for (int dj = 0; dj < depth && layer0 + dj < (int64_t) routers_.size(); ++dj) {
        layer = layer0 + dj;
        want.clear();
        wconf.clear();
        const bool scored = src_->wants_scores();
        // The router dot is AVX2 (kq_avx2.cpp); a CPU without AVX2 (the experimental older-CPU builds) takes the AVX1
        // one (kq_avx1.cpp) or, without AVX, the same sums in plain C++.  From the Strata_Dirigo fork (rwkeyes): an
        // AVX-only Xeon E5-2687W died here (vpmovzxwd) on its first request.  An estimate only (which experts to
        // prefetch); the order of the additions differs between the three, the output does not depend on it.
        if (strata::kernels::cpu::cpu_avx2_ok()) {
            strata::kernels::cpu::bf16_rows_dot_multi(routers_[(size_t) layer].data(), (int) n_expert_, (int) n_embd_,
                                                      x_.data(), (int) nt, logits.data());
        } else if (strata::kernels::cpu::cpu_avx1_ok()) {
            strata::kernels::cpu::bf16_rows_dot_multi_avx1(routers_[(size_t) layer].data(), (int) n_expert_,
                                                           (int) n_embd_, x_.data(), (int) nt, logits.data());
        } else {
            // a mul and an add, not std::fma: without an FMA instruction that is a libm call per element (74x
            // slower on a Xeon E5-2665, measured by the fork)
            const uint16_t* rw = routers_[(size_t) layer].data();
            for (int64_t r = 0; r < n_expert_; ++r) {
                const uint16_t* wr = rw + (size_t) r * (size_t) n_embd_;
                for (int64_t t = 0; t < nt; ++t) {
                    const float* xr = x_.data() + (size_t) t * (size_t) n_embd_;
                    float acc = 0.0f;
                    for (int64_t c = 0; c < n_embd_; ++c) {
                        const uint32_t bits = (uint32_t) wr[c] << 16;
                        float wf;
                        std::memcpy(&wf, &bits, sizeof wf);
                        acc += wf * xr[c];
                    }
                    logits[(size_t) t * (size_t) n_expert_ + (size_t) r] = acc;
                }
            }
        }
        const int kmax = stats_on_ ? (int) std::min<int64_t>(k_ + 10, n_expert_) : k_;
        const int ks[3] = {k_, (int) std::min<int64_t>(k_ + 4, n_expert_), kmax};
        std::vector<uint64_t> sm;
        if (stats_on_) sm.assign((size_t) (3 * words_), 0);
        for (int64_t t = 0; t < nt; ++t) {
            const float* lt = logits.data() + (size_t) (t * n_expert_);
            for (int64_t e = 0; e < n_expert_; ++e) order[(size_t) e] = (int32_t) e;
            std::partial_sort(order.begin(), order.begin() + kmax, order.end(),
                              [&](int32_t a, int32_t b) { return lt[(size_t) a] > lt[(size_t) b]; });
            for (int j = 0; j < kmax; ++j) {
                const int64_t e = order[(size_t) j];
                if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;   // on the GPU
                if (j < k_) {
                    const auto it = std::find(want.begin(), want.end(), e);
                    // confidence: 1 for a token's top pick down to 1/k_ for its k_-th, the best over the window's tokens,
                    // halved for every layer further ahead than the next (a deeper guess is less often right)
                    const float c = (float) (k_ - j) / (float) k_ * (dj == 0 ? 1.0f : std::pow(0.5f, (float) dj));
                    if (it == want.end()) { want.push_back(e); wconf.push_back(c); }
                    else if (scored && c > wconf[(size_t) (it - want.begin())]) wconf[(size_t) (it - want.begin())] = c;
                }
                if (stats_on_ && dj == 0)
                    for (int s = 0; s < 3; ++s)
                        if (j < ks[s]) sm[(size_t) (s * words_ + (e >> 6))] |= 1ull << (e & 63);
            }
        }
        if (stats_on_ && dj == 0) {
            // votes: how many tokens put each non-resident expert in their top k_
            std::vector<std::pair<int, int32_t>> votes;   // (sum of 64 - rank, expert)
            for (int64_t t = 0; t < nt; ++t) {
                const float* lt = logits.data() + (size_t) (t * n_expert_);
                for (int64_t e = 0; e < n_expert_; ++e) order[(size_t) e] = (int32_t) e;
                std::partial_sort(order.begin(), order.begin() + k_, order.end(),
                                  [&](int32_t a, int32_t b) { return lt[(size_t) a] > lt[(size_t) b]; });
                for (int j = 0; j < k_; ++j) {
                    const int32_t e = order[(size_t) j];
                    if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;
                    auto it = std::find_if(votes.begin(), votes.end(), [&](const auto& v) { return v.second == e; });
                    if (it == votes.end()) votes.emplace_back(64 - j, e);
                    else it->first += 64 - j;
                }
            }
            std::sort(votes.begin(), votes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (int i = 0; i < kVoteN; ++i)
                pvote_[(size_t) (layer * kVoteN + i)].store(i < (int) votes.size() ? votes[(size_t) i].second + 1 : 0,
                                                            std::memory_order_relaxed);
            for (int64_t i = 0; i < 3 * words_; ++i)
                pmask_[(size_t) (layer * 3 * words_ + i)].store(sm[(size_t) i], std::memory_order_relaxed);
            pstamp_[(size_t) layer].store(sgen + 1, std::memory_order_release);
        }
        if (scored) src_->warm_scored(layer, want.data(), wconf.data(), (int64_t) want.size(), dj + 1);
        else src_->warm(layer, want.data(), (int64_t) want.size());
        if (fs != nullptr && dj == 0)   // the next layer only: a deeper guess is too often wrong to spend a copy on
            for (const int64_t e : want) fs->predict(src_, layer, (int32_t) e);
        predicted_.fetch_add((int64_t) want.size(), std::memory_order_relaxed);
        }
        busy_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                           std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            busy_ = false;
            has_residency_ = false;
        }
        cv_.notify_all();
    }
} catch (...) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        worker_failed_ = true;
        busy_ = pending_ = false;
        has_residency_ = false;
    }
    cv_.notify_all();
    throw; // Preserve existing uncaught-worker fatal behavior; never signal success.
}

