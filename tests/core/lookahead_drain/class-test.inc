class RouterLookahead {
public:
    RouterLookahead() = default;
    // Exact additional logical residency-array payload; output unchanged on
    // overflow/invalid dimensions. Zero layers is valid for legacy empty start.
    static bool residency_snapshot_bytes(size_t layers, int64_t experts, uint64_t& out) noexcept;
    // Owned startup only. Consumes a genuine exclusive dedicated foreground
    // lease after validating exact host bytes and no other resource charges.
    // Caller uses the actual runtime ledger, not an untrusted aliased lease.
    // No WindowBudget operation is linked into the default source translation unit.
    bool configure_residency_snapshot(size_t layers, int64_t experts,
        std::shared_ptr<const ReservationLease>& lease) noexcept;
    ~RouterLookahead();
    // Terminal stop/join only: preserves source/FS targets and all buffers for
    // the caller's subsequent GPU drain. False requires retaining this owner.
    // Concurrent/reentrant/self close refuses; no join under worker mutex.
    bool close_for_retirement() noexcept;
    // Reversible writer boundary for the actual borrowed residency table/source.
    // Caller retains this RouterLookahead through guard destruction, serializes
    // guard access and freezes configuration. Never destroy the owner via a callback.
    class SubmittedDrain {
    public:
        SubmittedDrain() = default;
        ~SubmittedDrain() noexcept;
        void reset() noexcept; // known writer completion/cancellation only
        SubmittedDrain(const SubmittedDrain&) = delete;
        SubmittedDrain& operator=(const SubmittedDrain&) = delete;
        SubmittedDrain(SubmittedDrain&&) = delete;
        SubmittedDrain& operator=(SubmittedDrain&&) = delete;
    private:
        friend class RouterLookahead;
        RouterLookahead* owner_ = nullptr;
    };
    enum class DrainResult { drained, busy, refused };
    // No deadline means a nonblocking poll. A supplied absolute steady-clock
    // deadline bounds waiting; callers must not wait on work needing their reactor.
    // Busy retains the seal in guard for retry; destruction releases it. While
    // sealed submit drops speculative predictions (counts skipped), never queues.
    // Only drained authorizes mutation, and only while the same guard is alive.
    // This joins our submitted warm/predict callbacks, NOT downstream source or
    // Foresight GPU completion; those owners retain their own ordering contracts.
    DrainResult drain_submitted(SubmittedDrain& guard,
        std::optional<std::chrono::steady_clock::time_point> deadline = std::nullopt);
    RouterLookahead(const RouterLookahead&) = delete;
    RouterLookahead& operator=(const RouterLookahead&) = delete;
    /// `routers[l]`: layer l's ffn_gate_inp as BF16 bits, n_expert rows of n_embd.
    bool start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k, ExpertSource* src,
               std::string& err, bool allow_cold_source = false);
    /// One prefetch design (#1348): the predicted experts that no cache holds also go to the Foresight swap space
    /// (ForesightSwap::predict), which copies them to VRAM ahead of the layer.  nullptr detaches it (waits for a
    /// prediction in progress), so the swap space can be destroyed first.
    void set_foresight(ForesightSwap* fs);
    /// Layer `layer`'s MoE input for `n_tok` tokens (host floats): predict and warm layer + 1.  Never waits: a
    /// prediction still running for an earlier layer makes this one skip.
    /// Caller supplies full startup geometry and serializes writes through this
    /// synchronous copy. Only the private snapshot survives submit returning.
    void submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res);
    /// STRATA_IO_PREFETCH_DEPTH: how many layers ahead of `layer` are predicted and warmed (default 1, the next layer).
    /// A deeper one predicts layer + j from layer's input, which is less exact the further it looks.
    void set_depth(int d) { std::lock_guard<std::mutex> lock(mu_); if(!retirement_started_.load()) depth_ = d < 1 ? 1 : d > 4 ? 4 : d; }
    int depth() const { return depth_; }
    /// STRATA_LOOKAHEAD_STATS=1 (measurement only): remember each prediction (top k, k+4 and k+10 per token) and let the
    /// dispatch score it against the experts the next layer really routes outside the GPU cache.
    void enable_stats(int64_t n_layers);
    bool stats_on() const { return stats_on_; }
    void stats_window_start() { std::lock_guard<std::mutex> lock(mu_); if(!retirement_started_.load()) gen_.fetch_add(1, std::memory_order_relaxed); }
    void stats_score(int64_t layer, const int32_t* ids, int64_t n, const int32_t* host_res, int pcie_num = 0);
    int64_t predicted() const { return predicted_.load(std::memory_order_relaxed); }
    int64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }
    double busy_ms() const { return (double) busy_us_.load(std::memory_order_relaxed) / 1000.0; }

private:
    void run();
    std::mutex retirement_mu_;
    std::atomic<bool> retirement_started_{false};
    bool retirement_done_ = false, retirement_failed_ = false; // retirement_mu_
    std::vector<std::vector<uint16_t>> routers_;
    int64_t n_embd_ = 0, n_expert_ = 0;
    int k_ = 10;
    int depth_ = 1;
    ExpertSource* src_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    WakeCv cv_;
    bool quit_ = false, pending_ = false, busy_ = false;
    SubmittedDrain* submitted_drain_ = nullptr; // mu_, nonmovable guard identity
    std::thread::id worker_id_{}; // mu_, avoids racing thread_ join
    bool worker_failed_ = false; // sticky, never a successful drain proof
    int64_t layer_ = -1, n_tok_ = 0;
    // Charge declared before allocation: storage is destroyed before its lease.
    std::shared_ptr<const ReservationLease> residency_lease_;
    std::vector<int32_t> residency_snapshot_;
    size_t admitted_layers_ = 0;
    int64_t admitted_experts_ = 0;
    bool has_residency_ = false;
    uint64_t sub_gen_ = 0;
    ForesightSwap* fs_ = nullptr;                   ///< (under mu_)
    std::vector<float> x_;
    std::atomic<int64_t> predicted_{0}, skipped_{0};
    std::atomic<uint64_t> busy_us_{0};
    // STRATA_LOOKAHEAD_STATS
    bool stats_on_ = false;
    int64_t words_ = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> pmask_;   // [layer][3 sizes][words]
    std::unique_ptr<std::atomic<uint64_t>[]> pstamp_;  // [layer]: gen + 1 of the prediction (0 = none)
    std::unique_ptr<std::atomic<int32_t>[]> pvote_;    // [layer][kVoteN]: experts by votes, expert + 1 (0 = none)
    std::atomic<uint64_t> gen_{0};
    uint64_t st_layers_ = 0, st_nopred_ = 0, st_actual_ = 0, st_hit_[3] = {0, 0, 0}, st_pred_[3] = {0, 0, 0},
             st_predhit_[3] = {0, 0, 0};
    uint64_t st_windows_ = 0;
    uint64_t st_vhit_[3] = {0, 0, 0}, st_vpred_[3] = {0, 0, 0}, st_pch_ = 0, st_pcov_[3] = {0, 0, 0};
    static constexpr int kVoteN = 16;
};
