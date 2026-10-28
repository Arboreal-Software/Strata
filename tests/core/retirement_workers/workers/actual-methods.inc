namespace {
// Detect same-thread lifecycle/source callback reentry BEFORE touching a
// nonrecursive mutex. try_lock on an already-owned std::mutex is not a fence.
struct WorkerLifecycleScope {
    const void* owner;
    WorkerLifecycleScope* previous;
    static thread_local WorkerLifecycleScope* current;
    explicit WorkerLifecycleScope(const void* value) noexcept
        : owner(value), previous(current) { current=this; }
    ~WorkerLifecycleScope() { current=previous; }
    static bool active(const void* value) noexcept {
        for(auto* frame=current;frame;frame=frame->previous)
            if(frame->owner==value)return true;
        return false;
    }
};
thread_local WorkerLifecycleScope* WorkerLifecycleScope::current=nullptr;
} // namespace

bool RouterLookahead::start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k,
                            ExpertSource* src, std::string& err, bool allow_cold_source) {
    if(WorkerLifecycleScope::active(this)){err="worker lifecycle reentry";return false;}
    WorkerLifecycleScope lifecycle_frame(this);
    std::unique_lock<std::mutex> lifecycle(retirement_mu_,std::try_to_lock);
    if(!lifecycle.owns_lock()||retirement_started_.load(std::memory_order_acquire)||thread_.joinable()) {err="lookahead is started or retired";return false;}
    if (src == nullptr || (!allow_cold_source && !src->warms())) { err = "RouterLookahead: the expert source does not warm"; return false; }
    if (n_embd % 8 != 0) { err = "RouterLookahead: n_embd is not a multiple of 8"; return false; }
    for (const auto& r : routers)
        if (r.size() != (size_t) (n_embd * n_expert)) { err = "RouterLookahead: a router of another shape"; return false; }
    if(retirement_started_.load(std::memory_order_acquire)) {err="lookahead is retired";return false;}
    routers_ = std::move(routers);
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    k_ = k < 1 ? 1 : k > (int) n_expert ? (int) n_expert : k;
    src_ = src;
    x_.assign((size_t) (8 * n_embd), 0.f);
    thread_ = std::thread([this] {
        strata::platform::name_this_thread("strata-lookahead");
        strata::aux_cpus::pin_current_thread();
        run();
    });
    return true;
}

bool RouterLookahead::close_for_retirement() noexcept {
    retirement_started_.store(true,std::memory_order_release);
    if(WorkerLifecycleScope::active(this))return false;
    WorkerLifecycleScope lifecycle_frame(this);
    try {
        std::unique_lock<std::mutex> close(retirement_mu_,std::try_to_lock);
        if(!close.owns_lock()||retirement_failed_)return false;
        if(retirement_done_)return true;
        if(thread_.joinable()&&thread_.get_id()==std::this_thread::get_id())return false;
        { std::lock_guard<std::mutex> lock(mu_); quit_=true; }
        cv_.notify_all();
        try { if(thread_.joinable())thread_.join(); }
        catch(...) {retirement_failed_=true;return false;}
        retirement_done_=true;
        return true;
    } catch(...) {return false;}
}

void RouterLookahead::set_foresight(ForesightSwap* fs) {
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if(retirement_started_.load(std::memory_order_acquire))return;
            if (fs != nullptr || !busy_) { fs_ = fs; return; }
        }
        // detaching: a prediction in progress finishes first (the swap space is about to go away)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RouterLookahead::submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res) {
    if (layer + 1 >= (int64_t) routers_.size() || n_tok <= 0 || x == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if(retirement_started_.load(std::memory_order_acquire))return;
        if (busy_ || pending_) { skipped_.fetch_add(1, std::memory_order_relaxed); return; }
        n_tok_ = std::min<int64_t>(n_tok, 8);
        std::memcpy(x_.data(), x, (size_t) (n_tok_ * n_embd_) * sizeof(float));
        layer_ = layer + 1;
        host_res_ = host_res;
        sub_gen_ = gen_.load(std::memory_order_relaxed);
        pending_ = true;
    }
    cv_.notify_one();
}

void RouterLookahead::enable_stats(int64_t n_layers) {
    std::lock_guard<std::mutex> lock(mu_);
    if(retirement_started_.load(std::memory_order_acquire))return;
    words_ = (n_expert_ + 63) / 64;
    pmask_.reset(new std::atomic<uint64_t>[(size_t) (n_layers * 3 * words_)]);
    pstamp_.reset(new std::atomic<uint64_t>[(size_t) n_layers]);
    for (int64_t i = 0; i < n_layers * 3 * words_; ++i) pmask_[(size_t) i].store(0, std::memory_order_relaxed);
    for (int64_t i = 0; i < n_layers; ++i) pstamp_[(size_t) i].store(0, std::memory_order_relaxed);
    pvote_.reset(new std::atomic<int32_t>[(size_t) (n_layers * kVoteN)]);
    for (int64_t i = 0; i < n_layers * kVoteN; ++i) pvote_[(size_t) i].store(0, std::memory_order_relaxed);
    stats_on_ = true;
}

bool ForesightSwap::set_strict_retirement(bool enabled, const std::vector<int>& devices, std::string& error) {
    if(WorkerLifecycleScope::active(this)){error="worker lifecycle reentry";return false;}
    WorkerLifecycleScope lifecycle_frame(this);
    std::unique_lock<std::mutex> lifecycle(retirement_mu_,std::try_to_lock);
    if(!lifecycle.owns_lock()||retirement_started_.load(std::memory_order_acquire)){error="foresight is retired";return false;}
    if (retirement_init_started_) { error = "strict foresight retirement must be configured before init"; return false; }
#if defined(STRATA_GPU_METAL)
    if (enabled) { error = "strict foresight retirement is unsupported by Metal"; return false; }
#endif
    if (enabled && !valid_retirement_topology(devices, error)) return false;
    retirement_consumers_ = devices;
    strict_retirement_ = enabled;
    return true;
}

bool ForesightSwap::close_for_retirement() noexcept {
    retirement_started_.store(true,std::memory_order_release);
    if(WorkerLifecycleScope::active(this))return false;
    WorkerLifecycleScope lifecycle_frame(this);
    for(auto* frame=active_request_;frame;frame=frame->previous)
        if(frame->owner==this)return false; // source callback still owns smu_
    try {
        std::unique_lock<std::mutex> close(retirement_mu_,std::try_to_lock);
        if(!close.owns_lock()||retirement_failed_)return false;
        if(retirement_done_)return true;
        if(filler_.joinable()&&filler_.get_id()==std::this_thread::get_id())return false;
        {
            std::lock_guard<std::mutex> slots_lock(smu_);
            std::lock_guard<std::mutex> worker_lock(mu_);
            stop_=true;
        }
        cv_.notify_all();
        try {if(filler_.joinable())filler_.join();}
        catch(...) {retirement_failed_=true;return false;}
        retirement_done_=true;
        return true;
    } catch(...) {return false;}
}

bool ForesightSwap::take(int64_t layer, int32_t expert, unsigned long long& ptr) {
    std::lock_guard<std::mutex> smu_lk(smu_);
    if(retirement_started_.load(std::memory_order_acquire))return false;
    const int s = where[(size_t) (layer * n_expert + expert)];
    if (s < 0) return false;
    Slot& sl = slot[(size_t) (layer * slots + s)];
    if (!sl.ready) {
        if (!sl.loading || !sl.issued.load(std::memory_order_acquire)) { ++pending; return false; }
        ExpertResourceDevice selected(strict_retirement_, sl.event_device);
        const cudaError_t q = cudaEventQuery((cudaEvent_t) sl.ev);
        if (q != cudaSuccess) {
            if (strict_retirement_ && q != cudaErrorNotReady) retirement_failed("foresight event readiness");
            if (q != cudaErrorNotReady) cudaGetLastError();
            ++pending;
            return false;
        }
        sl.ready = true;
        sl.loading = false;
        if (verify) {   // test mode: the bytes the GPU will read must be the expert's own
            std::vector<uint8_t> back((size_t) sl.bytes);
            const OnDevice on(strict_retirement_ ? -1 : layer_dev[(size_t) layer]);
            const bool same = cudaMemcpy(back.data(), (const void*) sl.dptr, (size_t) sl.bytes, cudaMemcpyDeviceToHost) ==
                                  cudaSuccess && std::memcmp(back.data(), sl.src, (size_t) sl.bytes) == 0;
            if (same) ++verify_ok;
            else {
                cudaGetLastError();
                ++verify_bad;
                sl.ready = false;                           // never serve it
                where[(size_t) (layer * n_expert + expert)] = -1;
                sl.expert = -1;
                return false;
            }
        }
    }
    sl.last_ref = completed.load(std::memory_order_acquire);
    ptr = sl.dptr;
    ++hits;
    return true;
}

void ForesightSwap::hold(int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    if(retirement_started_.load(std::memory_order_acquire))return;
    const size_t key = (size_t) (layer * n_expert + expert);
    const int s = where[key];
    where[key] = kHeld;
    if (s < 0) return;
    Slot& sl = slot[(size_t) (layer * slots + s)];
    sl.expert = kDropped;   // `loading` and `last_ref` stay: a copy in flight and a window that read it are waited for
    sl.ready = false;
    ++dropped;
}

void ForesightSwap::unhold(int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    if(retirement_started_.load(std::memory_order_acquire))return;
    int16_t& w = where[(size_t) (layer * n_expert + expert)];
    if (w == kHeld) w = -1;
}

void ForesightSwap::note_miss(ExpertSource* src, int64_t layer, int32_t expert) {
    std::lock_guard<std::mutex> smu_lk(smu_);
    request_locked(src, layer, expert, false);
}

void ForesightSwap::predict(ExpertSource* src, int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    request_locked(src, layer, expert, true);
}

void ForesightSwap::request_locked(ExpertSource* src, int64_t layer, int32_t expert, bool predicted_copy) {
    if(retirement_started_.load(std::memory_order_acquire))return;
    WorkerLifecycleScope lifecycle_frame(this);
    ActiveRequest frame{this,active_request_};
    struct Restore { ActiveRequest*& head; ActiveRequest* previous; ~Restore(){head=previous;} } restore{active_request_,active_request_};
    active_request_=&frame;
    if (failed.load(std::memory_order_relaxed)) return;
    const size_t key = (size_t) (layer * n_expert + expert);
    if (where[key] != -1) return;                       // already on its way, or held by an exchange (#2134)
    const uint64_t now = completed.load(std::memory_order_acquire);
    if (admit > 1 && !predicted_copy) {
        if (now > miss_at[key] + 4) miss_cnt[key] = 0;
        miss_at[key] = now;
        if (miss_cnt[key] < 255) ++miss_cnt[key];
        if (miss_cnt[key] < admit) return;
    }
    const int c = layer_card[(size_t) layer];
    if (card_round[(size_t) c] != now) { card_round[(size_t) c] = now; card_used[(size_t) c] = 0; }
    if (card_used[(size_t) c] >= budget) { ++over_budget; return; }
    if (src == nullptr || !src->pinned(layer, expert)) return;   // async copies need the pinned arena
    if(retirement_started_.load(std::memory_order_acquire))return;
    const uint8_t* b = src->blob(layer, expert);
    if (b == nullptr || retirement_started_.load(std::memory_order_acquire)) return;
    int best = -1;
    uint64_t best_ref = ~0ull;
    for (int s = 0; s < slots; ++s) {   // an empty slot, else the one used longest ago that no window can still read
        Slot& sl = slot[(size_t) (layer * slots + s)];
        if (sl.loading) {   // copied in but never asked for since: landed long ago -> an ordinary candidate
            if (sl.last_ref + (uint64_t) depth > now || !sl.issued.load(std::memory_order_acquire)) continue;
            ExpertResourceDevice selected(strict_retirement_, sl.event_device);
            const cudaError_t q = cudaEventQuery((cudaEvent_t) sl.ev);
            if (q != cudaSuccess) {
                if (strict_retirement_ && q != cudaErrorNotReady) retirement_failed("foresight slot reuse readiness");
                if (q != cudaErrorNotReady) cudaGetLastError();
                continue;
            }
            sl.loading = false;
            sl.ready = true;
        }
        if (sl.expert == -1) { best = s; break; }      // never used (a dropped slot, kDropped, waits like a used one)
        if (sl.last_ref + (uint64_t) depth > now) continue;
        if (sl.last_ref < best_ref) { best_ref = sl.last_ref; best = s; }
    }
    if (best < 0) { ++busy; return; }
    Slot& sl = slot[(size_t) (layer * slots + best)];
    if (sl.expert >= 0) where[(size_t) (layer * n_expert + sl.expert)] = -1;
    sl.expert = expert;
    sl.ready = false;
    sl.loading = true;
    sl.issued.store(false, std::memory_order_relaxed);
    sl.src = b;
    sl.last_ref = now;
    where[key] = (int16_t) best;
    ++card_used[(size_t) c];
    ++copies;
    if (predicted_copy) ++predicted;
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(layer * slots + best);
    }
    cv_.notify_one();
}
