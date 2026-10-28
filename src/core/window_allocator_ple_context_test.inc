// Exact PleReadContext implementation extracted from src/kernels/ngram.cpp; mocked table/reader only.
struct PleReadContext::Impl {
    std::shared_ptr<PleTable> owner;
    std::shared_ptr<const void> lifetime_guard;
    uint64_t generation = 0;
    std::atomic<bool> retirement_started{false};
    std::atomic<bool> retirement_failed{false};
    bool retired = false; // protected by the retained table mutex
    bool pending = false;
    bool untracked = false;
    strata::ngram::PleReader::Ticket ticket;
    uint32_t rows[PLE_N_HEADS] = {};
    uint8_t raw[PLE_N_HEADS * PLE_ROW_BYTES_MAX] = {};
    Impl* next_quarantined = nullptr;
};
PleReadContext::PleReadContext(std::shared_ptr<PleTable> table,std::shared_ptr<const void> guard)
    : impl_(std::make_unique<Impl>()) {
    if(!table)throw std::invalid_argument("PLE read context requires a retained table");
    std::lock_guard<std::recursive_mutex> lock(table->impl_->mu);
    if(!table->is_open()||table->impl_->close_failed)throw std::invalid_argument("PLE read context requires an open healthy table");
    impl_->generation=table->impl_->generation;
    impl_->owner=std::move(table);impl_->lifetime_guard=std::move(guard);
}
PleReadContext::~PleReadContext() {
    if(!impl_)return;
    try {
        auto& table=*impl_->owner->impl_;
        std::lock_guard<std::recursive_mutex> lock(table.mu);
        if(impl_->retirement_started.load(std::memory_order_acquire)) {
            if(impl_->retirement_failed.load(std::memory_order_acquire)||!impl_->retired)throw 0;
            return; // explicit close already established completion; never retry
        }
        if(impl_->untracked||table.close_failed)throw 0;
        if(impl_->pending&&impl_->generation==table.generation&&table.mode==PleIo::Direct) {
            std::string error;
            // collect returns only once this ticket's IO has drained, including
            // ordinary IO failures. Explicit close already joined stale tickets.
            (void)table.reader.collect(impl_->ticket,error);
        }
    } catch(...) {
        // An exceptional join is not completion proof. Retain raw destinations,
        // shared table and caller's reservation guard until process restart.
        static std::mutex mutex;static Impl* quarantined=nullptr;
        std::lock_guard lock(mutex);
        impl_->next_quarantined=quarantined;quarantined=impl_.release();
    }
}
bool PleReadContext::close_for_retirement() noexcept {
    // Publish the seal even if acquiring the mutex itself throws. Aliases that
    // already entered issue/collect finish under that mutex before this join.
    impl_->retirement_started.store(true,std::memory_order_release);
    if(impl_->retirement_failed.load(std::memory_order_acquire))return false;
    try {
        auto& table=*impl_->owner->impl_;
        std::lock_guard<std::recursive_mutex> lock(table.mu);
        if(impl_->retirement_failed.load(std::memory_order_acquire))return false;
        if(impl_->retired)return true;
        if(impl_->untracked||table.close_failed) {
            impl_->retirement_failed.store(true,std::memory_order_release);return false;
        }
        if(impl_->pending&&impl_->generation==table.generation&&table.mode==PleIo::Direct) {
            std::string error;
            // False also means unknown ticket, not just a drained IO error.
            // Until the reader exposes typed completion, retain either case.
            if(!table.reader.collect(impl_->ticket,error)) {
                impl_->retirement_failed.store(true,std::memory_order_release);return false;
            }
        }
        // A changed healthy table generation means its close joined old reads.
        // Mmap issue only stores row indices and has no pending IO destination.
        impl_->pending=false;
        impl_->retired=true;
        return true;
    } catch(...) {
        impl_->retirement_failed.store(true,std::memory_order_release);
        return false;
    }
}
PleTable* PleReadContext::table() const {return impl_->owner.get();}
bool PleReadContext::pending() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->owner->impl_->mu);
    return impl_->pending;
}
bool PleReadContext::issue(const uint32_t* rows) {
    auto& table=*impl_->owner->impl_;
    std::lock_guard<std::recursive_mutex> lock(table.mu);
    if(impl_->retirement_started.load(std::memory_order_acquire)||!rows||impl_->pending||table.close_failed||impl_->generation!=table.generation||!impl_->owner->is_open())return false;
    std::memcpy(impl_->rows,rows,sizeof impl_->rows);
    // Mark pending before issue: exceptional partial submission must retain its
    // output buffer until reader close, not destroy it as an unstarted request.
    impl_->pending=true;
    if(table.mode==PleIo::Direct) {
        try {impl_->ticket=table.reader.issue(impl_->rows,PLE_N_HEADS,impl_->raw);}
        catch(...) {
            // No ticket was returned, so close joins every possible partial read
            // before this context's buffer can be reused or released.
            impl_->untracked=true;
            impl_->owner->close();
            impl_->untracked=false;impl_->pending=false;throw;
        }
    }
    return true;
}
bool PleReadContext::collect(float* output,std::string& error) {
    auto& table=*impl_->owner->impl_;
    std::lock_guard<std::recursive_mutex> lock(table.mu);
    if(impl_->retirement_started.load(std::memory_order_acquire)) {
        error="PLE context is retired or retirement failed";return false;
    }
    if(!output||!impl_->pending){error="PLE context collect without pending output";return false;}
    if(impl_->generation!=table.generation){impl_->pending=false;error="PLE context invalidated by table close";return false;}
    if(table.mode==PleIo::Direct) {
        const bool ok=table.reader.collect(impl_->ticket,error);
        impl_->pending=false;
        if(!ok)return false;
        for(int h=0;h<PLE_N_HEADS;++h)table.decode(impl_->raw+size_t(h)*table.rb,output+size_t(h)*PLE_HEAD_DIM);
        table.bytes_read+=uint64_t(PLE_N_HEADS)*table.rb;
    } else {
        for(int h=0;h<PLE_N_HEADS;++h)impl_->owner->read_row(impl_->rows[h],output+size_t(h)*PLE_HEAD_DIM);
        impl_->pending=false;
    }
    return true;
}

