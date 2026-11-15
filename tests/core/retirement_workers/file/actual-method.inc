namespace {
struct FileBackgroundStopFrame {
    const void* owner;
    FileBackgroundStopFrame* previous;
    static thread_local FileBackgroundStopFrame* current;
    explicit FileBackgroundStopFrame(const void* p) noexcept:owner(p),previous(current){current=this;}
    ~FileBackgroundStopFrame(){current=previous;}
    static bool active(const void* p) noexcept {
        for(auto* f=current;f;f=f->previous)if(f->owner==p)return true;
        return false;
    }
};
thread_local FileBackgroundStopFrame* FileBackgroundStopFrame::current=nullptr;
} // namespace

bool FileExpertSource::stop_background_after_callers_joined() noexcept {
    background_sealed_.store(true,std::memory_order_release);
    if(FileBackgroundStopFrame::active(this))return false;
    FileBackgroundStopFrame frame(this);
    try {
        std::unique_lock<std::mutex> close(background_close_mu_,std::try_to_lock);
        if(!close.owns_lock()||background_close_failed_.load(std::memory_order_acquire))return false;
        if(background_joined_)return true;
        const auto self=std::this_thread::get_id();
        for(const auto& t:async_th_)if(t.joinable()&&t.get_id()==self)return false;
        for(const auto& t:io_workers_)if(t.joinable()&&t.get_id()==self)return false;
        if(ram_thread_.joinable()&&ram_thread_.get_id()==self)return false;
        // Finish already-issued prefetches before stopping helper IO workers.
        // Do not call prefetch_end/ram_stop/io_stop: they also release storage.
        for(auto& t:async_th_)if(t.joinable())t.join();
        {std::lock_guard<std::mutex> lock(ram_mu_);ram_quit_=true;}
        ram_cv_.notify_all();
        if(ram_thread_.joinable())ram_thread_.join();
        {std::lock_guard<std::mutex> lock(io_mu_);io_quit_=true;}
        io_cv_.notify_all();
        for(auto& t:io_workers_)if(t.joinable())t.join();
        background_joined_=true;
        return true;
    } catch(...) {
        background_close_failed_.store(true,std::memory_order_release);
        return false;
    }
}
