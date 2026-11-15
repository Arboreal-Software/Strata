DetachedVmmChunk::DetachedVmmChunk(DetachedVmmChunk&& x) noexcept
    : handle_(std::exchange(x.handle_,0)), incarnation_(x.incarnation_),
      device_(x.device_), fresh_(x.fresh_), lease_(std::move(x.lease_)) {
    if(x.retirement_generation_)retirement_failed("moving sealed physical chunk");
}
DetachedVmmChunk::~DetachedVmmChunk() {
    if(retirement_generation_&&!retirement_complete_)retirement_failed("uncommitted sealed physical chunk");
    if(handle_)retire();
}
