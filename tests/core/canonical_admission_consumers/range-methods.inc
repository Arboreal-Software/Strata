bool VmmRange::managed_host_bound(uint64_t count,uint64_t& bytes) noexcept {
    constexpr uint64_t base=4096+sizeof(VmmRange);
    if(count>(UINT64_MAX-base)/(sizeof(ManagedSlot)+sizeof(VmmChunk)))return false;
    bytes=base+count*(sizeof(ManagedSlot)+sizeof(VmmChunk));return true;
}
void VmmRange::release_managed_metadata() noexcept {
    { std::vector<ManagedSlot> retired;retired.swap(managed_slots_); }
    { std::vector<VmmChunk> retired;retired.swap(h_); }
    metadata_.reset();managed_prepared_=false;managed_activation_attempted_=true;
}
bool VmmRange::prepare_managed_metadata(uint64_t count,uint64_t gran,
    std::shared_ptr<const ReservationLease> lease,int device,std::string& error) {
    try {
        uint64_t required=0;
        if(managed_||base_||!h_.empty()||device<0||!count||count>INT64_MAX||count>SIZE_MAX||
           !gran||count>SIZE_MAX/gran||count>std::vector<ManagedSlot>().max_size()||count>std::vector<VmmChunk>().max_size()||
           !managed_host_bound(count,required)||!lease||!lease->canonical_owner(lease)||lease.use_count()!=1||
           lease->purpose()!=ReservationPurpose::foreground||lease->resources().host_bytes<required||
           lease->resources().pinned_bytes||lease->resources().disk_bytes||lease->resources().io_slots||lease->resources().shadows){
            error="managed host preparation requires fresh range and dedicated bounded metadata";return false;
        }
        for(auto n:lease->resources().device_bytes)if(n){error="managed metadata contains device quota";return false;}
        std::vector<ManagedSlot> slots((size_t)count);
        std::vector<VmmChunk> handles((size_t)count,0);
        managed_slots_.swap(slots);h_.swap(handles);metadata_=std::move(lease);
        managed_device_=device;managed_granularity_=gran;managed_=true;managed_prepared_=true;
        strict_retirement_=true;retirement_device_=device;return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
}
