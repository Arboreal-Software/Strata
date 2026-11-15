#include "strata/core/window_buffer.hpp"
#include "strata/core/window_device.hpp"
#include <atomic>
#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#include <cstring>

namespace strata::core {
struct OwnedWindowBuffer::Storage {
    int device=-1;
    uint64_t bytes=0;
    void* pointer=nullptr;
    void* device_pointer=nullptr;
    WindowBufferMemory memory=WindowBufferMemory::device;
    std::vector<int> consumers;
    std::shared_ptr<const ReservationLease> reservation;
    bool uncertain=false;
    Storage* next=nullptr;
    static std::atomic<bool> failed;
    static std::mutex mutex;
    static Storage* quarantine;
    static std::atomic<uint64_t> count;
    static void destroy(Storage* storage) noexcept {
        bool safe=!storage->uncertain;
        try {
            if(safe&&storage->pointer) {
                for(int id:storage->consumers) {
                    WindowDeviceScope scope(id);
                    if(!scope.ok()||cudaDeviceSynchronize()!=cudaSuccess||!scope.finish()){safe=false;break;}
                }
                if(safe) {
                    WindowDeviceScope scope(storage->device);
                    if(!scope.ok()||(storage->memory==WindowBufferMemory::mapped_host?
                        cudaFreeHost(storage->pointer):cudaFree(storage->pointer))!=cudaSuccess) safe=false;
                    else {storage->pointer=nullptr;storage->device_pointer=nullptr;if(!scope.finish())safe=false;}
                }
            }
        } catch(...) {safe=false;}
        if(safe){delete storage;return;}
        failed.store(true);
        // No allocation and no device waits while the registry lock is held.
        std::lock_guard lock(mutex);
        storage->next=quarantine;quarantine=storage;++count;
    }
};
std::atomic<bool> OwnedWindowBuffer::Storage::failed{false};
std::mutex OwnedWindowBuffer::Storage::mutex;
OwnedWindowBuffer::Storage* OwnedWindowBuffer::Storage::quarantine=nullptr;
std::atomic<uint64_t> OwnedWindowBuffer::Storage::count{0};
void* OwnedWindowBuffer::data() const {return storage_?storage_->pointer:nullptr;}
void* OwnedWindowBuffer::device_data() const {return storage_?storage_->device_pointer:nullptr;}
uint64_t OwnedWindowBuffer::bytes() const {return storage_?storage_->bytes:0;}
uint64_t window_buffer_quarantine_count() {return OwnedWindowBuffer::Storage::count.load();}
bool allocate_window_buffer(int device,uint64_t bytes,const std::vector<int>& consumers,
    WindowBudget& budget,std::shared_ptr<const ReservationLease> parent,OwnedWindowBuffer& out,std::string& error,
    WindowBufferMemory memory) {
    using Storage=OwnedWindowBuffer::Storage;
    if(Storage::failed.load()){error="window buffer quarantine requires worker restart";return false;}
    const auto capacity=budget.snapshot().capacity;
    if((memory!=WindowBufferMemory::device&&memory!=WindowBufferMemory::mapped_host)||
       device<0||size_t(device)>=capacity.device_bytes.size()||!bytes||bytes>SIZE_MAX||
       consumers.empty()||consumers.size()>64||!parent||
       std::find(consumers.begin(),consumers.end(),device)==consumers.end()) {
        error="invalid window buffer allocation";return false;
    }
    for(size_t i=0;i<consumers.size();++i) {
        if(consumers[i]<0||size_t(consumers[i])>=capacity.device_bytes.size()||
           std::find(consumers.begin(),consumers.begin()+i,consumers[i])!=consumers.begin()+i) {
            error="invalid window buffer consumer topology";return false;
        }
    }
    try {
        auto storage=std::shared_ptr<Storage>(new Storage,Storage::destroy);
        storage->device=device;storage->bytes=bytes;storage->consumers=consumers;storage->memory=memory;
        WindowResources charge;charge.device_bytes.resize(capacity.device_bytes.size());
        // Bounded metadata headroom includes the control block and device list.
        charge.host_bytes=sizeof(Storage)+256+storage->consumers.capacity()*sizeof(int);
        if(memory==WindowBufferMemory::device)charge.device_bytes[size_t(device)]=bytes;
        else {
            if(bytes>UINT64_MAX-charge.host_bytes){error="mapped window buffer charge overflow";return false;}
            charge.host_bytes+=bytes;charge.pinned_bytes=bytes;
        }
        auto reserved=budget.reserve(charge,ReservationPurpose::private_attachment,std::move(parent));
        if(!reserved){error="window buffer reservation refused";return false;}
        storage->reservation=std::move(reserved.lease);
        WindowDeviceScope scope(device);
        if(!scope.ok()){error="window buffer device selection failed";return false;}
        storage->uncertain=true;
        const auto allocated=memory==WindowBufferMemory::mapped_host?
            cudaHostAlloc(&storage->pointer,size_t(bytes),cudaHostAllocMapped|cudaHostAllocPortable):
            cudaMalloc(&storage->pointer,size_t(bytes));
        if(allocated!=cudaSuccess||!storage->pointer) {
            error="window buffer allocation failed";return false;
        }
        if(memory==WindowBufferMemory::mapped_host) {
            if(cudaHostGetDevicePointer(&storage->device_pointer,storage->pointer,0)!=cudaSuccess||!storage->device_pointer) {
                error="window buffer mapping failed";return false;
            }
            std::memset(storage->pointer,0,size_t(bytes));
        } else storage->device_pointer=storage->pointer;
        if(!scope.finish()) {
            error="window buffer allocation/device restoration failed";return false;
        }
        if(Storage::failed.load()){error="window buffer poisoned during allocation";return false;}
        storage->uncertain=false;
        OwnedWindowBuffer candidate;candidate.storage_=std::move(storage);out=std::move(candidate);return true;
    } catch(const std::bad_alloc&) {error="window buffer host metadata unavailable";return false;}
}
}
