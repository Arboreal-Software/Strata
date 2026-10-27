#include "strata/core/strict_retirement.hpp"
#include "strata/core/vmm.hpp"
#include <atomic>
#include <limits>
#include <utility>
#include <new>

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>

namespace strata::core {
namespace {

struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        decltype(&cuDeviceGetAttribute) attr = nullptr;
        decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
        if (cudaGetDevice(&a.dev) != cudaSuccess) return;
        if (!resolve("cuDeviceGetAttribute", attr) || !resolve("cuMemGetAllocationGranularity", granularity) ||
            !resolve("cuMemAddressReserve", a.reserve) || !resolve("cuMemAddressFree", a.free_va) ||
            !resolve("cuMemCreate", a.create) || !resolve("cuMemRelease", a.release) || !resolve("cuMemMap", a.map) ||
            !resolve("cuMemUnmap", a.unmap) || !resolve("cuMemSetAccess", a.access))
            return;
        int vmm = 0;
        if (attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) a.dev) != CUDA_SUCCESS || !vmm)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = a.dev;
        size_t g = 0;
        if (granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || g == 0) return;
        a.gran = g;
        a.ok = true;
    });
    return a;
}

}  // namespace

static bool managed_domain(int device) noexcept {
    int current = -1;
    return api().ok && api().dev == device && cudaGetDevice(&current) == cudaSuccess && current == device;
}
static bool managed_free(VmmChunk h) noexcept {
    return api().ok && api().release((CUmemGenericAllocationHandle)h) == CUDA_SUCCESS;
}
bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new() {
    const Api& a = api();
    if (!a.ok) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, (size_t) a.gran, &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}
void vmm_chunk_retire(VmmChunk h, bool strict) {
    if (!strict) { vmm_chunk_free(h); return; }
    if (h && (!api().ok || api().release((CUmemGenericAllocationHandle)h) != CUDA_SUCCESS))
        retirement_failed("VMM physical allocation release");
}

bool VmmRange::activate_managed() noexcept {
    if(!managed_||!managed_prepared_||managed_activation_attempted_||base_||managed_unknown_)return false;
    if(!managed_domain(managed_device_)||api().gran!=managed_granularity_)return false;
    managed_activation_attempted_=true;
    const uint64_t bytes=uint64_t(h_.size())*managed_granularity_;
    CUdeviceptr p=0;
    if(api().reserve(&p,(size_t)bytes,0,0,0)!=CUDA_SUCCESS){
        if(p){base_=(unsigned long long)p;managed_unknown_=true;}return false;
    }
    base_=(unsigned long long)p;managed_prepared_=false;return true;
}
bool VmmRange::reserve(uint64_t bytes) {
    if (managed_) return false;
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    if (strict_retirement_ && a.dev != retirement_device_) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    if (strict_retirement_) h_.assign((size_t)n, 0);
    CUdeviceptr p = 0;
    if (a.reserve(&p, (size_t) (n * a.gran), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    if (!strict_retirement_) h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (managed_unknown_) retirement_failed("unknown managed VMM destruction");
    if (base_ == 0) { if (managed_) release_managed_metadata(); return; }
    StrictRetirementDevice retirement(strict_retirement_, retirement_device_);
    const Api& a = api();
    managed_internal_ = true;
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_retire(unmap(i), strict_retirement_);
    managed_internal_ = false;
    const auto freed = a.free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * a.gran));
    if (strict_retirement_ && freed != CUDA_SUCCESS) retirement_failed("VMM address release");
    base_ = 0;
    if (managed_) release_managed_metadata();
    else h_.clear();
    managed_internal_ = false;
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (a.map((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0, (CUmemGenericAllocationHandle) h, 0) !=
        CUDA_SUCCESS)
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = a.dev;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return a.access((CUdeviceptr) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_retire(unmap(c), strict_retirement_);
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (managed_ && !managed_internal_) return 0;
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (a.unmap((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran) != CUDA_SUCCESS) {
        if (strict_retirement_) retirement_failed("VMM unmap");
        return 0;
    }
    h_[(size_t) i] = 0;
    return h;
}

}  // namespace strata::core

#else  // not CUDA
// HIP (ROCm 6+): the same elastic range through hipMemAddressReserve / hipMemCreate / hipMemMap / hipMemSetAccess.  Available
// only where the device reports virtual memory management support (STRATA_HIP_VMM=0 turns it off): --kv-grow stays off
// otherwise, as before.  Everything else (Metal, gfx906, ROCm older than 6): no virtual memory, the elastic K/V stays off.
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#endif

#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 6
namespace strata::core {
namespace {
struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
};
void fail(const char* what, hipError_t e) {
    static int n = 0;
    if (n++ < 8) std::fprintf(stderr, "strata: HIP virtual memory: %s failed: %s\n", what, hipGetErrorString(e));
    (void) hipGetLastError();
}
hipMemAllocationProp alloc_prop(int dev) {
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.requestedHandleType = hipMemHandleTypeNone;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = dev;
    return p;
}
const Api& api() {
    static Api& a = *new Api;   // never destroyed: the K/V pools and the cache release their chunks from static destructors
    static std::once_flag once;
    std::call_once(once, [] {
        const char* off = std::getenv("STRATA_HIP_VMM");
        if (off != nullptr && off[0] == '0') return;
        if (hipGetDevice(&a.dev) != hipSuccess) { (void) hipGetLastError(); return; }
        int vmm = 0;
        if (hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, a.dev) != hipSuccess || !vmm) {
            (void) hipGetLastError();
            return;
        }
        const hipMemAllocationProp prop = alloc_prop(a.dev);
        size_t g = 0;
        if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum) != hipSuccess || g == 0) {
            (void) hipGetLastError();
            return;
        }
        // gfx1201 reports a 4 KiB minimum granularity; the elastic range works in 2 MiB chunks like CUDA's (any multiple of the minimum)
        const uint64_t want = 2ull << 20;
        a.gran = g >= want ? g : (want + g - 1) / g * g;
        a.ok = true;
    });
    return a;
}
}  // namespace

static bool managed_domain(int device) noexcept {
    int current = -1;
    return api().ok && api().dev == device && hipGetDevice(&current) == hipSuccess && current == device;
}
static bool managed_free(VmmChunk h) noexcept {
    return api().ok && hipMemRelease((hipMemGenericAllocationHandle_t)(uintptr_t)h) == hipSuccess;
}
bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new() {
    const Api& a = api();
    if (!a.ok) return 0;
    const hipMemAllocationProp prop = alloc_prop(a.dev);
    hipMemGenericAllocationHandle_t h = nullptr;
    if (const hipError_t e = hipMemCreate(&h, (size_t) a.gran, &prop, 0); e != hipSuccess) { fail("hipMemCreate", e); return 0; }
    return (VmmChunk) (uintptr_t) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) (void) hipMemRelease((hipMemGenericAllocationHandle_t) (uintptr_t) h);
}
void vmm_chunk_retire(VmmChunk h, bool strict) {
    if (!strict) { vmm_chunk_free(h); return; }
    if (h && (!api().ok || hipMemRelease((hipMemGenericAllocationHandle_t)(uintptr_t)h) != hipSuccess))
        retirement_failed("HIP VMM physical allocation release");
}

bool VmmRange::activate_managed() noexcept {
    if(!managed_||!managed_prepared_||managed_activation_attempted_||base_||managed_unknown_)return false;
    if(!managed_domain(managed_device_)||api().gran!=managed_granularity_)return false;
    managed_activation_attempted_=true;
    const uint64_t bytes=uint64_t(h_.size())*managed_granularity_;
    void* p=nullptr;
    if(hipMemAddressReserve(&p,(size_t)bytes,0,nullptr,0)!=hipSuccess){
        if(p){base_=(unsigned long long)(uintptr_t)p;managed_unknown_=true;}return false;
    }
    base_=(unsigned long long)(uintptr_t)p;managed_prepared_=false;return true;
}
bool VmmRange::reserve(uint64_t bytes) {
    if (managed_) return false;
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    if (strict_retirement_ && a.dev != retirement_device_) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    if (strict_retirement_) h_.assign((size_t)n, 0);
    void* p = nullptr;
    if (const hipError_t e = hipMemAddressReserve(&p, (size_t) (n * a.gran), 0, nullptr, 0); e != hipSuccess) { fail("hipMemAddressReserve", e); return false; }
    base_ = (unsigned long long) (uintptr_t) p;
    if (!strict_retirement_) h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (managed_unknown_) retirement_failed("unknown managed VMM destruction");
    if (base_ == 0) { if (managed_) release_managed_metadata(); return; }
    StrictRetirementDevice retirement(strict_retirement_, retirement_device_);
    const Api& a = api();
    managed_internal_ = true;
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_retire(unmap(i), strict_retirement_);
    managed_internal_ = false;
    const auto freed = hipMemAddressFree((void*) (uintptr_t) base_, (size_t) ((uint64_t) h_.size() * a.gran));
    if (strict_retirement_ && freed != hipSuccess) retirement_failed("HIP VMM address release");
    base_ = 0;
    if (managed_) release_managed_metadata();
    else h_.clear();
    managed_internal_ = false;
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (const hipError_t e = hipMemMap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0,
                                      (hipMemGenericAllocationHandle_t) (uintptr_t) h, 0);
        e != hipSuccess) {
        fail("hipMemMap", e);
        return false;
    }
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = a.dev;
    d.flags = hipMemAccessFlagsProtReadWrite;
    const hipError_t e = hipMemSetAccess((void*) (uintptr_t) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran),
                                         &d, 1);
    const bool ok = e == hipSuccess;
    if (!ok) fail("hipMemSetAccess", e);
    return ok;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_retire(unmap(c), strict_retirement_);
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (managed_ && !managed_internal_) return 0;
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (hipMemUnmap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran) != hipSuccess) { if (strict_retirement_) retirement_failed("HIP VMM unmap"); (void) hipGetLastError(); return 0; }
    h_[(size_t) i] = 0;
    return h;
}
}  // namespace strata::core
#else


namespace strata::core {
static bool managed_domain(int) noexcept { return false; }
static bool managed_free(VmmChunk) noexcept { return false; }
bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
void vmm_chunk_free(VmmChunk) {}
void vmm_chunk_retire(VmmChunk h, bool strict) { if (strict && h) retirement_failed("unsupported VMM chunk retirement"); }
bool VmmRange::activate_managed() noexcept { return false; }
bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() { if(managed_)release_managed_metadata(); }
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
}  // namespace strata::core

#endif
#endif

namespace strata::core {
DetachedVmmChunk::DetachedVmmChunk(DetachedVmmChunk&& x) noexcept
    : handle_(std::exchange(x.handle_,0)), incarnation_(x.incarnation_),
      device_(x.device_), fresh_(x.fresh_), lease_(std::move(x.lease_)) {}
bool DetachedVmmChunk::allocation_domain(int device) noexcept { return managed_domain(device); }
DetachedVmmChunk::~DetachedVmmChunk() { if (handle_) retire(); }
bool DetachedVmmChunk::retire() noexcept {
    if (!handle_) return true;
    // Explicit quiescence is the caller's obligation; destruction drains before
    // releasing an owned detached allocation, using existing worker policy.
    StrictRetirementDevice guard(true,device_);
    if (!managed_domain(device_) || !managed_free(handle_))
        retirement_failed("managed detached physical release");
    handle_=0;
    lease_.reset();
    return true;
}
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
           !managed_host_bound(count,required)||!lease||lease.use_count()!=1||
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
bool VmmRange::prepare_managed(std::shared_ptr<const ReservationLease> lease,int device,std::string& error) {
    uint64_t required=0;
    if(managed_ || !base_ || mapped_count()!=0 || !managed_domain(device) ||
       !lease || lease.use_count()!=1 || lease->purpose()!=ReservationPurpose::foreground || !managed_host_bound(h_.size(),required) ||
       lease->resources().host_bytes<required || lease->resources().pinned_bytes ||
       lease->resources().disk_bytes || lease->resources().io_slots || lease->resources().shadows) {
        error="managed range requires empty reserved range and dedicated metadata admission";return false;
    }
    for(auto n:lease->resources().device_bytes)if(n){error="metadata lease contains device quota";return false;}
    try {
        std::vector<ManagedSlot> slots(h_.size());
        managed_slots_.swap(slots);metadata_=std::move(lease);
        managed_device_=device;strict_retirement_=true;retirement_device_=device;managed_=true;
        return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
}
bool VmmRange::map_managed(int64_t lo,int64_t hi,std::span<DetachedVmmChunk> input,
                          std::span<VmmChunkOutcome> output) noexcept {
    if(!managed_ || !base_ || managed_unknown_ || !managed_domain(managed_device_) || lo<0 || hi<lo || hi>chunks())return false;
    size_t need=0;
    for(auto i=lo;i<hi;++i)need+=!mapped(i);
    if(input.size()!=need || output.size()<need)return false;
    for(auto& c:input)if(!c.handle_ || !c.lease_ || c.device_!=managed_device_)return false;
    size_t k=0;
    for(auto i=lo;i<hi;++i)if(!mapped(i)) {
        output[k]={i,input[k].incarnation_,input[k].fresh_,VmmDisposition::not_attempted};++k;
    }
    k=0;
    for(auto i=lo;i<hi;++i)if(!mapped(i)) {
        auto& src=input[k];auto& dst=managed_slots_[(size_t)i];auto& receipt=output[k++];
        const auto h=std::exchange(src.handle_,0);
        dst.incarnation=src.incarnation_;dst.fresh=src.fresh_;dst.lease=std::move(src.lease_);
        if(!map_one(i,h)) {
            if(!managed_free(h)) {
                // h_ is the only handle storage even for an uncertain detached
                // allocation. Poison forbids treating it as an accessible map.
                h_[(size_t)i]=h;managed_unknown_=true;receipt.disposition=VmmDisposition::unknown;
            } else {receipt.disposition=VmmDisposition::retired;dst.lease.reset();}
            return false;
        }
        if(!set_access(i,i+1)) {
            managed_unknown_=true;receipt.disposition=VmmDisposition::unknown;
            return false;
        }
        receipt.disposition=VmmDisposition::access_ready;
    }
    return true;
}
bool VmmRange::detach_managed(int64_t i,DetachedVmmChunk& out,VmmChunkOutcome& receipt) noexcept {
    if(!managed_ || !base_ || managed_unknown_ || !managed_domain(managed_device_) || !out.empty() || !mapped(i))return false;
    auto& slot=managed_slots_[(size_t)i];
    // unmap's strict backend policy fail-stops on uncertainty. It never clears
    // h_ before successful backend unmap.
    managed_internal_=true;const auto h=unmap(i);managed_internal_=false;
    if(!h){managed_unknown_=true;receipt={i,slot.incarnation,slot.fresh,VmmDisposition::unknown};return false;}
    out.handle_=h;out.incarnation_=slot.incarnation;out.device_=managed_device_;
    out.fresh_=false;out.lease_=std::move(slot.lease);
    receipt={i,out.incarnation_,false,VmmDisposition::detached};slot.incarnation=0;
    return true;
}
} // namespace strata::core
