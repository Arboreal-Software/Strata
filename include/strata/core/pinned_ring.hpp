// #1996 (0.1.43.1): host-to-device uploads whose host side is pageable go up through pinned bounce slots.
//
// A cudaMemcpyAsync from pageable memory is staged by the runtime, and the staging is not ordered the way a pinned copy
// is.  On gfx1100 (ROCm 7.2.4) a text prompt after a picture one read stale bytes that way (#1712, fixed for the step
// records and the token ids in 0.1.43); a 4-hour multi-slot load (#1996) still faulted after 31 minutes, with the
// image rows, the draft tokens and bounds and the non-mapped routing tables still going up from caller memory.  The
// analysis of which copies those are, and the idea of a ring of pinned slots with one event each, are
// mantovaniluca91's (patch C on #1712); this is our own form of it.
//
// A PinnedRing owns a few pinned slots, each with an event.  h2d() copies the bytes into the next slot (after the copy
// that last read that slot is done), queues the DMA from the slot and records the slot's event.  So the caller may
// overwrite or free its source the moment the call returns, as a pageable copy allowed.  With no pinned RAM (or
// STRATA_TEST_PAGEABLE=1) the copy is made from the caller's memory and the stream is waited for, so the source is
// free on return all the same.  A ring belongs to ONE device (the one current when it is first used) and is used by one
// thread at a time.
#pragma once

#include <cuda_runtime.h>
#include "strata/core/strict_retirement.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

class PinnedRing {
public:
    static constexpr int kSlots = 4;
    static constexpr size_t minimum_slot_bytes() noexcept { return kMinCap; }

    PinnedRing() = default;
    PinnedRing(const PinnedRing&) = delete;
    PinnedRing& operator=(const PinnedRing&) = delete;
    ~PinnedRing() { release(); }
    void set_strict_retirement(bool enabled) {
        if(owned_&&!enabled){owned_fail(cudaErrorInvalidValue);return;}
        strict_retirement_ = enabled;
    }
    bool owned_mode() const noexcept {return owned_;}
    bool owned_ready() const noexcept {return owned_&&frozen_&&first_error_==cudaSuccess;}
    cudaError_t owned_error() const noexcept {return first_error_;}
    cudaError_t reject_owned(cudaError_t error=cudaErrorInvalidValue) noexcept {return owned_fail(error);}
    // Completion knowledge only: a prior first error remains a failed operation.
    // On false, retain every slot/uncertain flag until checked retirement.
    bool drain_owned() {
        if(!owned_||dev_<0)return true;
        OwnedScratch state;state.result=cudaGetDevice(&state.device);
        if(state.result!=cudaSuccess||state.device!=dev_){owned_fail(state.result==cudaSuccess?cudaErrorInvalidValue:state.result);return false;}
        for(Slot& sl:slots_) {
            if(sl.uncertain) {
                state.result=cudaStreamSynchronize(sl.stream);
                if(state.result!=cudaSuccess){owned_fail(state.result);return false;}
                sl.uncertain=false;
            }
            if(sl.pending) {
                if(!sl.ev)owned_fail(cudaErrorInvalidValue);
                state.result=sl.ev?cudaEventSynchronize(sl.ev):cudaStreamSynchronize(sl.stream);
                if(state.result!=cudaSuccess){owned_fail(state.result);return false;}
                sl.pending=false;
            }
            sl.stream=nullptr;
        }
        return true;
    }
    static constexpr size_t owned_transient_bytes() noexcept {
        // At most three nested owned-call scratch records plus checked retirement
        // scope, wrapper/returned stage pointers and first-error result. The
        // shared quote scratch/output and caller device scopes are priced by
        // the actual workspace producer separately; no heap state is added.
        return 3*sizeof(OwnedScratch)+sizeof(StrictRetirementDevice)+2*sizeof(void*)+sizeof(cudaError_t);
    }
    bool prepare_owned(size_t bytes,cudaStream_t stream=nullptr) {
        if(owned_){owned_fail(cudaErrorInvalidValue);return false;}
        owned_=true;strict_retirement_=true;
        if(!bytes||bytes<kMinCap||bytes>SIZE_MAX/kSlots||acquired_||dev_>=0||owned_pageable_forced()) {
            owned_fail(cudaErrorInvalidValue);return false;
        }
        OwnedScratch state;state.bytes=bytes;
        state.result=cudaGetDevice(&state.device);
        if(state.result!=cudaSuccess||state.device<0){owned_fail(state.result==cudaSuccess?cudaErrorInvalidValue:state.result);return false;}
        dev_=state.device;owned_cap_=bytes;owned_stream_=stream;
        for(Slot& sl:slots_) {
            if(sl.p||sl.ev||sl.cap||sl.pending||sl.uncertain){owned_fail(cudaErrorInvalidValue);return false;}
            state.result=cudaHostAlloc(&sl.p,bytes,cudaHostAllocPortable);
            if(sl.p)sl.cap=bytes; // preserve any partial result for checked cleanup
            if(state.result!=cudaSuccess||!sl.p){owned_fail(state.result==cudaSuccess?cudaErrorInvalidValue:state.result);return false;}
            state.result=cudaEventCreateWithFlags(&sl.ev,cudaEventDisableTiming);
            if(state.result!=cudaSuccess||!sl.ev){owned_fail(state.result==cudaSuccess?cudaErrorInvalidValue:state.result);return false;}
            // Bind the actual stage stream to an event created on this device.
            // A failed record is uncertain even without a payload submission.
            sl.uncertain=true;sl.stream=stream;
            state.result=cudaEventRecord(sl.ev,stream);
            if(state.result!=cudaSuccess){owned_fail(state.result);return false;}
            sl.pending=true;sl.uncertain=false;
            state.result=cudaEventSynchronize(sl.ev);
            if(state.result!=cudaSuccess){owned_fail(state.result);return false;}
            sl.pending=false;sl.stream=nullptr;
        }
        frozen_=true;
        return owned_check(bytes);
    }

    // dst (device) <- bytes from src (host, any kind of memory); src is free to change when this returns
    cudaError_t h2d(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        if(owned_) {
            if(!owned_check(bytes))return first_error_;
            if(!bytes)return cudaSuccess;
            if(!dst||!src||s!=owned_stream_)return owned_fail(cudaErrorInvalidValue);
            void* stage=acquire(bytes);if(!stage)return first_error_;
            std::memcpy(stage,src,bytes);return submit(dst,stage,bytes,s);
        }
        if (bytes == 0) return cudaSuccess;
        void* stage = acquire(bytes);
        if (stage == nullptr) return direct(dst, src, bytes, s);
        std::memcpy(stage, src, bytes);
        return submit(dst, stage, bytes, s);
    }

    // For a caller that gathers many small pieces into one upload: acquire() hands out a pinned slot of at least
    // `bytes` (nullptr: none could be had - use direct()), the caller fills it, then submit() queues the copy and
    // guards the slot.  An acquired slot must be submitted before the next acquire().
    void* acquire(size_t bytes) {
        if(owned_)return acquire_owned(bytes);
        if (bytes == 0 || pageable_forced()) return nullptr;
        int cur = -1;
        if (cudaGetDevice(&cur) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        if (dev_ < 0) dev_ = cur;
        if (cur != dev_) return nullptr;   // a ring is for one device
        Slot& sl = slots_[next_];
        next_ = (next_ + 1) % kSlots;
        if (sl.pending) {                  // the copy that last read this slot is done before it is refilled
            sl.pending = false;
            if (retirement_result(strict_retirement_, cudaEventSynchronize(sl.ev), "pinned ring prior upload") != cudaSuccess) { cudaGetLastError(); return nullptr; }
        }
        if (sl.cap < bytes) {
            if (sl.p) { retirement_check(strict_retirement_, cudaFreeHost(sl.p), "pinned ring replacement"); sl.p = nullptr; sl.cap = 0; }   // idle: its last copy was waited for above
            const size_t cap = bytes < kMinCap ? kMinCap : bytes;
            if (cudaHostAlloc(&sl.p, cap, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                sl.p = nullptr;
                return nullptr;
            }
            sl.cap = cap;
        }
        if (sl.ev == nullptr && cudaEventCreateWithFlags(&sl.ev, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            sl.ev = nullptr;
            return nullptr;
        }
        acquired_ = &sl;
        return sl.p;
    }

    cudaError_t submit(void* dst, const void* stage, size_t bytes, cudaStream_t s) {
        if(owned_)return submit_owned(dst,stage,bytes,s);
        Slot* sl = acquired_;
        acquired_ = nullptr;
        cudaError_t e = cudaMemcpyAsync(dst, stage, bytes, cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "strata: pinned upload of %zu bytes failed: %s\n", bytes, cudaGetErrorString(e));
            return e;
        }
        if (sl != nullptr && cudaEventRecord(sl->ev, s) == cudaSuccess) {
            sl->pending = true;
            return cudaSuccess;
        }
        cudaGetLastError();                // no event: the copy is waited for here instead
        return cudaStreamSynchronize(s);
    }

    // the same copy without a pinned slot: from the caller's memory, the stream waited for
    static cudaError_t direct(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        if (bytes == 0) return cudaSuccess;
        cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) std::fprintf(stderr, "strata: upload of %zu bytes failed: %s\n", bytes, cudaGetErrorString(e));
        return e;
    }

    // dst (pageable host) <- device bytes; the stream is drained and the bytes are in dst on return
    cudaError_t d2h_sync(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        if(owned_) {
            if(!owned_check(bytes))return first_error_;
            if(!bytes)return cudaSuccess;
            if(!dst||!src||s!=owned_stream_)return owned_fail(cudaErrorInvalidValue);
            OwnedScratch state;state.bytes=bytes;state.stream=s;
            void* stage=acquire(bytes);if(!stage)return first_error_;
            state.slot=acquired_;state.slot->uncertain=true;state.slot->stream=s;
            acquired_=nullptr;acquired_bytes_=0;
            state.result=cudaMemcpyAsync(stage,src,bytes,cudaMemcpyDeviceToHost,s);
            if(state.result!=cudaSuccess)return owned_fail(state.result);
            state.result=cudaStreamSynchronize(s);
            if(state.result!=cudaSuccess)return owned_fail(state.result);
            state.slot->uncertain=false;state.slot->stream=nullptr;
            std::memcpy(dst,stage,bytes);return cudaSuccess;
        }
        if (bytes == 0) return cudaSuccess;
        void* stage = acquire(bytes);
        cudaError_t e = cudaMemcpyAsync(stage != nullptr ? stage : dst, src, bytes, cudaMemcpyDeviceToHost, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e == cudaSuccess && stage != nullptr) std::memcpy(dst, stage, bytes);
        if (stage != nullptr) acquired_ = nullptr;   // no event: the stream was drained
        if (e != cudaSuccess) std::fprintf(stderr, "strata: download of %zu bytes failed: %s\n", bytes, cudaGetErrorString(e));
        return e;
    }

    // Every slot holds at least `bytes` from here on: grown now, by a caller that must not allocate later (a pinned
    // allocation or free takes the driver's lock and can wait for the device: STRATA_PIPED_CONCURRENT grows the rings
    // before a read whose windows run beside it).  On the device that is current, as acquire().  False: a slot could
    // not be had (its uploads then grow it, or go up from the caller's memory, as before).
    bool reserve(size_t bytes) {
        if(owned_){
            if(!owned_check(bytes))return false;
            if(bytes&&acquired_){owned_fail(cudaErrorInvalidValue);return false;}
            return true;
        }
        if (bytes == 0 || pageable_forced()) return true;
        int cur = -1;
        if (cudaGetDevice(&cur) != cudaSuccess) { cudaGetLastError(); return false; }
        if (dev_ < 0) dev_ = cur;
        if (cur != dev_) return false;
        for (Slot& sl : slots_) {
            if (sl.cap >= bytes && sl.ev != nullptr) continue;
            if (sl.pending) {
                sl.pending = false;
                if (retirement_result(strict_retirement_, cudaEventSynchronize(sl.ev), "pinned ring prior upload") != cudaSuccess) { cudaGetLastError(); return false; }
            }
            if (sl.cap < bytes) {
                if (sl.p) { retirement_check(strict_retirement_, cudaFreeHost(sl.p), "pinned ring replacement"); sl.p = nullptr; sl.cap = 0; }
                const size_t cap = bytes < kMinCap ? kMinCap : bytes;
                if (cudaHostAlloc(&sl.p, cap, cudaHostAllocPortable) != cudaSuccess) {
                    cudaGetLastError();
                    sl.p = nullptr;
                    return false;
                }
                sl.cap = cap;
            }
            if (sl.ev == nullptr && cudaEventCreateWithFlags(&sl.ev, cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                sl.ev = nullptr;
                return false;
            }
        }
        return true;
    }

    // Caller seals producers; failure is terminal before a slot can be reused.
    void retire_strict() {
        if(owned_){retire_owned();return;}
        StrictRetirementDevice device(dev_ >= 0, dev_);
        for (Slot& sl : slots_) {
            if (sl.pending && sl.ev) retirement_check(true, cudaEventSynchronize(sl.ev), "pinned ring drain");
            if (sl.ev) retirement_check(true, cudaEventDestroy(sl.ev), "pinned ring event");
            if (sl.p) retirement_check(true, cudaFreeHost(sl.p), "pinned ring memory");
            sl = Slot{};
        }
        acquired_ = nullptr; dev_ = -1; next_ = 0;
    }

    void release() {
        if(owned_){retire_owned();return;}
        if (strict_retirement_) { retire_strict(); return; }
        for (Slot& sl : slots_) {
            if (sl.pending && sl.ev) cudaEventSynchronize(sl.ev);
            if (sl.ev) cudaEventDestroy(sl.ev);
            if (sl.p) cudaFreeHost(sl.p);
            sl = Slot{};
        }
        cudaGetLastError();
        acquired_ = nullptr;
        dev_ = -1;
        next_ = 0;
    }

private:
    bool strict_retirement_ = false;
    bool owned_=false,frozen_=false;
    size_t owned_cap_=0,acquired_bytes_=0;
    cudaStream_t owned_stream_=nullptr;
    cudaError_t first_error_=cudaSuccess;
    static constexpr size_t kMinCap = 64 * 1024;
    struct Slot {
        void* p = nullptr;
        size_t cap = 0;
        cudaEvent_t ev = nullptr;
        bool pending = false;
        bool uncertain = false;
        cudaStream_t stream = nullptr;
    };
    struct OwnedScratch {
        Slot* slot=nullptr;
        size_t bytes=0;
        cudaStream_t stream=nullptr;
        int device=-1;
        cudaError_t result=cudaSuccess;
    };
    cudaError_t owned_fail(cudaError_t error) noexcept {
        if(first_error_==cudaSuccess&&error!=cudaSuccess)first_error_=error;
        return first_error_;
    }
    static bool owned_pageable_forced() {
        const char* value=std::getenv("STRATA_TEST_PAGEABLE");return value&&value[0]=='1';
    }
    bool owned_check(size_t bytes) {
        if(first_error_!=cudaSuccess)return false;
        if(!frozen_||bytes>owned_cap_||owned_pageable_forced()){owned_fail(cudaErrorInvalidValue);return false;}
        OwnedScratch state;state.result=cudaGetDevice(&state.device);
        if(state.result!=cudaSuccess||state.device!=dev_){owned_fail(state.result==cudaSuccess?cudaErrorInvalidValue:state.result);return false;}
        for(const Slot& sl:slots_)if(!sl.p||!sl.ev||sl.cap!=owned_cap_){owned_fail(cudaErrorInvalidValue);return false;}
        return true;
    }
    void* acquire_owned(size_t bytes) {
        if(!owned_check(bytes))return nullptr;
        if(!bytes)return nullptr;
        if(acquired_){owned_fail(cudaErrorInvalidValue);return nullptr;}
        OwnedScratch state;state.slot=&slots_[next_];state.bytes=bytes;
        if(state.slot->uncertain){owned_fail(cudaErrorInvalidValue);return nullptr;}
        if(state.slot->pending) {
            state.result=cudaEventSynchronize(state.slot->ev);
            if(state.result!=cudaSuccess){owned_fail(state.result);return nullptr;}
            state.slot->pending=false;state.slot->stream=nullptr;
        }
        acquired_=state.slot;acquired_bytes_=bytes;next_=(next_+1)%kSlots;
        return state.slot->p;
    }
    cudaError_t submit_owned(void* dst,const void* stage,size_t bytes,cudaStream_t stream) {
        if(!owned_check(bytes))return first_error_;
        if(!bytes&&!acquired_)return cudaSuccess;
        if(!dst||!acquired_||stage!=acquired_->p||bytes!=acquired_bytes_||stream!=owned_stream_)return owned_fail(cudaErrorInvalidValue);
        OwnedScratch state;state.slot=acquired_;state.stream=stream;state.bytes=bytes;
        // Even a failing submission may be uncertain. Keep the stream/slot paid
        // until checked retirement; recording an error is never a drain receipt.
        state.slot->uncertain=true;state.slot->stream=stream;
        acquired_=nullptr;acquired_bytes_=0;
        state.result=cudaMemcpyAsync(dst,stage,bytes,cudaMemcpyHostToDevice,stream);
        if(state.result!=cudaSuccess)return owned_fail(state.result);
        state.result=cudaEventRecord(state.slot->ev,stream);
        if(state.result!=cudaSuccess)return owned_fail(state.result);
        state.slot->pending=true;state.slot->uncertain=false;
        return cudaSuccess;
    }
    void retire_owned() {
        StrictRetirementDevice device(dev_>=0,dev_);
        if(!drain_owned())retirement_failed("owned pinned ring uncertain submission drain");
        for(Slot& sl:slots_) {
            if(sl.ev)retirement_check(true,cudaEventDestroy(sl.ev),"owned pinned ring event");
            if(sl.p)retirement_check(true,cudaFreeHost(sl.p),"owned pinned ring memory");
            sl=Slot{};
        }
        acquired_=nullptr;acquired_bytes_=0;dev_=-1;next_=0;frozen_=false;
    }
    static bool pageable_forced() {
        static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
        return v;
    }
    Slot slots_[kSlots];
    Slot* acquired_ = nullptr;
    int next_ = 0;
    int dev_ = -1;
};

}  // namespace strata::core
