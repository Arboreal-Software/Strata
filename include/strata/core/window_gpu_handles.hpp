#pragma once
#include "strata/core/strict_retirement.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace strata::core {
// Owns actual driver handles. Callers close/join every producer before retirement
// (including destruction); successful device drain alone cannot stop a producer.
// Serialized caller only. No callbacks or registry locks. CUDA/HIP opaque driver
// allocation bytes are unknown: metadata_capacity_bytes is vector storage only.
class OwnedWindowGpuHandles {
public:
    OwnedWindowGpuHandles(size_t max_handles, size_t max_devices)
        : max_handles_(max_handles), max_devices_(max_devices) {
        handles_.reserve(max_handles); devices_.reserve(max_devices);
    }
    OwnedWindowGpuHandles(const OwnedWindowGpuHandles&) = delete;
    OwnedWindowGpuHandles& operator=(const OwnedWindowGpuHandles&) = delete;
    OwnedWindowGpuHandles(OwnedWindowGpuHandles&&) = delete;
    OwnedWindowGpuHandles& operator=(OwnedWindowGpuHandles&&) = delete;
    ~OwnedWindowGpuHandles() { retire_after_producers_closed(); }

    bool declare_consumer_device(int device, std::string& error) {
        if (sealed_) { error="GPU handle owner is sealed"; return false; }
        if (device<0) { error="negative GPU consumer device"; return false; }
        for (int existing:devices_) if (existing==device) return true;
        if (devices_.size()==max_devices_) { error="GPU consumer device limit"; return false; }
        devices_.push_back(device); // capacity was prepared before any device work
        return true;
    }
    bool create_stream(int device, unsigned flags, cudaStream_t& out, std::string& error) {
        if (!prepare(device,error)) return false;
        Handle& node=handles_.back(); node.kind=Kind::stream;
        const int previous=select(device);
        const auto status=cudaStreamCreateWithFlags(&node.stream,flags);
        if (status!=cudaSuccess) {
            if (node.stream || status!=cudaErrorMemoryAllocation)
                retirement_failed("uncertain stream creation");
            check(cudaSetDevice(previous),"restore after stream allocation refusal");
            handles_.pop_back(); error="stream allocation refused"; return false;
        }
        if (!node.stream) retirement_failed("stream creation returned null");
        check(cudaSetDevice(previous),"restore after stream creation");
        out=node.stream; return true;
    }
    bool create_event(int device, unsigned flags, cudaEvent_t& out, std::string& error) {
        if (!prepare(device,error)) return false;
        Handle& node=handles_.back(); node.kind=Kind::event;
        const int previous=select(device);
        const auto status=cudaEventCreateWithFlags(&node.event,flags);
        if (status!=cudaSuccess) {
            if (node.event || status!=cudaErrorMemoryAllocation)
                retirement_failed("uncertain event creation");
            check(cudaSetDevice(previous),"restore after event allocation refusal");
            handles_.pop_back(); error="event allocation refused"; return false;
        }
        if (!node.event) retirement_failed("event creation returned null");
        check(cudaSetDevice(previous),"restore after event creation");
        out=node.event; return true;
    }
    // Seals creation and drains without destroying borrowed streams/events.
    // No producer may submit work after this call; consumers may synchronize
    // these still-live handles while retiring their own resources afterward.
    void drain_after_producers_closed() noexcept {
        if (drained_) return;
        sealed_=true;
        if (!handles_.empty()) {
            int previous=-1; check(cudaGetDevice(&previous),"get GPU handle drain device");
            for (int device:devices_) {
                check(cudaSetDevice(device),"select GPU handle consumer device");
                check(cudaDeviceSynchronize(),"drain GPU handle consumer device");
            }
            check(cudaSetDevice(previous),"restore GPU handle drain device");
        }
        drained_=true;
    }
    void retire_after_producers_closed() noexcept {
        if (retired_) return;
        drain_after_producers_closed();
        if (!handles_.empty()) {
            int previous=-1; check(cudaGetDevice(&previous),"get GPU handle retirement device");
            for (auto& node:handles_) {
                check(cudaSetDevice(node.device),"select GPU handle creation device");
                if (node.kind==Kind::event && node.event) {
                    check(cudaEventDestroy(node.event),"destroy owned event"); node.event=nullptr;
                } else if (node.kind==Kind::stream && node.stream) {
                    check(cudaStreamDestroy(node.stream),"destroy owned stream"); node.stream=nullptr;
                }
            }
            check(cudaSetDevice(previous),"restore GPU handle retirement device");
        }
        handles_.clear(); retired_=true;
    }
    size_t stream_count() const noexcept {size_t n=0;for(const auto& h:handles_)if(h.stream)++n;return n;}
    size_t event_count() const noexcept {size_t n=0;for(const auto& h:handles_)if(h.event)++n;return n;}
    size_t device_count() const noexcept {return devices_.size();}
    size_t metadata_capacity_bytes() const noexcept {
        return handles_.capacity()*sizeof(Handle)+devices_.capacity()*sizeof(int);
    }
private:
    enum class Kind {stream,event};
    struct Handle {Kind kind=Kind::stream;int device=-1;cudaStream_t stream=nullptr;cudaEvent_t event=nullptr;};
    static void check(cudaError_t status,const char* operation) noexcept {
        if(status!=cudaSuccess)retirement_failed(operation);
    }
    static int select(int device) noexcept {
        int previous=-1;check(cudaGetDevice(&previous),"get GPU handle creation device");
        check(cudaSetDevice(device),"select GPU handle creation device");return previous;
    }
    bool prepare(int device,std::string& error) {
        if(sealed_){error="GPU handle owner is sealed";return false;}
        if(handles_.size()==max_handles_){error="GPU handle limit";return false;}
        if(!declare_consumer_device(device,error))return false;
        handles_.emplace_back();handles_.back().device=device;return true;
    }
    size_t max_handles_,max_devices_;
    std::vector<Handle> handles_;
    std::vector<int> devices_;
    bool sealed_=false, drained_=false, retired_=false;
};
} // namespace strata::core
