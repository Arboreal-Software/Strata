#pragma once
#include <cstdint>
#include <string>
namespace strata::core {
enum class CodecBackend : uint32_t { raw=0, nvcomp=1, hipcomp=2 };
enum class CodecKind : uint32_t { raw=0, lz4=1 };
struct CodecIdentity {
    CodecBackend backend=CodecBackend::raw;
    CodecKind kind=CodecKind::raw;
    uint32_t sdk_major=0,sdk_minor=0,sdk_patch=0,framing=1;
    // Versioned byte-oriented options; no vendor high-level frame is accepted.
    uint32_t options=0;
    std::string build_pin; // Exact SDK package/source variant, not just semantic version.
    bool operator==(const CodecIdentity&) const=default;
};
struct CodecCapability {
    CodecIdentity identity;
    bool compiled=false,abi_verified=false,planning=false;
    // Execution remains gated until actual hardware qualification.
    bool execution=false;
    std::string reason;
};
struct CodecQuery {int device=-1;void* stream=nullptr;uint64_t chunks=0,max_chunk_bytes=0,total_bytes=0;};
struct CodecBackendPlan {
    CodecCapability capability;
    CodecQuery query; // Sizing is valid only for this exact batch/device/stream.
    uint64_t input_alignment=0,output_alignment=0,temp_alignment=0;
    uint64_t max_output_chunk_bytes=0,encode_workspace_bytes=0,decode_workspace_bytes=0;
};
// Explicit device/stream queries only. Disabled adapters never initialize a GPU.
// The caller supplies its owned stream; these methods select/restore the device.
// True proves SDK plan-query compatibility, not hardware execution qualification.
CodecCapability nvcomp_codec_capability();
CodecCapability hipcomp_codec_capability();
bool query_nvcomp_codec(const CodecQuery&,CodecBackendPlan&,std::string&);
bool query_hipcomp_codec(const CodecQuery&,CodecBackendPlan&,std::string&);
}

// Async runtime boundary: each implementation uses its own SDK types. These
// methods never throw; false may follow a partial effect and requires drain.
#include <cstddef>
#include <memory>
namespace strata::core {
enum class CodecCopy { host_to_device, device_to_host };
enum class CodecEventPoll { complete, pending, failed };
struct CodecDeviceBatch {
    const void* const* inputs=nullptr;
    const size_t* input_bytes=nullptr;
    const size_t* capacities=nullptr;
    void* const* outputs=nullptr;
    size_t* actual_bytes=nullptr;
    uint32_t* statuses=nullptr;
    size_t chunks=0,max_input_chunk=0;
};
class CodecRuntime {
public:
    virtual ~CodecRuntime()=default;
    virtual CodecCapability capability() const=0;
    virtual bool hardware_qualified() const noexcept{return false;}
    virtual bool compression_statuses() const noexcept=0;
    virtual bool get_device(int&) noexcept=0;
    virtual bool set_device(int) noexcept=0;
    virtual bool stream_create(void*&) noexcept=0;
    virtual bool stream_destroy(void*) noexcept=0;
    virtual bool stream_drain(void*) noexcept=0;
    virtual bool event_create(void*&) noexcept=0;
    virtual bool event_record(void*,void*) noexcept=0;
    virtual CodecEventPoll event_poll(void*) noexcept=0;
    virtual bool event_destroy(void*) noexcept=0;
    virtual bool device_allocate(size_t,void*&) noexcept=0;
    virtual bool device_free(void*) noexcept=0;
    virtual bool pinned_allocate(size_t,void*&) noexcept=0;
    virtual bool pinned_free(void*) noexcept=0;
    virtual bool copy(void*,const void*,size_t,CodecCopy,void*) noexcept=0;
    virtual bool query(const CodecQuery&,CodecBackendPlan&) noexcept=0;
    virtual bool compress(const CodecDeviceBatch&,void*,size_t,void*) noexcept=0;
    virtual bool decompress(const CodecDeviceBatch&,void*,size_t,void*) noexcept=0;
};
// Null when the individually pinned SDK was not compiled. Factories do not call
// GPU APIs. Real runtime hardware_qualified stays false until qualification.
std::shared_ptr<CodecRuntime> make_nvcomp_codec_runtime();
std::shared_ptr<CodecRuntime> make_hipcomp_codec_runtime();
}
