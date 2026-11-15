#pragma once
#include "strata/core/window_codec_backend.hpp"
#include "strata/core/window_budget.hpp"
#include "strata/core/window_hash.hpp"
#include <span>
#include <array>
#include <vector>
namespace strata::core {
struct CodecLimits {
    uint64_t max_chunks=0,max_chunk_bytes=0,max_total_bytes=0;
    uint64_t max_encoded_bytes=0,max_workspace_bytes=0,max_metadata_bytes=0;
};
struct CodecChunk {
    CodecKind kind=CodecKind::raw;
    uint64_t logical_offset=0,logical_bytes=0,encoded_offset=0,encoded_bytes=0;
    StateDigest digest{};
};
struct CodecEnvelope {
    CodecIdentity identity;
    uint64_t logical_bytes=0;
    std::vector<CodecChunk> chunks;
    std::vector<uint8_t> payload;
};
enum class CodecDirection { encode, decode };
struct CodecPlanChunk {
    uint64_t logical_bytes=0,encoded_bytes=0;
    CodecKind kind=CodecKind::raw;
};
struct CodecPlanOptions {
    CodecDirection direction=CodecDirection::encode;
    int device=-1;
    uint32_t device_count=0;
    void* intended_stream=nullptr;
    uint64_t control_host_bytes=0,control_device_bytes=0;
};
struct CodecBufferLayout {
    std::vector<uint64_t> raw_offsets,compressed_offsets;
    uint64_t raw_bytes=0,compressed_bytes=0,metadata_bytes=0,output_capacity=0;
    // Logical sizes; each nonempty allocation charges size+alignment-1.
    std::array<uint64_t,5> device_sizes{},device_alignments{};
    std::array<uint64_t,4> pinned_sizes{};
    uint64_t pinned_alignment=alignof(size_t);
    WindowResources control,transient,result;
};
struct CodecPlan {
    CodecIdentity identity;
    std::vector<uint64_t> chunk_bytes;
    uint64_t maximum_output_bytes=0,workspace_bytes=0,metadata_bytes=0;
    uint64_t input_alignment=1,output_alignment=1,temp_alignment=1,input_staging_bytes=0;
    CodecBufferLayout layout;
    // Concurrent incremental job charges: controls + transient + output owner.
    // Existing immutable input ownership remains separately charged. Caller SDK
    // control envelopes are declarations, not measured hidden SDK allocations.
    WindowResources incremental_peak;
    bool raw_fallback=false;
};
// Same checked host-control envelope used before any async control creation.
bool codec_control_host_charge(uint64_t chunks,const CodecLimits&,uint64_t requested,
                               uint64_t& charge,std::string&);
// Shared output-owner charge; chunks is zero for a plain decoded byte vector.
bool codec_output_host_charge(uint64_t chunks,uint64_t capacity,uint64_t& charge) noexcept;
// Exact declared async-job layout, bound to the intended SDK batch/device/stream.
// Decode includes ALL chunks; only LZ4 chunks enter the SDK query. Missing SDK
// selects independent RAW encode fallback; compressed decode cannot fall back.
bool plan_window_codec(CodecBackend,std::span<const CodecPlanChunk>,const CodecLimits&,
                       const CodecPlanOptions&,const CodecBackendPlan*,CodecPlan&,std::string&);
// Structural validation only: decode must also verify every uncompressed digest.
// Validation uses trusted total extent and exact allowed identity. No implicit
// cross-vendor/version compatibility, and no decompressed-size allocation query.
bool validate_codec_envelope(const CodecEnvelope&,uint64_t expected_bytes,const CodecLimits&,
                             const CodecIdentity& allowed,std::string&);
class RawCodecHandle {
public:
    const CodecEnvelope* get() const{return envelope_.get();}
    explicit operator bool()const{return bool(envelope_);}
private:
    std::shared_ptr<const ReservationLease> lease_;
    std::shared_ptr<const CodecEnvelope> envelope_;
    friend bool encode_window_raw(std::span<const uint8_t>,uint64_t,const CodecLimits&,WindowBudget&,RawCodecHandle&,std::string&);
};
class CodecJob;
class RawDecodeHandle {
public:
    std::span<const uint8_t> bytes()const{return data_?std::span<const uint8_t>(*data_):std::span<const uint8_t>{};}
    explicit operator bool()const{return bool(data_);}
private:
    std::shared_ptr<const ReservationLease> lease_;
    std::shared_ptr<const std::vector<uint8_t>> data_;
    friend class CodecJob;
    friend bool decode_window_raw(const CodecEnvelope&,uint64_t,const CodecLimits&,WindowBudget&,RawDecodeHandle&,std::string&);
};
// These operations never call a GPU/SDK. Returned immutable data retains its
// reservation. Failed admission/validation/allocation leaves output unchanged.
// Host charges are conservative declared envelopes, not allocator/RSS measures.
bool encode_window_raw(std::span<const uint8_t>,uint64_t chunk_bytes,const CodecLimits&,
                       WindowBudget&,RawCodecHandle&,std::string&);
bool decode_window_raw(const CodecEnvelope&,uint64_t expected_bytes,const CodecLimits&,
                       WindowBudget&,RawDecodeHandle&,std::string&);
}
