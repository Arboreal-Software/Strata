#pragma once
#include "strata/core/window_codec.hpp"
#include <atomic>
namespace strata::core {
class CodecJob;
class CodecEncodedHandle {
public:
    const CodecEnvelope* get()const;
    explicit operator bool()const{return get()!=nullptr;}
private:
    struct Owned;
    std::shared_ptr<const Owned> owned_;
    friend class CodecJob;
    friend bool freeze_codec_encoded(const CodecEnvelope&,uint64_t,const CodecLimits&,const CodecIdentity&,WindowBudget&,CodecEncodedHandle&,std::string&);
};
// Deep immutable copy, admitted before allocation. This does not establish SDK
// validity: bounded LZ4 prevalidation and actual decode checks happen in the job.
bool freeze_codec_encoded(const CodecEnvelope&,uint64_t expected,const CodecLimits&,const CodecIdentity&,WindowBudget&,CodecEncodedHandle&,std::string&);
enum class CodecJobState { pending, ready, cancelled, failed, quarantined };
struct CodecJobOptions {
    int device=-1;
    // Explicit caller-priced control-handle envelope, before stream/event create.
    // Runtime hidden allocation peaks require separate hardware qualification.
    uint64_t control_host_bytes=0,control_device_bytes=0;
    bool allow_unqualified_hardware=false;
    // Optional cancellation sampled at progress/publication boundaries. For an
    // atomic race against output publication, use CodecJob::cancel().
    std::shared_ptr<std::atomic<bool>> cancellation;
};
struct CodecStartResult;
class CodecJob {
public:
    ~CodecJob();
    CodecJob(const CodecJob&)=delete;
    CodecJob& operator=(const CodecJob&)=delete;
    static CodecStartResult begin_encode(RawCodecHandle,std::shared_ptr<CodecRuntime>,const CodecLimits&,WindowBudget,const CodecJobOptions&);
    static CodecStartResult begin_decode(CodecEncodedHandle,std::shared_ptr<CodecRuntime>,const CodecLimits&,WindowBudget,const CodecJobOptions&);
    // poll/drain/take are serialized by caller. cancel alone is thread-safe;
    // cancel and take linearize on one atomic gate. A completed take wins.
    CodecJobState poll() noexcept;
    CodecJobState drain() noexcept;
    void cancel() noexcept;
    CodecJobState state() const noexcept;
    const char* error() const noexcept;
    bool take_encoded(CodecEncodedHandle&) noexcept;
    bool take_decoded(RawDecodeHandle&) noexcept;
private:
    struct Storage;
    Storage* storage_=nullptr;
    explicit CodecJob(Storage*);
    static CodecStartResult begin(RawCodecHandle,CodecEncodedHandle,std::shared_ptr<CodecRuntime>,const CodecLimits&,WindowBudget,const CodecJobOptions&,bool);
};
struct CodecStartResult {
    std::unique_ptr<CodecJob> job;
    const char* error=nullptr;
    explicit operator bool()const{return job&&!error;}
};
// Retained until worker restart, with no destructor retry. No GPU handle or
// reservation in a quarantined node is dropped or retried by this component.
uint64_t codec_job_quarantine_count() noexcept;
}
