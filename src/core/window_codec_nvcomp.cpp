#include "strata/core/window_codec_backend.hpp"
#include <algorithm>
#include <cstddef>
#if defined(STRATA_WINDOW_NVCOMP_600)
#include <nvcomp.h>
#include <nvcomp/lz4.h>
static_assert(NVCOMP_VER_MAJOR==6&&NVCOMP_VER_MINOR==0&&NVCOMP_VER_PATCH==0&&NVCOMP_VER_BUILD==12&&CUDART_VERSION/1000==13,"nvCOMP adapter requires pinned6.0.0 SDK");
#endif
namespace strata::core {
CodecCapability nvcomp_codec_capability(){
    CodecCapability c;c.identity={CodecBackend::nvcomp,CodecKind::lz4,6,0,0,1,0,"nvcomp-6.0.0.12-cu13"};
#if defined(STRATA_WINDOW_NVCOMP_600)
    c.compiled=true;c.reason="SDK linked; explicit per-device sizing required; hardware execution unqualified";
#else
    c.reason="nvCOMP6.0.0 SDK adapter not compiled; independent RAW available";
#endif
    return c;
}
bool query_nvcomp_codec(const CodecQuery& q,CodecBackendPlan& out,std::string& error){
#if !defined(STRATA_WINDOW_NVCOMP_600)
    (void)q;(void)out;error=nvcomp_codec_capability().reason;return false;
#else
    if(q.device<0||!q.chunks||!q.max_chunk_bytes||q.max_chunk_bytes>nvcompLZ4CompressionMaxAllowedChunkSize||q.chunks>SIZE_MAX||q.total_bytes>SIZE_MAX||q.total_bytes<q.max_chunk_bytes||q.chunks>q.total_bytes||(q.total_bytes-1)/q.chunks>=q.max_chunk_bytes){error="invalid nvCOMP sizing request";return false;}
    nvcompProperties_t properties{};
    if(nvcompGetProperties(&properties)!=nvcompSuccess||properties.version!=NVCOMP_VER||properties.cudart_version/1000!=CUDART_VERSION/1000){error="nvCOMP runtime/header/CUDA version mismatch";return false;}
    int previous=-1;if(cudaGetDevice(&previous)!=cudaSuccess){error="nvCOMP current device query failed";return false;}
    if(previous!=q.device&&cudaSetDevice(q.device)!=cudaSuccess){
        const auto restored=cudaSetDevice(previous);error=restored==cudaSuccess?"nvCOMP device selection failed":"nvCOMP device selection/restore failed";return false;
    }
    const auto compression=nvcompBatchedLZ4CompressDefaultOpts;
    auto decompression=nvcompBatchedLZ4DecompressDefaultOpts;
    // Never silently route malformed data to a hardware decompression engine.
    decompression.backend=NVCOMP_DECOMPRESS_BACKEND_CUDA;
    nvcompAlignmentRequirements_t ca{},da{};size_t ce=0,de=0,maximum=0;
    const bool ok=nvcompBatchedLZ4CompressGetRequiredAlignments(compression,&ca)==nvcompSuccess&&
        nvcompBatchedLZ4DecompressGetRequiredAlignments(decompression,&da)==nvcompSuccess&&
        nvcompBatchedLZ4CompressGetTempSize(size_t(q.chunks),size_t(q.max_chunk_bytes),compression,&ce,size_t(q.total_bytes),static_cast<cudaStream_t>(q.stream))==nvcompSuccess&&
        nvcompBatchedLZ4DecompressGetTempSize(size_t(q.chunks),size_t(q.max_chunk_bytes),decompression,&de,size_t(q.total_bytes),static_cast<cudaStream_t>(q.stream))==nvcompSuccess&&
        nvcompBatchedLZ4CompressGetMaxOutputChunkSize(size_t(q.max_chunk_bytes),compression,&maximum)==nvcompSuccess;
    if(previous!=q.device&&cudaSetDevice(previous)!=cudaSuccess){error="nvCOMP sizing device restore failed";return false;}
    if(!ok){error="nvCOMP SDK sizing failed";return false;}
    CodecBackendPlan p;p.query=q;p.capability=nvcomp_codec_capability();p.capability.abi_verified=true;p.capability.planning=true;
    p.input_alignment=std::max(ca.input,da.output);p.output_alignment=std::max(ca.output,da.input);p.temp_alignment=std::max(ca.temp,da.temp);
    p.encode_workspace_bytes=ce;p.decode_workspace_bytes=de;p.max_output_chunk_bytes=maximum;out=std::move(p);return true;
#endif
}
}

namespace strata::core {
#if defined(STRATA_WINDOW_NVCOMP_600)
namespace {
class Runtime final:public CodecRuntime {
public:
 CodecCapability capability()const override{return nvcomp_codec_capability();}
 bool compression_statuses()const noexcept override{return true;}
 bool get_device(int& d)noexcept override{return cudaGetDevice(&d)==cudaSuccess;}
 bool set_device(int d)noexcept override{return cudaSetDevice(d)==cudaSuccess;}
 bool stream_create(void*& p)noexcept override{return cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t*>(&p),cudaStreamNonBlocking)==cudaSuccess;}
 bool stream_destroy(void* p)noexcept override{return cudaStreamDestroy(static_cast<cudaStream_t>(p))==cudaSuccess;}
 bool stream_drain(void* p)noexcept override{return cudaStreamSynchronize(static_cast<cudaStream_t>(p))==cudaSuccess;}
 bool event_create(void*& p)noexcept override{return cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t*>(&p),cudaEventDisableTiming)==cudaSuccess;}
 bool event_record(void* p,void* stream)noexcept override{return cudaEventRecord(static_cast<cudaEvent_t>(p),static_cast<cudaStream_t>(stream))==cudaSuccess;}
 CodecEventPoll event_poll(void* p)noexcept override{auto e=cudaEventQuery(static_cast<cudaEvent_t>(p));return e==cudaSuccess?CodecEventPoll::complete:(e==cudaErrorNotReady?CodecEventPoll::pending:CodecEventPoll::failed);}
 bool event_destroy(void* p)noexcept override{return cudaEventDestroy(static_cast<cudaEvent_t>(p))==cudaSuccess;}
 bool device_allocate(size_t n,void*& p)noexcept override{return cudaMalloc(&p,n)==cudaSuccess;}
 bool device_free(void* p)noexcept override{return cudaFree(p)==cudaSuccess;}
 bool pinned_allocate(size_t n,void*& p)noexcept override{return cudaHostAlloc(&p,n,cudaHostAllocDefault)==cudaSuccess;}
 bool pinned_free(void* p)noexcept override{return cudaFreeHost(p)==cudaSuccess;}
 bool copy(void* d,const void* s,size_t n,CodecCopy kind,void* stream)noexcept override{return cudaMemcpyAsync(d,s,n,kind==CodecCopy::host_to_device?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,static_cast<cudaStream_t>(stream))==cudaSuccess;}
 bool query(const CodecQuery& q,CodecBackendPlan& p)noexcept override{try{std::string error;return query_nvcomp_codec(q,p,error);}catch(...){return false;}}
 bool compress(const CodecDeviceBatch& b,void* temp,size_t bytes,void* stream)noexcept override{return nvcompBatchedLZ4CompressAsync(b.inputs,b.input_bytes,b.max_input_chunk,b.chunks,temp,bytes,b.outputs,b.actual_bytes,nvcompBatchedLZ4CompressDefaultOpts,reinterpret_cast<nvcompStatus_t*>(b.statuses),static_cast<cudaStream_t>(stream))==nvcompSuccess;}
 bool decompress(const CodecDeviceBatch& b,void* temp,size_t bytes,void* stream)noexcept override{auto opts=nvcompBatchedLZ4DecompressDefaultOpts;opts.backend=NVCOMP_DECOMPRESS_BACKEND_CUDA;return nvcompBatchedLZ4DecompressAsync(b.inputs,b.input_bytes,b.capacities,b.actual_bytes,b.chunks,temp,bytes,b.outputs,opts,reinterpret_cast<nvcompStatus_t*>(b.statuses),static_cast<cudaStream_t>(stream))==nvcompSuccess;}
};
static_assert(sizeof(nvcompStatus_t)==sizeof(uint32_t));
}
#endif
std::shared_ptr<CodecRuntime> make_nvcomp_codec_runtime(){
#if defined(STRATA_WINDOW_NVCOMP_600)
 return std::make_shared<Runtime>();
#else
 return {};
#endif
}
}
