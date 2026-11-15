#include "strata/core/window_codec_backend.hpp"
#include <cstddef>
#if defined(STRATA_WINDOW_HIPCOMP_230)
#include <hipcomp.h>
#include <hipcomp/lz4.h>
static_assert(HIPCOMP_MAJOR_VERSION==2&&HIPCOMP_MINOR_VERSION==3&&HIPCOMP_PATCH_VERSION==0,"hipCOMP adapter requires pinned2.3.0 SDK");
#endif
namespace strata::core {
CodecCapability hipcomp_codec_capability(){
    CodecCapability c;c.identity={CodecBackend::hipcomp,CodecKind::lz4,2,3,0,1,0,"hipcomp-22cc762f54fba7cdfca74a4c50c00f2aac4ace7a"};
#if defined(STRATA_WINDOW_HIPCOMP_230)
    c.compiled=true;c.reason="Pinned hipCOMP2.3.0 linked; explicit sizing required; hardware execution unqualified";
#else
    c.reason="hipCOMP2.3.0 SDK adapter not compiled; independent RAW available";
#endif
    return c;
}
bool query_hipcomp_codec(const CodecQuery& q,CodecBackendPlan& out,std::string& error){
#if !defined(STRATA_WINDOW_HIPCOMP_230)
    (void)q;(void)out;error=hipcomp_codec_capability().reason;return false;
#else
    if(q.device<0||!q.chunks||!q.max_chunk_bytes||q.max_chunk_bytes>(1u<<24)||q.chunks>SIZE_MAX||q.total_bytes>SIZE_MAX||q.total_bytes<q.max_chunk_bytes||q.chunks>q.total_bytes||(q.total_bytes-1)/q.chunks>=q.max_chunk_bytes){error="invalid hipCOMP sizing request";return false;}
    int previous=-1;if(hipGetDevice(&previous)!=hipSuccess){error="hipCOMP current device query failed";return false;}
    if(previous!=q.device&&hipSetDevice(q.device)!=hipSuccess){const auto restored=hipSetDevice(previous);error=restored==hipSuccess?"hipCOMP device selection failed":"hipCOMP device selection/restore failed";return false;}
    size_t ce=0,de=0,maximum=0;
    const bool ok=hipcompBatchedLZ4CompressGetTempSize(size_t(q.chunks),size_t(q.max_chunk_bytes),hipcompBatchedLZ4DefaultOpts,&ce)==hipcompSuccess&&
        hipcompBatchedLZ4DecompressGetTempSize(size_t(q.chunks),size_t(q.max_chunk_bytes),&de)==hipcompSuccess&&
        hipcompBatchedLZ4CompressGetMaxOutputChunkSize(size_t(q.max_chunk_bytes),hipcompBatchedLZ4DefaultOpts,&maximum)==hipcompSuccess;
    if(previous!=q.device&&hipSetDevice(previous)!=hipSuccess){error="hipCOMP sizing device restore failed";return false;}
    if(!ok){error="hipCOMP SDK sizing failed";return false;}
    // hipCOMP has no inspected runtime-version query. Build/package provenance
    // must keep the pinned header/library pair together; successful queries alone
    // are not a cross-version or cross-vendor format compatibility declaration.
    CodecBackendPlan p;p.query=q;p.capability=hipcomp_codec_capability();p.capability.abi_verified=true;p.capability.planning=true;
    p.input_alignment=p.output_alignment=p.temp_alignment=hipcompLZ4RequiredAlignment;
    p.encode_workspace_bytes=ce;p.decode_workspace_bytes=de;p.max_output_chunk_bytes=maximum;out=std::move(p);return true;
#endif
}
}

namespace strata::core {
#if defined(STRATA_WINDOW_HIPCOMP_230)
namespace {
class Runtime final:public CodecRuntime {
public:
 CodecCapability capability()const override{return hipcomp_codec_capability();}
 bool compression_statuses()const noexcept override{return false;}
 bool get_device(int& d)noexcept override{return hipGetDevice(&d)==hipSuccess;}
 bool set_device(int d)noexcept override{return hipSetDevice(d)==hipSuccess;}
 bool stream_create(void*& p)noexcept override{return hipStreamCreateWithFlags(reinterpret_cast<hipStream_t*>(&p),hipStreamNonBlocking)==hipSuccess;}
 bool stream_destroy(void* p)noexcept override{return hipStreamDestroy(static_cast<hipStream_t>(p))==hipSuccess;}
 bool stream_drain(void* p)noexcept override{return hipStreamSynchronize(static_cast<hipStream_t>(p))==hipSuccess;}
 bool event_create(void*& p)noexcept override{return hipEventCreateWithFlags(reinterpret_cast<hipEvent_t*>(&p),hipEventDisableTiming)==hipSuccess;}
 bool event_record(void* p,void* stream)noexcept override{return hipEventRecord(static_cast<hipEvent_t>(p),static_cast<hipStream_t>(stream))==hipSuccess;}
 CodecEventPoll event_poll(void* p)noexcept override{auto e=hipEventQuery(static_cast<hipEvent_t>(p));return e==hipSuccess?CodecEventPoll::complete:(e==hipErrorNotReady?CodecEventPoll::pending:CodecEventPoll::failed);}
 bool event_destroy(void* p)noexcept override{return hipEventDestroy(static_cast<hipEvent_t>(p))==hipSuccess;}
 bool device_allocate(size_t n,void*& p)noexcept override{return hipMalloc(&p,n)==hipSuccess;}
 bool device_free(void* p)noexcept override{return hipFree(p)==hipSuccess;}
 bool pinned_allocate(size_t n,void*& p)noexcept override{return hipHostMalloc(&p,n,hipHostMallocDefault)==hipSuccess;}
 bool pinned_free(void* p)noexcept override{return hipHostFree(p)==hipSuccess;}
 bool copy(void* d,const void* s,size_t n,CodecCopy kind,void* stream)noexcept override{return hipMemcpyAsync(d,s,n,kind==CodecCopy::host_to_device?hipMemcpyHostToDevice:hipMemcpyDeviceToHost,static_cast<hipStream_t>(stream))==hipSuccess;}
 bool query(const CodecQuery& q,CodecBackendPlan& p)noexcept override{try{std::string error;return query_hipcomp_codec(q,p,error);}catch(...){return false;}}
 bool compress(const CodecDeviceBatch& b,void* temp,size_t bytes,void* stream)noexcept override{return hipcompBatchedLZ4CompressAsync(b.inputs,b.input_bytes,b.max_input_chunk,b.chunks,temp,bytes,b.outputs,b.actual_bytes,hipcompBatchedLZ4DefaultOpts,static_cast<hipStream_t>(stream))==hipcompSuccess;}
 bool decompress(const CodecDeviceBatch& b,void* temp,size_t bytes,void* stream)noexcept override{return hipcompBatchedLZ4DecompressAsync(b.inputs,b.input_bytes,b.capacities,b.actual_bytes,b.chunks,temp,bytes,b.outputs,reinterpret_cast<hipcompStatus_t*>(b.statuses),static_cast<hipStream_t>(stream))==hipcompSuccess;}
};
static_assert(sizeof(hipcompStatus_t)==sizeof(uint32_t));
}
#endif
std::shared_ptr<CodecRuntime> make_hipcomp_codec_runtime(){
#if defined(STRATA_WINDOW_HIPCOMP_230)
 return std::make_shared<Runtime>();
#else
 return {};
#endif
}
}
