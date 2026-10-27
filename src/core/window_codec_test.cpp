#include "strata/core/window_codec.hpp"
#include <cassert>
#include <iostream>
#include <limits>
using namespace strata::core;
WindowBudget budget(uint64_t n=1<<20){return WindowBudget({{n,n},n,n,0,0,0},{{0,0},0,0,0,0,0});}
CodecLimits limits(){return {1024,65536,1<<20,2<<20,1<<20,1<<20};}
int main(){
    std::string error;auto b=budget();auto l=limits();std::vector<uint8_t> input(131);
    for(size_t i=0;i<input.size();++i)input[i]=uint8_t(i*79);
    RawCodecHandle encoded;RawDecodeHandle decoded;
    assert(encode_window_raw(input,64,l,b,encoded,error));assert(encoded.get()->chunks.size()==3);
    assert(decode_window_raw(*encoded.get(),input.size(),l,b,decoded,error));assert(std::equal(input.begin(),input.end(),decoded.bytes().begin()));
    const auto retained=b.snapshot().reserved.host_bytes;assert(retained>input.size()*2);
    const auto* prior=encoded.get();auto scarce=budget(1);assert(!encode_window_raw(input,64,l,scarce,encoded,error));assert(encoded.get()==prior&&scarce.snapshot().reserved.host_bytes==0);
    const auto* bytes=decoded.bytes().data();auto bad=*encoded.get();bad.payload[0]^=1;assert(!decode_window_raw(bad,input.size(),l,b,decoded,error));assert(decoded.bytes().data()==bytes&&b.snapshot().reserved.host_bytes==retained);
    bad=*encoded.get();bad.chunks[0].encoded_bytes=UINT64_MAX;assert(!decode_window_raw(bad,input.size(),l,b,decoded,error));
    bad=*encoded.get();bad.chunks[1].logical_offset--;assert(!decode_window_raw(bad,input.size(),l,b,decoded,error));
    bad=*encoded.get();bad.identity.framing++;assert(!decode_window_raw(bad,input.size(),l,b,decoded,error));
    bad=*encoded.get();bad.chunks[0].kind=CodecKind::lz4;assert(!decode_window_raw(bad,input.size(),l,b,decoded,error));
    bad=*encoded.get();bad.payload.push_back(0);assert(!decode_window_raw(bad,input.size(),l,b,decoded,error));
    assert(!decode_window_raw(*encoded.get(),input.size()-1,l,b,decoded,error));
    auto tiny=l;tiny.max_metadata_bytes=1;assert(!encode_window_raw(input,64,tiny,b,encoded,error));
    std::vector<CodecPlanChunk> extents{{64,64},{64,64},{3,3}};CodecPlan plan;
    CodecPlanOptions opts{CodecDirection::encode,0,2,reinterpret_cast<void*>(1),4096,256};
    for(auto vendor:{CodecBackend::nvcomp,CodecBackend::hipcomp}){assert(plan_window_codec(vendor,extents,l,opts,nullptr,plan,error));assert(plan.raw_fallback&&plan.identity.backend==CodecBackend::raw&&plan.maximum_output_bytes==131);}
    auto saved=plan.maximum_output_bytes;std::vector<CodecPlanChunk> overflow{{UINT64_MAX,UINT64_MAX},{1,1}};assert(!plan_window_codec(CodecBackend::raw,overflow,l,opts,nullptr,plan,error));assert(plan.maximum_output_bytes==saved);
    assert(!plan_window_codec(CodecBackend(99),extents,l,opts,nullptr,plan,error));
    {auto invalid=opts;invalid.device_count=UINT32_MAX;assert(!plan_window_codec(CodecBackend::raw,extents,l,invalid,nullptr,plan,error));}
    CodecBackendPlan sdk;sdk.query={0,opts.intended_stream,3,64,131};sdk.capability={{CodecBackend::nvcomp,CodecKind::lz4,6,0,0,1,0,"nvcomp-6.0.0.12-cu13"},true,true,true,false,{}};sdk.input_alignment=sdk.output_alignment=sdk.temp_alignment=4;sdk.max_output_chunk_bytes=80;sdk.encode_workspace_bytes=128;sdk.decode_workspace_bytes=64;
    assert(plan_window_codec(CodecBackend::nvcomp,extents,l,opts,&sdk,plan,error));assert(!plan.raw_fallback&&plan.maximum_output_bytes==240&&plan.incremental_peak.device_bytes[0]>240&&plan.incremental_peak.pinned_bytes>0&&plan.incremental_peak.host_bytes>=plan.incremental_peak.pinned_bytes);
    assert(plan.input_staging_bytes==132);
    {auto mismatched=opts;mismatched.intended_stream=reinterpret_cast<void*>(2);const auto prior=plan.incremental_peak;
     assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,mismatched,&sdk,plan,error));assert(plan.incremental_peak==prior);}
    assert(plan.layout.device_sizes[0]==132&&plan.layout.device_sizes[2]==132);
    assert(plan.incremental_peak.host_bytes==plan.layout.control.host_bytes+plan.layout.transient.host_bytes+plan.layout.result.host_bytes);
    assert(plan.incremental_peak.pinned_bytes==plan.layout.transient.pinned_bytes);
    {const auto previous=plan.incremental_peak;
     auto huge=opts;huge.control_host_bytes=UINT64_MAX;
     assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,huge,&sdk,plan,error));assert(plan.incremental_peak==previous);
     huge=opts;huge.control_device_bytes=UINT64_MAX;
     assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,huge,&sdk,plan,error));assert(plan.incremental_peak==previous);
     auto huge_sdk=sdk;huge_sdk.input_alignment=uint64_t(1)<<63;
     assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,opts,&huge_sdk,plan,error));assert(plan.incremental_peak==previous);
     huge_sdk=sdk;huge_sdk.max_output_chunk_bytes=UINT64_MAX;
     assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,opts,&huge_sdk,plan,error));assert(plan.incremental_peak==previous);
     auto absent=opts;absent.intended_stream=nullptr;
     assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,absent,&sdk,plan,error));assert(plan.incremental_peak==previous);
     auto bad_kind=extents;bad_kind[0].kind=CodecKind::lz4;
     assert(!plan_window_codec(CodecBackend::nvcomp,bad_kind,l,opts,&sdk,plan,error));assert(plan.incremental_peak==previous);
     uint64_t charge=17;assert(!codec_output_host_charge(UINT64_MAX,1,charge)&&charge==17);
    }
    sdk.query.chunks=1;assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,opts,&sdk,plan,error));sdk.query.chunks=3;
    sdk.output_alignment=3;assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,opts,&sdk,plan,error));sdk.output_alignment=4;
    sdk.encode_workspace_bytes=UINT64_MAX;assert(!plan_window_codec(CodecBackend::nvcomp,extents,l,opts,&sdk,plan,error));
    CodecEnvelope mixed;mixed.identity={CodecBackend::nvcomp,CodecKind::lz4,6,0,0,1,0,"nvcomp-6.0.0.12-cu13"};mixed.logical_bytes=8;mixed.payload.resize(6);mixed.chunks={{CodecKind::raw,0,4,0,4,{}},{CodecKind::lz4,4,4,4,2,{}}};
    assert(validate_codec_envelope(mixed,8,l,mixed.identity,error));auto foreign=mixed.identity;foreign.backend=CodecBackend::hipcomp;assert(!validate_codec_envelope(mixed,8,l,foreign,error));assert(!decode_window_raw(mixed,8,l,b,decoded,error));mixed.identity.build_pin="unknown";assert(!validate_codec_envelope(mixed,8,l,mixed.identity,error));
    // Disabled adapters must provide usable RAW fallback without any SDK/runtime.
    assert(!nvcomp_codec_capability().compiled&&!hipcomp_codec_capability().compiled);
    CodecBackendPlan untouched;untouched.max_output_chunk_bytes=123;assert(!query_nvcomp_codec({},untouched,error)&&untouched.max_output_chunk_bytes==123);assert(!query_hipcomp_codec({},untouched,error)&&untouched.max_output_chunk_bytes==123);
    RawCodecHandle empty;RawDecodeHandle empty_decoded;assert(encode_window_raw({},64,l,b,empty,error));assert(decode_window_raw(*empty.get(),0,l,b,empty_decoded,error));assert(empty_decoded.bytes().empty());
    {auto copies=encoded;encoded={};assert(copies&&b.snapshot().reserved.host_bytes>0);}
    decoded={};empty={};empty_decoded={};assert(b.snapshot().reserved.host_bytes==0);
    std::cout<<"codec RAW/planner/envelope/reservation/disabled-backend tests passed\n";
}
