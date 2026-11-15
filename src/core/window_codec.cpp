#include "strata/core/window_codec.hpp"
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
namespace strata::core {
namespace {
bool refuse(std::string& error,const char* why){error=why;return false;}
bool add(uint64_t a,uint64_t b,uint64_t& out){if(b>UINT64_MAX-a)return false;out=a+b;return true;}
bool mul(uint64_t a,uint64_t b,uint64_t& out){if(a&&b>UINT64_MAX/a)return false;out=a*b;return true;}
bool alignment(uint64_t a){return a&&!(a&(a-1));}
bool raw_identity(const CodecIdentity& i){return i==CodecIdentity{};}
bool supported_identity(const CodecIdentity& i){
    if(raw_identity(i))return true;
    return i.kind==CodecKind::lz4&&i.framing==1&&i.options==0&&i.sdk_patch==0&&
        ((i.backend==CodecBackend::nvcomp&&i.sdk_major==6&&i.sdk_minor==0&&i.build_pin=="nvcomp-6.0.0.12-cu13")||
         (i.backend==CodecBackend::hipcomp&&i.sdk_major==2&&i.sdk_minor==3&&i.build_pin=="hipcomp-22cc762f54fba7cdfca74a4c50c00f2aac4ace7a"));
}
bool limits_ok(const CodecLimits& l){return l.max_chunks&&l.max_chunk_bytes&&l.max_total_bytes&&l.max_encoded_bytes&&l.max_metadata_bytes&&l.max_chunks<=SIZE_MAX&&l.max_total_bytes<=SIZE_MAX&&l.max_encoded_bytes<=SIZE_MAX;}
StateDigest digest(std::span<const uint8_t> bytes){WindowHasher h;h.update(bytes.data(),bytes.size());return h.digest();}
bool metadata(uint64_t n,uint64_t& bytes){return mul(n,sizeof(CodecChunk)+sizeof(uint64_t),bytes)&&add(bytes,512,bytes);}
WindowResources host_resources(const WindowBudget& b,uint64_t bytes){WindowResources r;r.device_bytes.resize(b.snapshot().capacity.device_bytes.size());r.host_bytes=bytes;return r;}
}
bool codec_output_host_charge(uint64_t chunks,uint64_t capacity,uint64_t& charge) noexcept {
    uint64_t meta=0,result=0;
    if(!mul(chunks,sizeof(CodecChunk),meta)||!add(meta,512,meta)||!add(meta,capacity,result))return false;
    charge=result;return true;
}
bool codec_control_host_charge(uint64_t chunks,const CodecLimits& limits,uint64_t requested,
    uint64_t& charge,std::string& error){
    uint64_t metadata_charge=0;
    if(chunks>limits.max_chunks||!mul(chunks,512,metadata_charge)||!add(metadata_charge,4096,metadata_charge)||metadata_charge>limits.max_metadata_bytes)
        return refuse(error,"codec host metadata limit");
    charge=std::max(metadata_charge,requested);return true;
}
bool plan_window_codec(CodecBackend wanted,std::span<const CodecPlanChunk> chunks,const CodecLimits& limits,
    const CodecPlanOptions& o,const CodecBackendPlan* sdk,CodecPlan& out,std::string& error){
    try {
        if(!limits_ok(limits)||chunks.size()>limits.max_chunks||o.device_count>64||
           (o.direction!=CodecDirection::encode&&o.direction!=CodecDirection::decode))return refuse(error,"invalid codec plan bounds");
        if(wanted!=CodecBackend::raw&&wanted!=CodecBackend::nvcomp&&wanted!=CodecBackend::hipcomp)return refuse(error,"unknown codec backend");
        const bool encode=o.direction==CodecDirection::encode;
        CodecPlan p;auto& v=p.layout;uint64_t total=0,largest=0,gpu_total=0,gpu_count=0,encoded_total=0;
        for(const auto& c:chunks){
            const auto n=c.logical_bytes;
            if(!n||n>limits.max_chunk_bytes||!add(total,n,total)||total>limits.max_total_bytes||
               (c.kind!=CodecKind::raw&&c.kind!=CodecKind::lz4)||
               (encode&&c.kind!=CodecKind::raw)||
               (c.kind==CodecKind::raw&&c.encoded_bytes!=n)||
               (c.kind==CodecKind::lz4&&(!c.encoded_bytes||c.encoded_bytes>=n))||
               !add(encoded_total,c.encoded_bytes,encoded_total)||encoded_total>limits.max_encoded_bytes)
                return refuse(error,"invalid chunk extent");
            if(encode||c.kind==CodecKind::lz4){++gpu_count;largest=std::max(largest,n);if(!add(gpu_total,n,gpu_total))return refuse(error,"codec GPU extent overflow");}
        }
        if(!codec_control_host_charge(chunks.size(),limits,o.control_host_bytes,v.control.host_bytes,error))return false;
        p.metadata_bytes=v.control.host_bytes;
        for(auto* r:{&v.control,&v.transient,&v.result,&p.incremental_peak})r->device_bytes.resize(o.device_count);
        const bool available=sdk&&sdk->capability.compiled&&sdk->capability.abi_verified&&sdk->capability.planning;
        const bool raw=chunks.empty()||(!encode&&!gpu_count)||wanted==CodecBackend::raw||!available;
        p.raw_fallback=encode&&wanted!=CodecBackend::raw&&raw;
        if(raw){
            if(!encode&&gpu_count)return refuse(error,"compressed decoder unavailable");
            p.maximum_output_bytes=total;v.output_capacity=total;
            if(!codec_output_host_charge(encode?chunks.size():0,total,v.result.host_bytes))return refuse(error,"RAW reservation overflow");
        }else{
            const auto& id=sdk->capability.identity;
            if(id.backend!=wanted||!supported_identity(id)||o.device<0||uint32_t(o.device)>=o.device_count||
               !o.intended_stream||sdk->query.stream!=o.intended_stream||sdk->query.device!=o.device||
               sdk->query.chunks!=gpu_count||sdk->query.max_chunk_bytes!=largest||sdk->query.total_bytes!=gpu_total||
               o.control_host_bytes<4096||!o.control_device_bytes)
                return refuse(error,"SDK plan identity/device/stream/control mismatch");
            if(!alignment(sdk->input_alignment)||!alignment(sdk->output_alignment)||!alignment(sdk->temp_alignment)||sdk->max_output_chunk_bytes<largest)
                return refuse(error,"invalid SDK output/alignment bounds");
            p.identity=id;p.input_alignment=sdk->input_alignment;p.output_alignment=sdk->output_alignment;p.temp_alignment=sdk->temp_alignment;
            v.raw_offsets.reserve(chunks.size());v.compressed_offsets.reserve(chunks.size());
            auto append_aligned=[&](uint64_t n,uint64_t align,uint64_t& sum){uint64_t padded=0;return add(n,align-1,padded)&&add(sum,padded&~(align-1),sum);};
            for(const auto& c:chunks){
                v.raw_offsets.push_back(v.raw_bytes);v.compressed_offsets.push_back(v.compressed_bytes);
                if(!append_aligned(c.logical_bytes,p.input_alignment,v.raw_bytes))return refuse(error,"input staging overflow");
                if((encode||c.kind==CodecKind::lz4)&&!append_aligned(encode?sdk->max_output_chunk_bytes:c.encoded_bytes,p.output_alignment,v.compressed_bytes))return refuse(error,"encoded staging overflow");
            }
            p.input_staging_bytes=v.raw_bytes;p.maximum_output_bytes=v.compressed_bytes;
            p.workspace_bytes=std::max(sdk->encode_workspace_bytes,sdk->decode_workspace_bytes);
            if(p.workspace_bytes>limits.max_workspace_bytes||
               !mul(gpu_count,2*sizeof(void*)+3*sizeof(size_t)+sizeof(uint32_t),v.metadata_bytes))return refuse(error,"codec workspace/metadata limit");
            v.device_sizes={encode?v.raw_bytes:0,v.compressed_bytes,v.raw_bytes,p.workspace_bytes,v.metadata_bytes};
            v.device_alignments={p.input_alignment,p.output_alignment,p.input_alignment,p.temp_alignment,alignof(size_t)};
            v.pinned_sizes={encode?v.raw_bytes:0,v.compressed_bytes,v.raw_bytes,v.metadata_bytes};
            uint64_t gpu=0,pin=0;
            for(size_t i=0;i<v.device_sizes.size();++i)if(v.device_sizes[i]){uint64_t n=0;if(!add(v.device_sizes[i],v.device_alignments[i]-1,n)||n>SIZE_MAX||!add(gpu,n,gpu))return refuse(error,"codec device allocation overflow");}
            for(auto size:v.pinned_sizes)if(size){uint64_t n=0;if(!add(size,v.pinned_alignment-1,n)||n>SIZE_MAX||!add(pin,n,pin))return refuse(error,"codec pinned allocation overflow");}
            v.control.device_bytes[size_t(o.device)]=o.control_device_bytes;
            v.transient.device_bytes[size_t(o.device)]=gpu;v.transient.host_bytes=pin;v.transient.pinned_bytes=pin;
            v.output_capacity=encode?std::max(total,v.compressed_bytes):total;
            if(!codec_output_host_charge(chunks.size(),v.output_capacity,v.result.host_bytes))return refuse(error,"codec output owner overflow");
        }
        if(p.maximum_output_bytes>limits.max_encoded_bytes)return refuse(error,"encoded output limit");
        for(const auto* r:{&v.control,&v.transient,&v.result}){
            if(!add(p.incremental_peak.host_bytes,r->host_bytes,p.incremental_peak.host_bytes)||!add(p.incremental_peak.pinned_bytes,r->pinned_bytes,p.incremental_peak.pinned_bytes))return refuse(error,"codec host peak overflow");
            for(size_t i=0;i<o.device_count;++i)if(!add(p.incremental_peak.device_bytes[i],r->device_bytes[i],p.incremental_peak.device_bytes[i]))return refuse(error,"codec device peak overflow");
        }
        p.chunk_bytes.reserve(chunks.size());for(const auto& c:chunks)p.chunk_bytes.push_back(c.logical_bytes);
        out=std::move(p);return true;
    }catch(const std::bad_alloc&){return refuse(error,"codec plan allocation failed");}
     catch(const std::length_error&){return refuse(error,"codec plan extent unavailable");}
}
bool validate_codec_envelope(const CodecEnvelope& e,uint64_t expected,const CodecLimits& l,const CodecIdentity& allowed,std::string& error){
    if(!limits_ok(l)||e.identity!=allowed||expected>l.max_total_bytes||e.logical_bytes!=expected||e.chunks.size()>l.max_chunks||e.payload.size()>l.max_encoded_bytes)
        return refuse(error,"codec identity/extent/bounds mismatch");
    if(!supported_identity(allowed))
        return refuse(error,"unsupported codec identity");
    uint64_t meta=0;if(!metadata(e.chunks.size(),meta)||meta>l.max_metadata_bytes)return refuse(error,"codec metadata limit");
    uint64_t logical=0,encoded=0;
    for(const auto& c:e.chunks){
        if(!c.logical_bytes||c.logical_bytes>l.max_chunk_bytes||!c.encoded_bytes||c.logical_offset!=logical||c.encoded_offset!=encoded||
           !add(logical,c.logical_bytes,logical)||logical>expected||!add(encoded,c.encoded_bytes,encoded)||encoded>e.payload.size())return refuse(error,"invalid codec chunk extents");
        if(c.kind==CodecKind::raw){if(c.logical_bytes!=c.encoded_bytes)return refuse(error,"RAW extent mismatch");}
        else if(c.kind!=CodecKind::lz4||allowed.kind!=CodecKind::lz4||c.encoded_bytes>=c.logical_bytes)return refuse(error,"unknown/nonbeneficial chunk encoding");
    }
    return logical==expected&&encoded==e.payload.size()?true:refuse(error,"incomplete codec envelope");
}
bool encode_window_raw(std::span<const uint8_t> input,uint64_t chunk_size,const CodecLimits& l,WindowBudget& budget,RawCodecHandle& out,std::string& error){
    try{
        if(!limits_ok(l)||!chunk_size||chunk_size>l.max_chunk_bytes||input.size()>l.max_total_bytes||input.size()>l.max_encoded_bytes)return refuse(error,"invalid RAW encode bounds");
        const uint64_t count=input.size()/chunk_size+(input.size()%chunk_size!=0);uint64_t meta=0,charge=0;
        if(count>l.max_chunks||!metadata(count,meta)||meta>l.max_metadata_bytes||!add(meta,input.size(),charge))return refuse(error,"RAW encode reservation overflow/limit");
        auto lease=budget.reserve(host_resources(budget,charge),ReservationPurpose::foreground);
        if(!lease)return refuse(error,"RAW encode admission refused");
        auto e=std::make_shared<CodecEnvelope>();e->logical_bytes=input.size();e->chunks.reserve(size_t(count));e->payload.assign(input.begin(),input.end());
        for(uint64_t offset=0;offset<input.size();){const uint64_t n=std::min<uint64_t>(chunk_size,input.size()-offset);e->chunks.push_back({CodecKind::raw,offset,n,offset,n,digest(input.subspan(size_t(offset),size_t(n)))});offset+=n;}
        RawCodecHandle candidate;candidate.lease_=std::move(lease.lease);candidate.envelope_=std::move(e);std::swap(out.lease_,candidate.lease_);std::swap(out.envelope_,candidate.envelope_);return true;
    }catch(const std::bad_alloc&){return refuse(error,"RAW encode allocation failed");}
     catch(const std::length_error&){return refuse(error,"RAW encode extent unavailable");}
}
bool decode_window_raw(const CodecEnvelope& e,uint64_t expected,const CodecLimits& l,WindowBudget& budget,RawDecodeHandle& out,std::string& error){
    try{
        if(!validate_codec_envelope(e,expected,l,CodecIdentity{},error))return false;
        uint64_t charge=0;if(!add(expected,512,charge))return refuse(error,"RAW decode reservation overflow");
        auto lease=budget.reserve(host_resources(budget,charge),ReservationPurpose::foreground);if(!lease)return refuse(error,"RAW decode admission refused");
        for(const auto& c:e.chunks)if(digest(std::span<const uint8_t>(e.payload).subspan(size_t(c.encoded_offset),size_t(c.encoded_bytes)))!=c.digest)return refuse(error,"RAW chunk digest mismatch");
        auto data=std::make_shared<const std::vector<uint8_t>>(e.payload);RawDecodeHandle candidate;candidate.lease_=std::move(lease.lease);candidate.data_=std::move(data);std::swap(out.lease_,candidate.lease_);std::swap(out.data_,candidate.data_);return true;
    }catch(const std::bad_alloc&){return refuse(error,"RAW decode allocation failed");}
     catch(const std::length_error&){return refuse(error,"RAW decode extent unavailable");}
}
}
