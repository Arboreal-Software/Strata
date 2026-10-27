#include "strata/core/window_codec_job.hpp"
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
namespace {
struct Fake final:CodecRuntime {
    uint64_t device_allocated_bytes=0,pinned_allocated_bytes=0;
    int calls=0,fail=0,current=1;bool pending=false,corrupt=false,bad_status=false,bad_size=false;
    std::set<void*> allocations;std::vector<std::string> trace;
    bool step(const char* name){trace.emplace_back(name);return ++calls!=fail;}
    CodecCapability capability()const override{return {{CodecBackend::nvcomp,CodecKind::lz4,6,0,0,1,0,"nvcomp-6.0.0.12-cu13"},true,true,true,false,{}};}
    bool hardware_qualified()const noexcept override{return true;}
    bool compression_statuses()const noexcept override{return true;}
    bool get_device(int& d)noexcept override{if(!step("get"))return false;d=current;return true;}
    bool set_device(int d)noexcept override{if(!step("set"))return false;current=d;return true;}
    bool stream_create(void*& p)noexcept override{p=reinterpret_cast<void*>(1);return step("stream_create");}
    bool stream_destroy(void*)noexcept override{return step("stream_destroy");}
    bool stream_drain(void*)noexcept override{return step("drain");}
    bool event_create(void*& p)noexcept override{p=reinterpret_cast<void*>(2);return step("event_create");}
    bool event_record(void*,void*)noexcept override{return step("record");}
    CodecEventPoll event_poll(void*)noexcept override{if(!step("poll"))return CodecEventPoll::failed;return pending?CodecEventPoll::pending:CodecEventPoll::complete;}
    bool event_destroy(void*)noexcept override{return step("event_destroy");}
    bool alloc(size_t n,void*& p,const char* name){p=std::malloc(n);assert(p);allocations.insert(p);return step(name);}
    bool release(void* p,const char* name){if(!step(name))return false;assert(allocations.erase(p)==1);std::free(p);return true;}
    bool device_allocate(size_t n,void*& p)noexcept override{device_allocated_bytes+=n;return alloc(n,p,"malloc");}
    bool device_free(void* p)noexcept override{return release(p,"free");}
    bool pinned_allocate(size_t n,void*& p)noexcept override{pinned_allocated_bytes+=n;return alloc(n,p,"pin");}
    bool pinned_free(void* p)noexcept override{return release(p,"unpin");}
    bool copy(void* d,const void* s,size_t n,CodecCopy,void*)noexcept override{if(!step("copy"))return false;std::memcpy(d,s,n);return true;}
    bool query(const CodecQuery& q,CodecBackendPlan& p)noexcept override{if(!step("query"))return false;p.capability=capability();p.query=q;p.input_alignment=p.output_alignment=p.temp_alignment=16;p.max_output_chunk_bytes=q.max_chunk_bytes+64;p.encode_workspace_bytes=128;p.decode_workspace_bytes=64;return true;}
    bool compress(const CodecDeviceBatch& b,void*,size_t,void*)noexcept override{
        if(!step("compress"))return false;
        for(size_t j=0;j<b.chunks;++j){auto* in=static_cast<const uint8_t*>(b.inputs[j]);auto* out=static_cast<uint8_t*>(b.outputs[j]);size_t n=b.input_bytes[j],k=0;
            bool same=n>=13;for(size_t i=1;i<n;++i)same&=in[i]==in[0];
            auto ext=[&](size_t x){while(x>=255){out[k++]=255;x-=255;}out[k++]=uint8_t(x);};
            if(same){const size_t match=n-6;out[k++]=uint8_t(16+std::min<size_t>(15,match-4));out[k++]=in[0];out[k++]=1;out[k++]=0;if(match-4>=15)ext(match-19);out[k++]=0x50;std::memcpy(out+k,in+n-5,5);k+=5;}
            else{out[k++]=uint8_t(std::min<size_t>(15,n)<<4);if(n>=15)ext(n-15);std::memcpy(out+k,in,n);k+=n;}
            assert(k<=b.capacities[j]);b.actual_bytes[j]=k;b.statuses[j]=0;
        }return true;
    }
    bool decompress(const CodecDeviceBatch& b,void*,size_t,void*)noexcept override{
        if(!step("decompress"))return false;
        for(size_t j=0;j<b.chunks;++j){auto* in=static_cast<const uint8_t*>(b.inputs[j]);auto* out=static_cast<uint8_t*>(b.outputs[j]);size_t at=0,k=0;
            auto len=[&](size_t x){if(x==15){uint8_t c;do{assert(at<b.input_bytes[j]);c=in[at++];x+=c;}while(c==255);}return x;};
            while(at<b.input_bytes[j]){auto t=in[at++];auto n=len(t>>4);assert(at+n<=b.input_bytes[j]&&k+n<=b.capacities[j]);std::memcpy(out+k,in+at,n);at+=n;k+=n;if(at==b.input_bytes[j])break;size_t off=in[at]|size_t(in[at+1])<<8;at+=2;n=len(t&15)+4;assert(off&&off<=k&&k+n<=b.capacities[j]);while(n--){out[k]=out[k-off];++k;}}
            if(corrupt)out[0]^=1;
            b.actual_bytes[j]=bad_size?k+1:k;b.statuses[j]=bad_status?1:0;
        }return true;
    }
};
WindowBudget budget(uint64_t n=1<<22){return WindowBudget({{n,n},n,n,0,0,0},{{0,0},0,0,0,0,0});}
CodecLimits limits(){return {1024,65536,1<<20,2<<20,1<<20,1<<20};}
CodecJobOptions options(){CodecJobOptions o;o.device=0;o.control_host_bytes=4096;o.control_device_bytes=256;return o;}
RawCodecHandle input(WindowBudget& b){std::vector<uint8_t> p(131,42);p[130]=3;RawCodecHandle r;std::string e;assert(encode_window_raw(p,64,limits(),b,r,e));return r;}
int run(int failure,bool poll,bool decode=false){auto b=budget();auto r=input(b);auto f=std::make_shared<Fake>();f->fail=failure;CodecEncodedHandle initial;
    if(decode){f->fail=0;auto prep=CodecJob::begin_encode(r,f,limits(),b,options());assert(prep&&prep.job->drain()==CodecJobState::ready&&prep.job->take_encoded(initial));f->calls=0;f->trace.clear();f->fail=failure;}
    auto job=decode?CodecJob::begin_decode(initial,f,limits(),b,options()):CodecJob::begin_encode(r,f,limits(),b,options());assert(job.job);
    if(job.job->state()==CodecJobState::pending){if(poll){while(job.job->poll()==CodecJobState::pending){}}else job.job->drain();}
    int calls=f->calls;const auto state=job.job->state();CodecEncodedHandle out;
    if(failure){assert(state==CodecJobState::failed||state==CodecJobState::quarantined);assert(!job.job->take_encoded(out));}
    else if(decode){assert(state==CodecJobState::ready);RawDecodeHandle decoded;assert(job.job->take_decoded(decoded));assert(std::equal(decoded.bytes().begin(),decoded.bytes().end(),r.get()->payload.begin()));}
    else{assert(state==CodecJobState::ready);assert(job.job->take_encoded(out));assert(!job.job->take_encoded(out));assert(out.get()->identity.backend==CodecBackend::nvcomp);assert(out.get()->chunks.back().kind==CodecKind::raw);}
    if(state==CodecJobState::quarantined){auto before=b.snapshot().reserved;job.job.reset();assert(f->calls==calls);assert(b.snapshot().reserved==before);assert(codec_job_quarantine_count());}
    else{job.job.reset();assert(f->allocations.empty());assert(f->current==1);}
    return calls;
}
void planned_peak(bool decode,size_t count,bool raw_fallback=false){
    auto b=budget();auto l=limits();std::string error;
    std::vector<uint8_t> data(count,42);data.back()=3;
    RawCodecHandle raw;assert(encode_window_raw(data,64,l,b,raw,error));
    CodecEncodedHandle encoded;const CodecEnvelope* envelope=raw.get();
    if(decode){
        if(raw_fallback)assert(freeze_codec_encoded(*raw.get(),count,l,CodecIdentity{},b,encoded,error));
        else{auto prep=CodecJob::begin_encode(raw,std::make_shared<Fake>(),l,b,options());
            assert(prep&&prep.job->drain()==CodecJobState::ready&&prep.job->take_encoded(encoded));}
        envelope=encoded.get();
    }
    const auto before=b.snapshot().reserved;
    auto f=std::make_shared<Fake>();auto o=options();
    auto job=decode?CodecJob::begin_decode(encoded,f,l,b,o):CodecJob::begin_encode(raw,raw_fallback?nullptr:f,l,b,o);
    assert(job);const auto after=b.snapshot().reserved;
    std::vector<CodecPlanChunk> chunks;uint64_t total_gpu=0,largest=0,gpu_count=0;
    for(const auto& c:envelope->chunks){chunks.push_back({c.logical_bytes,c.encoded_bytes,c.kind});
        if(!decode||c.kind==CodecKind::lz4){++gpu_count;total_gpu+=c.logical_bytes;largest=std::max(largest,c.logical_bytes);}}
    CodecBackendPlan sdk;assert(f->query({0,reinterpret_cast<void*>(1),gpu_count,largest,total_gpu},sdk));
    CodecPlanOptions po{decode?CodecDirection::decode:CodecDirection::encode,0,2,reinterpret_cast<void*>(1),o.control_host_bytes,o.control_device_bytes};
    CodecPlan plan;assert(plan_window_codec(CodecBackend::nvcomp,chunks,l,po,raw_fallback?nullptr:&sdk,plan,error));
    assert(after.host_bytes-before.host_bytes==plan.incremental_peak.host_bytes);
    assert(after.pinned_bytes-before.pinned_bytes==plan.incremental_peak.pinned_bytes);
    assert(after.device_bytes[0]-before.device_bytes[0]==plan.incremental_peak.device_bytes[0]);
    assert(plan.layout.transient.device_bytes[0]==f->device_allocated_bytes);
    assert(plan.layout.transient.pinned_bytes==f->pinned_allocated_bytes);
    if(decode&&!raw_fallback){assert(gpu_count<chunks.size());assert(plan.layout.device_sizes[0]==0);
        assert(plan.layout.raw_bytes>=count&&plan.layout.metadata_bytes==gpu_count*(2*sizeof(void*)+3*sizeof(size_t)+sizeof(uint32_t)));}
    if(!raw_fallback){auto mismatch=po;mismatch.intended_stream=reinterpret_cast<void*>(3);
        const auto previous=plan.incremental_peak;assert(!plan_window_codec(CodecBackend::nvcomp,chunks,l,mismatch,&sdk,plan,error));assert(plan.incremental_peak==previous);}
    std::cout<<"exact peak "<<(decode?"decode":"encode")<<" bytes="<<count<<" RAW="<<raw_fallback
        <<" chunks="<<chunks.size()<<" sdk_chunks="<<gpu_count<<" host="<<plan.incremental_peak.host_bytes
        <<" pinned="<<plan.incremental_peak.pinned_bytes<<" device="<<plan.incremental_peak.device_bytes[0]<<'\n';
    assert(job.job->drain()==CodecJobState::ready);
    if(decode){RawDecodeHandle out;assert(job.job->take_decoded(out));assert(std::equal(out.bytes().begin(),out.bytes().end(),data.begin()));}
    else{CodecEncodedHandle out;assert(job.job->take_encoded(out));}
}
void fork_failure(int n,bool poll,bool decode=false){auto pid=fork();assert(pid>=0);if(!pid){run(n,poll,decode);_Exit(0);}int status;assert(waitpid(pid,&status,0)==pid);if(!WIFEXITED(status)||WEXITSTATUS(status)){std::cerr<<"failure step "<<n<<" poll "<<poll<<'\n';std::abort();}}
}
int main(){
    for(size_t n:{size_t(131),size_t(1293)}){planned_peak(false,n);planned_peak(true,n);planned_peak(false,n,true);planned_peak(true,n,true);}
    const int drain_calls=run(0,false),poll_calls=run(0,true);
    for(int i=1;i<=drain_calls;++i)fork_failure(i,false);
    for(int i=1;i<=poll_calls;++i)fork_failure(i,true);
    const int decode_drain=run(0,false,true),decode_poll=run(0,true,true);
    for(int i=1;i<=decode_drain;++i)fork_failure(i,false,true);
    for(int i=1;i<=decode_poll;++i)fork_failure(i,true,true);
    {
        auto b=budget();auto raw=input(b);auto f=std::make_shared<Fake>();auto a=CodecJob::begin_encode(raw,f,limits(),b,options());assert(a);f->pending=true;assert(a.job->poll()==CodecJobState::pending);f->pending=false;assert(a.job->drain()==CodecJobState::ready);CodecEncodedHandle enc;assert(a.job->take_encoded(enc));auto d=CodecJob::begin_decode(enc,f,limits(),b,options());assert(d&&d.job->drain()==CodecJobState::ready);RawDecodeHandle out;assert(d.job->take_decoded(out));assert(std::equal(out.bytes().begin(),out.bytes().end(),raw.get()->payload.begin()));
        auto bad=*enc.get();bad.payload[2]=0;bad.payload[3]=0;CodecEncodedHandle malformed;std::string e;assert(freeze_codec_encoded(bad,131,limits(),bad.identity,b,malformed,e));auto before=f->calls;auto rejected=CodecJob::begin_decode(malformed,f,limits(),b,options());assert(!rejected&&f->calls==before);
        for(int mode=0;mode<3;++mode){f->corrupt=mode==0;f->bad_size=mode==1;f->bad_status=mode==2;auto j=CodecJob::begin_decode(enc,f,limits(),b,options());assert(j);assert(j.job->drain()==CodecJobState::failed);const auto* p=out.bytes().data();assert(!j.job->take_decoded(out)&&out.bytes().data()==p);}
    }
    {auto b=budget();auto raw=input(b);auto f=std::make_shared<Fake>();auto j=CodecJob::begin_encode(raw,f,limits(),b,options());assert(j);j.job->cancel();assert(j.job->drain()==CodecJobState::cancelled);assert(f->allocations.empty());CodecEncodedHandle out;assert(!j.job->take_encoded(out));}
    {auto b=budget();auto raw=input(b);auto f=std::make_shared<Fake>();auto o=options();o.cancellation=std::make_shared<std::atomic<bool>>(true);auto j=CodecJob::begin_encode(raw,f,limits(),b,o);assert(!j&&f->calls==0);}
    {auto b=budget();auto raw=input(b);auto j=CodecJob::begin_encode(raw,nullptr,limits(),b,options());assert(j&&j.job->state()==CodecJobState::ready);CodecEncodedHandle enc;assert(j.job->take_encoded(enc));assert(enc.get()->identity.backend==CodecBackend::raw);auto d=CodecJob::begin_decode(enc,nullptr,limits(),b,options());assert(d&&d.job->state()==CodecJobState::ready);}
    {auto b=budget();auto raw=input(b);auto tiny=budget(4096);auto f=std::make_shared<Fake>();auto j=CodecJob::begin_encode(raw,f,limits(),tiny,options());assert(!j&&f->calls==0);}
    {auto b=budget();auto raw=input(b);auto f=std::make_shared<Fake>();auto j=CodecJob::begin_encode(raw,f,limits(),b,options());assert(j);assert(j.job->poll()==CodecJobState::pending);j.job->cancel();assert(j.job->drain()==CodecJobState::cancelled);assert(f->allocations.empty());}
    {auto b=budget();auto raw=input(b);auto f=std::make_shared<Fake>();auto o=options();o.device=1;auto j=CodecJob::begin_encode(raw,f,limits(),b,o);assert(j&&j.job->drain()==CodecJobState::ready);j.job->cancel();CodecEncodedHandle out;assert(!j.job->take_encoded(out));assert(f->current==1);}
    {auto b=budget();auto raw=input(b);auto f=std::make_shared<Fake>();auto j=CodecJob::begin_encode(raw,f,limits(),b,options());assert(j);const auto peak=b.snapshot().reserved;auto limited=WindowBudget({{peak.device_bytes[0]-1,1<<22},1<<22,1<<22,0,0,0},{{0,0},0,0,0,0,0});auto other=std::make_shared<Fake>();auto denied=CodecJob::begin_encode(raw,other,limits(),limited,options());assert(!denied);assert(std::find(other->trace.begin(),other->trace.end(),"compress")==other->trace.end());assert(other->allocations.empty());}
    std::cout<<"codec async ownership passed "<<drain_calls+poll_calls+decode_drain+decode_poll<<" injected API failures, mixed roundtrip, cancellation, corruption, RAW fallback\n";
}
