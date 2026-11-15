#include "strata/core/window_codec_job.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
namespace strata::core {
namespace {
bool plus(uint64_t a,uint64_t b,uint64_t& c){if(b>UINT64_MAX-a)return false;c=a+b;return true;}
StateDigest hash(std::span<const uint8_t> p){WindowHasher h;h.update(p.data(),p.size());return h.digest();}
WindowResources resources(const WindowBudget& b,uint64_t host,uint64_t pinned,int device,uint64_t gpu){WindowResources r;r.device_bytes.resize(b.snapshot().capacity.device_bytes.size());if(device>=0&&size_t(device)<r.device_bytes.size())r.device_bytes[size_t(device)]=gpu;r.host_bytes=host;r.pinned_bytes=pinned;return r;}
// Bounded standard LZ4 block grammar, including end-of-block conditions. This
// admits no vendor high-level frame and never derives an allocation from input.
bool lz4_valid(std::span<const uint8_t> in,uint64_t expected) noexcept {
    size_t at=0;uint64_t produced=0,last_match=0;bool matched=false;
    auto length=[&](uint64_t base,uint64_t& result){result=base;if(base!=15)return true;for(;;){if(at==in.size())return false;auto b=in[at++];if(!plus(result,b,result))return false;if(b!=255)return true;}};
    while(at<in.size()){
        auto token=in[at++];uint64_t literals=0;if(!length(token>>4,literals)||literals>in.size()-at||produced>expected||literals>expected-produced)return false;
        at+=size_t(literals);produced+=literals;
        if(at==in.size())return produced==expected&&(!matched||(literals>=5&&expected>=12&&last_match<=expected-12));
        if(in.size()-at<2)return false;
        const uint32_t offset=uint32_t(in[at])|(uint32_t(in[at+1])<<8);at+=2;
        if(!offset||offset>produced)return false;
        uint64_t match=0;if(!length(token&15,match)||!plus(match,4,match)||match>expected-produced)return false;
        last_match=produced;matched=true;produced+=match;
    }
    return false;
}
std::atomic<uint64_t> quarantined_count{0};
}
struct CodecEncodedHandle::Owned {
    std::shared_ptr<const ReservationLease> lease;
    CodecEnvelope envelope;
};
const CodecEnvelope* CodecEncodedHandle::get()const{return owned_?&owned_->envelope:nullptr;}
bool freeze_codec_encoded(const CodecEnvelope& input,uint64_t expected,const CodecLimits& l,const CodecIdentity& allowed,WindowBudget& b,CodecEncodedHandle& out,std::string& error){
    try{
        if(!validate_codec_envelope(input,expected,l,allowed,error))return false;
        uint64_t charge=0;if(!codec_output_host_charge(input.chunks.size(),input.payload.size(),charge)){error="encoded owner size overflow";return false;}
        auto lease=b.reserve(resources(b,charge,0,-1,0),ReservationPurpose::foreground);if(!lease){error="encoded owner admission refused";return false;}
        auto owner=std::make_shared<CodecEncodedHandle::Owned>();owner->lease=std::move(lease.lease);owner->envelope=input;
        CodecEncodedHandle candidate;candidate.owned_=std::move(owner);std::swap(out.owned_,candidate.owned_);return true;
    }catch(const std::bad_alloc&){error="encoded owner allocation failed";return false;}
     catch(const std::length_error&){error="encoded owner extent unavailable";return false;}
}
struct CodecJob::Storage {
    enum class Phase { compression, verification, decoding } phase=Phase::compression;
    struct Block {void* base=nullptr;uint8_t* data=nullptr;uint64_t bytes=0;};
    // Input, compressed output, decoded verification/output, workspace, metadata.
    std::array<Block,5> device;
    // Pinned source, encoded output, decoded output, metadata.
    std::array<Block,4> pinned;
    WindowBudget budget;
    CodecJobOptions options;
    CodecLimits limits;
    std::shared_ptr<CodecRuntime> runtime;
    RawCodecHandle raw_input;
    CodecEncodedHandle encoded_input;
    std::shared_ptr<const ReservationLease> control_lease,device_control_lease,transient_lease,decoded_lease;
    std::shared_ptr<CodecEncodedHandle::Owned> encoded_result;
    std::shared_ptr<std::vector<uint8_t>> decoded_result;
    CodecBackendPlan sdk;
    std::vector<uint64_t> logical,raw_offsets,compressed_offsets;
    std::vector<size_t> gpu_indices;
    std::vector<std::unique_ptr<WindowHasher>> hashers;
    void* stream=nullptr;void* event=nullptr;
    CodecJobState status=CodecJobState::failed;
    const char* why="codec job not initialized";
    // 0 unclaimed, 1 cancelled, 2 output publication committed.
    std::atomic<uint8_t> completion_gate{0};
    bool encode=false,submitted=false,taken=false,quarantined=false;
    uint64_t raw_bytes=0,compressed_bytes=0,metadata_bytes=0,logical_total=0;
    Storage* next=nullptr;
    explicit Storage(WindowBudget b):budget(std::move(b)){}
    static std::atomic<Storage*> quarantine;
    bool is_cancelled()const noexcept{return completion_gate.load(std::memory_order_acquire)==1||(options.cancellation&&options.cancellation->load(std::memory_order_acquire));}
    void poison(const char* reason)noexcept{
        why=reason;status=CodecJobState::quarantined;
        if(quarantined)return;
        quarantined=true;Storage* head=quarantine.load(std::memory_order_relaxed);
        do{next=head;}while(!quarantine.compare_exchange_weak(head,this,std::memory_order_release,std::memory_order_relaxed));
        quarantined_count.fetch_add(1,std::memory_order_relaxed);
    }
    bool enter(int& previous)noexcept{
        if(!runtime->get_device(previous)){poison("codec current device unknown");return false;}
        if(previous!=options.device&&!runtime->set_device(options.device)){poison("codec device selection uncertain");return false;}return true;
    }
    bool leave(int previous)noexcept{
        if(previous!=options.device&&!runtime->set_device(previous)){poison("codec device restoration uncertain");return false;}return true;
    }
    bool retire_selected()noexcept{
        if(stream&&!runtime->stream_drain(stream)){poison("codec drain uncertain");return false;}
        if(event){if(!runtime->event_destroy(event)){poison("codec event destruction uncertain");return false;}event=nullptr;}
        for(auto& b:device)if(b.base){if(!runtime->device_free(b.base)){poison("codec device free uncertain");return false;}b.base=nullptr;b.data=nullptr;}
        for(auto& b:pinned)if(b.base){if(!runtime->pinned_free(b.base)){poison("codec pinned free uncertain");return false;}b.base=nullptr;b.data=nullptr;}
        if(stream){if(!runtime->stream_destroy(stream)){poison("codec stream destruction uncertain");return false;}stream=nullptr;}
        return true;
    }
    void fail_selected(const char* reason)noexcept{why=reason;if(retire_selected())status=is_cancelled()?CodecJobState::cancelled:CodecJobState::failed;}
    void release_retired()noexcept{if(status!=CodecJobState::quarantined&&status!=CodecJobState::pending){transient_lease.reset();device_control_lease.reset();raw_input={};encoded_input={};if(status!=CodecJobState::ready){encoded_result.reset();decoded_result.reset();decoded_lease.reset();}}}
    bool allocate(Block& b,uint64_t n,uint64_t a,bool host)noexcept{
        if(!n)return true;
        if(!a||(a&(a-1))||!plus(n,a-1,b.bytes)||b.bytes>SIZE_MAX)return false;
        const bool ok=host?runtime->pinned_allocate(size_t(b.bytes),b.base):runtime->device_allocate(size_t(b.bytes),b.base);
        if(!ok)return false; // A nonnull partial allocation is still retained.
        if(!b.base)return false;
        const auto addr=reinterpret_cast<uintptr_t>(b.base);if(addr>UINTPTR_MAX-(a-1))return false;
        b.data=reinterpret_cast<uint8_t*>((addr+uintptr_t(a-1))&~uintptr_t(a-1));return true;
    }
    size_t count()const noexcept{return gpu_indices.size();}
    CodecDeviceBatch batch()const noexcept{
        auto* m=device[4].data;const size_t n=count();CodecDeviceBatch b;
        b.inputs=reinterpret_cast<const void*const*>(m);b.outputs=reinterpret_cast<void*const*>(m+n*sizeof(void*));
        b.input_bytes=reinterpret_cast<const size_t*>(m+2*n*sizeof(void*));b.capacities=b.input_bytes+n;
        b.actual_bytes=const_cast<size_t*>(b.capacities+n);b.statuses=reinterpret_cast<uint32_t*>(b.actual_bytes+n);b.chunks=n;
        b.max_input_chunk=size_t(sdk.query.max_chunk_bytes);return b;
    }
    size_t* actual()const noexcept{return reinterpret_cast<size_t*>(pinned[3].data+2*count()*sizeof(void*)+2*count()*sizeof(size_t));}
    uint32_t* statuses()const noexcept{return reinterpret_cast<uint32_t*>(actual()+count());}
    void descriptors(bool compression)noexcept{
        auto* inputs=reinterpret_cast<const void**>(pinned[3].data);
        auto* outputs=reinterpret_cast<void**>(pinned[3].data+count()*sizeof(void*));auto* sizes=reinterpret_cast<size_t*>(outputs+count());auto* caps=sizes+count();
        for(size_t j=0;j<count();++j){const size_t i=gpu_indices[j];
            inputs[j]=compression?device[0].data+raw_offsets[i]:device[1].data+compressed_offsets[i];
            outputs[j]=compression?device[1].data+compressed_offsets[i]:device[2].data+raw_offsets[i];
            sizes[j]=compression?size_t(logical[i]):(encode?actual()[j]:size_t(encoded_input.get()->chunks[i].encoded_bytes));
            caps[j]=compression?size_t(sdk.max_output_chunk_bytes):size_t(logical[i]);
        }
        std::fill(actual(),actual()+count(),size_t(0));std::fill(statuses(),statuses()+count(),UINT32_MAX);
    }
    bool enqueue(bool compression)noexcept{
        const auto direction=CodecCopy::host_to_device;submitted=true;
        if(compression){if(!runtime->copy(device[0].data,pinned[0].data,size_t(raw_bytes),direction,stream))return false;}
        else if(!encode){if(!runtime->copy(device[1].data,pinned[1].data,size_t(compressed_bytes),direction,stream))return false;}
        if(!runtime->copy(device[4].data,pinned[3].data,size_t(metadata_bytes),direction,stream))return false;
        if(!(compression?runtime->compress(batch(),device[3].data,size_t(sdk.encode_workspace_bytes),stream):runtime->decompress(batch(),device[3].data,size_t(sdk.decode_workspace_bytes),stream)))return false;
        if(!runtime->copy(compression?static_cast<void*>(pinned[1].data):static_cast<void*>(pinned[2].data),compression?static_cast<void*>(device[1].data):static_cast<void*>(device[2].data),size_t(compression?compressed_bytes:raw_bytes),CodecCopy::device_to_host,stream)||
           !runtime->copy(pinned[3].data,device[4].data,size_t(metadata_bytes),CodecCopy::device_to_host,stream)||!runtime->event_record(event,stream))return false;
        status=CodecJobState::pending;return true;
    }
    bool compressed_ok()const noexcept{
        for(size_t j=0;j<count();++j){if((runtime->compression_statuses()&&statuses()[j]!=0)||!actual()[j]||actual()[j]>sdk.max_output_chunk_bytes)return false;
            const auto i=gpu_indices[j];if(!lz4_valid({pinned[1].data+compressed_offsets[i],actual()[j]},logical[i]))return false;
        }return true;
    }
    bool decoded_ok()noexcept{
        const auto* source=encode?raw_input.get():encoded_input.get();
        for(size_t j=0;j<count();++j){const auto i=gpu_indices[j];if(statuses()[j]!=0||actual()[j]!=logical[i])return false;
            const std::span<const uint8_t> bytes{pinned[2].data+raw_offsets[i],size_t(logical[i])};
            try{hashers[j]->update(bytes.data(),bytes.size());if(hashers[j]->digest()!=source->chunks[i].digest)return false;}catch(...){return false;}
            if(encode&&std::memcmp(bytes.data(),source->payload.data()+source->chunks[i].encoded_offset,bytes.size())!=0)return false;
        }return true;
    }
    void build_result()noexcept{
        if(encode){
            auto& e=encoded_result->envelope;uint64_t selected=0;
            // Compression sizes were preserved in the preallocated result chunks.
            for(size_t i=0;i<logical.size();++i)selected+=std::min<uint64_t>(e.chunks[i].encoded_bytes,logical[i]);
            const bool benefit=selected<logical_total&&e.identity.build_pin.size()<logical_total-selected;
            uint64_t offset=0;
            for(size_t i=0;i<logical.size();++i){auto& c=e.chunks[i];const bool compressed=benefit&&c.encoded_bytes<logical[i];
                const uint64_t n=compressed?c.encoded_bytes:logical[i];const auto* from=compressed?pinned[1].data+compressed_offsets[i]:raw_input.get()->payload.data()+c.logical_offset;
                std::memcpy(e.payload.data()+offset,from,size_t(n));c.kind=compressed?CodecKind::lz4:CodecKind::raw;c.encoded_offset=offset;c.encoded_bytes=n;offset+=n;
            }
            e.payload.resize(size_t(offset));if(!benefit)e.identity=CodecIdentity{};
        }else{
            const auto& e=*encoded_input.get();for(size_t i=0;i<logical.size();++i){const auto& c=e.chunks[i];const auto* from=c.kind==CodecKind::raw?e.payload.data()+c.encoded_offset:pinned[2].data+raw_offsets[i];std::memcpy(decoded_result->data()+c.logical_offset,from,size_t(c.logical_bytes));}
        }
    }
    void completed_selected()noexcept{
        if(is_cancelled()){why="codec job cancelled";if(retire_selected())status=CodecJobState::cancelled;return;}
        if(phase==Phase::compression){
            if(!compressed_ok()){fail_selected("codec compression status/size/grammar invalid");return;}
            for(size_t j=0;j<count();++j)encoded_result->envelope.chunks[gpu_indices[j]].encoded_bytes=actual()[j];
            descriptors(false);phase=Phase::verification;
            if(!enqueue(false))fail_selected("codec verification enqueue failed");
            return;
        }
        if(!decoded_ok()){fail_selected("codec decode status/size/digest mismatch");return;}
        build_result();if(retire_selected()){why=nullptr;status=CodecJobState::ready;}
    }
};
std::atomic<CodecJob::Storage*> CodecJob::Storage::quarantine{nullptr};
CodecJob::CodecJob(Storage* s):storage_(s){}
uint64_t codec_job_quarantine_count()noexcept{return quarantined_count.load(std::memory_order_acquire);}
CodecJob::~CodecJob(){if(!storage_)return;if(!storage_->quarantined){cancel();if(storage_->status==CodecJobState::pending)drain();else if(storage_->stream||storage_->event){int previous=-1;if(storage_->enter(previous)){storage_->fail_selected("codec abandoned during construction");storage_->leave(previous);}}}if(!storage_->quarantined)delete storage_;}
CodecJobState CodecJob::state()const noexcept{return storage_->status;}
const char* CodecJob::error()const noexcept{return storage_->why;}
void CodecJob::cancel()noexcept{uint8_t expected=0;storage_->completion_gate.compare_exchange_strong(expected,1,std::memory_order_acq_rel);}
CodecJobState CodecJob::poll()noexcept{
    auto& s=*storage_;if(s.status!=CodecJobState::pending)return s.status;
    int previous=-1;if(!s.enter(previous))return s.status;
    const auto p=s.runtime->event_poll(s.event);
    if(p==CodecEventPoll::failed)s.fail_selected("codec event query failed");
    else if(p==CodecEventPoll::complete)s.completed_selected();
    if(!s.quarantined)s.leave(previous);
    s.release_retired();return s.status;
}
CodecJobState CodecJob::drain()noexcept{
    auto& s=*storage_;if(s.status!=CodecJobState::pending)return s.status;
    int previous=-1;if(!s.enter(previous))return s.status;
    while(s.status==CodecJobState::pending){if(!s.runtime->stream_drain(s.stream)){s.poison("codec drain uncertain");break;}s.completed_selected();}
    if(!s.quarantined)s.leave(previous);
    s.release_retired();return s.status;
}
bool CodecJob::take_encoded(CodecEncodedHandle& out)noexcept{
    auto& s=*storage_;if(s.status!=CodecJobState::ready||!s.encode||s.taken||s.is_cancelled())return false;
    uint8_t expected=0;if(!s.completion_gate.compare_exchange_strong(expected,2,std::memory_order_acq_rel))return false;
    CodecEncodedHandle result;result.owned_=std::move(s.encoded_result);std::swap(out.owned_,result.owned_);s.taken=true;return true;
}
bool CodecJob::take_decoded(RawDecodeHandle& out)noexcept{
    auto& s=*storage_;if(s.status!=CodecJobState::ready||s.encode||s.taken||s.is_cancelled())return false;
    uint8_t expected=0;if(!s.completion_gate.compare_exchange_strong(expected,2,std::memory_order_acq_rel))return false;
    RawDecodeHandle result;result.lease_=std::move(s.decoded_lease);result.data_=std::move(s.decoded_result);
    std::swap(out.lease_,result.lease_);std::swap(out.data_,result.data_);s.taken=true;return true;
}
CodecStartResult CodecJob::begin_encode(RawCodecHandle input,std::shared_ptr<CodecRuntime> runtime,const CodecLimits& l,WindowBudget b,const CodecJobOptions& o){return begin(std::move(input),{},std::move(runtime),l,std::move(b),o,true);}
CodecStartResult CodecJob::begin_decode(CodecEncodedHandle input,std::shared_ptr<CodecRuntime> runtime,const CodecLimits& l,WindowBudget b,const CodecJobOptions& o){return begin({},std::move(input),std::move(runtime),l,std::move(b),o,false);}
CodecStartResult CodecJob::begin(RawCodecHandle raw,CodecEncodedHandle encoded,std::shared_ptr<CodecRuntime> runtime,const CodecLimits& l,WindowBudget budget,const CodecJobOptions& o,bool encode){
    CodecStartResult result;int previous=-1;bool selected=false;
    try{
        const auto* input=encode?raw.get():encoded.get();if(!input)return {{},"missing immutable codec input"};
        std::string error;if(!validate_codec_envelope(*input,input->logical_bytes,l,encode?CodecIdentity{}:input->identity,error))return {{},"invalid codec input envelope"};
        uint64_t metadata_charge=0;
        if(!codec_control_host_charge(input->chunks.size(),l,o.control_host_bytes,metadata_charge,error))return {{},"codec host metadata limit"};
        auto control=budget.reserve(resources(budget,metadata_charge,0,-1,0),ReservationPurpose::foreground);
        if(!control)return {{},"codec control admission refused"};
        auto node=std::make_unique<Storage>(budget);node->control_lease=std::move(control.lease);
        result.job=std::unique_ptr<CodecJob>(new CodecJob(node.get()));node.release();auto& s=*result.job->storage_;
        s.raw_input=std::move(raw);s.encoded_input=std::move(encoded);s.options=o;s.limits=l;s.runtime=std::move(runtime);s.encode=encode;s.logical_total=input->logical_bytes;
        auto fail=[&](const char* why){s.why=why;s.status=s.is_cancelled()?CodecJobState::cancelled:CodecJobState::failed;result.error=why;};
        if(s.is_cancelled()){fail("codec job cancelled before creation");s.release_retired();return result;}
        bool all_raw=true;for(const auto& c:input->chunks)all_raw&=c.kind==CodecKind::raw;
        const bool available=s.runtime&&s.runtime->capability().compiled&&(s.runtime->hardware_qualified()||o.allow_unqualified_hardware);
        if((encode&&!available)||(!encode&&all_raw)||input->chunks.empty()){
            for(const auto& c:input->chunks)if(c.kind!=CodecKind::raw||hash({input->payload.data()+c.encoded_offset,size_t(c.encoded_bytes)})!=c.digest){fail("RAW fallback digest mismatch");s.release_retired();return result;}
            if(encode){CodecEncodedHandle output;if(!freeze_codec_encoded(*input,input->logical_bytes,l,CodecIdentity{},budget,output,error)){fail("RAW fallback admission failed");s.release_retired();return result;}s.encoded_result=std::const_pointer_cast<CodecEncodedHandle::Owned>(output.owned_);}
            else{uint64_t host=0;if(!codec_output_host_charge(0,input->logical_bytes,host)){fail("RAW output overflow");s.release_retired();return result;}auto lease=budget.reserve(resources(budget,host,0,-1,0),ReservationPurpose::foreground);if(!lease){fail("RAW output admission refused");s.release_retired();return result;}s.decoded_lease=std::move(lease.lease);s.decoded_result=std::make_shared<std::vector<uint8_t>>(input->payload);}
            s.status=CodecJobState::ready;s.why=nullptr;s.release_retired();return result;
        }
        if(!available){fail("compressed decoder unavailable/unqualified");s.release_retired();return result;}
        const auto dimensions=budget.snapshot().capacity.device_bytes.size();
        if(o.device<0||size_t(o.device)>=dimensions||dimensions>64||o.control_host_bytes<4096||!o.control_device_bytes){fail("invalid codec device/control envelope");s.release_retired();return result;}
        // Add device control allowance before creating any stream/event handle.
        auto device_control=budget.reserve(resources(budget,0,0,o.device,o.control_device_bytes),ReservationPurpose::foreground);
        if(!device_control){fail("codec device control admission refused");s.release_retired();return result;}
        // Storage owns the device lease separately until all handles retire.
        s.device_control_lease=std::move(device_control.lease);
        s.logical.reserve(input->chunks.size());s.raw_offsets.resize(input->chunks.size());s.compressed_offsets.resize(input->chunks.size());s.gpu_indices.reserve(input->chunks.size());
        uint64_t largest=0,total_gpu=0;
        for(size_t i=0;i<input->chunks.size();++i){const auto& c=input->chunks[i];s.logical.push_back(c.logical_bytes);
            if(!encode&&c.kind==CodecKind::raw){if(hash({input->payload.data()+c.encoded_offset,size_t(c.encoded_bytes)})!=c.digest){fail("mixed RAW digest mismatch");s.release_retired();return result;}continue;}
            if(!encode&&!lz4_valid({input->payload.data()+c.encoded_offset,size_t(c.encoded_bytes)},c.logical_bytes)){fail("compressed block grammar invalid");s.release_retired();return result;}
            s.gpu_indices.push_back(i);largest=std::max(largest,c.logical_bytes);if(!plus(total_gpu,c.logical_bytes,total_gpu)){fail("codec extent overflow");s.release_retired();return result;}
        }
        if(!s.enter(previous)){result.error=s.why;return result;}selected=true;
        if(!s.runtime->stream_create(s.stream)||!s.stream||!s.runtime->event_create(s.event)||!s.event){s.fail_selected("codec control creation failed");if(!s.quarantined)s.leave(previous);s.release_retired();result.error=s.why;return result;}
        CodecQuery query{o.device,s.stream,s.count(),largest,total_gpu};
        if(!s.runtime->query(query,s.sdk)||s.sdk.query.device!=o.device||s.sdk.query.stream!=s.stream||s.sdk.query.chunks!=s.count()||s.sdk.query.total_bytes!=total_gpu||s.sdk.query.max_chunk_bytes!=largest||(!encode&&s.sdk.capability.identity!=input->identity))throw std::runtime_error("codec sizing/identity failed");
        CodecPlan checked;std::vector<CodecPlanChunk> plan_chunks;plan_chunks.reserve(input->chunks.size());
        for(const auto& c:input->chunks)plan_chunks.push_back({c.logical_bytes,c.encoded_bytes,c.kind});
        CodecPlanOptions plan_options{encode?CodecDirection::encode:CodecDirection::decode,o.device,uint32_t(dimensions),s.stream,o.control_host_bytes,o.control_device_bytes};
        if(!plan_window_codec(s.sdk.capability.identity.backend,plan_chunks,l,plan_options,&s.sdk,checked,error)||checked.raw_fallback)throw std::runtime_error("codec plan refused");
        const auto& layout=checked.layout;
        s.raw_offsets=std::move(checked.layout.raw_offsets);s.compressed_offsets=std::move(checked.layout.compressed_offsets);
        s.raw_bytes=layout.raw_bytes;s.compressed_bytes=layout.compressed_bytes;s.metadata_bytes=layout.metadata_bytes;
        const auto& sizes=layout.device_sizes;const auto& alignments=layout.device_alignments;const auto& host_sizes=layout.pinned_sizes;
        auto transient=budget.reserve(layout.transient,ReservationPurpose::foreground);if(!transient)throw std::runtime_error("codec full reservation refused");s.transient_lease=std::move(transient.lease);
        const uint64_t output_capacity=layout.output_capacity;
        auto output=budget.reserve(layout.result,ReservationPurpose::foreground);if(!output)throw std::runtime_error("codec output admission refused");
        if(encode){s.encoded_result=std::make_shared<CodecEncodedHandle::Owned>();s.encoded_result->lease=std::move(output.lease);s.encoded_result->envelope.identity=s.sdk.capability.identity;s.encoded_result->envelope.logical_bytes=s.logical_total;s.encoded_result->envelope.chunks=input->chunks;s.encoded_result->envelope.payload.resize(size_t(output_capacity));}
        else{s.decoded_lease=std::move(output.lease);s.decoded_result=std::make_shared<std::vector<uint8_t>>(size_t(output_capacity));}
        s.hashers.reserve(s.count());for(size_t j=0;j<s.count();++j)s.hashers.push_back(std::make_unique<WindowHasher>());
        for(size_t i=0;i<sizes.size();++i)if(!s.allocate(s.device[i],sizes[i],alignments[i],false))throw std::runtime_error("codec device allocation failed");
        for(size_t i=0;i<host_sizes.size();++i)if(!s.allocate(s.pinned[i],host_sizes[i],layout.pinned_alignment,true))throw std::runtime_error("codec pinned allocation failed");
        for(auto& block:s.pinned)if(block.data)std::memset(block.data,0,size_t(block.bytes-(reinterpret_cast<uintptr_t>(block.data)-reinterpret_cast<uintptr_t>(block.base))));
        for(size_t i=0;i<input->chunks.size();++i){const auto& c=input->chunks[i];if(encode)std::memcpy(s.pinned[0].data+s.raw_offsets[i],input->payload.data()+c.encoded_offset,size_t(c.encoded_bytes));else if(c.kind==CodecKind::lz4)std::memcpy(s.pinned[1].data+s.compressed_offsets[i],input->payload.data()+c.encoded_offset,size_t(c.encoded_bytes));}
        if(s.is_cancelled()){s.fail_selected("codec cancelled before enqueue");}
        else{s.descriptors(encode);s.phase=encode?Storage::Phase::compression:Storage::Phase::decoding;if(!s.enqueue(encode))s.fail_selected("codec enqueue failed");}
        if(!s.quarantined)s.leave(previous);
        selected=false;s.release_retired();if(s.status!=CodecJobState::pending&&s.status!=CodecJobState::ready)result.error=s.why;return result;
    }catch(...){
        result.error="codec preparation allocation/validation failed";
        if(result.job){auto& s=*result.job->storage_;if(selected&&!s.quarantined){s.fail_selected(result.error);if(!s.quarantined)s.leave(previous);}else if(!s.quarantined){s.why=result.error;s.status=CodecJobState::failed;}s.release_retired();if(s.quarantined)result.error=s.why;}
        return result;
    }
}
}
