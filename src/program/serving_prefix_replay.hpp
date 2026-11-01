#pragma once
// Program-private owned transcript. No cache bytes, borrowed callbacks or executor ownership.
#include "strata/core/window_state.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strata::program {
struct PrefixReplayEvent {
    enum class Kind : uint8_t { window, prefill };
    Kind kind=Kind::window;
    uint64_t from=0,to=0;
    int64_t chunk=0,lend_tokens=0,lend_exact=0;
    bool planned=false,single=false,chunked=false,lend=false;
};
struct PrefixReplayParticipant {
    const void* cache=nullptr;const void* prefill=nullptr;const void* session=nullptr;
    const void* offsets=nullptr;
    int device=-1;int64_t begin=0,end=0,slots=0,chunk=0,lent_chunk=0;
    int32_t first=-1;
    bool loan_seen=false;
    std::vector<std::pair<int32_t,int32_t>> loan;
};
struct PrefixReplayPlan {
    // Credit also retained by ServingQualification; payloads always die first.
    std::shared_ptr<const void> credit;
    std::vector<int32_t> base;
    std::vector<int64_t> tokens;
    std::vector<PrefixReplayEvent> events;
    std::array<PrefixReplayParticipant,8> participants{};
    std::array<PrefixReplayEvent,8> leading{};
    const void* source=nullptr;const void* runtime=nullptr;
    uint64_t request=0,boundary=0,reached=0;
    uint64_t origin_runtime_incarnation=0,origin_residency_generation=0;
    core::WindowKey origin_key{};
    std::array<char,65> origin_binary{},origin_source{};
    double pipe=0;
    size_t participant_count=0,leading_count=0;
    core::StateDigest conditioning{},digest{};
    bool active=false,sealed=false,refused=false;
    const char* reason=nullptr;
    bool reject(const char* why) noexcept {if(!refused)reason=why;refused=true;return false;}
    static bool quote(uint64_t boundary,uint64_t entries,uint64_t& bytes) noexcept {
        auto add=[&](uint64_t count,uint64_t unit){
            if(unit&&count>(UINT64_MAX-bytes)/unit)return false;
            bytes+=count*unit;return bytes<=SIZE_MAX;
        };
        if(boundary<=64)return true;
        if(boundary>(UINT64_MAX-4)/4)return false;
        return add(1,sizeof(PrefixReplayPlan)+16384)&&
            add(boundary*4+4,sizeof(PrefixReplayEvent))&&add(boundary,sizeof(int64_t))&&
            add(entries,sizeof(int32_t)+sizeof(std::pair<int32_t,int32_t>));
    }
    bool prepare(std::shared_ptr<const void> admitted,std::span<const int32_t> mapping) {
        if(active||refused||sealed||!admitted||boundary<=64||mapping.empty()||boundary>SIZE_MAX/4-4)
            return reject("prefix transcript preparation bounds");
        credit=std::move(admitted);
        base.assign(mapping.begin(),mapping.end());tokens.resize(size_t(boundary));
        events.reserve(size_t(boundary)*4+4);
        for(size_t i=0;i<leading_count;++i)events.push_back(leading[i]);
        active=true;return true;
    }
    bool record(const PrefixReplayEvent& e) {
        if(refused||sealed||e.from!=reached||e.to<=e.from||e.to>boundary||
           (e.kind!=PrefixReplayEvent::Kind::window&&e.kind!=PrefixReplayEvent::Kind::prefill))
            return reject("actual prefix calls are not a complete ordered partition");
        if(e.kind==PrefixReplayEvent::Kind::prefill&&!active)
            return reject("actual batched prefix lacks pre-admitted transcript");
        if(active){
            if(events.size()==events.capacity())return reject("prefix transcript event capacity");
            events.push_back(e);
        }else{
            if(leading_count==leading.size())return reject("leading window transcript capacity");
            leading[leading_count++]=e;
        }
        reached=e.to;return true;
    }
    static core::StateDigest prefix_digest(std::span<const int32_t> h,uint64_t count) {
        core::WindowHasher hash;constexpr char domain[]="serving-prefix-conditioning-v1";
        hash.update(domain,sizeof(domain));hash.update(&count,sizeof(count));
        for(uint64_t i=0;i<count;++i)hash.update(&h[size_t(i)],sizeof(int32_t));
        return hash.digest();
    }
    core::StateDigest transcript_digest() const {
        core::WindowHasher hash;hash.update(conditioning.data(),conditioning.size());
        auto add=[&](auto value){hash.update(&value,sizeof(value));};
        add(request);add(boundary);add(pipe);add(participant_count);add(events.size());add(base.size());
        add(origin_runtime_incarnation);add(origin_residency_generation);
        hash.update(origin_binary.data(),origin_binary.size());hash.update(origin_source.data(),origin_source.size());
        add(origin_key.engine_generation);add(origin_key.window_generation);add(origin_key.revision);
        add(origin_key.append_epoch);add(origin_key.selection_epoch);add(origin_key.provenance_epoch);
        for(const auto* text:{&origin_key.session_id,&origin_key.thread_id,&origin_key.window_id}){
            add(text->size());hash.update(text->data(),text->size());}
        hash.update(origin_key.model_profile.data(),origin_key.model_profile.size());
        hash.update(origin_key.topology.data(),origin_key.topology.size());
        for(const auto& e:events){add(uint8_t(e.kind));add(e.from);add(e.to);add(e.chunk);
            add(e.lend_tokens);add(e.lend_exact);add(e.planned);add(e.single);add(e.chunked);add(e.lend);}
        for(auto value:base)add(value);
        for(size_t i=0;i<participant_count;++i){const auto& p=participants[i];
            add(p.device);add(p.begin);add(p.end);add(p.slots);add(p.first);add(p.chunk);add(p.lent_chunk);
            add(p.loan_seen);add(p.loan.size());for(const auto& pair:p.loan){add(pair.first);add(pair.second);}}
        return hash.digest();
    }
    bool seal(std::span<const int32_t> history) {
        if(refused||sealed||reached!=boundary||history.size()<boundary||participant_count>participants.size())
            return reject("actual prefix transcript incomplete at import");
        if(!active){sealed=true;return true;} // whollyT1: existing no-plan oracle
        conditioning=prefix_digest(history,boundary);digest=transcript_digest();sealed=true;return true;
    }
    bool origin_matches(const core::WindowKey& current,uint64_t runtime_incarnation,
                        uint64_t residency_generation) const noexcept {
        return origin_runtime_incarnation && origin_runtime_incarnation==runtime_incarnation &&
            origin_residency_generation==residency_generation &&
            origin_key.engine_generation==current.engine_generation && origin_key.window_generation==current.window_generation &&
            origin_key.session_id==current.session_id && origin_key.thread_id==current.thread_id &&
            origin_key.window_id==current.window_id && origin_key.revision==current.revision &&
            origin_key.provenance_epoch==current.provenance_epoch &&
            origin_key.model_profile==current.model_profile && origin_key.topology==current.topology;
    }
    bool matches(std::span<const int32_t> history,uint64_t ordinal,uint64_t forwarded) const {
        if(!active||!sealed||refused||request!=ordinal||!boundary||boundary>forwarded||
           history.size()<boundary||participant_count>participants.size()||conditioning!=prefix_digest(history,boundary)||
           events.empty()||digest!=transcript_digest())return false;
        uint64_t at=0;bool batch=false;
        for(const auto& e:events){if(e.from!=at||e.to<=at||e.to>boundary)return false;
            batch|=e.kind==PrefixReplayEvent::Kind::prefill;at=e.to;}
        return at==boundary&&batch;
    }
};
// The holder is allocated under origin admission BEFORE native mutation. Moving the
// sealed transcript at park allocates nothing and retains its own metadata credit.
inline bool retain_recorded_prefix_origin(std::shared_ptr<PrefixReplayPlan>& holder,PrefixReplayPlan& recorded,
    std::span<const int32_t> history,uint64_t ordinal,uint64_t forwarded,const core::WindowKey& key,
    uint64_t runtime_incarnation,uint64_t residency_generation) {
    if(!holder || holder->active || holder->sealed || holder->credit ||
       !recorded.matches(history,ordinal,forwarded) ||
       !recorded.origin_matches(key,runtime_incarnation,residency_generation))return false;
    *holder=std::move(recorded);
    recorded=PrefixReplayPlan{};
    return true;
}
struct PrefixReplayAbort {void* user=nullptr;bool (*requested)(void*)=nullptr;
    bool operator()()const{return requested&&requested(user);}};
struct PrefixReplayProgress {std::array<uint64_t,8> stage_end{};size_t stages=0;int64_t stopped_at=-1;};
struct PrefixReplayCallbacks {
    void* user=nullptr;
    bool (*check)(void*,const PrefixReplayPlan&,std::string&)=nullptr;
    bool (*run_piece)(void*,const PrefixReplayEvent&,std::span<const int32_t>,
                      PrefixReplayAbort,PrefixReplayProgress&,std::string&)=nullptr;
    bool (*refill)(void*,std::string&)=nullptr;
};
struct PrefixReplayView {
    const PrefixReplayPlan* plan=nullptr;const PrefixReplayCallbacks* calls=nullptr;
    uint64_t origin_request=0; // zero means this request; a retained origin never gets relabelled
};
inline bool activation_prefix_view_matches(uint64_t required_boundary,const PrefixReplayView& view,
    const core::SnapshotDescriptor& actual,uint64_t runtime_incarnation,uint64_t residency_generation) {
    if(required_boundary<=64)return !view.plan;
    return view.plan && view.calls && view.calls->user && view.calls->check && view.calls->run_piece && view.calls->refill &&
        view.origin_request==2 && view.plan->request==view.origin_request && view.plan->boundary==required_boundary &&
        view.plan->origin_matches(actual.key,runtime_incarnation,residency_generation) &&
        view.plan->matches(actual.accepted_tokens,view.origin_request,actual.cut.forwarded);
}
} // namespace strata::program
