#pragma once
// Opt-in serving reactor adapter. Private stdin is a trusted manager transport,
// never user authentication. Core priority/dependency identities are authoritative;
// physical ownership remains with the actual OwnedBatch/RAW callers.
#include "serving_repair_scheduler.hpp"
#include "strata/core/thread_priority.hpp"
#include "strata/core/thread_dependencies.hpp"
#include "strata/core/window_budget.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string_view>
#if defined(__linux__)
#include <sys/random.h>
#endif

namespace strata::program {
inline constexpr const char* kServingThreadPolicy="strict-priority-tier-then-class-lottery-v1";
template<class Read> inline uint64_t serving_thread_epoch_from_os(Read&& read) noexcept {
    uint64_t epoch=0;
    // Nonblocking exact-size admission. No PRNG/counter/time/PID fallback and
    // no partial entropy publication; even a valid all-zero draw refuses.
    const auto bytes=read(&epoch,sizeof(epoch));
    return bytes==sizeof(epoch)?epoch:0;
}
inline uint64_t serving_thread_epoch() noexcept {
#if defined(__linux__)
    return serving_thread_epoch_from_os([](void* p,size_t n) noexcept {
        const auto r=::getrandom(p,n,GRND_NONBLOCK);return r<0?size_t(0):size_t(r);
    });
#else
    return 0; // explicit OS-entropy adapter required before opt-in on other OSes
#endif
}
struct ServingThreadLimits {
    core::PriorityLimits priority{};
    core::DependencyLimits dependency{};
    size_t max_frame_bytes=0,max_control_lines=0;
    uint64_t metadata_bytes=0;
};
struct ServingThreadConfig { bool enabled=false,valid=true;ServingThreadLimits limits; };
template<class T> inline bool serving_thread_number(std::string_view v,T& out) noexcept {
    if(v.empty()||v.front()=='+'||(v.size()>1&&v.front()=='0')||v=="-0"||
       (v.size()>2&&v[0]=='-'&&v[1]=='0'))return false;
    const auto r=std::from_chars(v.data(),v.data()+v.size(),out);
    return r.ec==std::errc{}&&r.ptr==v.data()+v.size();
}
inline bool serving_thread_limits_valid(const ServingThreadLimits& l) noexcept {
    const auto& p=l.priority;const auto& d=l.dependency;
    if(p.min_base>p.max_base||p.max_activity_boost<0||!p.max_activity_ttl_ms||
       !p.max_threads||p.max_threads>4096||!p.max_activity_leases||p.max_activity_leases>256||
       !p.max_id_bytes||p.max_id_bytes>64||!d.max_operations||d.max_operations>4096||!d.max_edges||d.max_edges>16384||
       l.max_frame_bytes<128||l.max_frame_bytes>4096||!l.max_control_lines||l.max_control_lines>256)return false;
    // Conservative explicit workspace envelope, including graph/registry refresh
    // clones, response snapshots and input/replay strings. Not a total RSS claim.
    const uint64_t lower=65536+uint64_t(p.max_threads)*(512+uint64_t(p.max_activity_leases)*256)*4+
        uint64_t(d.max_operations)*(1024+uint64_t(p.max_activity_leases)*256)*4+
        uint64_t(d.max_edges)*128+uint64_t(l.max_control_lines+4)*l.max_frame_bytes*4;
    return l.metadata_bytes>=lower;
}
inline ServingThreadConfig parse_serving_thread_config(const char* policy,const char* text) noexcept {
    ServingThreadConfig c;if(!policy&&!text)return c;
    if(!policy||!text||std::string_view(policy)!=kServingThreadPolicy){c.valid=false;return c;}
    std::array<std::string_view,12> words{};std::string_view input(text);
    for(size_t i=0;i<words.size();++i){const auto comma=input.find(',');words[i]=input.substr(0,comma);
        if(i+1==words.size()){if(comma!=input.npos){c.valid=false;return c;}}
        else if(comma==input.npos){c.valid=false;return c;}
        if(comma!=input.npos)input.remove_prefix(comma+1);
    }
    auto& l=c.limits;auto& p=l.priority;auto& d=l.dependency;
    c.valid=serving_thread_number(words[0],p.min_base)&&serving_thread_number(words[1],p.max_base)&&
        serving_thread_number(words[2],p.max_activity_boost)&&serving_thread_number(words[3],p.max_activity_ttl_ms)&&
        serving_thread_number(words[4],p.max_threads)&&serving_thread_number(words[5],p.max_activity_leases)&&
        serving_thread_number(words[6],p.max_id_bytes)&&serving_thread_number(words[7],d.max_operations)&&
        serving_thread_number(words[8],d.max_edges)&&serving_thread_number(words[9],l.max_frame_bytes)&&
        serving_thread_number(words[10],l.max_control_lines)&&serving_thread_number(words[11],l.metadata_bytes)&&
        serving_thread_limits_valid(l);
    c.enabled=c.valid;return c;
}
inline ServingThreadConfig serving_thread_config() noexcept {
    return parse_serving_thread_config(std::getenv("STRATA_SERVING_THREAD_POLICY_V1"),
                                      std::getenv("STRATA_SERVING_THREAD_LIMITS_V1"));
}
struct ServingThreadReply {
    uint64_t engine=0,sequence=0,graph_revision=0;
    const char* status="invalid_frame";
    std::array<char,65> thread{},actor{};
    uint64_t generation=0,base_version=0,accepted=0,applied=0,cut=0,incarnation=0,lease_version=0;
    uint64_t operation=0,result_revision=0;
    int32_t base=0,boost=0;int64_t effective=0;
    std::optional<uint64_t> expiry;
    std::array<char,2048> json() const noexcept {
        std::array<char,2048> out{};char next[32]="null";
        if(expiry)std::snprintf(next,sizeof(next),"%llu",(unsigned long long)*expiry);
        std::snprintf(out.data(),out.size(),
            "{\"schema\":1,\"engine_epoch\":%llu,\"sequence\":%llu,\"status\":\"%s\",\"graph_revision\":%llu,"
            "\"thread_id\":\"%s\",\"thread_generation\":%llu,\"base\":%d,\"base_version\":%llu,\"effective_priority\":%lld,"
            "\"activity_boost\":%d,\"accepted_revision\":%llu,\"applied_revision\":%llu,\"applied_cut\":%llu,"
            "\"actor_id\":\"%s\",\"actor_incarnation\":%llu,\"lease_version\":%llu,\"operation_serial\":%llu,"
            "\"result_revision\":%llu,\"next_expiry_ms\":%s}",
            (unsigned long long)engine,(unsigned long long)sequence,status,(unsigned long long)graph_revision,
            thread.data(),(unsigned long long)generation,base,(unsigned long long)base_version,(long long)effective,
            boost,(unsigned long long)accepted,(unsigned long long)applied,(unsigned long long)cut,actor.data(),
            (unsigned long long)incarnation,(unsigned long long)lease_version,(unsigned long long)operation,
            (unsigned long long)result_revision,next);return out;
    }
};
struct ServingThreadBinding {
    bool bound=false;uint64_t request=0,occupancy=0;
    core::OperationKey foreground,repair;
};
struct ServingThreadUnit {
    size_t slot=0;uint64_t request=0,occupancy=0;
    core::OperationKey operation;
    core::AdmissionUnit unit=core::AdmissionUnit::decode;
    ServingRepairSelectionCandidate repair;
    bool demand=false,resource_ready=false;
    uint64_t class_weight=0;
};
struct ServingThreadSelection {
    bool ready=false;size_t candidate=0;uint64_t total_weight=0,graph_revision=0;
    int64_t effective_priority=0;
    core::DependencyOperation operation;
    core::ThreadPrioritySnapshot inherited_source;
    uint64_t selected_at_ms=0;
    RepairAdmissionDecision admission;
};
struct ServingThreadPrefillBoundary {
    uint64_t source_window_generation=0,source_cut=0,from=0,to=0;
    std::span<const int64_t> stage_forwarded;
    bool pipeline_joined=false,devices_drained=false,request_fence_open=false,source_bootstrap=false;
    bool valid() const noexcept {
        if(!source_window_generation||(!source_cut&&!source_bootstrap)||(source_cut&&source_bootstrap)||
           from>=to||to>uint64_t(INT64_MAX)||stage_forwarded.empty()||stage_forwarded.size()>64||
           !pipeline_joined||!devices_drained||!request_fence_open)return false;
        for(const auto end:stage_forwarded)if(end<0||uint64_t(end)!=to)return false;
        return true;
    }
};
class ServingThreadBridge {
    // Metadata reservation outlives every graph/registry/snapshot/reply member.
    std::shared_ptr<const core::ReservationLease> reservation_;
    uint64_t engine_,last_sequence_=0,admission_cut_=0;
    ServingThreadLimits limits_;
    core::ThreadPriorities priorities_;
    core::ThreadDependencies dependencies_;
    core::DependencySnapshot graph_;
    std::array<ServingThreadBinding,2> bindings_{};
    std::vector<core::ThreadKey> closed_threads_;
    std::string last_frame_;
    ServingThreadReply last_reply_;
    static const char* error(core::PriorityError e) noexcept {
        using E=core::PriorityError;switch(e){case E::none:return "ok";case E::unknown_generation:return "unknown_generation";
        case E::stale_version:return "stale_version";case E::invalid_priority:return "invalid_priority";
        case E::invalid_time:return "invalid_time";case E::limit_exceeded:return "limit_exceeded";
        case E::closed:return "closed";case E::invalid_acknowledgement:return "invalid_acknowledgement";}return "invalid_argument";
    }
    static const char* error(core::DependencyError e) noexcept {
        using E=core::DependencyError;switch(e){case E::none:return "ok";case E::unknown_generation:return "unknown_generation";
        case E::stale_version:return "stale_version";case E::invalid_argument:return "invalid_argument";
        case E::limit_exceeded:return "limit_exceeded";case E::cycle:return "cycle";case E::terminal:return "terminal";
        case E::invalid_time:return "invalid_time";case E::stale_priority:return "stale_priority";
        case E::allocation_failed:return "allocation_failed";}return "invalid_argument";
    }
    static void copy_id(std::array<char,65>& out,const std::string& s) noexcept {
        std::snprintf(out.data(),out.size(),"%s",s.c_str());
    }
    bool id(std::string_view s) const noexcept {
        if(s.empty()||s.size()>limits_.priority.max_id_bytes)return false;
        for(char c:s)if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||
                         c=='_'||c=='.'||c==':'||c=='-'))return false;return true;
    }
    void priority_reply(ServingThreadReply& reply,core::PriorityResult&& result,std::string_view actor={}) {
        reply.status=error(result.error);const auto& s=result.snapshot;
        copy_id(reply.thread,s.key.id);reply.generation=s.key.generation;reply.base=s.base;
        reply.base_version=s.base_version;reply.accepted=s.accepted_revision;reply.applied=s.applied_revision;
        reply.cut=s.applied_cut;reply.effective=s.local_effective;reply.boost=int32_t(s.local_effective-s.base);
        for(const auto& a:s.activity)if(a.lease_id==actor){copy_id(reply.actor,a.lease_id);
            reply.incarnation=a.incarnation;reply.lease_version=a.version;}
    }
    bool used(const core::OperationKey& key) const noexcept {
        for(const auto& b:bindings_)if(b.bound&&(b.foreground==key||b.repair==key))return true;return false;
    }
    ServingThreadReply dispatch(std::span<const std::string_view> w,uint64_t now,ServingThreadReply r) {
        auto number=[&](size_t i,auto& v){return i<w.size()&&serving_thread_number(w[i],v);};
        const auto verb=w[4];core::ThreadKey thread;thread.engine_generation=engine_;
        if(verb=="CREATE"){
            int32_t base;if(w.size()!=7||!id(w[5])||!number(6,base))return r;
            for(const auto& closed:closed_threads_)if(closed.id==w[5]){r.status="not_ready";return r;}
            priority_reply(r,priorities_.create_thread(std::string(w[5]),base,now));return r;
        }
        if(verb=="BASE"||verb=="ACTOR"||verb=="BOOST"||verb=="DISCONNECT"||
           verb=="RETIRE_ACTOR"||verb=="SNAPSHOT"||verb=="ACTOR_SNAPSHOT"||verb=="CLOSE"||verb=="OPEN"){
            if(w.size()<7||!id(w[5])||!number(6,thread.generation))return r;
            thread.id=std::string(w[5]);uint64_t version=0,incarnation=0,ttl=0;int32_t value=0;
            if(verb=="SNAPSHOT"&&w.size()==7)priority_reply(r,priorities_.snapshot(thread,now));
            else if(verb=="ACTOR_SNAPSHOT"&&w.size()==9&&id(w[7])&&number(8,incarnation)){
                auto view=priorities_.snapshot(thread,now);
                if(view){bool found=false;for(const auto& actor:view.snapshot.activity)
                    if(actor.lease_id==w[7]&&actor.incarnation==incarnation)found=true;
                    if(!found){r.status="unknown_generation";return r;}}
                priority_reply(r,std::move(view),w[7]);
            }
            else if(verb=="BASE"&&w.size()==9&&number(7,version)&&number(8,value))
                priority_reply(r,priorities_.set_base(thread,version,value,now));
            else if(verb=="CLOSE"&&w.size()==8&&number(7,version)){
                // Prepare the remembered identity before closing. Capacity was
                // reserved at construction; later physical retirement reclaims
                // the registry record, never merely close acceptance.
                const bool remembered=std::find(closed_threads_.begin(),closed_threads_.end(),thread)!=closed_threads_.end();
                core::ThreadKey prepared=thread;
                auto result=priorities_.close(thread,version,now);
                if(result&&!remembered)closed_threads_.push_back(std::move(prepared));
                priority_reply(r,std::move(result));
            }
            else if(verb=="ACTOR"&&w.size()==9&&id(w[7])&&number(8,version))
                priority_reply(r,priorities_.register_activity(thread,std::string(w[7]),version,now),w[7]);
            else if((verb=="BOOST"||verb=="DISCONNECT"||verb=="RETIRE_ACTOR")&&
                    w.size()==(verb=="BOOST"?12:10)&&id(w[7])&&number(8,incarnation)&&number(9,version)){
                core::ActivityKey actor{thread,std::string(w[7]),incarnation};
                if(verb=="BOOST"){
                    if(!number(10,value)||!number(11,ttl))return r;
                    priority_reply(r,priorities_.set_activity(actor,version,value,ttl,now),w[7]);
                }else if(verb=="DISCONNECT")priority_reply(r,priorities_.disconnect(actor,version,now),w[7]);
                else priority_reply(r,priorities_.retire_activity(actor,version,now),w[7]);
            }else if(verb=="OPEN"&&w.size()==9&&number(7,version)){
                core::DependencyWorkClass kind;
                if(w[8]=="foreground")kind=core::DependencyWorkClass::foreground;
                else if(w[8]=="live_repair")kind=core::DependencyWorkClass::live_repair;
                else if(w[8]=="saved_maintenance")kind=core::DependencyWorkClass::saved_maintenance;else return r;
                auto out=dependencies_.register_operation(thread,version,kind,graph_.revision,now);
                r.status=error(out.error);r.operation=out.key.serial;r.result_revision=out.key.result_revision;r.graph_revision=out.revision;
                copy_id(r.thread,thread.id);r.generation=thread.generation;
            }
            return r;
        }
        if(verb=="EDGE"||verb=="CANCEL"||verb=="RETIRE"){
            uint64_t revision=0;core::OperationKey a{engine_,0,0},b{engine_,0,0};unsigned add=0;
            if(!number(5,revision)||!number(6,a.serial)||!number(7,a.result_revision))return r;
            core::DependencyResult result;bool observed=false;
            if(verb=="CANCEL"&&w.size()==8){result=dependencies_.finish(a,core::DependencyState::cancelled,revision);observed=true;}
            else if(verb=="RETIRE"&&w.size()==8){
                if(used(a))r.status="not_owned";else {result=dependencies_.retire(a,revision);observed=true;}
            }
            else if(verb=="EDGE"&&w.size()==11&&number(8,b.serial)&&number(9,b.result_revision)&&number(10,add)&&add<=1){
                if(add&&used(a))r.status="not_ready";
                else {result=dependencies_.update_dependencies({{a,b,add!=0}},revision);observed=true;}
            }
            if(observed){r.status=error(result.error);r.graph_revision=result.revision;}
            r.operation=a.serial;r.result_revision=a.result_revision;return r;
        }
        if(verb=="BIND"){
            unsigned slot=2;uint64_t request=0;core::OperationKey foreground{engine_,0,0},repair{engine_,0,0};
            if(w.size()!=11||!number(5,slot)||slot>=bindings_.size()||w[6].size()!=12||
               !number(7,foreground.serial)||!number(8,foreground.result_revision)||
               !number(9,repair.serial)||!number(10,repair.result_revision))return r;
            for(char c:w[6]){unsigned n;if(c>='0'&&c<='9')n=unsigned(c-'0');else if(c>='a'&&c<='f')n=unsigned(c-'a'+10);else return r;request=(request<<4)|n;}++request;
            if(bindings_[slot].bound){r.status="slot_busy";return r;}
            const auto* fg=operation(foreground);const auto* bg=operation(repair);
            if(!fg||!bg){r.status="unknown_generation";return r;}
            if(fg->thread.key!=bg->thread.key||fg->work_class!=core::DependencyWorkClass::foreground||
               bg->work_class!=core::DependencyWorkClass::live_repair||used(foreground)||used(repair)){
                r.status="not_owned";return r;
            }
            if(fg->state!=core::DependencyState::pending||bg->state!=core::DependencyState::pending||fg->thread.closed||bg->thread.closed){r.status="terminal";return r;}
            if(!fg->dependencies_satisfied||!bg->dependencies_satisfied){r.status="not_ready";return r;}
            for(const auto& b:bindings_)if(b.bound&&b.request==request){r.status="not_owned";return r;}
            bindings_[slot]={true,request,0,foreground,repair};r.status="ok";
            copy_id(r.thread,fg->thread.key.id);r.generation=fg->thread.key.generation;return r;
        }
        return r;
    }
public:
    ServingThreadBridge(uint64_t engine,ServingThreadLimits limits,std::shared_ptr<const core::ReservationLease> reservation)
        :reservation_(std::move(reservation)),engine_(engine),limits_(limits),priorities_(engine,limits.priority),
         dependencies_(engine,limits.dependency,priorities_) {
        if(!engine||!serving_thread_limits_valid(limits)||!reservation_||
           !reservation_->canonical_owner(reservation_)||reservation_->resources().host_bytes<limits.metadata_bytes)
            throw std::invalid_argument("explicit paid serving thread policy required");
        last_frame_.reserve(limits_.max_frame_bytes);
        closed_threads_.reserve(limits_.priority.max_threads);
    }
    const ServingThreadLimits& limits() const noexcept{return limits_;}
    uint64_t engine() const noexcept{return engine_;}
    bool refresh(uint64_t now) {
        if(!priorities_.advance_time(now)||!dependencies_.refresh_priorities(now))return false;
        graph_=dependencies_.snapshot();
        for(size_t i=0;i<closed_threads_.size();){const auto key=closed_threads_[i];bool owned=false;
            for(const auto& binding:bindings_)if(binding.bound){
                const auto* fg=operation(binding.foreground);const auto* bg=operation(binding.repair);
                if((fg&&fg->thread.key==key)||(bg&&bg->thread.key==key))owned=true;
            }
            if(owned){++i;continue;}
            // Closed thread invalidates its standalone result handles as well.
            // Failure propagation already ran in the real core refresh above.
            for(;;){core::OperationKey terminal{};
                for(const auto& op:graph_.operations)if(op.thread.key==key){
                    if(op.state==core::DependencyState::pending)return false;
                    terminal=op.key;break;
                }
                if(!terminal.serial)break;
                if(!dependencies_.retire(terminal,graph_.revision))return false;
                graph_=dependencies_.snapshot();
            }
            auto snapshot=priorities_.snapshot(key,now);
            if(!snapshot||!priorities_.retire_closed_thread(key,snapshot.snapshot.accepted_revision,now))return false;
            closed_threads_.erase(closed_threads_.begin()+std::ptrdiff_t(i));
        }
        return true;
    }
    std::optional<uint64_t> next_expiry() const{return priorities_.next_activity_expiry_ms();}
    const core::DependencyOperation* operation(core::OperationKey key) const noexcept {
        for(const auto& op:graph_.operations)if(op.key==key)return &op;return nullptr;
    }
    static bool runnable(const core::DependencyOperation* op) noexcept {
        return op&&op->state==core::DependencyState::pending&&op->dependencies_satisfied&&!op->thread.closed;
    }
    const ServingThreadBinding* binding(size_t slot) const noexcept {
        return slot<bindings_.size()&&bindings_[slot].bound?&bindings_[slot]:nullptr;
    }
    bool request_ready(size_t slot,uint64_t request) const noexcept {
        const auto* b=binding(slot);return b&&b->request==request&&!b->occupancy&&runnable(operation(b->foreground));
    }
    std::optional<int64_t> request_priority(size_t slot,uint64_t request) const noexcept {
        if(!request_ready(slot,request))return {};
        return operation(bindings_[slot].foreground)->effective_priority;
    }
    std::optional<int64_t> active_priority() const noexcept {
        std::optional<int64_t> result;
        for(const auto& b:bindings_)if(b.bound&&b.occupancy){const auto* op=operation(b.foreground);
            if(runnable(op)&&(!result||op->effective_priority>*result))result=op->effective_priority;}
        return result;
    }
    bool cancellation_pending(size_t slot) const noexcept {
        const auto* b=binding(slot);if(!b)return false;
        for(const auto& key:{b->foreground,b->repair}){const auto* op=operation(key);
            if(!op||op->thread.closed||op->state==core::DependencyState::cancelled||op->state==core::DependencyState::failed||
               op->state==core::DependencyState::dependency_failed)return true;}
        return false;
    }
    bool attach(size_t slot,uint64_t request,uint64_t occupancy) noexcept {
        if(!occupancy||!request_ready(slot,request))return false;
        bindings_[slot].occupancy=occupancy;return true;
    }
    bool foreground_finished(size_t slot,bool cancel,uint64_t now) {
        const auto* b=binding(slot);if(!b)return false;
        const auto* op=operation(b->foreground);if(!op)return false;
        if(op->state==core::DependencyState::pending&&
           !dependencies_.finish(b->foreground,cancel?core::DependencyState::cancelled:core::DependencyState::complete,graph_.revision))return false;
        return refresh(now);
    }
    // Caller must prove its actual native owner fully retired before releasing.
    // This is not a drain API and never infers retirement from graph terminal state.
    bool release(size_t slot,uint64_t request,uint64_t occupancy,bool cancel,uint64_t now) {
        const auto* b=binding(slot);if(!b||b->request!=request||b->occupancy!=occupancy)return false;
        const auto keys=std::array{b->foreground,b->repair};
        for(const auto& key:keys){const auto* op=operation(key);if(!op)return false;
            if(op->state==core::DependencyState::pending){
                auto result=dependencies_.finish(key,cancel?core::DependencyState::cancelled:core::DependencyState::complete,graph_.revision);
                if(!result||!refresh(now))return false;
            }
        }
        for(const auto& key:keys)if(!dependencies_.retire(key,graph_.revision)||!refresh(now))return false;
        bindings_[slot]={};return true;
    }
    ServingThreadReply command(std::string_view frame,uint64_t now) {
        ServingThreadReply reply;reply.engine=engine_;
        if(frame.size()>limits_.max_frame_bytes)return reply;
        std::array<std::string_view,20> words{};size_t count=0;auto tail=frame;
        while(!tail.empty()&&count<words.size()){
            const auto pos=tail.find(' ');auto word=tail.substr(0,pos);if(word.empty())return reply;
            words[count++]=word;if(pos==tail.npos){tail={};break;}
            tail.remove_prefix(pos+1);if(tail.empty())return reply;
        }
        uint64_t version=0,engine=0,sequence=0;
        if(!tail.empty()||count<5||words[0]!="SCHED"||!serving_thread_number(words[1],version)||version!=1||
           !serving_thread_number(words[2],engine)||!serving_thread_number(words[3],sequence)||!sequence)return reply;
        reply.sequence=sequence;
        if(engine!=engine_){reply.status="stale_engine";return reply;}
        if(sequence==last_sequence_&&last_frame_==frame)return last_reply_;
        if(last_sequence_==UINT64_MAX){reply.status="sequence_exhausted";return reply;}
        if(sequence!=last_sequence_+1){reply.status="stale_sequence";return reply;}
        // Allocate exact replay input BEFORE any mutation; all reply fields and
        // publication are fixed storage. No uncertain changed-frame retry.
        std::string prepared(frame);
        try {
            if(!refresh(now))reply.status="stale_priority";
            else {reply.graph_revision=graph_.revision;reply=dispatch(std::span(words.data(),count),now,reply);}
        }catch(const std::bad_alloc&){reply.status="allocation_failed";}
        // No allocation needed for these scalar receipt fields. Refresh on the
        // next ordered boundary independently of reply delivery.
        // No graph snapshot or other allocation after dispatch may commit.
        // Mutating dependency calls return their actual revision in fixed POD;
        // registry calls fully prepare the core response before committing.
        reply.expiry=priorities_.next_activity_expiry_ms();
        last_frame_=std::move(prepared);last_reply_=reply;last_sequence_=sequence;return reply;
    }
    ServingThreadSelection select(const ServingRepairAdmissionPolicy& policy,
            const ServingRepairSelectionBoundary& boundary,std::span<const ServingThreadUnit> units,
            uint64_t now_ms,uint64_t ticket) {
        ServingThreadSelection out;if(!refresh(now_ms)||!boundary.safe_boundary||boundary.terminal_input||boundary.other_lane_work||units.size()>4)return out;
        std::array<const core::DependencyOperation*,4> ops{};std::array<bool,4> eligible{};
        bool foreground=false,live=false;
        for(size_t i=0;i<units.size();++i){const auto& u=units[i];const auto* b=binding(u.slot);
            if(!b||b->request!=u.request||b->occupancy!=u.occupancy)continue;
            const auto* op=operation(u.operation);if(!runnable(op)||!u.demand||!u.resource_ready)continue;
            if(u.operation!=b->foreground&&u.operation!=b->repair)continue;
            if(op->work_class==core::DependencyWorkClass::foreground){
                if(u.unit!=core::AdmissionUnit::decode&&u.unit!=core::AdmissionUnit::prefill)continue;
                ops[i]=op;eligible[i]=true;foreground=true;
            }else if(u.unit==core::AdmissionUnit::repair||u.unit==core::AdmissionUnit::save)ops[i]=op;
        }
        auto actual=boundary;actual.foreground.current_request_generating=foreground;
        for(size_t i=0;i<units.size();++i)if(ops[i]&&ops[i]->work_class==core::DependencyWorkClass::live_repair){
            auto c=units[i].repair;c.work.work_class=ops[i]->work_class;c.work.unit=units[i].unit;
            c.work.demand=units[i].demand;c.work.resource_ready=units[i].resource_ready;
            if(serving_repair_selection_admission(policy,actual,c,false).admitted())live=true;
        }
        int64_t tier=std::numeric_limits<int64_t>::min();
        std::array<RepairAdmissionDecision,4> admissions{};
        for(size_t i=0;i<units.size();++i)if(ops[i]){
            if(ops[i]->work_class!=core::DependencyWorkClass::foreground){
                auto c=units[i].repair;c.work.work_class=ops[i]->work_class;c.work.unit=units[i].unit;
                c.work.demand=units[i].demand;c.work.resource_ready=units[i].resource_ready;
                admissions[i]=serving_repair_selection_admission(policy,actual,c,foreground||live);
                eligible[i]=admissions[i].admitted();
            }
            if(eligible[i])tier=std::max(tier,ops[i]->effective_priority);
        }
        for(size_t i=0;i<units.size();++i)if(eligible[i]&&ops[i]->effective_priority==tier){
            if(!units[i].class_weight||UINT64_MAX-out.total_weight<units[i].class_weight)return {};
            out.total_weight+=units[i].class_weight;
        }
        if(ticket>=out.total_weight)return out;
        for(size_t i=0;i<units.size();++i)if(eligible[i]&&ops[i]->effective_priority==tier){
            if(ticket<units[i].class_weight){
                out.ready=true;out.candidate=i;out.graph_revision=graph_.revision;out.operation=*ops[i];
                out.effective_priority=tier;out.selected_at_ms=now_ms;out.admission=admissions[i];
                const auto* source=operation(ops[i]->priority_source);if(!source)return {};
                out.inherited_source=source->thread;return out;
            }
            ticket-=units[i].class_weight;
        }
        return out;
    }
    // A foreground lottery winner admits the genuine shared native window for
    // every independently eligible foreground peer in the SAME priority tier.
    // The winner never authorizes a blocked/lower-tier peer implicitly.
    ServingThreadSelection foreground_peer(const ServingThreadSelection& winner,const ServingThreadUnit& unit) const {
        if(!winner.ready||winner.operation.work_class!=core::DependencyWorkClass::foreground||
           winner.graph_revision!=graph_.revision||!unit.demand||!unit.resource_ready||
           (unit.unit!=core::AdmissionUnit::decode&&unit.unit!=core::AdmissionUnit::prefill))return {};
        const auto* b=binding(unit.slot);const auto* op=operation(unit.operation);
        if(!b||b->foreground!=unit.operation||b->request!=unit.request||b->occupancy!=unit.occupancy||
           !runnable(op)||op->effective_priority!=winner.effective_priority)return {};
        const auto* source=operation(op->priority_source);if(!source)return {};
        auto peer=winner;peer.operation=*op;peer.inherited_source=source->thread;return peer;
    }
    // Called ONLY with a real completed native unit and its actual slot/cut.
    // applied_cut is an ordered executor admission serial (as in WindowAdmission),
    // not the independent per-window native StateCut id. Both are reported.
    bool acknowledge_unit(const ServingThreadSelection& selected,const ServingThreadUnit& unit,
                          uint64_t actual_occupancy,uint64_t native_cut) {
        const auto* b=binding(unit.slot);
        if(!selected.ready||admission_cut_==UINT64_MAX||!native_cut||!b||b->request!=unit.request||
           b->occupancy!=actual_occupancy||actual_occupancy!=unit.occupancy||
           selected.operation.key!=unit.operation||selected.graph_revision!=graph_.revision)return false;
        const uint64_t cut=++admission_cut_;
        auto ack=priorities_.acknowledge(selected.operation.thread.key,selected.operation.thread.accepted_revision,cut,selected.selected_at_ms);
        if(!ack)return false;
        if(selected.inherited_source.key!=selected.operation.thread.key&&
           !priorities_.acknowledge(selected.inherited_source.key,selected.inherited_source.accepted_revision,cut,selected.selected_at_ms))return false;
        std::fprintf(stderr,"SCHED_UNIT v=1 engine=%llu admission_cut=%llu slot=%zu request=%012llx occupancy=%llu native_cut=%llu operation=%llu result=%llu priority=%lld class_weight=%llu accepted=%llu source_accepted=%llu\n",
            (unsigned long long)engine_,(unsigned long long)cut,unit.slot,(unsigned long long)(unit.request-1),
            (unsigned long long)actual_occupancy,(unsigned long long)native_cut,(unsigned long long)unit.operation.serial,
            (unsigned long long)unit.operation.result_revision,(long long)selected.effective_priority,
            (unsigned long long)unit.class_weight,(unsigned long long)selected.operation.thread.accepted_revision,
            (unsigned long long)selected.inherited_source.accepted_revision);return true;
    }
    bool acknowledge_prefill(const ServingThreadSelection& selected,const ServingThreadUnit& unit,
                             const ServingThreadPrefillBoundary& actual) {
        const auto* b=binding(unit.slot);
        if(!actual.valid()||!selected.ready||!b||b->request!=unit.request||b->occupancy||unit.occupancy||
           unit.unit!=core::AdmissionUnit::prefill||unit.operation!=b->foreground||
           selected.operation.key!=unit.operation||selected.graph_revision!=graph_.revision||admission_cut_==UINT64_MAX)return false;
        const uint64_t cut=++admission_cut_;
        if(!priorities_.acknowledge(selected.operation.thread.key,selected.operation.thread.accepted_revision,cut,selected.selected_at_ms))return false;
        if(selected.inherited_source.key!=selected.operation.thread.key&&
           !priorities_.acknowledge(selected.inherited_source.key,selected.inherited_source.accepted_revision,cut,selected.selected_at_ms))return false;
        // This is actual admitted executor progress, NOT a fabricated native
        // StateCut/tensor certificate. The original RAW source cut is an anchor;
        // all stage endpoints come from real callbacks joined by Prefill::run.
        std::fprintf(stderr,"SCHED_PREFILL_UNIT v=1 engine=%llu admission_cut=%llu slot=%zu request=%012llx source_window=%llu source_cut=%llu source_bootstrap=%d from=%llu to=%llu stages=%zu priority=%lld class_weight=%llu accepted=%llu source_accepted=%llu\n",
            (unsigned long long)engine_,(unsigned long long)cut,unit.slot,(unsigned long long)(unit.request-1),
            (unsigned long long)actual.source_window_generation,(unsigned long long)actual.source_cut,int(actual.source_bootstrap),
            (unsigned long long)actual.from,(unsigned long long)actual.to,actual.stage_forwarded.size(),
            (long long)selected.effective_priority,(unsigned long long)unit.class_weight,
            (unsigned long long)selected.operation.thread.accepted_revision,(unsigned long long)selected.inherited_source.accepted_revision);
        return true;
    }
    void ready(std::FILE* output) const {
        const auto& p=limits_.priority;const auto& d=limits_.dependency;
        std::fprintf(output,"SCHED_READY {\"schema\":1,\"engine_epoch\":%llu,\"policy\":\"%s\",\"limits\":{\"min_base\":%d,\"max_base\":%d,\"max_activity_boost\":%d,\"max_activity_ttl_ms\":%llu,\"max_threads\":%zu,\"max_activity_leases\":%zu,\"max_id_bytes\":%zu,\"max_operations\":%zu,\"max_edges\":%zu,\"max_frame_bytes\":%zu,\"max_control_lines\":%zu,\"metadata_bytes\":%llu},\"max_reply_bytes\":2048,\"capabilities\":[\"managed_threads\",\"managed_activity\",\"dependencies\",\"request_bindings\",\"accepted_applied\",\"idle_expiry\",\"strict_tier_class_lottery\"]}\n",
            (unsigned long long)engine_,kServingThreadPolicy,p.min_base,p.max_base,p.max_activity_boost,
            (unsigned long long)p.max_activity_ttl_ms,p.max_threads,p.max_activity_leases,p.max_id_bytes,
            d.max_operations,d.max_edges,limits_.max_frame_bytes,limits_.max_control_lines,(unsigned long long)limits_.metadata_bytes);
        std::fflush(output);
    }
};
} // namespace strata::program
