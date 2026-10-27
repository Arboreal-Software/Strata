#include "strata/core/window_admission.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace strata::core {
using A=AdmissionStatus;
struct WindowAdmissionRecord {
    // Destroy last: receipt metadata, authorization and work stay charged through retirement.
    std::shared_ptr<const ReservationLease> reservation;
    const WindowAdmission* reactor=nullptr;
    A status=A::acknowledgement_pending;
    uint64_t cut=0, now=0, graph_revision=0;
    DependencyOperation priority;
    ThreadPrioritySnapshot source;
    AdmissionCandidate candidate;
    ManagedWindowWork work;
    // Deliberate one-per-slot retention until exact retained work resolves.
    std::shared_ptr<WindowAdmissionRecord> pending_owner;
};
AdmissionStatus WindowAdmissionHandle::status() const{return record_?record_->status:A::invalid;}
uint64_t WindowAdmissionHandle::cut() const{return record_?record_->cut:0;}
const DependencyOperation* WindowAdmissionHandle::priority() const{return record_?&record_->priority:nullptr;}
const ThreadPrioritySnapshot* WindowAdmissionHandle::inherited_source() const{return record_?&record_->source:nullptr;}
const ManagedWindowWork* WindowAdmissionHandle::work() const{return record_&&record_->status==A::ready?&record_->work:nullptr;}
const ManagedWindowToken* WindowAdmissionHandle::token() const{return record_&&record_->status!=A::released?&record_->work.token():nullptr;}
WindowWork WindowAdmissionHandle::recovery_lease() const{return record_?record_->work.lease():WindowWork{};}
namespace {
WindowResources host(WindowBudget& budget,uint64_t bytes){
    WindowResources r;r.device_bytes.resize(budget.snapshot().capacity.device_bytes.size());r.host_bytes=bytes;return r;
}
bool active(const DependencyOperation& op){return op.state==DependencyState::pending&&op.dependencies_satisfied&&!op.thread.closed;}
// Conservative logical payload lower bound; caller envelopes additionally price
// STL allocation/control-block overhead and transactional registry refresh copies.
bool charge(uint64_t& remaining,uint64_t n){if(n>remaining)return false;remaining-=n;return true;}
bool thread_cost(uint64_t& remaining,const ThreadPrioritySnapshot& t){
    if(!charge(remaining,sizeof(t))||!charge(remaining,t.key.id.size()+1))return false;
    for(const auto& a:t.activity)if(!charge(remaining,sizeof(a))||!charge(remaining,a.lease_id.size()+1))return false;
    return true;
}
bool operation_cost(uint64_t& remaining,const DependencyOperation& op){
    if(!charge(remaining,sizeof(op))||!thread_cost(remaining,op.thread)||!charge(remaining,op.priority_source_thread.id.size()+1))return false;
    for(const auto& key:op.prerequisites)if(!charge(remaining,sizeof(key)))return false;
    return true;
}
bool window_cost(uint64_t& remaining,const WindowKey& key){
    return charge(remaining,key.session_id.size()+1)&&charge(remaining,key.thread_id.size()+1)&&charge(remaining,key.window_id.size()+1);
}
}
WindowAdmission::WindowAdmission(AdmissionLimits l,ThreadPriorities& p,ThreadDependencies& d,WindowManager& m,WindowBudget& b)
    :limits_(l),priorities_(p),dependencies_(d),manager_(m),budget_(b),cut_(l.last_admission_cut){
    if(!l.max_candidates||!l.max_outstanding||!l.workspace_bytes||l.record_bytes<sizeof(WindowAdmissionRecord))
        throw std::invalid_argument("bounded admission requires explicit metadata envelopes");
    auto r=b.reserve(host(b,l.workspace_bytes),ReservationPurpose::foreground);
    if(!r)throw std::runtime_error("admission workspace reservation refused");
    workspace_=std::move(r.lease);records_.resize(l.max_outstanding);
}
bool WindowAdmission::pending() const {
    for(const auto& r:records_)if(r&&(r->status==A::acknowledgement_pending||r->status==A::acknowledgement_blocked))return true;
    return false;
}
AdmissionDecision WindowAdmission::refresh(uint64_t now){
    AdmissionDecision answer;answer.next_expiry_ms=next_expiry_;
    if(pending()){answer.status=A::acknowledgement_pending;return answer;}
    if(now<now_){answer.status=A::invalid;return answer;}
    try {
        if(!priorities_.advance_time(now)){answer.status=A::invalid;return answer;}
        auto refreshed=dependencies_.refresh_priorities(now);
        if(!refreshed){answer.status=refreshed.error==DependencyError::allocation_failed?A::allocation_failed:A::stale;return answer;}
        auto graph=dependencies_.snapshot();
        // Activity may exist before a thread has any operation in the graph.
        // Wake for the entire registry; graph-only deadlines leave idle actors
        // boosted indefinitely when the reactor has no other reason to wake.
        auto next=priorities_.next_activity_expiry_ms();
        auto bytes=limits_.workspace_bytes;
        for(const auto& op:graph.operations){
            if(!operation_cost(bytes,op)){answer.status=A::budget;return answer;}
            if(op.thread.applied_cut>cut_){answer.status=A::stale;return answer;}
        }
        graph_=std::move(graph);now_=now;next_expiry_=next;
        answer.graph_revision=graph_.revision;answer.next_expiry_ms=next;
    }catch(const std::bad_alloc&){answer.status=A::allocation_failed;}
    return answer;
}
AdmissionStatus WindowAdmission::acknowledge(const std::shared_ptr<WindowAdmissionRecord>& r){
    if(r->status!=A::acknowledgement_pending)return r->status;
    try {
        auto ack=priorities_.acknowledge(r->priority.thread.key,r->priority.thread.accepted_revision,r->cut,r->now);
        if(!ack){r->status=A::acknowledgement_blocked;return r->status;}
        if(r->source.key!=r->priority.thread.key){
            ack=priorities_.acknowledge(r->source.key,r->source.accepted_revision,r->cut,r->now);
            if(!ack){r->status=A::acknowledgement_blocked;return r->status;}
        }
        // These scalar observations are published only after all acknowledgements.
        r->priority.thread.applied_revision=r->priority.thread.accepted_revision;
        r->priority.thread.applied_cut=r->cut;
        r->source.applied_revision=r->source.accepted_revision;r->source.applied_cut=r->cut;
        r->status=A::ready;
    }catch(const std::bad_alloc&){
        // Even if a registry prepares its reply before committing its own
        // acknowledgement, manager admission has already happened. Preserve
        // this exact work and cut; a second inherited-source acknowledgement
        // may also fail after the first one succeeded.
        return A::acknowledgement_pending;
    }
    return r->status;
}
AdmissionDecision WindowAdmission::admit(uint64_t now,std::span<const AdmissionCandidate> candidates,
                                         bool other_demand,WindowAdmissionHandle& out){
    auto answer=refresh(now);if(answer.status!=A::no_work)return answer;
    if(candidates.size()>limits_.max_candidates){answer.status=A::capacity;return answer;}
    try {
        auto slot=std::find(records_.begin(),records_.end(),nullptr);
        if(slot==records_.end()){answer.status=A::capacity;return answer;}
        if(cut_==UINT64_MAX){answer.status=A::exhausted;return answer;}
        const DependencyOperation* best=nullptr;size_t chosen=0;
        auto operation=[&](const OperationKey& key)->const DependencyOperation* {
            for(const auto& op:graph_.operations)if(op.key==key)return &op;
            return nullptr;
        };
        bool nonmaintenance_demand=other_demand;
        for(size_t i=0;i<candidates.size();++i){
            const auto& c=candidates[i];const auto* op=operation(c.operation);
            const auto kind=static_cast<size_t>(c.unit);
            if(!op||!c.unit_serial||kind>=limits_.max_units.size()||!c.amount||c.amount>limits_.max_units[kind]||
               c.window.registration.engine!=op->key.engine_generation||c.window.selection.key.thread_id!=op->thread.key.id){
                answer.status=A::invalid;return answer;
            }
            for(size_t j=0;j<i;++j)if(candidates[j].operation==c.operation||
                (candidates[j].window.registration==c.window.registration)){
                answer.status=A::invalid;return answer;
            }
            if(active(*op)&&c.demand&&op->work_class!=DependencyWorkClass::saved_maintenance)nonmaintenance_demand=true;
        }
        for(size_t i=0;i<candidates.size();++i){
            const auto& c=candidates[i];const auto* op=operation(c.operation);
            if(!active(*op)||!c.demand||!c.resource_ready||!c.authorization)continue;
            if(op->work_class==DependencyWorkClass::saved_maintenance&&
                (nonmaintenance_demand||!c.idle_authorized||!c.foreground_return_protected))continue;
            if(!best||op->work_class<best->work_class||
                (op->work_class==best->work_class&&(op->effective_priority>best->effective_priority||
                 (op->effective_priority==best->effective_priority&&op->key.serial<best->key.serial)))){
                best=op;chosen=i;
            }
        }
        if(!best)return answer;
        answer.candidate=chosen;
        const auto& c=candidates[chosen];
        for(const auto& r:records_)if(r&&(r->priority.key==c.operation||r->work.token().registration==c.window.registration)){
            answer.status=A::unresolved;return answer;
        }
        ManagedWindowSnapshot actual;
        if(manager_.entry_snapshot(c.window.registration,actual)!=WindowManagerStatus::ok||actual.token!=c.window){answer.status=A::stale;return answer;}
        if(actual.execution.in_flight||actual.busy||actual.execution.phase==WindowExecutionPhase::cancelling||
           actual.execution.phase==WindowExecutionPhase::aborted||actual.execution.phase==WindowExecutionPhase::unhealthy){answer.status=A::unresolved;return answer;}
        if(!actual.lane||c.lane.closed||!c.lane.window||*c.lane.window!=c.window){answer.status=A::routing_needed;return answer;}
        WindowLaneToken lane;
        if(manager_.lane_snapshot(*actual.lane,lane)!=WindowManagerStatus::ok||lane!=c.lane){answer.status=A::stale;return answer;}
        const auto* source=operation(best->priority_source);
        if(!source){answer.status=A::stale;return answer;}
        auto bytes=limits_.record_bytes;
        if(!charge(bytes,sizeof(WindowAdmissionRecord))||!operation_cost(bytes,*best)||!thread_cost(bytes,source->thread)||
           !window_cost(bytes,c.window.selection.key)||!window_cost(bytes,c.lane.window->selection.key)){
            answer.status=A::budget;return answer;
        }
        auto reservation=budget_.reserve(host(budget_,limits_.record_bytes),ReservationPurpose::foreground);
        if(!reservation){answer.status=A::budget;return answer;}
        auto record=std::make_shared<WindowAdmissionRecord>();
        record->reactor=this;record->reservation=std::move(reservation.lease);record->candidate=c;
        record->priority=*best;record->now=now;record->cut=cut_+1;record->graph_revision=graph_.revision;
        record->source=source->thread;
        // All allocating receipt copies precede actual admission.
        if(manager_.admit(c.lane,record->work)!=WindowManagerStatus::ok){answer.status=A::stale;return answer;}
        ++cut_;record->pending_owner=record;*slot=record;
        answer.status=acknowledge(record);
        WindowAdmissionHandle result;result.record_=std::move(record);out=std::move(result);
    }catch(const std::bad_alloc&){answer.status=A::allocation_failed;}
    return answer;
}
AdmissionStatus WindowAdmission::recover_acknowledgement(const WindowAdmissionHandle& handle){
    auto r=handle.record_;
    if(!r||r->reactor!=this||r->status==A::released||
       std::find(records_.begin(),records_.end(),r)==records_.end())return A::invalid;
    if(r->status!=A::acknowledgement_pending)return r->status;
    try {
        ManagedWindowSnapshot view;
        if(manager_.entry_snapshot(r->work.token().registration,view)!=WindowManagerStatus::ok||
           view.token!=r->work.token()||view.execution.operation!=r->work.lease().operation()||
           !view.execution.in_flight||view.execution.phase!=WindowExecutionPhase::running){
            r->status=A::acknowledgement_blocked;return r->status;
        }
    }catch(const std::bad_alloc&){return A::acknowledgement_pending;}
    return acknowledge(r);
}
AdmissionStatus WindowAdmission::release(const WindowAdmissionHandle& handle){
    auto r=handle.record_;
    if(!r||r->reactor!=this||r->status==A::released||
       std::find(records_.begin(),records_.end(),r)==records_.end())return A::invalid;
    // Do not consult manager: it may select a distinct restored execution now.
    if(!r->work.lease().resolved())return A::unresolved;
    for(auto& slot:records_)if(slot==r){slot.reset();break;}
    r->status=A::released;r->pending_owner.reset();
    // No lock or half-mutated registry spans arbitrary owner retirement.
    r->candidate.authorization.reset();r->work={};
    return A::released;
}
bool WindowAdmission::outstanding(size_t slot,WindowAdmissionHandle& out) const {
    if(slot>=records_.size()||!records_[slot])return false;
    WindowAdmissionHandle result;result.record_=records_[slot];out=std::move(result);return true;
}
} // namespace strata::core
