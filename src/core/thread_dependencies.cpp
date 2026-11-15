#include "strata/core/thread_dependencies.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::core {
ThreadDependencies::ThreadDependencies(uint64_t engine,DependencyLimits limits,ThreadPriorities& priorities)
    :engine_(engine),limits_(limits),priorities_(priorities) {
    if(!engine||!limits.max_operations||!limits.max_edges)throw std::invalid_argument("invalid dependency limits");
}
DependencyResult ThreadDependencies::result(DependencyError error,OperationKey key) const{return {error,revision_,key};}
bool ThreadDependencies::can_change(uint64_t expected) const{return expected==revision_&&revision_!=UINT64_MAX;}
bool ThreadDependencies::matches(const Nodes& nodes,const OperationKey& key) const {
    auto it=nodes.find(key.serial);return key.engine_generation==engine_&&it!=nodes.end()&&it->second.key==key;
}
bool ThreadDependencies::order(const Nodes& nodes,std::vector<uint64_t>& out) {
    std::map<uint64_t,size_t> incoming;
    for(const auto& [id,n]:nodes){(void)n;incoming.emplace(id,0);}
    for(const auto& [id,n]:nodes){(void)id;for(auto p:n.prerequisites)++incoming.at(p);}
    out.clear();out.reserve(nodes.size());
    for(const auto& [id,count]:incoming)if(!count)out.push_back(id);
    for(size_t i=0;i<out.size();++i)for(auto p:nodes.at(out[i]).prerequisites)if(!--incoming.at(p))out.push_back(p);
    return out.size()==nodes.size();
}
void ThreadDependencies::propagate_failures(Nodes& nodes) {
    std::vector<uint64_t> sorted;
    if(!order(nodes,sorted))throw std::logic_error("dependency graph cycle");
    for(auto it=sorted.rbegin();it!=sorted.rend();++it){
        auto& n=nodes.at(*it);if(n.state!=DependencyState::pending)continue;
        for(auto p:n.prerequisites){auto state=nodes.at(p).state;
            if(state!=DependencyState::pending&&state!=DependencyState::complete){n.state=DependencyState::dependency_failed;break;}}
    }
}
DependencyResult ThreadDependencies::register_operation(const ThreadKey& thread,uint64_t milestone,
        DependencyWorkClass kind,uint64_t expected,uint64_t now) {
    std::lock_guard lock(mutex_);
    if(expected!=revision_)return result(DependencyError::stale_version);
    if(!can_change(expected)||!next_serial_||nodes_.size()>=limits_.max_operations)return result(DependencyError::limit_exceeded);
    if(!milestone||thread.engine_generation!=engine_||
       (kind!=DependencyWorkClass::foreground&&kind!=DependencyWorkClass::live_repair&&kind!=DependencyWorkClass::saved_maintenance))return result(DependencyError::invalid_argument);
    try {
        auto priority=priorities_.snapshot(thread,now);
        if(!priority)return result(priority.error==PriorityError::invalid_time?DependencyError::invalid_time:DependencyError::unknown_generation);
        if(priority.snapshot.closed)return result(DependencyError::unknown_generation);
        OperationKey key{engine_,next_serial_,milestone};
        nodes_.emplace(key.serial,Node{key,std::move(priority.snapshot),kind,DependencyState::pending,{}});
        next_serial_=next_serial_==UINT64_MAX?0:next_serial_+1;++revision_;return result(DependencyError::none,key);
    }catch(const std::bad_alloc&){return result(DependencyError::allocation_failed);}
}
DependencyResult ThreadDependencies::update_dependencies(const std::vector<DependencyChange>& changes,uint64_t expected) {
    std::lock_guard lock(mutex_);
    if(expected!=revision_)return result(DependencyError::stale_version);
    if(!can_change(expected)||changes.size()>limits_.max_edges)return result(DependencyError::limit_exceeded);
    try {
        Nodes candidate=nodes_;
        for(const auto& c:changes){
            if(!matches(candidate,c.consumer)||!matches(candidate,c.prerequisite))return result(DependencyError::unknown_generation);
            auto& n=candidate.at(c.consumer.serial);
            if(n.state!=DependencyState::pending)return result(DependencyError::terminal);
            if(c.add)n.prerequisites.insert(c.prerequisite.serial);else n.prerequisites.erase(c.prerequisite.serial);
        }
        size_t edges=0;
        for(const auto& [id,n]:candidate){(void)id;if(n.prerequisites.size()>limits_.max_edges-edges)return result(DependencyError::limit_exceeded);edges+=n.prerequisites.size();}
        std::vector<uint64_t> sorted;if(!order(candidate,sorted))return result(DependencyError::cycle);
        propagate_failures(candidate);nodes_.swap(candidate);++revision_;return result();
    }catch(const std::bad_alloc&){return result(DependencyError::allocation_failed);}
}
DependencyResult ThreadDependencies::finish(const OperationKey& key,DependencyState outcome,uint64_t expected) {
    std::lock_guard lock(mutex_);
    if(expected!=revision_)return result(DependencyError::stale_version);
    if(!can_change(expected))return result(DependencyError::limit_exceeded);
    if(!matches(nodes_,key))return result(DependencyError::unknown_generation);
    if(outcome!=DependencyState::complete&&outcome!=DependencyState::failed&&outcome!=DependencyState::cancelled)return result(DependencyError::invalid_argument);
    if(nodes_.at(key.serial).state!=DependencyState::pending)return result(DependencyError::terminal);
    if(outcome==DependencyState::complete)for(auto p:nodes_.at(key.serial).prerequisites)
        if(nodes_.at(p).state!=DependencyState::complete)return result(DependencyError::invalid_argument);
    try {
        auto candidate=nodes_;candidate.at(key.serial).state=outcome;propagate_failures(candidate);
        nodes_.swap(candidate);++revision_;return result();
    }catch(const std::bad_alloc&){return result(DependencyError::allocation_failed);}
}
DependencyResult ThreadDependencies::retire(const OperationKey& key,uint64_t expected) {
    std::lock_guard lock(mutex_);
    if(expected!=revision_)return result(DependencyError::stale_version);
    if(!can_change(expected))return result(DependencyError::limit_exceeded);
    if(!matches(nodes_,key))return result(DependencyError::unknown_generation);
    if(nodes_.at(key.serial).state==DependencyState::pending)return result(DependencyError::invalid_argument);
    // Failure was already propagated and complete incoming edges are satisfied.
    // Erasing these edges preserves consumer state while reclaiming all capacity.
    for(auto& [id,n]:nodes_){(void)id;n.prerequisites.erase(key.serial);}
    nodes_.erase(key.serial);++revision_;return result();
}
DependencyResult ThreadDependencies::refresh_priorities(uint64_t now) {
    std::lock_guard lock(mutex_);
    if(revision_==UINT64_MAX)return result(DependencyError::limit_exceeded);
    try {
        auto candidate=nodes_;bool changed=false;
        for(auto& [id,n]:candidate){
            (void)id;if(n.state!=DependencyState::pending)continue;
            auto r=priorities_.snapshot(n.thread.key,now);
            if(r.error==PriorityError::invalid_time)return result(DependencyError::invalid_time);
            if(!r||r.snapshot.closed){n.state=DependencyState::cancelled;changed=true;continue;}
            if(r.snapshot.accepted_revision<n.thread.accepted_revision||r.snapshot.applied_revision<n.thread.applied_revision||r.snapshot.applied_cut<n.thread.applied_cut)
                return result(DependencyError::stale_priority);
            if(r.snapshot.accepted_revision!=n.thread.accepted_revision||r.snapshot.applied_revision!=n.thread.applied_revision||r.snapshot.applied_cut!=n.thread.applied_cut){
                n.thread=std::move(r.snapshot);changed=true;
            }
        }
        if(changed){propagate_failures(candidate);nodes_.swap(candidate);++revision_;}
        return result();
    }catch(const std::bad_alloc&){return result(DependencyError::allocation_failed);}
}
DependencySnapshot ThreadDependencies::snapshot() const {
    std::lock_guard lock(mutex_);DependencySnapshot view;view.revision=revision_;
    std::map<uint64_t,std::pair<int64_t,uint64_t>> priority;
    for(const auto& [id,n]:nodes_)priority.emplace(id,std::pair{n.thread.local_effective,id});
    std::vector<uint64_t> sorted;order(nodes_,sorted);
    for(auto id:sorted){const auto& n=nodes_.at(id);if(n.state!=DependencyState::pending)continue;
        for(auto p:n.prerequisites){if(nodes_.at(p).state!=DependencyState::pending)continue;
            auto& dest=priority.at(p);const auto src=priority.at(id);
            if(src.first>dest.first||(src.first==dest.first&&src.second<dest.second))dest=src;
        }
    }
    view.operations.reserve(nodes_.size());
    for(const auto& [id,n]:nodes_){
        DependencyOperation op;op.key=n.key;op.thread=n.thread;op.work_class=n.work_class;op.state=n.state;
        const auto [effective,source]=priority.at(id);op.effective_priority=effective;op.priority_source=nodes_.at(source).key;op.priority_source_thread=nodes_.at(source).thread.key;
        op.dependencies_satisfied=n.state==DependencyState::pending;
        for(auto p:n.prerequisites){op.prerequisites.push_back(nodes_.at(p).key);if(nodes_.at(p).state!=DependencyState::complete)op.dependencies_satisfied=false;}
        view.operations.push_back(std::move(op));
    }
    return view;
}
} // namespace strata::core
