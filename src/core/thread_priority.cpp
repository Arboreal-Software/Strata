#include "strata/core/thread_priority.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace strata::core {
static_assert(std::is_nothrow_move_constructible_v<PriorityResult>);
static_assert(std::is_nothrow_move_assignable_v<ActivityPriority>);
namespace {
void effective(ThreadPrioritySnapshot& snapshot) noexcept {
    int32_t boost=0;
    for(const auto& actor:snapshot.activity)if(actor.active&&!snapshot.closed)boost=std::max(boost,actor.boost);
    snapshot.local_effective=int64_t(snapshot.base)+boost;
}
// Operates only on a private response candidate, before registry publication.
void project_actor(ThreadPrioritySnapshot& snapshot,const ActivityPriority& actor) {
    auto at=std::lower_bound(snapshot.activity.begin(),snapshot.activity.end(),actor.lease_id,
        [](const ActivityPriority& a,const std::string& id){return a.lease_id<id;});
    if(at!=snapshot.activity.end()&&at->lease_id==actor.lease_id)*at=actor;
    else snapshot.activity.insert(at,actor);
    effective(snapshot);
}
}

ThreadPriorities::ThreadPriorities(uint64_t engine, PriorityLimits limits)
    : engine_generation_(engine), limits_(limits) {
    if (!engine || limits.min_base > limits.max_base || limits.max_activity_boost < 0 ||
        !limits.max_activity_ttl_ms || !limits.max_threads || !limits.max_activity_leases || !limits.max_id_bytes)
        throw std::invalid_argument("invalid thread priority limits");
}
bool ThreadPriorities::valid_id(const std::string& id) const {
    return !id.empty() && id.size() <= limits_.max_id_bytes;
}
ThreadPriorities::Record* ThreadPriorities::find(const ThreadKey& key) {
    auto it = threads_.find(key.id);
    return key.engine_generation == engine_generation_ && it != threads_.end() && it->second.view.key == key
        ? &it->second : nullptr;
}
bool ThreadPriorities::bump(Record& r) {
    if (r.view.accepted_revision == UINT64_MAX) return false;
    ++r.view.accepted_revision;
    return true;
}
bool ThreadPriorities::advance_locked(uint64_t now) {
    if (now < now_ms_) return false;
    now_ms_ = now;
    if (!expiry_pending_ || now < next_expiry_ms_) return true;
    expiry_pending_ = false;
    next_expiry_ms_ = UINT64_MAX;
    for (auto& [id,r] : threads_) {
        (void)id;
        bool changed = false;
        for (auto& [name,lease] : r.leases) {
            (void)name;
            if (lease.active && now >= lease.expires_at_ms) {
                lease.active = false; lease.boost = 0;
                if (lease.version != UINT64_MAX) ++lease.version;
                changed = true;
            }
            if (lease.active) {
                expiry_pending_ = true;
                next_expiry_ms_ = std::min(next_expiry_ms_,lease.expires_at_ms);
            }
        }
        if (changed && !bump(r)) {
            r.view.closed = true;
            for (auto& [name,lease] : r.leases) { (void)name; lease.active = false; lease.boost = 0; }
        }
    }
    return true;
}
bool ThreadPriorities::advance_time(uint64_t now) {
    std::lock_guard lock(mutex_);
    return advance_locked(now);
}
std::optional<uint64_t> ThreadPriorities::next_activity_expiry_ms() const {
    std::lock_guard lock(mutex_);
    return expiry_pending_ ? std::optional<uint64_t>(next_expiry_ms_) : std::nullopt;
}
PriorityResult ThreadPriorities::result(const Record& r, PriorityError error) const {
    auto v = r.view;
    int32_t boost = 0;
    for (const auto& [id,a] : r.leases) {
        (void)id;
        v.activity.push_back(a);
        if (a.active && !v.closed) boost = std::max(boost,a.boost);
    }
    v.local_effective = int64_t(v.base) + boost;
    return {error,std::move(v)};
}
PriorityResult ThreadPriorities::register_thread(const ThreadKey& key, int32_t base, uint64_t now) {
    std::lock_guard lock(mutex_);
    if (!advance_locked(now)) return {PriorityError::invalid_time,{}};
    return register_locked(key,base);
}
PriorityResult ThreadPriorities::register_locked(const ThreadKey& key,int32_t base) {
    if (key.engine_generation != engine_generation_ || !key.generation || !valid_id(key.id))
        return {PriorityError::unknown_generation,{}};
    if (base < limits_.min_base || base > limits_.max_base) return {PriorityError::invalid_priority,{}};
    auto it = threads_.find(key.id);
    if (it != threads_.end()) {
        if (it->second.view.key == key)
            return result(it->second,it->second.view.closed ? PriorityError::closed :
                          it->second.registered_base == base ? PriorityError::none : PriorityError::stale_version);
        if (!it->second.view.closed || key.generation <= it->second.view.key.generation || key.generation <= retired_generation_floor_)
            return result(it->second,PriorityError::unknown_generation);
    } else {
        if (key.generation <= retired_generation_floor_) return {PriorityError::unknown_generation,{}};
        if (threads_.size() >= limits_.max_threads) return {PriorityError::limit_exceeded,{}};
    }
    Record r;
    r.view.key = key; r.view.base = base; r.view.base_version = 1; r.view.accepted_revision = 1;
    r.registered_base = base;
    auto out=result(r);
    // New map nodes allocate before insertion; replacing a closed record moves
    // its fully constructed candidate without throwing.
    static_assert(std::is_nothrow_move_assignable_v<Record>);
    threads_.insert_or_assign(key.id,std::move(r));
    highest_thread_generation_=std::max(highest_thread_generation_,key.generation);
    return out;
}
PriorityResult ThreadPriorities::set_base(const ThreadKey& key, uint64_t version, int32_t base, uint64_t now) {
    std::lock_guard lock(mutex_);
    if (!advance_locked(now)) return {PriorityError::invalid_time,{}};
    auto* r = find(key);
    if (!r) return {PriorityError::unknown_generation,{}};
    if (r->view.closed) return result(*r,PriorityError::closed);
    if (r->view.base_version != version) return result(*r,PriorityError::stale_version);
    if (base < limits_.min_base || base > limits_.max_base) return result(*r,PriorityError::invalid_priority);
    if (version == UINT64_MAX || r->view.accepted_revision==UINT64_MAX) return result(*r,PriorityError::limit_exceeded);
    auto out=result(*r);out.snapshot.base=base;++out.snapshot.base_version;++out.snapshot.accepted_revision;
    effective(out.snapshot);
    r->view.base=base;++r->view.base_version;bump(*r);
    return out;
}
PriorityResult ThreadPriorities::set_activity(const ThreadKey& key, const std::string& id, uint64_t version,
                                             int32_t boost, uint64_t ttl, uint64_t now) {
    std::lock_guard lock(mutex_);
    if (!advance_locked(now)) return {PriorityError::invalid_time,{}};
    auto* r = find(key);
    if (!r) return {PriorityError::unknown_generation,{}};
    if (r->view.closed) return result(*r,PriorityError::closed);
    auto legacy=r->leases.find(id);
    if ((legacy!=r->leases.end() && legacy->second.incarnation!=0) ||
        (legacy==r->leases.end() && r->managed_actors)) return result(*r,PriorityError::stale_version);
    return set_activity_locked(*r,id,version,boost,ttl,now);
}
PriorityResult ThreadPriorities::set_activity_locked(Record& record,const std::string& id,uint64_t version,int32_t boost,uint64_t ttl,uint64_t now) {
    auto* r=&record;
    if (!valid_id(id) || boost < 0 || boost > limits_.max_activity_boost || !ttl ||
        ttl > limits_.max_activity_ttl_ms || ttl > UINT64_MAX-now)
        return result(*r,PriorityError::invalid_priority);
    auto it = r->leases.find(id);
    if ((it == r->leases.end() && version != 0) || (it != r->leases.end() && it->second.version != version))
        return result(*r,PriorityError::stale_version);
    if (it == r->leases.end() && r->leases.size() >= limits_.max_activity_leases)
        return result(*r,PriorityError::limit_exceeded);
    if (version == UINT64_MAX || r->view.accepted_revision == UINT64_MAX)
        return result(*r,PriorityError::limit_exceeded);
    ActivityPriority activity{id,version+1,now+ttl,boost,true,it==r->leases.end()?0:it->second.incarnation};
    auto out=result(*r);project_actor(out.snapshot,activity);++out.snapshot.accepted_revision;
    // The complete response exists before map insertion. Insertion failure has
    // no effect; existing values use their noexcept move assignment.
    r->leases.insert_or_assign(id,std::move(activity));
    expiry_pending_ = true;
    next_expiry_ms_ = std::min(next_expiry_ms_,now+ttl);
    bump(*r);
    return out;
}
PriorityResult ThreadPriorities::disconnect(const ThreadKey& key, const std::string& id, uint64_t version, uint64_t now) {
    std::lock_guard lock(mutex_);
    if (!advance_locked(now)) return {PriorityError::invalid_time,{}};
    auto* r = find(key);
    if (!r) return {PriorityError::unknown_generation,{}};
    if (r->view.closed) return result(*r,PriorityError::closed);
    auto legacy=r->leases.find(id);
    if (legacy!=r->leases.end() && legacy->second.incarnation!=0) return result(*r,PriorityError::stale_version);
    return disconnect_locked(*r,id,version,now);
}
PriorityResult ThreadPriorities::disconnect_locked(Record& record,const std::string& id,uint64_t version,uint64_t now) {
    auto* r=&record;
    auto it = r->leases.find(id);
    if (it == r->leases.end() || it->second.version != version) return result(*r,PriorityError::stale_version);
    if (version == UINT64_MAX || r->view.accepted_revision==UINT64_MAX) return result(*r,PriorityError::limit_exceeded);
    auto out=result(*r);
    auto at=std::lower_bound(out.snapshot.activity.begin(),out.snapshot.activity.end(),id,
        [](const ActivityPriority& a,const std::string& name){return a.lease_id<name;});
    at->active=false;at->boost=0;at->expires_at_ms=now;++at->version;
    ++out.snapshot.accepted_revision;effective(out.snapshot);
    it->second.active=false;it->second.boost=0;it->second.expires_at_ms=now;++it->second.version;bump(*r);
    return out;
}
PriorityResult ThreadPriorities::snapshot(const ThreadKey& key, uint64_t now) {
    std::lock_guard lock(mutex_);
    if (!advance_locked(now)) return {PriorityError::invalid_time,{}};
    auto* r = find(key);
    return r ? result(*r) : PriorityResult{PriorityError::unknown_generation,{}};
}
PriorityResult ThreadPriorities::acknowledge(const ThreadKey& key, uint64_t revision, uint64_t cut, uint64_t now) {
    std::lock_guard lock(mutex_);
    if (!advance_locked(now)) return {PriorityError::invalid_time,{}};
    auto* r = find(key);
    if (!r) return {PriorityError::unknown_generation,{}};
    if (r->view.closed) return result(*r,PriorityError::closed);
    if (revision != r->view.accepted_revision || revision < r->view.applied_revision || !cut || cut < r->view.applied_cut ||
        (revision > r->view.applied_revision && cut <= r->view.applied_cut))
        return result(*r,PriorityError::invalid_acknowledgement);
    auto out=result(*r);out.snapshot.applied_revision=revision;out.snapshot.applied_cut=cut;
    r->view.applied_revision=revision;r->view.applied_cut=cut;
    return out;
}
PriorityResult ThreadPriorities::close(const ThreadKey& key, uint64_t revision, uint64_t now) {
    std::lock_guard lock(mutex_);
    if (!advance_locked(now)) return {PriorityError::invalid_time,{}};
    auto* r = find(key);
    if (!r) return {PriorityError::unknown_generation,{}};
    if (r->view.accepted_revision != revision) return result(*r,PriorityError::stale_version);
    if (r->view.closed) return result(*r);
    if (r->view.accepted_revision==UINT64_MAX) return result(*r,PriorityError::limit_exceeded);
    auto out=result(*r);out.snapshot.closed=true;++out.snapshot.accepted_revision;
    for(auto& a:out.snapshot.activity){a.active=false;a.boost=0;}effective(out.snapshot);
    bump(*r);r->view.closed=true;
    for (auto& [id,a] : r->leases) { (void)id; a.active = false; a.boost = 0; }
    return out;
}
PriorityLifecycle ThreadPriorities::lifecycle() const {
    std::lock_guard lock(mutex_);
    return {retired_generation_floor_,highest_thread_generation_==UINT64_MAX?0:highest_thread_generation_+1,next_actor_incarnation_};
}
PriorityResult ThreadPriorities::create_thread(const std::string& id,int32_t base,uint64_t now) {
    std::lock_guard lock(mutex_);
    if(!advance_locked(now))return {PriorityError::invalid_time,{}};
    if(highest_thread_generation_==UINT64_MAX)return {PriorityError::limit_exceeded,{}};
    return register_locked({engine_generation_,highest_thread_generation_+1,id},base);
}
PriorityResult ThreadPriorities::retire_closed_thread(const ThreadKey& key,uint64_t revision,uint64_t now) {
    std::lock_guard lock(mutex_);
    if(!advance_locked(now))return {PriorityError::invalid_time,{}};
    auto* r=find(key);if(!r)return {PriorityError::unknown_generation,{}};
    if(r->view.accepted_revision!=revision)return result(*r,PriorityError::stale_version);
    if(!r->view.closed)return result(*r,PriorityError::closed);
    auto out=result(*r);
    retired_generation_floor_=std::max(retired_generation_floor_,key.generation);
    threads_.erase(key.id);return out;
}
PriorityResult ThreadPriorities::register_activity(const ThreadKey& key,const std::string& id,uint64_t revision,uint64_t now) {
    std::lock_guard lock(mutex_);
    if(!advance_locked(now))return {PriorityError::invalid_time,{}};
    auto* r=find(key);if(!r)return {PriorityError::unknown_generation,{}};
    if(r->view.closed)return result(*r,PriorityError::closed);
    if(r->view.accepted_revision!=revision)return result(*r,PriorityError::stale_version);
    if(!valid_id(id))return result(*r,PriorityError::invalid_priority);
    if(r->leases.count(id))return result(*r,PriorityError::stale_version);
    if(!next_actor_incarnation_||revision==UINT64_MAX||r->leases.size()>=limits_.max_activity_leases)return result(*r,PriorityError::limit_exceeded);
    ActivityPriority a{id,1,now,0,false,next_actor_incarnation_};
    auto out=result(*r);project_actor(out.snapshot,a);++out.snapshot.accepted_revision;
    r->leases.emplace(id,std::move(a));
    next_actor_incarnation_=next_actor_incarnation_==UINT64_MAX?0:next_actor_incarnation_+1;
    r->managed_actors=true;bump(*r);return out;
}
PriorityResult ThreadPriorities::set_activity(const ActivityKey& key,uint64_t version,int32_t boost,uint64_t ttl,uint64_t now) {
    std::lock_guard lock(mutex_);
    if(!advance_locked(now))return {PriorityError::invalid_time,{}};
    auto* r=find(key.thread);if(!r)return {PriorityError::unknown_generation,{}};
    if(r->view.closed)return result(*r,PriorityError::closed);
    auto it=r->leases.find(key.lease_id);
    if(!key.incarnation||it==r->leases.end()||it->second.incarnation!=key.incarnation)return result(*r,PriorityError::unknown_generation);
    return set_activity_locked(*r,key.lease_id,version,boost,ttl,now);
}
PriorityResult ThreadPriorities::disconnect(const ActivityKey& key,uint64_t version,uint64_t now) {
    std::lock_guard lock(mutex_);
    if(!advance_locked(now))return {PriorityError::invalid_time,{}};
    auto* r=find(key.thread);if(!r)return {PriorityError::unknown_generation,{}};
    if(r->view.closed)return result(*r,PriorityError::closed);
    auto it=r->leases.find(key.lease_id);
    if(!key.incarnation||it==r->leases.end()||it->second.incarnation!=key.incarnation)return result(*r,PriorityError::unknown_generation);
    return disconnect_locked(*r,key.lease_id,version,now);
}
PriorityResult ThreadPriorities::retire_activity(const ActivityKey& key,uint64_t version,uint64_t now) {
    std::lock_guard lock(mutex_);
    if(!advance_locked(now))return {PriorityError::invalid_time,{}};
    auto* r=find(key.thread);if(!r)return {PriorityError::unknown_generation,{}};
    if(r->view.closed)return result(*r,PriorityError::closed);
    auto it=r->leases.find(key.lease_id);
    if(it==r->leases.end()||it->second.incarnation!=key.incarnation)return result(*r,PriorityError::unknown_generation);
    if(it->second.version!=version)return result(*r,PriorityError::stale_version);
    if(it->second.active)return result(*r,PriorityError::invalid_priority);
    if(r->view.accepted_revision==UINT64_MAX)return result(*r,PriorityError::limit_exceeded);
    auto out=result(*r);
    auto at=std::lower_bound(out.snapshot.activity.begin(),out.snapshot.activity.end(),key.lease_id,
        [](const ActivityPriority& a,const std::string& name){return a.lease_id<name;});
    out.snapshot.activity.erase(at);++out.snapshot.accepted_revision;effective(out.snapshot);
    // Also supports migration of inactive legacy (incarnation zero) leases. Once
    // any lease is retired, absent legacy version-zero registration is forbidden.
    r->managed_actors=true;r->leases.erase(it);bump(*r);return out;
}
} // namespace strata::core
