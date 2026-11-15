#include "strata/core/window_manager.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {
using Status=WindowManagerStatus;
constexpr uint64_t maximum=std::numeric_limits<uint64_t>::max();
struct Entry {
    WindowRegistrationId id;
    WindowKey expected;
    uint64_t runtime_class=0;
    std::shared_ptr<WindowSelection> selection;
    std::shared_ptr<const ReservationLease> metadata;
    bool busy=false, cancelled=false;
    std::optional<size_t> lane;
};
struct Lane { uint64_t publication=1; bool closed=true; std::shared_ptr<Entry> entry; };
}
struct WindowManagerState {
    mutable std::mutex mutex;
    WindowManagerLimits limits;
    uint64_t serial=0, generation=0;
    std::vector<std::shared_ptr<Entry>> entries;
    std::vector<Lane> lanes;
    std::shared_ptr<const ReservationLease> metadata;
    std::shared_ptr<Entry> find(WindowRegistrationId id) const {
        if(id.engine!=limits.engine_generation || id.manager!=limits.manager_incarnation || !id.serial) return {};
        for(const auto& e:entries) if(e && e->id==id) return e;
        return {};
    }
    ManagedWindowToken token(const Entry& e) const { return {e.id,e.selection->token()}; }
    WindowLaneToken lane_token(size_t i) const {
        const auto& l=lanes.at(i);
        WindowLaneToken result{limits.engine_generation,limits.manager_incarnation,l.publication,i,l.closed,{}};
        if(l.entry) result.window=token(*l.entry);
        return result;
    }
    Status checked(const ManagedWindowToken& t,std::shared_ptr<Entry>& e) const {
        e=find(t.registration);
        if(!e) return Status::stale;
        if(e->cancelled) return Status::cancelled;
        if(e->busy) return Status::busy;
        if(!e->selection) return Status::pending;
        return token(*e)==t?Status::ok:Status::stale;
    }
    Status checked(const WindowLaneToken& t) const {
        if(t.lane>=lanes.size() || lane_token(t.lane)!=t) return Status::stale;
        if(lanes[t.lane].entry && lanes[t.lane].entry->busy) return Status::busy;
        return Status::ok;
    }
    void detach(const std::shared_ptr<Entry>& e) {
        for(auto& slot:entries) if(slot==e) { slot.reset();return; }
    }
};
void WindowRegistrationTicket::reset() noexcept {
    // e is destroyed after the state mutex, including a cancelled publication race.
    std::shared_ptr<Entry> e;
    if(auto s=state_.lock()) {
        std::lock_guard lock(s->mutex);
        e=s->find(id_);
        if(e && !e->selection) {if(e->busy)e->cancelled=true;else s->detach(e);}
    }
    id_={};state_.reset();
}
WindowRegistrationTicket::~WindowRegistrationTicket(){reset();}
WindowRegistrationTicket::WindowRegistrationTicket(WindowRegistrationTicket&& other) noexcept
    :id_(std::exchange(other.id_,{})),state_(std::move(other.state_)){}
WindowRegistrationTicket& WindowRegistrationTicket::operator=(WindowRegistrationTicket&& other) noexcept {
    if(this!=&other){reset();id_=std::exchange(other.id_,{});state_=std::move(other.state_);}return *this;
}
WindowManager::WindowManager(WindowManagerLimits limits,WindowBudget& budget) {
    if(!limits.engine_generation || !limits.manager_incarnation || !limits.max_entries ||
       !limits.max_identity_bytes || limits.lanes.empty()) throw std::invalid_argument("invalid manager bounds");
    for(const auto& lane:limits.lanes) if(!lane.runtime_class) throw std::invalid_argument("invalid runtime class");
    // Conservative bounded metadata envelope, including temporary token copies.
    // Native storage/runtime has its own factory reservation; this is not GPU pricing.
    const uint64_t per=sizeof(Entry)+sizeof(WindowSelection)+512;
    if(limits.max_identity_bytes>(maximum-per)/8) throw std::invalid_argument("manager bounds overflow");
    const uint64_t entry_bytes=per+8*limits.max_identity_bytes;
    if(limits.max_entries>(maximum-sizeof(WindowManagerState))/entry_bytes) throw std::invalid_argument("manager bounds overflow");
    uint64_t bytes=sizeof(WindowManagerState)+limits.max_entries*entry_bytes;
    if(limits.lanes.size()>(maximum-bytes)/(sizeof(Lane)+sizeof(WindowLaneCompatibility))) throw std::invalid_argument("lane bounds overflow");
    bytes+=limits.lanes.size()*(sizeof(Lane)+sizeof(WindowLaneCompatibility));
    auto resources=budget.snapshot().capacity;
    resources.device_bytes.assign(resources.device_bytes.size(),0);
    resources.host_bytes=bytes;resources.pinned_bytes=resources.disk_bytes=resources.io_slots=resources.shadows=0;
    auto reserved=budget.reserve(resources,ReservationPurpose::foreground);
    if(!reserved) throw std::runtime_error("manager metadata budget refused");
    auto s=std::make_shared<WindowManagerState>();s->metadata=std::move(reserved.lease);
    s->serial=limits.last_serial;s->generation=limits.last_window_generation;
    s->entries.resize(limits.max_entries);s->lanes.resize(limits.lanes.size());s->limits=std::move(limits);
    state_=std::move(s);
}
Status WindowManager::reserve_registration(const WindowKey& key,uint64_t runtime_class,WindowRegistrationTicket& out) {
    auto s=state_;auto e=std::make_shared<Entry>();WindowRegistrationTicket ticket;
    if(key.engine_generation!=s->limits.engine_generation || !key.window_generation || key.window_id.empty() || !runtime_class)
        return Status::invalid;
    size_t size=0;
    for(const auto* str:{&key.session_id,&key.thread_id,&key.window_id}) {
        if(str->size()>s->limits.max_identity_bytes-size) return Status::invalid;
        size+=str->size();
    }
    e->expected=key;e->runtime_class=runtime_class;e->metadata=s->metadata;
    {
        std::lock_guard lock(s->mutex);
        for(const auto& old:s->entries) if(old && old->expected.window_generation==key.window_generation && old->expected.window_id==key.window_id)
            return Status::duplicate;
        if(s->generation==maximum) return Status::exhausted;
        if(key.window_generation<=s->generation) return Status::stale;
        if(s->serial==maximum) return Status::exhausted;
        auto slot=std::find(s->entries.begin(),s->entries.end(),nullptr);
        if(slot==s->entries.end()) return Status::capacity;
        e->id={s->limits.engine_generation,s->limits.manager_incarnation,s->serial+1};
        ticket.id_=e->id;ticket.state_=s;
        *slot=e;++s->serial;s->generation=key.window_generation;
    }
    out=std::move(ticket);return Status::ok;
}
Status WindowManager::publish_registration(WindowRegistrationTicket& ticket,PreparedNativeWindow prepared,ManagedWindowToken& out) {
    auto s=state_;std::shared_ptr<Entry> e;
    {
        std::lock_guard lock(s->mutex);
        if(ticket.state_.lock()!=s || !(e=s->find(ticket.id_))) return Status::stale;
        if(e->cancelled) return Status::cancelled;
        if(e->busy) return Status::busy;
        if(e->selection) return Status::invalid;
        e->busy=true;
    }
    std::shared_ptr<WindowSelection> selection;
    ManagedWindowToken token;
    Status result=Status::ok;
    try {
        selection=std::make_shared<WindowSelection>(std::move(prepared));
        token={e->id,selection->token()};
        if(token.selection.key!=e->expected || !selection->quiesce(token.selection)) result=Status::invalid;
    } catch(const std::invalid_argument&) {result=Status::invalid;}
      catch(...) {std::lock_guard lock(s->mutex);e->busy=false;if(e->cancelled)s->detach(e);throw;}
    {
        std::lock_guard lock(s->mutex);
        e->busy=false;
        if(e->cancelled || s->find(e->id)!=e) {s->detach(e);return Status::cancelled;}
        if(result!=Status::ok) return result;
        e->selection=std::move(selection);
        ticket.id_={};ticket.state_.reset();
    }
    out=std::move(token);return Status::ok;
}
Status WindowManager::cancel_registration(WindowRegistrationTicket& ticket) {
    if(ticket.state_.lock()!=state_) return Status::stale;
    ticket.reset();return Status::ok;
}
Status WindowManager::cancel_registration(WindowRegistrationId id) {
    auto s=state_;std::shared_ptr<Entry> retired;
    {
        std::lock_guard lock(s->mutex);retired=s->find(id);
        if(!retired)return Status::stale;
        if(retired->selection)return Status::invalid;
        if(retired->busy)retired->cancelled=true;else s->detach(retired);
    }
    return Status::ok;
}
Status WindowManager::entry_snapshot(WindowRegistrationId id,ManagedWindowSnapshot& out) const {
    auto s=state_;ManagedWindowSnapshot snapshot;
    {
        std::lock_guard lock(s->mutex);auto e=s->find(id);
        if(!e) return Status::stale;
        if(e->cancelled) return Status::cancelled;
        if(e->busy) return Status::busy;
        if(!e->selection) return Status::pending;
        snapshot.token=s->token(*e);snapshot.lane=e->lane;
        if(!e->selection->view(snapshot.token.selection,snapshot.execution)) return Status::stale;
    }
    out=std::move(snapshot);return Status::ok;
}
Status WindowManager::lane_snapshot(size_t lane,WindowLaneToken& out) const {
    WindowLaneToken token;
    {std::lock_guard lock(state_->mutex);if(lane>=state_->lanes.size())return Status::invalid;
     auto& l=state_->lanes[lane];if(l.entry && l.entry->busy)return Status::busy;token=state_->lane_token(lane);}
    out=std::move(token);return Status::ok;
}
bool WindowManager::matches_prepared(const ManagedWindowToken& token,const PreparedNativeWindow& prepared) const {
    std::lock_guard lock(state_->mutex);std::shared_ptr<Entry> e;
    return state_->checked(token,e)==Status::ok && e->selection->matches_prepared(token.selection,prepared);
}
Status WindowManager::request_park(const WindowLaneToken& expected) {
    auto s=state_;std::lock_guard lock(s->mutex);auto result=s->checked(expected);
    if(result!=Status::ok)return result;
    auto& lane=s->lanes[expected.lane];
    if(!lane.entry)return Status::invalid;
    if(lane.closed)return Status::ok;
    if(lane.publication==maximum)return Status::exhausted;
    if(!lane.entry->selection->quiesce(expected.window->selection))return Status::stale;
    lane.closed=true;++lane.publication;return Status::ok;
}
Status WindowManager::switch_active(const WindowLaneToken& expected,const ManagedWindowToken& target,WindowLaneToken& out) {
    auto s=state_;WindowLaneToken next;WindowRead source_read;
    {
        std::lock_guard lock(s->mutex);auto result=s->checked(expected);
        if(result!=Status::ok)return result;
        auto& lane=s->lanes[expected.lane];std::shared_ptr<Entry> e;
        if((result=s->checked(target,e))!=Status::ok)return result;
        if(!lane.closed)return Status::pending;
        if(e->lane && *e->lane!=expected.lane)return Status::bound;
        const auto& compatibility=s->limits.lanes[expected.lane];
        if(e->runtime_class!=compatibility.runtime_class || target.selection.key.model_profile!=compatibility.model_profile ||
           target.selection.key.topology!=compatibility.topology)return Status::incompatible;
        if(lane.publication==maximum)return Status::exhausted;
        WindowExecutionView view;
        if(!e->selection->view(target.selection,view))return Status::stale;
        if(view.phase==WindowExecutionPhase::unhealthy)return Status::unhealthy;
        if(view.phase==WindowExecutionPhase::aborted || view.phase==WindowExecutionPhase::cancelling)return Status::cancelled;
        if(view.phase!=WindowExecutionPhase::quiesced || view.in_flight)return Status::pending;
        // Same-entry resume needs no source witness: resume atomically checks gate/read holds.
        if(lane.entry && lane.entry!=e && !lane.entry->selection->acquire_read(expected.window->selection,source_read))return Status::pending;
        next={expected.engine,expected.manager,lane.publication+1,expected.lane,false,target};
        if(!e->selection->resume(target.selection))return Status::read_held;
        // Everything after successful resume is no-throw, with old entry still held by registry.
        if(lane.entry)lane.entry->lane.reset();
        lane.entry=e;e->lane=expected.lane;lane.closed=false;++lane.publication;
    }
    out=std::move(next);return Status::ok;
}
Status WindowManager::unbind(const WindowLaneToken& expected,WindowLaneToken& out) {
    auto s=state_;WindowRead read;WindowLaneToken next;
    {
        std::lock_guard lock(s->mutex);auto result=s->checked(expected);
        if(result!=Status::ok)return result;
        auto& lane=s->lanes[expected.lane];if(!lane.closed)return Status::pending;
        if(lane.publication==maximum)return Status::exhausted;
        if(lane.entry) {
            WindowExecutionView view;
            if(!lane.entry->selection->view(expected.window->selection,view))return Status::stale;
            if(view.in_flight)return Status::pending;
            if(view.phase!=WindowExecutionPhase::aborted && view.phase!=WindowExecutionPhase::unhealthy &&
               !lane.entry->selection->acquire_read(expected.window->selection,read))return Status::pending;
        }
        next={expected.engine,expected.manager,lane.publication+1,expected.lane,true,{}};
        if(lane.entry)lane.entry->lane.reset();
        lane.entry.reset();++lane.publication;
    }
    out=std::move(next);return Status::ok;
}
Status WindowManager::admit(const WindowLaneToken& expected,ManagedWindowWork& out) {
    auto s=state_;ManagedWindowWork work;
    {
        std::lock_guard lock(s->mutex);auto result=s->checked(expected);
        if(result!=Status::ok)return result;
        auto& lane=s->lanes[expected.lane];if(lane.closed || !lane.entry)return Status::pending;
        work.token_=*expected.window; // Copy identity before admission; no allocation may follow a successful admission.
        work.entry_=lane.entry;
        if(!lane.entry->selection->admit(expected.window->selection,work.work_))return Status::pending;
    }
    out=std::move(work);return Status::ok;
}
Status WindowManager::acquire_read(const ManagedWindowToken& expected,SelectedWindowRead& out) {
    auto s=state_;SelectedWindowRead read;
    {
        std::lock_guard lock(s->mutex);std::shared_ptr<Entry> e;auto result=s->checked(expected,e);
        if(result!=Status::ok)return result;
        if(!e->selection->acquire_read(expected.selection,read))return Status::pending;
    }
    out=std::move(read);return Status::ok;
}
Status WindowManager::request_cancel(const ManagedWindowToken& expected) {
    auto s=state_;std::lock_guard lock(s->mutex);std::shared_ptr<Entry> e;auto result=s->checked(expected,e);
    if(result!=Status::ok)return result;
    if(e->lane && s->lanes[*e->lane].publication==maximum)return Status::exhausted;
    if(!e->selection->cancel(expected.selection))return Status::unhealthy;
    if(e->lane){auto& lane=s->lanes[*e->lane];lane.closed=true;++lane.publication;}
    return Status::ok;
}
Status WindowManager::retire(const ManagedWindowToken& expected) {
    auto s=state_;std::shared_ptr<Entry> retired;
    {
        std::lock_guard lock(s->mutex);auto result=s->checked(expected,retired);
        if(result!=Status::ok)return result;
        if(retired->lane)return Status::bound;
        WindowExecutionView view;
        if(!retired->selection->view(expected.selection,view))return Status::stale;
        if(view.phase==WindowExecutionPhase::unhealthy)return Status::unhealthy;
        if(view.in_flight || view.phase!=WindowExecutionPhase::aborted)return Status::pending;
        s->detach(retired);
    }
    return Status::ok;
}
Status WindowManager::replace_parked(const ManagedWindowToken& expected,const WindowRead& read,
                                   const PreparedNativeWindow& replacement,ManagedWindowToken& out) {
    auto s=state_;std::shared_ptr<Entry> e;ManagedWindowToken next;
    const auto* descriptor=replacement.descriptor();
    if(!descriptor)return Status::invalid;
    next={expected.registration,{expected.selection.publication+1,descriptor->key,descriptor->cut}};
    {
        std::lock_guard lock(s->mutex);auto result=s->checked(expected,e);
        if(result!=Status::ok)return result;
        if(e->lane)return Status::bound;
        if(expected.selection.publication==maximum)return Status::exhausted;
        e->busy=true;
    }
    bool replaced=false;
    try {replaced=e->selection->replace(expected.selection,read,replacement);}
    catch(...) {std::lock_guard lock(s->mutex);e->busy=false;if(e->cancelled)s->detach(e);throw;}
    // Existing replacement creates a running gate; quiesce it before exposing this parked entry.
    if(replaced && !e->selection->quiesce(next.selection))std::terminate();
    {std::lock_guard lock(s->mutex);e->busy=false;}
    if(!replaced)return Status::invalid;
    out=std::move(next);return Status::ok;
}
} // namespace strata::core
