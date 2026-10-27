#include "strata/core/window_token_ledger.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::core {
using Status=WindowTokenStatus;
struct WindowTokenCandidateState {
    std::shared_ptr<WindowTokenLedgerState> owner;
    WindowTokenCandidateView view;
    ManagedWindowWork work;
    bool decode=false, ready=false, finish_succeeded=false;
};
struct WindowTokenLedgerState {
    mutable std::mutex mutex;
    WindowKey key;
    StateCut cut;
    WindowTokenLedgerLimits limits;
    uint64_t serial=0, borrows=0, import_serial=0;
    bool active=true, retired=false;
    std::vector<int32_t> tokens;
    std::shared_ptr<const ReservationLease> reservation;
    // Intentional bounded pending ownership: dropping caller handles is not a
    // proof that GPU work drained. pending_candidate exposes recovery.
    std::shared_ptr<WindowTokenCandidateState> pending;
};
struct WindowTokenCaptureStorage {
    std::shared_ptr<const ReservationLease> reservation;
    ManagedWindowToken token;
    SelectedWindowRead read;
    SnapshotDescriptor metadata;
};
bool WindowTokenCapture::usable() const { return storage_ && storage_->read.lease().usable(); }
const ManagedWindowToken& WindowTokenCapture::token() const {
    if(!storage_)throw std::logic_error("empty capture history");
    return storage_->token;
}
const SnapshotDescriptor& WindowTokenCapture::metadata() const {
    if(!storage_)throw std::logic_error("empty capture history");
    return storage_->metadata;
}
const SelectedWindowRead& WindowTokenCapture::selected_read() const {
    if(!storage_)throw std::logic_error("empty capture history");
    return storage_->read;
}
struct WindowTokenImportState {
    std::shared_ptr<WindowTokenLedgerState> old_state,new_state;
    ManagedWindowToken old_entry;
    BundleHandle source;
    SelectedWindowRead read;
    PreparedNativeWindow prepared;
    uint64_t serial=0;
    bool published=false, consumed=false;
};
WindowTokenImport::~WindowTokenImport(){reset();}
WindowTokenImport::WindowTokenImport(WindowTokenImport&& other) noexcept:state_(std::move(other.state_)){}
WindowTokenImport& WindowTokenImport::operator=(WindowTokenImport&& other) noexcept {
    if(this!=&other){reset();state_=std::move(other.state_);}return *this;
}
void WindowTokenImport::reset() noexcept {
    auto retired=std::move(state_);
    if(retired && !retired->consumed && retired->old_state) {
        auto& old=*retired->old_state;std::lock_guard lock(old.mutex);
        if(retired->serial && old.import_serial==retired->serial) {
            old.import_serial=0;
            // Only an unpublished preparation failure can restore writes without
            // observing manager identity. Published-ticket abandonment is fail-closed.
            if(!retired->published)old.active=true;
        }
    }
    // read/prepared/source owners retire only after releasing ledger mutex.
}
const WindowRead& WindowTokenImport::old_read() const {
    if(!state_)throw std::logic_error("empty import ticket");
    return state_->read.lease();
}
const BundleHandle& WindowTokenImport::source() const {
    if(!state_)throw std::logic_error("empty import ticket");
    return state_->source;
}
namespace {
constexpr uint64_t maximum=std::numeric_limits<uint64_t>::max();
struct Borrow {
    std::shared_ptr<WindowTokenLedgerState> state;
    ~Borrow(){if(state){std::lock_guard lock(state->mutex);--state->borrows;}}
};
bool same_base(const WindowTokenLedgerState& s,const ManagedWindowToken& token) {
    return s.key==token.selection.key && s.cut==token.selection.cut;
}
bool current(const std::shared_ptr<WindowTokenLedgerState>& s,const std::shared_ptr<WindowTokenCandidateState>& c) {
    return c && c->owner==s && s->pending==c;
}
bool terminal(const WindowTokenCandidateState& c,const ManagedWindowSnapshot& snapshot) {
    return snapshot.token==c.view.base && snapshot.execution.phase==WindowExecutionPhase::aborted &&
        !snapshot.execution.in_flight && snapshot.execution.operation==c.view.operation;
}
bool published(const WindowTokenCandidateState& c,const ManagedWindowSnapshot& snapshot) {
    return snapshot.token.registration==c.view.base.registration &&
        snapshot.token.selection.publication==c.view.base.selection.publication &&
        snapshot.token.selection.key==c.view.proposed_key && snapshot.token.selection.cut==c.view.proposed_cut &&
        !snapshot.execution.in_flight && snapshot.execution.operation==c.view.operation &&
        (snapshot.execution.phase==WindowExecutionPhase::running || snapshot.execution.phase==WindowExecutionPhase::quiesced);
}
}
namespace {
struct LedgerBudgetRefusal : std::runtime_error {LedgerBudgetRefusal():std::runtime_error("ledger host budget refused") {}};
std::shared_ptr<WindowTokenLedgerState> allocate_ledger_state(const WindowKey& key,const StateCut& cut,
    WindowTokenLedgerLimits limits,WindowBudget& budget) {
    if(!key.engine_generation || !key.window_generation || key.window_id.empty() || !limits.stages ||
       !limits.max_tokens || limits.max_tokens>size_t(-1)/sizeof(int32_t))throw std::invalid_argument("invalid ledger bounds");
    uint64_t strings=0;
    for(const auto* value:{&key.session_id,&key.thread_id,&key.window_id}) {
        if(value->size()>maximum-strings)throw std::invalid_argument("ledger identity overflow");
        strings+=value->size();
    }
    const uint64_t fixed=sizeof(WindowTokenLedgerState)+sizeof(WindowTokenCandidateState)+4096;
    if(strings>(maximum-fixed)/16)throw std::invalid_argument("ledger identity overflow");
    const uint64_t base_metadata=fixed+16*strings;
    const uint64_t stages_metadata=uint64_t(limits.stages)*(sizeof(StateCut)+sizeof(NativeSessionView));
    if(stages_metadata>maximum-base_metadata)throw std::invalid_argument("ledger stage storage overflow");
    const uint64_t metadata=base_metadata+stages_metadata;
    if(metadata<fixed || limits.max_tokens>(maximum-metadata)/sizeof(int32_t))throw std::invalid_argument("ledger storage overflow");
    auto resources=budget.snapshot().capacity;resources.device_bytes.assign(resources.device_bytes.size(),0);
    resources.host_bytes=metadata+limits.max_tokens*sizeof(int32_t);
    resources.pinned_bytes=resources.disk_bytes=resources.io_slots=resources.shadows=0;
    auto reservation=budget.reserve(resources,ReservationPurpose::foreground);
    if(!reservation)throw LedgerBudgetRefusal();
    auto s=std::make_shared<WindowTokenLedgerState>();s->reservation=std::move(reservation.lease);
    s->key=key;s->cut=cut;s->limits=limits;s->serial=limits.candidate_serial_seed;
    s->tokens.reserve(size_t(limits.max_tokens));return s;
}
bool import_monotonic(const WindowKey& old,const StateCut& cut,const SnapshotDescriptor& next) {
    const auto& k=next.key;const auto& c=next.cut;
    if(k.engine_generation!=old.engine_generation || k.window_generation!=old.window_generation ||
       k.session_id!=old.session_id || k.thread_id!=old.thread_id || k.window_id!=old.window_id ||
       k.model_profile!=old.model_profile || k.topology!=old.topology || k.provenance_epoch!=old.provenance_epoch ||
       k.revision<old.revision || k.selection_epoch<old.selection_epoch || k.append_epoch<old.append_epoch ||
       c.id<cut.id || c.output_watermark<cut.output_watermark)return false;
    if(k.revision==old.revision)
        return k.append_epoch==old.append_epoch && c.accepted==cut.accepted && c.forwarded==cut.forwarded &&
               c.history==cut.history && c.exact_frontier>=cut.exact_frontier;
    return (c.history==cut.history && c.accepted==cut.accepted) || k.append_epoch>old.append_epoch;
}
}
WindowTokenLedger::WindowTokenLedger(const WindowKey& key,const StateCut& cut,WindowTokenLedgerLimits limits,WindowBudget& budget) {
    StateCut initial;initial.id=1;initial.history=window_history_digest({});
    if(cut!=initial)throw std::invalid_argument("ledger requires canonical initial cut or validated import");
    state_=allocate_ledger_state(key,cut,limits,budget);
}
WindowTokenLedgerView WindowTokenLedger::view() const {
    std::lock_guard lock(state_->mutex);
    return {state_->key,state_->cut,state_->pending?state_->pending->view.serial:0,
        state_->pending?std::optional<WindowTokenPhase>(state_->pending->view.phase):std::nullopt,state_->active,state_->import_serial};
}
Status WindowTokenLedger::inspect(const WindowTokenCandidate& candidate,WindowTokenCandidateView& out) const {
    WindowTokenCandidateView result;
    {std::lock_guard lock(state_->mutex);if(!candidate.state_ || candidate.state_->owner!=state_)return Status::stale;
     result=candidate.state_->view;}
    out=std::move(result);return Status::ok;
}
Status WindowTokenLedger::pending_candidate(WindowTokenCandidate& out) const {
    WindowTokenCandidate next;
    {std::lock_guard lock(state_->mutex);if(!state_->pending)return Status::stale;next.state_=state_->pending;}
    out=std::move(next);return Status::ok;
}
Status WindowTokenLedger::borrow_published(const std::function<void(std::span<const int32_t>,size_t)>& fn) const {
    if(!fn)return Status::invalid;
    auto s=state_;Borrow borrow;std::span<const int32_t> tokens;size_t forwarded=0;
    {std::lock_guard lock(s->mutex);if(!s->active&&!s->retired&&!s->import_serial)return Status::inactive;
     if(s->borrows==maximum)return Status::exhausted;
     tokens={s->tokens.data(),size_t(s->cut.accepted)};forwarded=size_t(s->cut.forwarded);++s->borrows;borrow.state=s;}
    fn(tokens,forwarded);return Status::ok;
}
Status WindowTokenLedger::borrow_candidate(const WindowTokenCandidate& candidate,
    const std::function<void(std::span<const int32_t>,size_t)>& fn) const {
    if(!fn)return Status::invalid;
    auto s=state_;Borrow borrow;std::span<const int32_t> tokens;size_t forwarded=0;
    {std::lock_guard lock(s->mutex);if(!current(s,candidate.state_))return Status::stale;
     if(s->borrows==maximum)return Status::exhausted;
     tokens={s->tokens.data(),s->tokens.size()};forwarded=size_t(s->cut.forwarded);++s->borrows;borrow.state=s;}
    fn(tokens,forwarded);return Status::ok;
}
bool WindowTokenLedger::matches_history_prefix(std::span<const int32_t> prompt) const {
    std::lock_guard lock(state_->mutex);if(!state_->active&&!state_->retired&&!state_->import_serial)return false;
    const auto count=size_t(state_->cut.accepted);
    return prompt.size()>=count && std::equal(state_->tokens.begin(),state_->tokens.begin()+count,prompt.begin());
}
Status WindowTokenLedger::stage_accept(const ManagedWindowToken& expected,const ManagedWindowWork& work,
    std::span<const int32_t> suffix,WindowTokenCandidate& out) {
    if(suffix.empty())return Status::invalid;
    return stage(expected,work,suffix,0,false,out);
}
Status WindowTokenLedger::stage_forward(const ManagedWindowToken& expected,const ManagedWindowWork& work,
    uint64_t count,WindowTokenCandidate& out) {
    if(!count)return Status::invalid;
    return stage(expected,work,{},count,false,out);
}
Status WindowTokenLedger::stage_decode(const ManagedWindowToken& expected,const ManagedWindowWork& work,WindowTokenCandidate& out) {
    return stage(expected,work,{},1,true,out);
}
Status WindowTokenLedger::stage(const ManagedWindowToken& expected,const ManagedWindowWork& work,
    std::span<const int32_t> suffix,uint64_t forwarded,bool decode,WindowTokenCandidate& out) {
    auto s=state_;std::shared_ptr<WindowTokenCandidateState> candidate;WindowTokenCandidate next;
    if(work.token()!=expected || work.selected().token()!=expected.selection || !work.lease())return Status::stale;
    for(auto id:suffix)if(id<0)return Status::invalid;
    try {
        bool correct_stages=false;
        if(!work.selected().borrow([&](const auto& views,const auto&){correct_stages=views.size()==s->limits.stages;}) || !correct_stages)return Status::stale;
        // Copies can allocate: prepare them before binding a pending transaction.
        candidate=std::make_shared<WindowTokenCandidateState>();candidate->owner=s;candidate->work=work;
        candidate->decode=decode;candidate->view.base=expected;candidate->view.operation=work.lease().operation();
        candidate->view.proposed_key=expected.selection.key;candidate->view.proposed_cut=expected.selection.cut;
        std::lock_guard lock(s->mutex);
        if(s->import_serial)return Status::busy;
        if(!s->active)return Status::inactive;
        if(s->pending || s->borrows)return Status::busy;
        if(!same_base(*s,expected))return Status::stale;
        if(s->cut.exact_frontier!=s->cut.forwarded)return Status::invalid;
        if(decode && s->cut.accepted-s->cut.forwarded!=1)return Status::invalid;
        const auto appended=decode?uint64_t(1):uint64_t(suffix.size());
        if(appended>s->limits.max_tokens-s->cut.accepted || forwarded>s->cut.accepted-s->cut.forwarded)return Status::limit;
        if(s->serial==maximum || s->cut.id==maximum || (appended && s->key.append_epoch==maximum) ||
           (decode && s->cut.output_watermark==maximum))return Status::exhausted;
        auto& v=candidate->view;v.serial=s->serial+1;++v.proposed_cut.id;
        v.proposed_cut.accepted+=appended;v.proposed_cut.forwarded+=forwarded;v.proposed_cut.exact_frontier+=forwarded;
        if(appended)++v.proposed_key.append_epoch;
        if(decode){++v.proposed_cut.output_watermark;v.phase=WindowTokenPhase::awaiting_sample;}
        else {
            s->tokens.insert(s->tokens.end(),suffix.begin(),suffix.end());
            try {if(appended)v.proposed_cut.history=window_history_digest(s->tokens);}
            catch(...) {s->tokens.resize(size_t(s->cut.accepted));throw;}
            v.phase=WindowTokenPhase::ready;candidate->ready=true;
        }
        s->pending=candidate;++s->serial;next.state_=candidate;
    } catch(const std::bad_alloc&) {return Status::allocation_failed;}
      catch(const std::length_error&) {return Status::allocation_failed;}
    out=std::move(next);return Status::ok;
}
Status WindowTokenLedger::record_sample(const WindowTokenCandidate& candidate,int32_t sampled) {
    auto s=state_;std::lock_guard lock(s->mutex);const auto c=candidate.state_;
    if(!current(s,c))return Status::stale;
    if(s->borrows)return Status::busy;
    if(!c->decode || sampled<0)return Status::invalid;
    if(c->view.output && *c->view.output!=sampled)return Status::invalid;
    if(c->ready)return Status::ok;
    if(c->finish_succeeded)return Status::unresolved;
    if(!c->view.output){s->tokens.push_back(sampled);c->view.output=sampled;}
    try {c->view.proposed_cut.history=window_history_digest(s->tokens);}
    catch(const std::bad_alloc&){c->view.phase=WindowTokenPhase::unresolved;return Status::allocation_failed;}
    c->view.phase=WindowTokenPhase::ready;c->ready=true;return Status::ok;
}
Status WindowTokenLedger::finish(const WindowTokenCandidate& candidate,WindowManager& manager,
    const std::vector<StateCut>& cuts,bool joined,bool drained,WindowTokenReceipt& out) {
    auto s=state_;const auto c=candidate.state_;WindowTokenReceipt receipt;
    ManagedWindowWork retired_work;
    std::shared_ptr<WindowTokenCandidateState> retired_pending;
    {
        std::lock_guard lock(s->mutex);
        if(!current(s,c))return Status::stale;
        if(s->borrows)return Status::busy;
        if(c->view.phase==WindowTokenPhase::awaiting_sample)return Status::awaiting_sample;
        if(!c->ready)return Status::unresolved;
        // Prepare all public receipt storage before the device-state publication.
        try {receipt={c->view.proposed_key,c->view.proposed_cut,c->view.output};}
        catch(const std::bad_alloc&){return Status::allocation_failed;}
        ManagedWindowSnapshot snapshot;
        if(!c->finish_succeeded) {
            if(manager.entry_snapshot(c->view.base.registration,snapshot)!=WindowManagerStatus::ok ||
               snapshot.token!=c->view.base || !snapshot.execution.in_flight ||
               snapshot.execution.operation!=c->view.operation){c->view.phase=WindowTokenPhase::unresolved;return Status::unresolved;}
            if(cuts.size()!=s->limits.stages || !std::all_of(cuts.begin(),cuts.end(),[&](const auto& cut){return cut==c->view.proposed_cut;}))return Status::invalid;
            auto lease=c->work.lease();
            if(!lease.finish(c->view.operation,c->view.proposed_key,cuts,joined,drained)){c->view.phase=WindowTokenPhase::unresolved;return Status::unresolved;}
            c->finish_succeeded=true;
        }
        // Cancellation can return true without publishing a cut. Also detect any
        // violated reactor serialization before authorizing output publication.
        try {
            if(manager.entry_snapshot(c->view.base.registration,snapshot)!=WindowManagerStatus::ok || !published(*c,snapshot)) {
                c->view.phase=WindowTokenPhase::unresolved;return Status::unresolved;
            }
        } catch(...) {c->view.phase=WindowTokenPhase::unresolved;throw;}
        // No allocation or arbitrary owner destruction from this point to unlock.
        s->key.append_epoch=c->view.proposed_key.append_epoch;s->cut=c->view.proposed_cut;
        c->view.phase=WindowTokenPhase::committed;
        retired_work=std::move(c->work);retired_pending=std::move(s->pending);
    }
    out=std::move(receipt);return Status::ok;
}
Status WindowTokenLedger::reconcile_terminal(const WindowTokenCandidate& candidate,WindowManager& manager) {
    auto s=state_;const auto c=candidate.state_;ManagedWindowWork retired_work;
    std::shared_ptr<WindowTokenCandidateState> retired_pending;
    {
        std::lock_guard lock(s->mutex);if(!current(s,c))return Status::stale;
        if(s->borrows)return Status::busy;
        ManagedWindowSnapshot snapshot;
        if(manager.entry_snapshot(c->view.base.registration,snapshot)!=WindowManagerStatus::ok || !terminal(*c,snapshot))return Status::unresolved;
        s->tokens.resize(size_t(s->cut.accepted));c->view.phase=WindowTokenPhase::cancelled;
        retired_work=std::move(c->work);retired_pending=std::move(s->pending);
    }
    return Status::cancelled;
}
Status WindowTokenLedger::prepare_import(WindowManager& manager,const ManagedWindowToken& expected,
    const BundleHandle& source,const WindowKey& snapshot_key,const WindowStateLayout& layout,
    WindowTokenLedgerLimits limits,WindowBudget& budget,std::unique_ptr<WindowTokenLedger>& out,WindowTokenImport& ticket_out) {
    auto old=state_;WindowTokenImport ticket;std::unique_ptr<WindowTokenLedger> next;
    try {
        if(!source || !validate_window_state(*source.get(),snapshot_key,layout))return Status::invalid;
        if(limits.stages!=layout.stages.size() || source.get()->cut.accepted>limits.max_tokens)return Status::limit;
        auto data=std::make_shared<WindowTokenImportState>();data->old_state=old;data->old_entry=expected;data->source=source;
        ticket.state_=data;
        {
            std::lock_guard lock(old->mutex);
            if(old->pending || old->borrows || old->import_serial)return Status::busy;
            if(!old->active)return Status::inactive;
            if(!same_base(*old,expected))return Status::stale;
            if(!import_monotonic(old->key,old->cut,*source.get()))return Status::invalid;
            if(old->serial==maximum || expected.selection.publication==maximum)return Status::exhausted;
            if(manager.acquire_read(expected,data->read)!=WindowManagerStatus::ok)return Status::stale;
            data->serial=++old->serial;old->import_serial=data->serial;old->active=false;
        }
        limits.candidate_serial_seed=std::max(limits.candidate_serial_seed,data->serial);
        auto state=allocate_ledger_state(source.get()->key,source.get()->cut,limits,budget);
        state->tokens.assign(source.get()->accepted_tokens.begin(),source.get()->accepted_tokens.end());state->active=false;
        data->new_state=state;next=std::unique_ptr<WindowTokenLedger>(new WindowTokenLedger(std::move(state)));
        data->published=true;
    } catch(const LedgerBudgetRefusal&){return Status::budget_refused;}
      catch(const std::bad_alloc&){return Status::allocation_failed;}
      catch(const std::length_error&){return Status::allocation_failed;}
      catch(const std::invalid_argument&){return Status::invalid;}
    out=std::move(next);ticket_out=std::move(ticket);return Status::ok;
}
Status WindowTokenLedger::activate_import(WindowTokenImport& ticket,WindowManager& manager,
    const ManagedWindowToken& selected,const PreparedNativeWindow& prepared) {
    const auto data=ticket.state_;if(!data || data->consumed || data->new_state!=state_)return Status::stale;
    SelectedWindowRead read;std::shared_ptr<WindowTokenImportState> retired;
    try {
        std::scoped_lock lock(data->old_state->mutex,state_->mutex);
        auto& old=*data->old_state;auto& next=*state_;
        if(old.import_serial!=data->serial || !same_base(old,data->old_entry) || old.pending || next.pending)return Status::stale;
        if(old.borrows || next.borrows)return Status::busy;
        if(next.active || next.retired)return Status::stale;
        if(selected.registration!=data->old_entry.registration ||
           selected.selection.publication!=data->old_entry.selection.publication+1 ||
           selected.selection.key!=next.key || selected.selection.cut!=next.cut ||
           !prepared || prepared.descriptor()!=data->source.get())return Status::stale;
        if(manager.acquire_read(selected,read)!=WindowManagerStatus::ok ||
           !manager.matches_prepared(selected,prepared) ||
           (data->prepared && !manager.matches_prepared(selected,data->prepared)))return Status::stale;
        data->prepared=prepared; // external ticket retains actual selected owner across observation failure
        ManagedWindowSnapshot snapshot;
        if(manager.entry_snapshot(selected.registration,snapshot)!=WindowManagerStatus::ok || snapshot.token!=selected ||
           snapshot.execution.in_flight || snapshot.execution.phase!=WindowExecutionPhase::quiesced)return Status::unresolved;
        next.active=true;old.active=false;old.retired=true;old.import_serial=0;data->consumed=true;
        retired=std::move(ticket.state_);
    } catch(const std::bad_alloc&){return Status::allocation_failed;}
    return Status::ok;
}
Status WindowTokenLedger::abort_import(WindowTokenImport& ticket,WindowManager& manager) {
    const auto data=ticket.state_;if(!data || data->consumed || data->old_state!=state_)return Status::stale;
    std::shared_ptr<WindowTokenImportState> retired;
    try {
        std::scoped_lock lock(state_->mutex,data->new_state->mutex);
        if(state_->import_serial!=data->serial || state_->pending || state_->borrows || data->new_state->borrows)return Status::busy;
        ManagedWindowSnapshot snapshot;
        if(!data->read.lease().usable() || manager.entry_snapshot(data->old_entry.registration,snapshot)!=WindowManagerStatus::ok ||
           snapshot.token!=data->old_entry || snapshot.execution.in_flight || snapshot.execution.phase!=WindowExecutionPhase::quiesced)return Status::stale;
        state_->active=true;state_->import_serial=0;data->consumed=true;retired=std::move(ticket.state_);
    } catch(const std::bad_alloc&){return Status::allocation_failed;}
    return Status::ok;
}
Status WindowTokenLedger::export_capture(WindowManager& manager,const ManagedWindowToken& expected,bool steering,
    WindowBudget& budget,WindowTokenCapture& out,const BundleHandle& provenance) {
    auto s=state_;WindowTokenCapture result;
    std::shared_ptr<WindowTokenCaptureStorage> storage;
    // All read/source/budget owners declared outside the lock retire after unlock.
    ReservationResult reserved;SelectedWindowRead read;
    try {
        std::lock_guard lock(s->mutex);
        if(s->pending || s->import_serial)return Status::busy;
        if(!s->active)return Status::inactive;
        if(!same_base(*s,expected))return Status::stale;
        if(s->tokens.size()!=s->cut.accepted)return Status::invalid;
        const SnapshotDescriptor* origin=provenance.get();
        if(origin && (origin->key!=s->key || origin->cut!=s->cut || origin->steering_enabled!=steering))return Status::invalid;
        if(s->cut.exact_frontier!=s->cut.forwarded && !origin)return Status::invalid;
        const uint64_t regions=origin?origin->approximation.size():0;
        if(regions>s->limits.max_tokens)return Status::limit;
        uint64_t bytes=sizeof(WindowTokenCaptureStorage)+1024;
        auto add=[&](uint64_t n){if(n>maximum-bytes)return false;bytes+=n;return true;};
        for(const auto* text:{&s->key.session_id,&s->key.thread_id,&s->key.window_id})
            if(text->size()>maximum/4 || !add(uint64_t(text->size())*4))return Status::limit;
        const uint64_t pending=s->cut.accepted-s->cut.forwarded;
        if(s->cut.accepted>maximum/sizeof(int32_t) || pending>maximum/sizeof(int32_t) ||
           regions>maximum/sizeof(ApproximationRegion) ||
           !add(s->cut.accepted*sizeof(int32_t)) || !add(pending*sizeof(int32_t)) ||
           !add(regions*sizeof(ApproximationRegion)) ||
           !add(uint64_t(s->limits.stages)*sizeof(NativeReadSessionView)))return Status::limit;
        auto resources=budget.snapshot().capacity;resources.device_bytes.assign(resources.device_bytes.size(),0);
        resources.host_bytes=bytes;resources.pinned_bytes=resources.disk_bytes=resources.io_slots=resources.shadows=0;
        reserved=budget.reserve(resources,ReservationPurpose::foreground);
        if(!reserved)return Status::budget_refused;
        if(manager.acquire_read(expected,read)!=WindowManagerStatus::ok)return Status::stale;
        storage=std::make_shared<WindowTokenCaptureStorage>();storage->reservation=reserved.lease;storage->token=expected;
        storage->metadata.key=s->key;storage->metadata.cut=s->cut;storage->metadata.steering_enabled=steering;
        storage->metadata.accepted_tokens.assign(s->tokens.begin(),s->tokens.end());
        storage->metadata.pending_tokens.assign(s->tokens.begin()+size_t(s->cut.forwarded),s->tokens.end());
        if(origin)storage->metadata.approximation=origin->approximation;
        if(!read.lease().usable())return Status::stale;
        storage->read=std::move(read);result.storage_=std::move(storage);
    } catch(const std::bad_alloc&){return Status::allocation_failed;}
      catch(const std::length_error&){return Status::allocation_failed;}
    out=std::move(result);return Status::ok;
}
} // namespace strata::core
