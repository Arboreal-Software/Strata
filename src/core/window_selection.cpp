#include "strata/core/window_selection.hpp"
#include <limits>
#include <stdexcept>

namespace strata::core {
bool SelectedWindowWork::borrow(const std::function<void(const std::vector<NativeSessionView>&,
    const std::shared_ptr<NativeRuntimeAttachment>&)>& use) const {
    if(!execution_ || !work_ || !use) return false;
    const auto state=execution_->view();
    if(!state.in_flight || state.operation!=work_.operation() || state.key!=token_.key ||
       state.cut!=token_.cut || (state.phase!=WindowExecutionPhase::running &&
       state.phase!=WindowExecutionPhase::quiescing)) return false;
    const auto runtime=runtime_.lock();
    use(stages_,runtime);return true;
}
bool SelectedWindowRead::borrow(const std::function<void(const std::vector<NativeReadSessionView>&,
    const std::shared_ptr<const NativeRuntimeAttachment>&)>& use) const {
    if(!read_.usable() || !use) return false;
    const auto runtime=runtime_.lock();
    use(stages_,runtime);return true;
}
WindowSelection::WindowSelection(PreparedNativeWindow initial) : selected_(std::move(initial)) {
    execution_=selected_.execution();
    if(!execution_ || !selected_.claim_selection()) throw std::invalid_argument("selection requires an unselected prepared gang");
}
bool WindowSelection::matches(const WindowSelectionToken& token) const {
    return token.publication==publication_ && execution_->matches(token.key,token.cut);
}
WindowSelectionToken WindowSelection::token() const {
    std::lock_guard lock(mutex_);
    const auto state=execution_->view();
    return {publication_,state.key,state.cut};
}
bool WindowSelection::matches_prepared(const WindowSelectionToken& token,const PreparedNativeWindow& prepared) const {
    std::lock_guard lock(mutex_);
    return prepared.storage_ && selected_.storage_==prepared.storage_ && matches(token);
}
bool WindowSelection::admit(const WindowSelectionToken& token,WindowWork& work) {
    WindowWork admitted;
    {
        std::lock_guard lock(mutex_);
        if(!matches(token) || !execution_->admit(token.key,admitted,&token.cut)) return false;
    }
    work=std::move(admitted);return true;
}
bool WindowSelection::admit(const WindowSelectionToken& token,SelectedWindowWork& out) {
    SelectedWindowWork admitted;
    {
        std::lock_guard lock(mutex_);
        if(!matches(token)) return false;
        // Allocate/copy the view before admission: a failed copy cannot abandon work.
        admitted.token_=token;
        admitted.stages_=selected_.session_views();
        admitted.runtime_=selected_.runtime_attachment();
        admitted.execution_=execution_;
        if(!execution_->admit(token.key,admitted.work_,&token.cut)) return false;
    }
    // Releasing an earlier output's runtime may join producers; never under mutex_.
    out=std::move(admitted);return true;
}
bool WindowSelection::view(const WindowSelectionToken& token,WindowExecutionView& out) const {
    WindowExecutionView current;
    {
        std::lock_guard lock(mutex_);
        if(!matches(token)) return false;
        current=execution_->view();
        // Completion takes only the execution lock and can advance between reads.
        if(current.key!=token.key || current.cut!=token.cut) return false;
    }
    out=std::move(current);return true;
}
bool WindowSelection::cancel(const WindowSelectionToken& token) {
    std::lock_guard lock(mutex_);
    return matches(token) && execution_->cancel(token.key,&token.cut);
}
bool WindowSelection::quiesce(const WindowSelectionToken& token) {
    std::lock_guard lock(mutex_);
    return matches(token) && execution_->begin_quiesce(token.key,&token.cut);
}
bool WindowSelection::acquire_read(const WindowSelectionToken& token,WindowRead& read) {
    WindowRead held;
    {
        std::lock_guard lock(mutex_);
        if(!matches(token) || !execution_->acquire_read(token.key,held,&token.cut)) return false;
    }
    read=std::move(held);return true;
}
bool WindowSelection::acquire_read(const WindowSelectionToken& token,SelectedWindowRead& out) {
    SelectedWindowRead held;
    {
        std::lock_guard lock(mutex_);
        if(!matches(token)) return false;
        held.token_=token;
        for(const auto& stage:selected_.session_views()) held.stages_.push_back({stage.device,stage.session});
        held.runtime_=selected_.runtime_attachment();
        if(!execution_->acquire_read(token.key,held.read_,&token.cut)) return false;
    }
    out=std::move(held);return true;
}
bool WindowSelection::resume(const WindowSelectionToken& token) {
    std::lock_guard lock(mutex_);
    return matches(token) && execution_->resume(token.key);
}
bool WindowSelection::replace(const WindowSelectionToken& expected,const WindowRead& old_read,
                              const PreparedNativeWindow& replacement) {
    const auto* descriptor=replacement.descriptor();
    if(!descriptor) return false;
    const auto& key=descriptor->key;
    // Same-topology window transaction. Cross-window selection/fork registration
    // is a separate manager operation; never silently relabel existing bytes here.
    if(key.engine_generation!=expected.key.engine_generation || key.window_generation!=expected.key.window_generation ||
       key.window_id!=expected.key.window_id || key.thread_id!=expected.key.thread_id || key.session_id!=expected.key.session_id ||
       key.model_profile!=expected.key.model_profile || key.topology!=expected.key.topology ||
       key.provenance_epoch!=expected.key.provenance_epoch || key.revision<expected.key.revision ||
       key.selection_epoch<expected.key.selection_epoch || descriptor->cut.output_watermark<expected.cut.output_watermark)
        return false;
    if(key.revision==expected.key.revision &&
       (key.append_epoch!=expected.key.append_epoch || descriptor->cut.accepted!=expected.cut.accepted ||
        descriptor->cut.forwarded!=expected.cut.forwarded || descriptor->cut.history!=expected.cut.history ||
        descriptor->cut.exact_frontier<expected.cut.exact_frontier)) return false;
    auto next_execution=replacement.execution(); // Allocate before the publication lock.
    PreparedNativeWindow retired;
    std::shared_ptr<WindowExecution> old_execution;
    {
        std::lock_guard lock(mutex_);
        if(!matches(expected) || publication_==std::numeric_limits<uint64_t>::max() || !replacement.claim_selection()) return false;
        if(!execution_->retire_for_replacement(old_read,expected.key,expected.cut)) {
            replacement.release_selection_claim();return false;
        }
        // No allocation, CUDA/IO wait or old-owner destructor under this lock.
        std::swap(retired,selected_);selected_=replacement;
        old_execution=std::move(execution_);execution_=std::move(next_execution);
        ++publication_;
    }
    return true;
}
} // namespace strata::core
