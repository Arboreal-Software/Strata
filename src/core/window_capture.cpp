#include "strata/core/window_capture.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/window_device.hpp"
#include "strata/core/verify.hpp"

#include <limits>
#include <new>
#include <algorithm>

namespace strata::core {
struct NativeCaptureStorage {
    std::shared_ptr<const ReservationLease> reservation;
    WindowRead read;
    SavedConversation native;
    std::shared_ptr<NativeCaptureStorage> pending_owner;
};
void NativeCaptureFailure::retire_after_teardown() {
    std::lock_guard lock(mutex_);
    if (!storage_) return;
    storage_->pending_owner.reset();
    storage_.reset();
}
namespace {
NativeCaptureResult invalid(const std::string& why) { return {NativeCaptureStatus::invalid,why}; }
bool add(uint64_t& total, uint64_t n) {
    if (n > UINT64_MAX - total) return false;
    total += n; return true;
}
}
NativeCaptureResult capture_native_window(WindowRead read, const ModelGeometry& g,
                                          const std::vector<NativeCaptureStage>& stages,
                                          const SnapshotDescriptor& meta,
                                          uint64_t max_tokens, uint64_t max_bytes,
                                          WindowBudget& budget, BundleHandle& out) {
    if (!read.usable() || read.key() != meta.key || read.cut() != meta.cut || !meta.devices.empty() ||
        stages.empty() || stages.size() > size_t(g.n_layers) || meta.cut.accepted != meta.accepted_tokens.size() ||
        meta.cut.forwarded > meta.cut.accepted || meta.cut.accepted > max_tokens)
        return invalid("capture requires a held common cut and matching accepted history");
    if (!meta.speculation_disabled || meta.continuation.mode != ContinuationMode::new_generation_policy ||
        !meta.continuation.payload.empty() || !meta.continuation.adapter.empty())
        return invalid("capture requires disabled speculation and a new generation policy");
    try {
        if (meta.cut.history != window_history_digest(meta.accepted_tokens) ||
            meta.pending_tokens.size() != meta.cut.accepted-meta.cut.forwarded ||
            !std::equal(meta.pending_tokens.begin(),meta.pending_tokens.end(),meta.accepted_tokens.begin()+meta.cut.forwarded) ||
            std::any_of(meta.accepted_tokens.begin(),meta.accepted_tokens.end(),[](int32_t id){ return id < 0; }))
            return invalid("capture accepted/pending history mismatch");
        std::vector<NativeWindowStage> config;
        for (const auto& stage : stages) {
            if (!stage.session || !stage.verifier || stage.device < 0) return invalid("missing live capture stage");
            if (!stage.verifier->snapshot_binding_matches(stage.session,stage.device))
                return invalid("capture verifier is bound to another session or device");
            const auto& ss = *stage.session;
            if (!ss.qsa_states || ss.qsa_alloc <= 0 || ss.qsa_alloc > g.n_qsa_layers() ||
                ss.qsa_ord0 < 0 || ss.qsa_ord0 > g.n_qsa_layers()-ss.qsa_alloc)
                return invalid("invalid live QSA allocation range");
            NativeWindowStage resolved{ss.layer_lo,ss.layer_hi,{},ss.ple_hist != nullptr};
            for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
                const auto& q = ss.qsa_states[ss.qsa_ord0+i];
                resolved.qsa_formats.push_back((q.kv_hybrid ? 3 : qsa_kv_format(q)) +
                    (q.kv_rot && !q.kv_q4 && !q.kv_hybrid ? 16 : 0));
            }
            config.push_back(std::move(resolved));
        }
        WindowStateLayout layout;
        auto checked = native_window_layout(g,config,meta.key,meta.cut,max_tokens,max_bytes,layout,meta.steering_enabled);
        if (!checked) return invalid(checked.detail);
        // Bound metadata before creating copies. Reserve three payloads (native,
        // descriptor, immutable copy), directories and conservative allocator slack.
        if (meta.approximation.size() > max_tokens || meta.pending_tokens.size() > max_tokens)
            return invalid("capture metadata exceeds configured limits");
        uint64_t metadata = sizeof(SnapshotDescriptor);
        for (size_t n : {meta.key.session_id.size(),meta.key.thread_id.size(),meta.key.window_id.size(),meta.native_layout.size()})
            if (!add(metadata,n)) return invalid("capture metadata overflow");
        if (metadata > 1024*1024 || max_tokens > UINT64_MAX/128 || stages.size() > UINT64_MAX/(1024*1024))
            return invalid("capture metadata limit exceeded");
        uint64_t bytes = max_tokens*128;
        if (!add(bytes,metadata*4) || !add(bytes,stages.size()*1024*1024)) return invalid("capture budget overflow");
        ConversationBuffer empty;
        for (const auto& stage : layout.stages) for (const auto& tensor : stage.tensors) {
            const uint64_t peak = empty.allocation_peak(tensor.bytes);
            if (peak > UINT64_MAX/3 || !add(bytes,peak*3) || !add(bytes,4096)) return invalid("capture payload overflow");
        }
        WindowResources resources;
        resources.device_bytes.resize(budget.snapshot().capacity.device_bytes.size(),0);
        resources.host_bytes = bytes; resources.shadows = 1;
        auto reserved = budget.reserve(resources,ReservationPurpose::private_work);
        if (!reserved) return {NativeCaptureStatus::budget_refused,"host capture capacity unavailable"};
        std::string error;
        bool pending = false;
        for (const auto& stage : stages) {
            auto drain = stage.verifier->poll_snapshot_drain(error);
            if (drain == Verifier::SnapshotDrain::unhealthy) return {NativeCaptureStatus::unhealthy,error};
            if (drain == Verifier::SnapshotDrain::unsupported) return invalid(error);
            pending = pending || drain == Verifier::SnapshotDrain::pending;
        }
        if (pending) return {NativeCaptureStatus::pending,"capture waits for verifier streams"};
        std::vector<int32_t> forwarded(meta.accepted_tokens.begin(),meta.accepted_tokens.begin()+meta.cut.forwarded);
        const std::vector<ConversationImageKey> images;
        const std::vector<ConversationCheckpoint> checkpoints;
        ConversationView view{forwarded,images,checkpoints,meta.steering_enabled};
        auto storage = std::make_shared<NativeCaptureStorage>();
        storage->reservation = reserved.lease;
        storage->read = read;
        auto failure = std::make_shared<NativeCaptureFailure>();
        failure->storage_ = storage;
        auto& native = storage->native;
        native.stage_images.resize(stages.size()-1);
        std::vector<NativePleTokens> ple;
        for (size_t i = 0; i < stages.size(); ++i) {
            if (!read.usable()) return invalid("capture cancelled before next stage");
            const auto& stage = stages[i];
            storage->pending_owner = storage;
            WindowDeviceScope device(stage.device);
            if (!device.ok()) {
                read.mark_unhealthy();
                return {NativeCaptureStatus::unhealthy,"capture could not select owning device",failure};
            }
            auto& image = i == 0 ? native : native.stage_images[i-1];
            // No temporary owns transfer destinations. A failed copy keeps the
            // whole transaction alive even if the caller drops the failure result.
            storage->pending_owner = storage;
            bool copied = false;
            try {
                copied = conversation_snapshot_save_private(image,view,*stage.session,g,error);
            } catch (...) {
                // Keep storage quarantined; exception could follow partial enqueue.
                read.mark_unhealthy();
                return {NativeCaptureStatus::unhealthy,"exception during native capture; teardown required",failure};
            }
            if (!copied) {
                read.mark_unhealthy();
                return {NativeCaptureStatus::unhealthy,error,failure};
            }
            if (!device.finish()) {
                read.mark_unhealthy();
                return {NativeCaptureStatus::unhealthy,"capture could not restore caller device",failure};
            }
            storage->pending_owner.reset();
            ple.push_back({{stage.session->ple_prev[0],stage.session->ple_prev[1]},stage.session->ple_token});
        }
        SnapshotDescriptor descriptor;
        if (!read.usable()) return invalid("capture cancelled before publication");
        checked = describe_native_window(native,g,layout,meta,ple,descriptor);
        if (!checked) return invalid(checked.detail);
        checked = BundleHandle::freeze(descriptor,meta.key,layout,out,reserved.lease);
        if (!checked) return invalid(checked.detail);
        return {NativeCaptureStatus::captured,{}};
    } catch (const std::bad_alloc&) {
        return {NativeCaptureStatus::budget_refused,"host allocation failed during capture"};
    }
}
} // namespace strata::core
