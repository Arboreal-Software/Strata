#pragma once

#include "suffix_experiment_input.hpp"
#include "suffix_experiment_metrics.hpp"
#include "suffix_raw_runtime.hpp"
#include "serving_prefix_replay.hpp"
#include "strata/core/window_suffix.hpp"
#include "strata/core/resident_fault_receipt.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <charconv>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::program {
// Dedicated disposable-process controller. Context and SerialUnit are the real
// generate.cpp bindings, not public callbacks supplied by a request. Include
// after its concrete runtime types, alongside the checked execution members.
class SuffixExperimentDriver final {
    template<class T> struct TransactionAllocator {
        using value_type=T;
        std::shared_ptr<const void> credit;
        explicit TransactionAllocator(std::shared_ptr<const void> c):credit(std::move(c)){}
        template<class U> TransactionAllocator(const TransactionAllocator<U>& other) noexcept:credit(other.credit){}
        template<class U> struct rebind {using other=TransactionAllocator<U>;};
        T* allocate(size_t count){return std::allocator<T>{}.allocate(count);}
        void deallocate(T* p,size_t count) noexcept {
            auto retained=credit;
            std::allocator<T>{}.deallocate(p,count);
        }
        template<class U> bool operator==(const TransactionAllocator<U>& other)const noexcept {
            return !credit.owner_before(other.credit)&&!other.credit.owner_before(credit);
        }
        template<class U> bool operator!=(const TransactionAllocator<U>& other)const noexcept {return !(*this==other);}
    };
    struct TransactionDelete {
        std::shared_ptr<const void> credit;
        void operator()(SuffixRawTransaction* tx)const noexcept {delete tx;}
    };
    struct Step { core::StateCut cut; int32_t next=-1; };
    struct AdmittedInput {
        std::shared_ptr<const void> credit;
        SuffixExperimentInput input;
    };
    struct Buffers {
        // Last member destruction first: every payload precedes its credit.
        std::shared_ptr<const void> credit;
        std::vector<float> exact_logits, approximate_logits;
        std::vector<int32_t> history, exact_output, approximate_output;
    };
    using Clock=std::chrono::steady_clock;
    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    }
    [[noreturn]] static void failstop(const char* why) noexcept {
        // No stack unwinding may release external callback/DMA buffers after
        // an uncertain attempted operation in this disposable process.
        std::fprintf(stderr,"suffix experiment terminal execution failure: %s\n",why);
        std::fflush(stderr);
        std::_Exit(70);
    }
    static void checked(const SuffixRawResult& result) {
        if(result.status==SuffixRawStatus::poisoned) failstop("poisoned RAW transaction");
    }
    template<class Operation> static SuffixRawResult raw_call(Operation&& operation) {
        // Even constructing an error string can throw after a transfer. Catch
        // here, before the enclosing driver's input/diagnostic buffers unwind.
        try {auto result=operation();checked(result);return result;}
        catch(...) {failstop("RAW transaction exception");}
    }
    static bool add_product(uint64_t& total,uint64_t count,uint64_t unit) {
        if(unit && count>(UINT64_MAX-total)/unit)return false;
        total+=count*unit;return total<=SIZE_MAX;
    }
    template<class Context>
    static SuffixRawResult read_input(SuffixRawTransaction& tx,const Context& c,AdmittedInput& out) {
        const auto& o=c.options;
        if(out.credit || !out.input.old_tokens.empty() || !out.input.edited_tokens.empty() ||
           o.max_new<=0 || o.max_new>64 || o.max_context<=o.max_new ||
           c.geometry.n_vocab<=0 || c.geometry.n_vocab>INT32_MAX)
            return {SuffixRawStatus::invalid,"invalid input bounds",{}};
        const uint64_t tokens=uint64_t(o.max_context-o.max_new);
        uint64_t text=128,peak=65536;
        if(!add_product(text,tokens,24) || !add_product(peak,text,3) || !add_product(peak,tokens,16))
            return {SuffixRawStatus::invalid,"input peak overflow",{}};
        // File growth can overlap the old M string and a new 2M allocation.
        // This is a pinned libstdc++ capacity quote, not a general RSS bound.
        auto admitted=tx.admit_controller_host(peak,out.credit);
        if(!admitted)return admitted;
        const auto status=read_suffix_experiment_input(o.suffix_experiment_file.c_str(),tokens,text,out.input);
        if(status!=SuffixInputReadStatus::ok)
            return {SuffixRawStatus::invalid,"bounded input read/parse refused",{}};
        for(const auto* ids:{&out.input.old_tokens,&out.input.edited_tokens})
            if(ids->size()>tokens || std::any_of(ids->begin(),ids->end(),[&](int32_t t){return t<0 || int64_t(t)>=c.geometry.n_vocab;}))
                return {SuffixRawStatus::invalid,"input outside runtime capacity/vocabulary",{}};
        return {SuffixRawStatus::ok,{},{}};
    }
    template<class Context>
    static core::WindowKey identity(const Context& c) {
        core::WindowKey key;
        key.engine_generation=c.owner->remote_policy_incarnation;
        key.window_generation=1;
        key.session_id="suffix-private";key.thread_id="standalone";key.window_id="one";
        key.revision=key.append_epoch=key.selection_epoch=key.provenance_epoch=1;
        // These are opaque, process-local binding digests. They are NOT a
        // portable content profile. Only this controller creates operands;
        // its retained immutable model/runtime cannot be replaced or exported.
        core::WindowHasher model,topology;
        constexpr char model_domain[]="suffix-fixed-owned-runtime-v1";
        constexpr char topology_domain[]="suffix-fixed-stage-binding-v1";
        model.update(model_domain,sizeof(model_domain));
        topology.update(topology_domain,sizeof(topology_domain));
        const auto runtime=reinterpret_cast<uintptr_t>(c.owner.get());
        const auto weights=reinterpret_cast<uintptr_t>(c.owner->model.get());
        model.update(&key.engine_generation,sizeof(key.engine_generation));
        model.update(&runtime,sizeof(runtime));model.update(&weights,sizeof(weights));
        for(const auto& stage:c.owner->sessions.stages()) {
            const auto session=reinterpret_cast<uintptr_t>(stage.session);
            topology.update(&session,sizeof(session));topology.update(&stage.device,sizeof(stage.device));
            topology.update(&stage.session->layer_lo,sizeof(stage.session->layer_lo));
            topology.update(&stage.session->layer_hi,sizeof(stage.session->layer_hi));
        }
        key.model_profile=model.digest();key.topology=topology.digest();return key;
    }
    static core::SnapshotDescriptor metadata(const core::WindowKey& key,const core::StateCut& cut,
                                              const std::vector<int32_t>& tokens) {
        core::SnapshotDescriptor value;
        value.key=key;value.cut=cut;value.accepted_tokens=tokens;
        value.pending_tokens.assign(tokens.begin()+cut.forwarded,tokens.end());
        value.steering_enabled=false;value.speculation_disabled=true;
        return value;
    }
    template<class Context> static bool validate(const Context&,std::string&);
    template<class Context> static bool validate_serving_qualification_execution(const Context&,std::string&);
    template<class Context> static Step reset(Context&,SuffixRawTransaction&,uint64_t&,uint64_t output_watermark=0);
    template<class Context,class SerialUnit>
    static Step forward(Context&,SerialUnit&,SuffixRawTransaction&,uint64_t&,int32_t,uint64_t,
                        uint64_t,core::StateDigest,uint64_t,std::span<float>);
    template<class Context>
    static Step replay_prefill_piece(Context&,SuffixRawTransaction&,uint64_t&,const PrefixReplayEvent&,
        const PrefixReplayCallbacks&,std::span<const int32_t>,uint64_t,core::StateDigest,PrefixReplayAbort,bool&);

    template<class Context,class SerialUnit>
    static Step replay(Context& c,SerialUnit& serial,SuffixRawTransaction& tx,uint64_t& next_cut,
                       const std::vector<int32_t>& tokens,uint64_t count,uint64_t accepted,
                       std::span<float> logits={}) {
        const auto history=core::window_history_digest(tokens);
        Step step;
        for(uint64_t position=0;position<count;++position)
            step=forward(c,serial,tx,next_cut,tokens[size_t(position)],position,accepted,history,
                         position+1,position+1==count?logits:std::span<float>{});
        return step;
    }
    template<class Context,class SerialUnit>
    static bool compare_clean_recomputation(Context& c,SerialUnit& serial,SuffixRawTransaction& tx,
        uint64_t& next_cut,const core::BundleHandle& repaired,Buffers& buffers,bool catchup_only=false) {
        if(!repaired||repaired.get()->cut!=tx.cut_||
           repaired.get()->cut.exact_frontier!=repaired.get()->cut.forwarded||
           repaired.get()->pending_tokens.size()!=1||!repaired.get()->approximation.empty())
            failstop("clean oracle requires current complete exact state and one pending token");
        const auto& original=*repaired.get();
        const auto accepted=original.cut.accepted,forwarded=original.cut.forwarded;
        if(!forwarded||accepted!=forwarded+1||accepted>uint64_t(c.options.max_context)||
           tx.cut_.id==UINT64_MAX)failstop("clean oracle cut bounds");
        // All history comes from the actual retained publication. No repaired
        // tensor or prefix seeds this reset/replay; output receipts are inherited
        // metadata only, and no recomputed pick is emitted to a user.
        buffers.history=original.accepted_tokens;
        next_cut=std::max(next_cut,tx.cut_.id+1);
        reset(c,tx,next_cut,original.cut.output_watermark);
        const auto clean_step=replay(c,serial,tx,next_cut,buffers.history,forwarded,accepted);
        auto clean_metadata=metadata(original.key,clean_step.cut,buffers.history);
        core::BundleHandle clean;
        if(!raw_call([&]{return tx.capture(clean_metadata,clean);}))failstop("clean recomputation capture refused");
        auto compare=[&](const char* phase,const core::BundleHandle& left,const core::BundleHandle& right) {
            core::WindowStateLayout layout;
            auto valid=core::native_window_layout(c.geometry,tx.initial_config_,original.key,
                left.get()->cut,uint64_t(c.options.max_context),c.options.suffix_experiment_host_bytes,layout,false);
            if(!valid)failstop("clean oracle actual native layout refused");
            clean_recompute_qualification::Report report;
            const bool equal=clean_recompute_qualification::compare(left,right,layout,report);
            std::printf("suffix_repair_clean phase=%s completed=%u equal=%u tensors=%llu bytes=%llu mismatches=%llu oracle=reset_replay_identical_accepted_history\n",
                phase,unsigned(report.completed),unsigned(equal),(unsigned long long)report.tensors_compared,
                (unsigned long long)report.bytes_compared,(unsigned long long)report.mismatches);
            for(size_t i=0;i<std::min<uint64_t>(report.mismatches,report.first.size());++i){const auto& m=report.first[i];
                std::printf("suffix_repair_clean_difference phase=%s reason=%u stage=%u layer=%lld kind=%u offset=%llu left=%llu right=%llu\n",
                    phase,unsigned(m.reason),m.stage,(long long)m.layer,unsigned(m.kind),
                    (unsigned long long)m.offset,(unsigned long long)m.left,(unsigned long long)m.right);
            }
            return equal;
        };
        const bool state_equal=compare(catchup_only?"catchup":"endpoint",repaired,clean);
        if(catchup_only) {
            // This pending token was sampled by the approximate foreground
            // BEFORE full catchup. Preserve the accepted transcript: its fresh
            // exact greedy prediction is not an equality requirement.
            std::printf("suffix_repair_clean_catchup forwarded=%llu accepted=%llu output_watermark=%llu pending_preserved=%u emitted=0\n",
                (unsigned long long)forwarded,(unsigned long long)accepted,
                (unsigned long long)original.cut.output_watermark,
                unsigned(clean.get()->pending_tokens==original.pending_tokens));
            return state_equal;
        }
        // Pending-token identity is checked against a fresh greedy prediction,
        // not merely copied into a descriptor that would validate itself.
        const bool pending_equal=clean_step.next==original.pending_tokens.front();
        std::printf("suffix_repair_clean_pending equal=%u recorded=%d recomputed=%d\n",
            unsigned(pending_equal),original.pending_tokens.front(),clean_step.next);
        if(!state_equal||!pending_equal)return false;
        clean={}; // its bytes were compared; release before continuation captures
        const int32_t pending=original.pending_tokens.front();
        const auto clean_next=forward(c,serial,tx,next_cut,pending,forwarded,accepted,
            original.cut.history,accepted,buffers.exact_logits);
        core::BundleHandle clean_continuation;
        auto next_metadata=metadata(original.key,clean_next.cut,buffers.history);
        if(!raw_call([&]{return tx.capture(next_metadata,clean_continuation);}))failstop("clean continuation capture refused");
        if(!raw_call([&]{return tx.restore(repaired,original.key);}))failstop("repaired endpoint restore for continuation refused");
        const auto repaired_next=forward(c,serial,tx,next_cut,pending,forwarded,accepted,
            original.cut.history,accepted,buffers.approximate_logits);
        core::BundleHandle repaired_continuation;
        next_metadata=metadata(original.key,repaired_next.cut,buffers.history);
        if(!raw_call([&]{return tx.capture(next_metadata,repaired_continuation);}))failstop("repaired continuation capture refused");
        SuffixLogitComparison logits;
        if(!compare_suffix_logits(buffers.exact_logits,buffers.approximate_logits,logits))failstop("clean continuation logits invalid");
        const bool continuation_equal=compare("continuation",repaired_continuation,clean_continuation);
        const bool picks_equal=clean_next.next==repaired_next.next&&
            logits.exact_argmax==clean_next.next&&logits.approximate_argmax==repaired_next.next;
        const bool logits_equal=logits.max_absolute_error==0;
        std::printf("suffix_repair_clean_continuation picks_equal=%u logits_equal=%u exact_argmax=%d repaired_argmax=%d rms=%.12g max_abs=%.12g kl=%.12g emitted=0\n",
            unsigned(picks_equal),unsigned(logits_equal),logits.exact_argmax,logits.approximate_argmax,
            logits.root_mean_square_error,logits.max_absolute_error,logits.kl_exact_to_approximate);
        return continuation_equal&&picks_equal&&logits_equal;
    }
    template<class Context,class SerialUnit>
    static void generate_tail(Context& c,SerialUnit& serial,SuffixRawTransaction& tx,uint64_t& next_cut,
                              Step step,const std::vector<int32_t>& prompt,uint64_t exact_frontier,
                              bool approximate,Buffers& buffers,std::vector<int32_t>& output) {
        buffers.history.assign(prompt.begin(),prompt.end());
        for(int64_t i=0;i<c.options.max_new;++i) {
            if(step.next<0 || int64_t(step.next)>=c.geometry.n_vocab)failstop("invalid sampled token");
            output.push_back(step.next);
            if(i+1==c.options.max_new)break;
            buffers.history.push_back(step.next);
            const auto digest=core::window_history_digest(buffers.history);
            const uint64_t count=buffers.history.size();
            step=forward(c,serial,tx,next_cut,step.next,count-1,count,digest,
                         approximate?exact_frontier:count,{});
        }
    }
    static bool cycle_observation_bytes(uint64_t& bytes){
        bytes=65536; // bounded report/string/control scratch, logical ABI profile
        return add_product(bytes,2,sizeof(resident_repair_detail::ResidentRepairController::CycleOwners))&&
            add_product(bytes,64,12*sizeof(uint64_t)); // bounded overlapping baseline/after snapshot vectors
    }
    static void cycle_resources(unsigned cycle,const char* phase,const core::WindowBudgetSnapshot& snapshot){
        const auto& r=snapshot.reserved;
        std::printf("suffix_repair_cycle_resources cycle=%u phase=%s host=%llu pinned=%llu disk=%llu io=%llu shadows=%llu devices=",
            cycle,phase,(unsigned long long)r.host_bytes,(unsigned long long)r.pinned_bytes,
            (unsigned long long)r.disk_bytes,(unsigned long long)r.io_slots,(unsigned long long)r.shadows);
        for(size_t i=0;i<r.device_bytes.size();++i)std::printf("%s%llu",i?",":"",(unsigned long long)r.device_bytes[i]);
        std::printf(" overcommitted=%u\n",unsigned(snapshot.overcommitted));
    }
    static void cycle_diagnostic_host(SuffixRawTransaction& tx,unsigned cycle,const char* phase)noexcept {
        SuffixRawTransaction::HostUsage usage;
        if(tx.observe_host_usage(usage))
            std::fprintf(stderr,"resident cycle H cycle=%u phase=%s observed=1 limit=%llu live=%llu claims=%u\n",
                cycle,phase,(unsigned long long)usage.limit,(unsigned long long)usage.live,usage.claims);
        else std::fprintf(stderr,"resident cycle H cycle=%u phase=%s observed=0\n",cycle,phase);
        std::fflush(stderr);
    }
    template<class Context,class SerialUnit>
    static int run_repair_cycles(Context& c,SerialUnit& serial,SuffixRawTransaction& tx,
        const std::shared_ptr<SuffixRawTransaction>& tx_owner,uint64_t& next_cut,Buffers& buffers,
        const resident_repair_detail::ResidentConstructionInputs& construction,
        const core::SnapshotDescriptor& initial,
        const std::shared_ptr<const core::NativeRepairSuffixData>& roots,const core::BundleHandle& seed,uint64_t boundary){
        using Controller=resident_repair_detail::ResidentRepairController;
        std::string error;
        const auto& parent=tx.allocation_.reservation();
        if(boundary!=32||initial.cut.accepted!=129||initial.cut.forwarded!=128||
           !parent||!parent->canonical_owner(parent)||parent->resources().device_bytes.empty()||
           parent->resources().device_bytes.size()>64){
            std::fputs("cycle profile requires actual129-token/B32 bounded native parent\n",stderr);return 2;
        }
        // Fixed observer/snapshot metadata was explicitly added to the SAME
        // driver buffers admission before its first allocation. Reuse that
        // genuine retained credit: a separate ticket would waste a fixed H
        // claim slot needed by second-attachment donor/prefix/splice/restore.
        if(!buffers.credit){std::fputs("cycle observation admission absent\n",stderr);return 2;}
        core::BundleHandle start_image;
        if(!raw_call([&]{return tx.capture(initial,start_image);}))return 2;
        const auto baseline=tx.budget_.snapshot();SuffixRawTransaction::HostUsage baseline_h;
        if(baseline.capacity.device_bytes.empty()||baseline.capacity.device_bytes.size()>64||baseline.overcommitted||
           !tx.observe_host_usage(baseline_h)||core::native_session_quarantine_count()||
           core::window_buffer_quarantine_count()||core::model_stage_quarantine_count()||core::model_gang_quarantine_count()){
            std::fputs("cycle actual warmed baseline unavailable\n",stderr);return 2;
        }
        cycle_resources(0,"baseline",baseline);
        std::printf("suffix_repair_cycle_baseline cycles=2 scenario=normal h_live=%llu h_claims=%u retained=foreground_model_live_raw_input_buffers_start_seed_roots_observation\n",
            (unsigned long long)baseline_h.live,baseline_h.claims);
        for(unsigned cycle=1;cycle<=2;++cycle){
            // Historical private source restore is not a new external ledger
            // publication. NEW zero/forward cuts use the retained mint floor.
            if(!raw_call([&]{return tx.restore(start_image,initial.key);}))return 2;
            const uint64_t mint_begin=next_cut;
            std::printf("suffix_repair_cycle_begin cycle=%u scenario=normal mint_floor=%llu\n",cycle,(unsigned long long)mint_begin);
            std::shared_ptr<Controller> repair;
            if(!Controller::create(c.owner,tx_owner,construction,initial,roots,seed,repair,error,next_cut)){
                std::fprintf(stderr,"cycle construction refused: %s\n",error.c_str());return 2;
            }
            typename Controller::CycleOwners owners;
            if(!repair->observe_cycle_owners(repair,owners,error)){
                if(!repair->retire(error))failstop("unobserved cycle private retirement failed");
                repair.reset();owners.reset();return 2;
            }
            cycle_diagnostic_host(tx,cycle,"before_execute");
            const bool completed=repair->execute_two_attachments(boundary+4,boundary+8,error,Controller::Scenario::normal);
            // Retirement/drain methods may rewrite error. Preserve the ORIGINAL
            // execute refusal now using already admitted fixed report scratch.
            std::array<char,4096> execute_error{};
            const size_t execute_error_bytes=std::min(error.size(),execute_error.size()-1);
            if(!completed){
                std::copy_n(error.data(),execute_error_bytes,execute_error.data());
                std::fprintf(stderr,"resident cycle execute refused cycle=%u detail=%s\n",cycle,execute_error.data());
                cycle_diagnostic_host(tx,cycle,"after_execute_return");
                cycle_resources(cycle,"execute_refusal",tx.budget_.snapshot());
                std::fflush(stdout);std::fflush(stderr);
            }
            repair->report(stdout);
            core::BundleHandle repaired_final,repaired_catchup;
            if(completed){
                if(!repair->receipt_trace_valid(error))failstop("cycle actual nine-receipt trace differs");
                if(!raw_call([&]{return tx.capture(repair->published_->metadata,repaired_final);}))
                    failstop("cycle final capture refused");
                repaired_catchup=repair->catchup_snapshot_;
                if(!repaired_catchup)failstop("cycle installed catchup capture missing");
            }
            next_cut=std::max(next_cut,repair->next_cut_);
            if(!repair->retire(error))failstop("cycle actual retirement refused");
            const bool retired_checked=repair->observe_retired_cycle(owners,error);
            if(!retired_checked)failstop("cycle actual retirement observations refused");
            owners.release_inspection(); // no retained binding before owner-expiry proof
            repair.reset(); // private Cuts, roots, splice images and RAW die here
            const bool private_expired=owners.strong_owners_expired();
            owners.drop_owner_controlblocks(); // releases weak-retained setup/gate credits
            const bool charges_expired=owners.charge_owners_expired();
            owners.reset(); // ALL weak observers gone BEFORE final ledger/H read
            if(!private_expired||!charges_expired){
                std::printf("suffix_repair_cycle_retirement cycle=%u checked=1 private_expired=%u charges_expired=%u weak_reset=1 pass=0\n",
                    cycle,unsigned(private_expired),unsigned(charges_expired));return 2;
            }
            if(!completed){
                std::fprintf(stderr,"resident cycle stopped after checked retirement cycle=%u original_execute_detail=%s\n",cycle,execute_error.data());
                cycle_diagnostic_host(tx,cycle,"after_refusal_retirement");return 2;
            }
            if(!raw_call([&]{return tx.restore(repaired_catchup,repaired_catchup.get()->key);}))
                failstop("cycle catchup oracle restore refused");
            if(!compare_clean_recomputation(c,serial,tx,next_cut,repaired_catchup,buffers,true))return 2;
            repaired_catchup={};
            if(!raw_call([&]{return tx.restore(repaired_final,repaired_final.get()->key);}))
                failstop("cycle final oracle restore refused");
            if(!compare_clean_recomputation(c,serial,tx,next_cut,repaired_final,buffers))return 2;
            repaired_final={}; // final clean/continuation temporaries already destroyed
            const auto after=tx.budget_.snapshot();SuffixRawTransaction::HostUsage after_h;
            if(!tx.observe_host_usage(after_h)){
                std::fputs("cycle post-retirement H observation unavailable\n",stderr);return 2;
            }
            const bool ledger_equal=after.capacity==baseline.capacity&&after.baseline==baseline.baseline&&
                after.reserved==baseline.reserved&&!after.overcommitted;
            const bool h_equal=after_h.limit==baseline_h.limit&&after_h.live==baseline_h.live&&after_h.claims==baseline_h.claims;
            const bool foreground_ok=c.owner->sessions.healthy()&&c.owner->model->healthy()&&
                c.owner->background_retirement.load(std::memory_order_acquire)==0;
            const bool quarantine_zero=!core::native_session_quarantine_count()&&!core::window_buffer_quarantine_count()&&
                !core::model_stage_quarantine_count()&&!core::model_gang_quarantine_count();
            cycle_resources(cycle,"after",after);
            const bool pass=ledger_equal&&h_equal&&foreground_ok&&quarantine_zero&&next_cut>mint_begin;
            std::printf("suffix_repair_cycle_retirement cycle=%u checked=1 private_expired=%u charges_expired=%u weak_reset=1 registry_absent=%u service_idle=%u ledger_equal=%u h_equal=%u h_live=%llu h_claims=%u quarantine_zero=%u foreground_healthy=%u mint_next=%llu pass=%u\n",
                cycle,unsigned(private_expired),unsigned(charges_expired),unsigned(retired_checked),unsigned(retired_checked),unsigned(ledger_equal),unsigned(h_equal),
                (unsigned long long)after_h.live,after_h.claims,unsigned(quarantine_zero),unsigned(foreground_ok),
                (unsigned long long)next_cut,unsigned(pass));
            if(!pass)return 2;
            std::printf("suffix_repair_cycle_end cycle=%u pass=1\n",cycle);
        }
        std::puts("suffix_repair_cycles_result cycles=2 scenario=normal pass=1 scope=dedicated_serial_private_retirement_not_physical_rss");
        return 0;
    }
public:
#include "serving_suffix_qualification.inc"
    template<class Context,class SerialUnit>
    static int run(Context& c,SerialUnit& serial) {
        std::string error;
        if(!validate(c,error)) {std::fprintf(stderr,"suffix runtime refused: %s\n",error.c_str());return 2;}
        auto key=identity(c);
        std::vector<core::Verifier*> verifiers{&c.primary_verifier};
        for(const auto& stage:c.secondary_stages)verifiers.push_back(&stage->ver);
        // Fixed <=8-stage bootstrap above is bounded setup metadata. All input,
        // image, control, diagnostic and output payloads are admitted below.
        // The resident controller retains this genuine transaction owner. A
        // no-op alias to a stack transaction would not preserve its host cap
        // and completion state across retained repair receipts.
        auto bootstrap=std::unique_ptr<SuffixRawTransaction>(new SuffixRawTransaction(
            c.geometry,key,c.owner->sessions,c.owner,std::move(verifiers),c.budget,
            uint64_t(c.options.max_context),c.options.suffix_experiment_host_bytes,
            c.options.suffix_experiment_host_bytes,false));
        auto& tx=*bootstrap;
        auto initialized=tx.initialize();checked(initialized);
        if(!initialized){std::fprintf(stderr,"suffix setup refused: %s\n",initialized.detail.c_str());return 2;}
        // Keep admitted metadata charged through both object deletion and the
        // last weak control-block deallocation, not merely member destruction.
        std::shared_ptr<const void> tx_credit=tx.setup_credit_;
        auto tx_owner=std::shared_ptr<SuffixRawTransaction>(bootstrap.release(),
            TransactionDelete{tx_credit},TransactionAllocator<SuffixRawTransaction>{tx_credit});
        AdmittedInput admitted;
        auto read=read_input(tx,c,admitted);checked(read);
        if(!read){std::fprintf(stderr,"suffix input refused: %s\n",read.detail.c_str());return 2;}
        const auto& input=admitted.input;
        Buffers buffers;
        uint64_t control=1ULL<<20;
        if(!add_product(control,uint64_t(c.options.max_context),128) ||
           !add_product(control,uint64_t(c.geometry.n_vocab),8) ||
           !add_product(control,uint64_t(c.options.max_new),16))return 2;
        if(c.options.suffix_repair_cycles==2){
            uint64_t cycle_metadata=0;
            if(!cycle_observation_bytes(cycle_metadata)||!add_product(control,1,cycle_metadata))return 2;
        }
        auto room=tx.admit_controller_host(control,buffers.credit);checked(room);
        if(!room){std::fprintf(stderr,"suffix diagnostics admission refused\n");return 2;}
        buffers.exact_logits.resize(size_t(c.geometry.n_vocab));
        buffers.approximate_logits.resize(size_t(c.geometry.n_vocab));
        buffers.history.reserve(size_t(c.options.max_context));
        buffers.exact_output.reserve(size_t(c.options.max_new));
        buffers.approximate_output.reserve(size_t(c.options.max_new));
        uint64_t next_cut=1;
        const uint64_t n=input.edited_tokens.size(),f=n-1,b=input.boundary;
        if(c.options.suffix_repair_experiment&&
           (b%4||f%4||b>=f||f-b<12||c.options.max_new<9)) {
            std::fputs("resident repair requires aligned prefix/suffix and room for two repairs plus nine foreground tokens\n",stderr);
            return 2;
        }
        // Warm the exact policy/graphs before its measured run. This is bounded
        // by the same prompt/output limits and is reported, never hidden in a
        // purported cold-start latency comparison. Reset clears session state;
        // model/expert caches remain warm and are not a cold-IO measurement.
        reset(c,tx,next_cut);
        auto warm=replay(c,serial,tx,next_cut,input.edited_tokens,n,n);
        generate_tail(c,serial,tx,next_cut,warm,input.edited_tokens,n,false,buffers,buffers.exact_output);
        buffers.exact_output.clear();
        // Exact reference is available even when the import shape is refused.
        reset(c,tx,next_cut);
        auto start=Clock::now();
        auto exact=replay(c,serial,tx,next_cut,input.edited_tokens,n,n,buffers.exact_logits);
        const double exact_prompt_ms=elapsed(start);
        generate_tail(c,serial,tx,next_cut,exact,input.edited_tokens,n,false,buffers,buffers.exact_output);
        const double exact_total_ms=elapsed(start);
        const auto shape=suffix_experiment_fallback_reason(input);
        auto report_fallback=[&](const char* reason) {
            std::printf("suffix_experiment mode=exact_fallback reason=%s reused=0 fresh=%llu identity=process_local output_policy=fixed_length_greedy\n",
                        reason,static_cast<unsigned long long>(n));
            std::printf("exact_output");for(auto token:buffers.exact_output)std::printf(" %d",token);std::printf("\n");
            return 0;
        };
        if(!shape.empty())return report_fallback(shape.data());
        core::BundleHandle donor,prefix;
        reset(c,tx,next_cut);
        auto old=replay(c,serial,tx,next_cut,input.old_tokens,f,n);
        auto old_metadata=metadata(key,old.cut,input.old_tokens);
        auto captured=raw_call([&]{return tx.capture(old_metadata,donor);});
        if(!captured)return report_fallback("donor_capture_refused");
        auto target_key=key;
        ++target_key.revision;++target_key.append_epoch;++target_key.selection_epoch;
        reset(c,tx,next_cut);
        std::vector<int32_t> prefix_tokens(input.edited_tokens.begin(),input.edited_tokens.begin()+b);
        start=Clock::now();
        auto fresh=replay(c,serial,tx,next_cut,prefix_tokens,b,b);
        auto prefix_metadata=metadata(target_key,fresh.cut,prefix_tokens);
        captured=raw_call([&]{return tx.capture(prefix_metadata,prefix);});
        if(!captured)return report_fallback("prefix_capture_refused");
        const uint64_t now=uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        if(!next_cut || next_cut==UINT64_MAX || !now)return report_fallback("cut_or_age_exhausted");
        core::NativeSuffixTarget target{target_key,input.edited_tokens,next_cut++,now};
        core::NativeSuffixLimits limits{uint64_t(c.options.max_context),c.options.suffix_experiment_host_bytes,control};
        uint64_t splice_bytes=0;
        auto quote=core::quote_int8_native_suffix(donor,key,prefix,target_key,c.geometry,tx.initial_config_,target,limits,splice_bytes);
        if(!quote) {
            std::fprintf(stderr,"suffix quote status=%u reason=%u\n",unsigned(quote.status),unsigned(quote.reason));
            return report_fallback("native_suffix_quote_refused");
        }
        std::shared_ptr<const void> splice_credit;
        core::PreparedNativeSuffix splice; // dies BEFORE aggregate credit
        auto slot=tx.claim_external_splice(splice_bytes,splice_credit);checked(slot);
        if(!slot)return report_fallback("aggregate_splice_budget_refused");
        auto prepared=core::prepare_int8_native_suffix(donor,key,prefix,target_key,c.geometry,tx.initial_config_,target,limits,c.budget,splice);
        if(!prepared) {
            std::fprintf(stderr,"suffix prepare status=%u reason=%u\n",unsigned(prepared.status),unsigned(prepared.reason));
            return report_fallback("native_suffix_prepare_refused");
        }
        auto restored=raw_call([&]{return tx.restore(splice.bundle(),target_key);});
        if(!restored)return report_fallback("restore_preflight_refused");
        if(c.options.suffix_repair_experiment) {
            using namespace resident_repair_detail;
            // This explicitly bounded first runtime profile is experimental.
            // The envelope reserves logical capacity; sampled GPU/host limits
            // must separately qualify CUDA's hidden runtime/graph overhead.
            ResidentConstructionInputs construction;
            construction.max_cells=c.primary_session.qsa_states[c.primary_session.qsa_primary()].max_cells;
            construction.platform_and_control.device_bytes.resize(
                c.owner->sessions.reservation()->resources().device_bytes.size());
            for(const auto& stage:c.owner->sessions.stages()) {
                construction.platform_and_control.device_bytes[size_t(stage.device)]=512ULL<<20;
                if(stage.session->layer_lo<=1&&1<stage.session->layer_hi)
                    construction.native_ple_key=stage.session->ple.w.key_native_data!=nullptr;
            }
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION)
            core::resident_fault_receipt::File fault_receipt;
            if(c.options.suffix_repair_fault_second_arena||c.options.suffix_repair_fault_second_copy||c.options.suffix_repair_fault_deferred_trap) {
                if((unsigned(c.options.suffix_repair_fault_second_arena)+unsigned(c.options.suffix_repair_fault_second_copy)+unsigned(c.options.suffix_repair_fault_deferred_trap)!=1)||
                   c.options.suffix_repair_cancel||c.options.suffix_repair_prepared_cancel||c.options.suffix_repair_cycles!=1||c.owner->sessions.stages().size()!=2||
                   c.owner->sessions.stages()[0].device==c.owner->sessions.stages()[1].device||
                   !fault_receipt.open_new(c.options.suffix_repair_fault_receipt.c_str()))
                    failstop("explicit resident fault profile/receipt refused");
                if(c.options.suffix_repair_fault_second_arena) {
                    construction.fault_second_arena=true;construction.fault_receipt_fd=fault_receipt.get();
                }
            }
#endif
            construction.platform_and_control.host_bytes=128ULL<<20;
            construction.buffer_metadata_host_ceiling=4096;
            auto current=ResidentRepairController::metadata_only(*splice.bundle().get());
            std::shared_ptr<const core::NativeRepairSuffixData> roots;
            if(!ResidentRepairController::make_initial_roots(c.owner,tx,donor,current,roots,error)) {
                std::fprintf(stderr,"resident ancestry refused: %s\n",error.c_str());return 2;
            }
            // The selected restore is already drained, and current/root data
            // owns its metadata. These old payloads are no longer consumers;
            // free their charges before constructing the second resident gang.
            donor={};splice={};splice_credit.reset();
            if(c.options.suffix_repair_cycles==2)
                return run_repair_cycles(c,serial,tx,tx_owner,next_cut,buffers,construction,current,roots,prefix,b);
            std::shared_ptr<ResidentRepairController> repair;
            if(!ResidentRepairController::create(c.owner,tx_owner,construction,current,roots,prefix,repair,error)) {
                std::fprintf(stderr,"resident construction refused: %s\n",error.c_str());return 2;
            }
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION)
            if(c.options.suffix_repair_fault_deferred_trap) {
                repair->fault_deferred_trap_=true;repair->fault_receipt_fd_=fault_receipt.get();
            }
            if(c.options.suffix_repair_fault_second_copy) {
                repair->fault_second_copy_=true;repair->fault_receipt_fd_=fault_receipt.get();
            }
#endif
            // Copies and restores are complete. Release old experiment images
            // before the controller captures its CURRENT donor/checkpoints.
            roots.reset();prefix={};
            const auto scenario=c.options.suffix_repair_prepared_cancel?
                ResidentRepairController::Scenario::cancel_after_second_prepare:
                c.options.suffix_repair_cancel?
                ResidentRepairController::Scenario::cancel_before_second_attachment:
                ResidentRepairController::Scenario::normal;
            const bool completed=repair->execute_two_attachments(b+4,b+8,error,scenario);
            const bool cancelled=repair->cancelled();
            repair->report(stdout);
            if(!completed)std::fprintf(stderr,"resident repair refused: %s\n",error.c_str());
            core::BundleHandle repaired_final;
            if(completed&&!cancelled&&!raw_call([&]{return tx.capture(repair->published_->metadata,repaired_final);}))
                failstop("repaired final exact capture refused");
            core::BundleHandle repaired_catchup;
            if(completed&&!cancelled) {
                if(!repair->receipt_trace_valid(error))failstop("resident accepted/output receipt trace differs");
                repaired_catchup=repair->catchup_snapshot_;
                if(!repaired_catchup)failstop("resident pre-forward catchup capture missing");
            }
            // Preserve the actual controller's mint high-water, including its
            // private cuts, before destroying it. Never derive it from the
            // earlier saved catchup descriptor.
            next_cut=std::max(next_cut,repair->next_cut_);
            if(!repair->retire(error))failstop("resident private retirement refused");
            repair.reset(); // no private gang participates in the clean oracle
            if(cancelled)return completed?0:2; // distinct cancellation receipt, never exactness success
            if(!completed)return 2;
            // Both captures own real global + shared-H credits. The catchup
            // reset starts from native zero state, never a repaired prefix.
            if(!raw_call([&]{return tx.restore(repaired_catchup,repaired_catchup.get()->key);}))
                failstop("pre-forward catchup selection for clean oracle refused");
            if(!compare_clean_recomputation(c,serial,tx,next_cut,repaired_catchup,buffers,true))return 2;
            repaired_catchup={}; // release before the endpoint/continuation images
            if(!raw_call([&]{return tx.restore(repaired_final,repaired_final.get()->key);}))
                failstop("final repaired selection for clean oracle refused");
            return compare_clean_recomputation(c,serial,tx,next_cut,repaired_final,buffers)?0:2;
        }
        auto approximate=forward(c,serial,tx,next_cut,input.edited_tokens[size_t(f)],f,n,
            core::window_history_digest(input.edited_tokens),b,buffers.approximate_logits);
        const double reuse_prompt_ms=elapsed(start);
        generate_tail(c,serial,tx,next_cut,approximate,input.edited_tokens,b,true,buffers,buffers.approximate_output);
        const double reuse_total_ms=elapsed(start);
        SuffixLogitComparison comparison;
        if(!compare_suffix_logits(buffers.exact_logits,buffers.approximate_logits,comparison))return 2;
        std::printf("suffix_experiment mode=approximate identity=process_local exported=0 boundary=%llu reused=%llu fresh=%llu pending_forwarded=1 output_policy=fixed_length_greedy\n",
            static_cast<unsigned long long>(b),static_cast<unsigned long long>(f-b),static_cast<unsigned long long>(b+1));
        std::printf("suffix_timing exact_prompt_ms=%.6f reuse_prompt_ms=%.6f exact_total_ms=%.6f reuse_total_ms=%.6f donor_setup_excluded=1 exact_warmup_runs=1 order=exact_then_reuse\n",
            exact_prompt_ms,reuse_prompt_ms,exact_total_ms,reuse_total_ms);
        std::printf("suffix_logits kl_exact_to_approx=%.12g rms=%.12g max_abs=%.12g exact_argmax=%d approximate_argmax=%d\n",
            comparison.kl_exact_to_approximate,comparison.root_mean_square_error,comparison.max_absolute_error,
            comparison.exact_argmax,comparison.approximate_argmax);
        // Descendant outputs retain this unresolved ancestry and age. No
        // exact-cache certificate is produced by freshly forwarding a token.
        const auto& lineage=splice.bundle().get()->approximation.front();
        std::printf("suffix_lineage first_unresolved_unix_ms=%llu ancestor=",
            static_cast<unsigned long long>(lineage.first_unresolved_unix_ms));
        for(uint8_t byte:lineage.ancestor)std::printf("%02x",unsigned(byte));
        std::printf(" descendants=approximate exact_frontier=%llu\n",static_cast<unsigned long long>(b));
        std::printf("exact_output");for(auto token:buffers.exact_output)std::printf(" %d",token);std::printf("\n");
        std::printf("approximate_output");for(auto token:buffers.approximate_output)std::printf(" %d",token);std::printf("\n");
        return 0;
    }
};
} // namespace strata::program
