#pragma once
#include "strata/core/expert_source.hpp"
#include "strata/core/window_selection.hpp"
namespace strata::core {
// Implemented by the actual selected runtime, never an alias wrapper. The
// implementation checks its real dispatch/source/pool ownership graph.
class ExpertHostRuntimeBinding {
public:
    virtual ~ExpertHostRuntimeBinding()=default;
    virtual bool owns_expert_dispatch(const ExpertDispatch*) const noexcept=0;
};
struct ExpertHostExecutionLimits {
    uint64_t layers=0,rows=0,batches=0,rows_per_layer=0,key_bytes=0,stages=0;
};
struct ExpertHostExecutionContext {
    std::shared_ptr<NativeRuntimeAttachment> runtime;
    std::shared_ptr<const ExpertSourceCapture> source;
    std::shared_ptr<const ExpertLayoutCapture> layout;
    std::shared_ptr<const VerifiedDriverBindings> inputs;
    const ExpertDispatch* dispatch=nullptr;
    // Caller-supplied diagnostic labels, not authenticated policy revisions.
    uint64_t incarnation=0,policy_revision=0;
    int64_t layer_begin=0,layer_end=0,experts=0;
};
enum class ExpertHostLayerStatus : uint8_t { active, returned, aborted };
struct ExpertHostLayerRecord {
    uint64_t ticket=0,tokens=0,width=0,row_begin=0,row_count=0,batch_begin=0,batch_count=0;
    int64_t layer=0;bool multi=false;
    ExpertHostLayerStatus status=ExpertHostLayerStatus::active;
};
struct ExpertHostBatchRecord {
    uint64_t ticket=0,ordinal=0,jobs=0,row_begin=0,row_count=0;
    ExpertHostPoolEntry entry=ExpertHostPoolEntry::single;
    bool completed=false;
};
struct ExpertHostExecutionView {
    // Borrowed diagnostics only. No GPU, inner-kernel, weighted reduction or
    // complete-work coverage. Addresses in rows must never be dereferenced.
    const WindowSelectionToken* selection=nullptr;
    uint64_t operation=0,incarnation=0,policy_revision=0;
    const ExpertHostLayerRecord* layers=nullptr;size_t layer_count=0;
    const ExpertHostCpuRow* rows=nullptr;size_t row_count=0;
    const ExpertHostBatchRecord* batches=nullptr;size_t batch_count=0;
    bool failed=false,host_closed=false;
};
bool expert_host_execution_host_bound(const ExpertHostExecutionLimits&,uint64_t&,std::string&);
class ExpertHostExecutionCapture final:public ExpertHostExecutionObserver {
public:
    static bool create(const ExpertHostExecutionLimits&,std::shared_ptr<const ReservationLease>,
        const SelectedWindowWork&,const ExpertHostExecutionContext&,
        std::unique_ptr<ExpertHostExecutionCapture>&,std::string&);
    ~ExpertHostExecutionCapture();
    ExpertHostExecutionCapture(const ExpertHostExecutionCapture&)=delete;
    ExpertHostExecutionCapture& operator=(const ExpertHostExecutionCapture&)=delete;
    ExpertHostExecutionCapture(ExpertHostExecutionCapture&&)=delete;
    ExpertHostExecutionCapture& operator=(ExpertHostExecutionCapture&&)=delete;
    bool begin_layer(const void*,const void*,const void*,int64_t,uint64_t,uint64_t,bool,uint64_t&) noexcept override;
    // Prepaid constant-time row/job validation; submission scans each batch once.
    bool cpu_row(uint64_t,const ExpertHostCpuRow&) noexcept override;
    bool cpu_batch_submitted(uint64_t,uint64_t,uint64_t,ExpertHostPoolEntry) noexcept override;
    void cpu_batch_completed(uint64_t,uint64_t) noexcept override;
    void end_host_dispatch(uint64_t) noexcept override;
    void abort_layer(uint64_t) noexcept override;
    // Trusted host closure ONLY, never device completion. Caller must remove raw
    // observer pointers and join producers BEFORE destruction/this call. Keeping
    // a backing arena alive cannot prevent calls into a destroyed observer.
    void close_after_producers_joined() noexcept;
    // Caller serializes with callbacks. Failed/aborted diagnostics stay visible.
    ExpertHostExecutionView view() const noexcept;
private:
    ExpertHostExecutionCapture()=default;
    struct Storage;
    struct Entry;
    std::unique_ptr<Storage> storage_;
};
}
