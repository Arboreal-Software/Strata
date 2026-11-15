#pragma once
#include "strata/core/layer.hpp"
#include "strata/core/vmm.hpp"
#include <memory>
namespace strata::core {
class PreparedVmmAllocationBatch;
// Initial mapping owner. Growth/donation/shrink and execution frontier checks
// remain separate required integration; never use legacy raw-handle functions.
// Serialized runtime-owner API. Callers close all host producers/graphs before
// retirement; its device drain alone does not stop future host submissions.
// QsaState borrows this owner; the NativeSessionAllocation retains it until
// state/graph borrowers retire. Direct callers must enforce the same ordering.
class OwnedQsaElasticPool final:public QsaElasticView {
public:
    static bool host_bound(const QsaElasticPlan&,uint64_t& owner_bytes,uint64_t& range_bytes) noexcept;
    // Host only. batch owns its genuine physical and batch-metadata admission;
    // range/owner leases are separate. All metadata and output slots are created
    // before any GPU work. Caller admits the complete gang before initialize.
    static bool prepare(const ModelGeometry&,const QsaElasticPlan&,
        std::shared_ptr<const ReservationLease> owner_metadata,
        std::shared_ptr<const ReservationLease> range_metadata,
        std::unique_ptr<PreparedVmmAllocationBatch>& batch,
        std::shared_ptr<OwnedQsaElasticPool>& out,std::string&);
    ~OwnedQsaElasticPool();
    OwnedQsaElasticPool(const OwnedQsaElasticPool&)=delete;
    OwnedQsaElasticPool& operator=(const OwnedQsaElasticPool&)=delete;
    const QsaElasticPlan& plan()const noexcept override;
    uint8_t* base()const noexcept override;
    int64_t committed_slots()const noexcept override;
    bool healthy()const noexcept override;
    bool initialize_checked(std::string&) override;
    bool claim_state(QsaState*) noexcept override;
    // Host-only seal retains bound state identity and every admitted owner.
    // False after seal is terminal: caller retains the context/owner, no retry.
    bool prepare_retirement(const NativeRetirementContext&) noexcept override;
    // Requires exact checked gang context; no device synchronization here.
    // False may follow successful frees: residual owners stay retained, no rollback.
    bool commit_retirement(const NativeRetirementContext&) noexcept override;
    bool retire_checked() noexcept override;
private:
    OwnedQsaElasticPool()=default;
    struct Storage;
    std::unique_ptr<Storage> storage_;
};
// Concrete host-only NativeSessionRequest::prepare_elastic target. Physical
// attachments retain the actual same-ledger arena parent through donation.
bool prepare_owned_qsa_elastic_pools(const ModelGeometry&,const std::vector<QsaElasticPlan>&,
    WindowBudget&,const std::shared_ptr<const ReservationLease>& parent,
    std::vector<std::shared_ptr<QsaElasticView>>& out,std::string&);

}
