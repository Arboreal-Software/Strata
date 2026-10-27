#pragma once
#include <array>
#include <cstdint>
#include <string>
namespace strata::core {
struct QsaState;
enum class QsaElasticFormat : uint8_t { f16, int8, q4, hybrid };
// Explicit immutable sizing/binding contract; no global mode/index or budget ABI.
struct QsaElasticPlan {
    QsaElasticFormat format=QsaElasticFormat::f16;
    bool rotate=false;
    int device=-1;
    int64_t cells=0,initial_cells=0,slots=0,page_size=0;
    uint64_t granularity=0,virtual_bytes=0,initial_chunks=0;
    uint64_t first_bytes=0,other_bytes=0,pinned_bytes=0,first_upload=0,other_upload=0;
    uint64_t arrays=0;
    std::array<uint64_t,4> offsets{},per_slot{};
};
inline bool qsa_elastic_plan_equal(const QsaElasticPlan&a,const QsaElasticPlan&b) noexcept {
    return a.format==b.format&&a.rotate==b.rotate&&a.device==b.device&&a.cells==b.cells&&a.initial_cells==b.initial_cells&&
        a.slots==b.slots&&a.page_size==b.page_size&&a.granularity==b.granularity&&a.virtual_bytes==b.virtual_bytes&&
        a.initial_chunks==b.initial_chunks&&a.first_bytes==b.first_bytes&&a.other_bytes==b.other_bytes&&
        a.pinned_bytes==b.pinned_bytes&&a.first_upload==b.first_upload&&a.other_upload==b.other_upload&&
        a.arrays==b.arrays&&a.offsets==b.offsets&&a.per_slot==b.per_slot;
}
class QsaElasticView {
public:
    virtual ~QsaElasticView()=default;
    virtual const QsaElasticPlan& plan()const noexcept=0;
    virtual uint8_t* base()const noexcept=0;
    virtual int64_t committed_slots()const noexcept=0;
    virtual bool healthy()const noexcept=0;
    // Host-prepared owner; initialize only after the entire gang is admitted.
    virtual bool initialize_checked(std::string&)=0;
    virtual bool claim_state(QsaState*) noexcept=0;
    virtual bool retire_checked() noexcept=0;
};
}
