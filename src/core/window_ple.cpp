#include "strata/core/window_ple.hpp"
#include <stdexcept>

namespace strata::core {
namespace {
struct PleOwners {
    std::shared_ptr<const ReservationLease> reservation;
    std::shared_ptr<const void> weights;
};
}
bool bind_private_ple(PleRun& destination,
    std::shared_ptr<strata::kernels::PleTable> table,
    const strata::kernels::PleWeights& weights,
    const strata::kernels::PleConsts& constants,
    std::shared_ptr<const void> weight_owner, WindowBudget& budget,
    std::shared_ptr<const ReservationLease> session_reservation,
    std::string& error) {
    if(destination.table || destination.read_context || !table || !table->is_open() || !weight_owner) {
        error="private PLE binding requires an unbound destination and retained table/weights";return false;
    }
    if(weights.key_native_q8_1 || weights.pre_key || weights.pre_value) {
        error="private PLE native-key/prefill scratch requires separate owned binding";return false;
    }
    if(weights.key_native_data && (!destination.w.key_native_q8_1 ||
       (weights.key_native_type!=42 && weights.key_native_type!=18 &&
        weights.key_native_type!=23 && weights.key_native_type!=8))) {
        error="private PLE native key requires owned scratch and a supported format";return false;
    }
    auto candidate=destination;
    candidate.table=table.get();candidate.w=weights;candidate.consts=constants;
    candidate.w.key_native_q8_1=destination.w.key_native_q8_1;
    if(!candidate.ready()) {error="private PLE buffers or immutable weights are incomplete";return false;}
    try {
        WindowResources required;
        required.device_bytes.resize(budget.snapshot().capacity.device_bytes.size());
        required.host_bytes=private_ple_host_headroom;required.io_slots=private_ple_io_slots;
        auto reserved=budget.reserve(required,ReservationPurpose::private_attachment,std::move(session_reservation));
        if(!reserved) {error="private PLE context capacity unavailable";return false;}
        auto owners=std::make_shared<PleOwners>();
        owners->reservation=std::move(reserved.lease);owners->weights=std::move(weight_owner);
        candidate.read_context=std::make_shared<strata::kernels::PleReadContext>(std::move(table),std::move(owners));
        destination=std::move(candidate);
        return true;
    } catch(const std::bad_alloc&) {
        error="private PLE context host allocation failed";return false;
    } catch(const std::invalid_argument& e) {
        error=e.what();return false;
    }
}
} // namespace strata::core
