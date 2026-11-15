#pragma once
#include "strata/core/window_budget.hpp"
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
enum class WindowBufferMemory { device, mapped_host };
class OwnedWindowBuffer {
public:
    // Host address for mapped storage, device address for device storage.
    void* data() const;
    void* device_data() const;
    uint64_t bytes() const;
private:
    struct Storage;
    std::shared_ptr<Storage> storage_;
    friend uint64_t window_buffer_quarantine_count();
    friend bool allocate_window_buffer(int,uint64_t,const std::vector<int>&,WindowBudget&,
        std::shared_ptr<const ReservationLease>,OwnedWindowBuffer&,std::string&,WindowBufferMemory);
};
// Internal runtime allocation: parent is the same-ledger private session lease.
// All devices that can use the pointer must be supplied. Owners must close/join
// producers before dropping the last handle; destruction drains those devices.
// Uncertain completion/free retains allocation and reservation until restart.
bool allocate_window_buffer(int device,uint64_t bytes,const std::vector<int>& consumers,
    WindowBudget&,std::shared_ptr<const ReservationLease> parent,OwnedWindowBuffer& out,std::string& error,
    WindowBufferMemory memory=WindowBufferMemory::device);
uint64_t window_buffer_quarantine_count();
}
