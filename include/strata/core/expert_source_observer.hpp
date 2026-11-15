#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
namespace strata::core {
enum class ExpertSourceTransfer {mapped_packed,mapped_roles,read_packed,scatter_roles};
// Internal synchronous observer; the caller retains it until all reader workers
// join. No budget/verified-input dependency or C++20 requirement in CUDA readers.
class ExpertSourceReadObserver {
public:
    virtual ~ExpertSourceReadObserver()=default;
    virtual bool validate_fd(const std::string&,int) const noexcept=0;
    virtual bool range_allowed(const std::string&,uint64_t,uint64_t,uint64_t) const noexcept=0;
    virtual uint64_t experts() const noexcept=0;
    virtual uint64_t max_workers() const noexcept=0;
    virtual bool record(uint64_t,uint64_t,const std::string&,uint64_t,uint64_t,uint64_t,
                        uint64_t,uint64_t,ExpertSourceTransfer,uint64_t mapped_file_ordinal=UINT64_MAX) noexcept=0;
    static void close_failed() noexcept {
        std::fprintf(stderr,"strata: verified expert source close uncertain; retaining authority until process exit\n");
        std::_Exit(1);
    }
};
}
