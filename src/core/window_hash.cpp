#include "strata/core/window_hash.hpp"

#include <stdexcept>
#include <limits>

// Private symbol names keep this pinned implementation independent of any hash
// library an optional backend already links.
#define sha256_init strata_window_sha256_init
#define sha256_update strata_window_sha256_update
#define sha256_final strata_window_sha256_final
#define sha256_hash strata_window_sha256_hash
#include "window_hash_vendor/sha256.c"

namespace strata::core {
struct WindowHasher::Impl { sha256_t state{}; uint64_t bytes = 0; };
WindowHasher::WindowHasher() : state_(std::make_unique<Impl>()) { sha256_init(&state_->state); }
WindowHasher::~WindowHasher() = default;
void WindowHasher::update(const void* data, size_t n) {
    if ((!data && n) || n > UINT64_MAX/8 - state_->bytes)
        throw std::length_error("invalid SHA-256 input extent");
    if (n) sha256_update(&state_->state,static_cast<const uint8_t*>(data),n);
    state_->bytes += n;
}
StateDigest WindowHasher::digest() const {
    auto copy = state_->state;
    StateDigest out{};
    sha256_final(&copy,out.data());
    return out;
}
StateDigest window_history_digest(const std::vector<int32_t>& ids) {
    WindowHasher h;
    constexpr char domain[] = "strata-window-history-v1";
    h.update(domain,sizeof(domain)); // includes terminating zero for an unambiguous domain
    uint8_t count[8];
    for (unsigned j=0;j<8;++j) count[j] = uint8_t(uint64_t(ids.size())>>(8*j));
    h.update(count,sizeof(count));
    std::array<uint8_t,4096> buffer{};
    size_t used = 0;
    for (int32_t id : ids) {
        for (unsigned j=0;j<4;++j) buffer[used++] = uint8_t(uint32_t(id)>>(8*j));
        if (used == buffer.size()) { h.update(buffer.data(),used); used=0; }
    }
    h.update(buffer.data(),used);
    return h.digest();
}
} // namespace strata::core
