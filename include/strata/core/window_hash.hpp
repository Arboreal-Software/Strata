#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::core {
using StateDigest = std::array<uint8_t,32>;
class WindowHasher {
public:
    WindowHasher();
    ~WindowHasher();
    WindowHasher(const WindowHasher&) = delete;
    WindowHasher& operator=(const WindowHasher&) = delete;
    void update(const void*, size_t);
    StateDigest digest() const;
private:
    struct Impl;
    std::unique_ptr<Impl> state_;
};
// Domain-separated SHA-256 over token count and little-endian int32 token IDs.
// Text-only identity; media/template/model meanings are separately bound by the
// profile digest. Negative IDs are rejected by the window-state validator.
StateDigest window_history_digest(const std::vector<int32_t>&);
} // namespace strata::core
