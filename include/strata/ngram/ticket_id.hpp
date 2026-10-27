#pragma once
#include <cstdint>
#include <stdexcept>

namespace strata::ngram::detail {
// Never recycle an ID during a reader's lifetime, including close/reopen.
// Zero is invalid and marks exhaustion after the final representable ID.
inline uint32_t take_ticket_id(uint32_t& next) {
    if(!next)throw std::overflow_error("PLE reader ticket IDs exhausted; recreate worker");
    return next++;
}
} // namespace strata::ngram::detail
