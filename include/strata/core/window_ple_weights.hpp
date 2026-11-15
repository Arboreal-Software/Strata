#pragma once
#include "strata/core/weights.hpp"
#include "strata/kernels/ple.hpp"
#include <array>

namespace strata::core {
// Order: key, value, norm_key, norm_query, norm_conv, conv1d.
// Metadata validation only: caller retains the actual loaded model allocations
// and validates device/model identity. Output is unchanged on refusal.
bool private_ple_weights(const std::array<const WeightRef*,6>&, bool native_key,
                         strata::kernels::PleWeights& out, std::string& error);
inline bool private_ple_weights(const WeightTable& table, bool native_key,
                               strata::kernels::PleWeights& out, std::string& error) {
    return private_ple_weights({table.find("blk.1.ple_key.weight"),table.find("blk.1.ple_value.weight"),
        table.find("blk.1.ple_norm_key.weight"),table.find("blk.1.ple_norm_query.weight"),
        table.find("blk.1.ple_norm_conv.weight"),table.find("blk.1.ple_conv1d.weight")},native_key,out,error);
}
} // namespace strata::core
