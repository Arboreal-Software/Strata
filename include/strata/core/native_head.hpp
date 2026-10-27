#pragma once

#include "strata/core/head_embed_load_inventory.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Opt-in native output projection. Uploads unchanged GGUF blocks once at startup and owns its
/// activation scratch. The normal canonical head remains available for numerical A/B comparisons.
/// Plan v0.3 P6: any type the native MMVQ takes (Q5_K in the Q2_0 / IQ3_XXS files, IQ4_XS in IQ2_XS).
/// `load` takes the model's shards (strata::gguf_split_paths): the architecture is checked on the first, the
/// tensor is read from the shard that holds it.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    // Trusted synchronous borrow: caller retains owner and excludes retirement.
    bool enable_inventory_capture(const NativeInventoryLimits&, std::shared_ptr<const ReservationLease>,
                                  std::shared_ptr<const void>, std::string&);
    const HeadEmbedLoadInventory* loaded_inventory() const noexcept {
        return load_complete_ && !release_started_ && !release_failed_ ? inventory_.get() : nullptr;
    }
    bool load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err);
    void unload();
    bool run(const float* mixed, float* logits, void* stream, std::string& err) const;
    uint64_t weight_bytes() const { return bytes_; }
    bool loaded() const { return weights_ != nullptr && load_complete_ && !release_started_; }
    bool ready_for_adoption() const { return loaded() && !release_failed_; }
    uint64_t allocated_bytes() const;
    /// Caller has drained all users and selected device(). Failure retains the
    /// remaining resources; an uncertain free is sticky and must not be retried.
    bool release_checked(std::string& err);
    /// Plan v0.3 P6: the GGUF blocks, for the verify window's multi-column head.
    const void* weights() const { return weights_; }
    int type() const { return type_; }
    /// The device whose memory holds the weights (the one current when `load` ran; -1 before).
    int device() const { return device_; }
    /// Bytes of one vocabulary row.
    size_t row_bytes() const { return n_out_ > 0 ? (size_t) (bytes_ / (uint64_t) n_out_) : 0; }

private:
    std::unique_ptr<HeadEmbedLoadInventory> inventory_;
    bool inventory_required_=false, inventory_attempted_=false;
    uint64_t inventory_generation_=0;
    void* weights_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    uint64_t scratch_bytes_ = 0;
    uint64_t allocated_weight_bytes_ = 0; // actual allocation, including backend layout padding
    int n_in_ = 0, n_out_ = 0;
    int type_ = -1;
    int device_ = -1;
    bool load_complete_ = false, release_started_ = false, release_failed_ = false;
};

/// Plan v0.3 P6: `token_embd.weight` in its GGUF form (the IQ model files), in mapped pinned host memory: a row
/// is read over PCIe per token, so the table costs no VRAM.  `embed_row`, the verify window and the MTP drafter
/// use it instead of the canonical table when it is set (`set_native_embed`).
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;
    // Trusted synchronous borrow: caller retains owner and excludes retirement.
    bool enable_inventory_capture(const NativeInventoryLimits&, std::shared_ptr<const ReservationLease>,
                                  std::shared_ptr<const void>, std::string&);
    const HeadEmbedLoadInventory* loaded_inventory() const noexcept {
        return load_complete_ && !release_started_ && !release_failed_ ? inventory_.get() : nullptr;
    }
    bool load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err);
    /// Rows for device token ids.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
    /// One row for a host token id.
    void gather_one(int64_t token, float* out, void* stream) const;
    uint64_t bytes() const { return bytes_; }
    int type() const { return type_; }
    bool ready_for_adoption() const { return load_complete_ && !release_started_ && !release_failed_; }
    uint64_t allocated_device_bytes() const { return device_owned_ && dev_ ? bytes_ : 0; }
    uint64_t allocated_host_bytes() const { return host_ ? bytes_ : 0; }
    uint64_t allocated_pinned_bytes() const { return allocated_host_bytes(); }
    int device() const { return device_; }
    /// All device users of the mapped table must be drained by the caller.
    bool release_checked(std::string& err);

private:
    std::unique_ptr<HeadEmbedLoadInventory> inventory_;
    bool inventory_required_=false, inventory_attempted_=false;
    uint64_t inventory_generation_=0;
    void* host_ = nullptr;
    const void* dev_ = nullptr;
    uint64_t bytes_ = 0;
    size_t row_ = 0;
    int64_t n_embd_ = 0, n_vocab_ = 0;
    int type_ = -1;
    int device_ = -1;
    bool device_owned_ = false;
    bool load_complete_ = false, release_started_ = false, release_failed_ = false;
};
void set_native_embed(const NativeEmbed* e);
const NativeEmbed* native_embed();

}  // namespace strata::core
