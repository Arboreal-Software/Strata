#pragma once

#include <cstdint>
#include <string_view>
#include "strata/core/native_load_inventory.hpp"
#include <set>
#include <string>
#include <vector>
#include <utility>

namespace strata::core {
class WeightTable;
struct WeightRef;

class VerifiedDriverBindings;
// Short-lived serialized startup result. The setup budget and supplied path
// strings outlive destruction/reporting; genuine typed bindings retain the
// actual descriptors. Success exposes scalar limits, never parser payloads.
class NativeInventoryPreflight {
public:
    NativeInventoryPreflight() = default;
    ~NativeInventoryPreflight();
    NativeInventoryPreflight(const NativeInventoryPreflight&) = delete;
    NativeInventoryPreflight& operator=(const NativeInventoryPreflight&) = delete;
    NativeInventoryPreflight(NativeInventoryPreflight&&) noexcept;
    NativeInventoryPreflight& operator=(NativeInventoryPreflight&&) = delete;
    explicit operator bool() const noexcept {return okay_;}
    const NativeInventoryLimits& limits() const noexcept {return limits_;}
    uint64_t workspace_bytes() const noexcept {return charge_.bytes();}
    uint64_t requested_bytes() const noexcept {return requested_;}
    uint64_t allocation_attempts() const noexcept {return allocation_attempts_;}
    uint64_t parser_constructions() const noexcept {return parser_constructions_;}
    void report_error(const char* prefix) const noexcept;
private:
    friend class NativeDense;
    GgufHostWorkspace charge_; // destroyed after the explicit arena free
    std::shared_ptr<const VerifiedDriverBindings> bindings_;
    void* arena_=nullptr;
    std::string_view path_;
    NativeInventoryLimits limits_{};
    uint64_t requested_=0,allocation_attempts_=0,parser_constructions_=0;
    uint64_t argument_=0,name_offset_=0,name_bytes_=0;
    uint32_t error_=0;
    int fd_=-1;
    bool okay_=false;
};

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    // Explicit opt-in BEFORE load, using the early caller budget and real input
    // owner. Reservation is dedicated to this inventory; caller must not pledge
    // the same lease to another allocation. Default path allocates no inventory.
    // Refusal changes no state.
    // Source/table/filter preflight only: no conversion/device allocations.
    // On failure the supplied limits are unchanged. Must use the same stage
    // range/table/source bindings as load; load rechecks the admitted maximum.
    static uint64_t inventory_bf16_control_bytes() noexcept;
    static uint64_t inventory_preflight_setup_bytes() noexcept;
    // Serialized verified-input startup only; dedicate the exact fixed setup
    // baseline first. Scope the result through diagnostics, then destroy it
    // before the separate final inventory admission. Input limits are unchanged.
    static NativeInventoryPreflight inventory_conversion_limits_admitted(
        const std::vector<std::string>&,WeightTable&,bool,const NativeInventoryLimits&,
        WindowBudget&,std::shared_ptr<const VerifiedDriverBindings>,int64_t layer_lo=0,int64_t layer_hi=-1);

    static bool inventory_conversion_limits(const std::vector<std::string>&,WeightTable&,bool,
        NativeInventoryLimits&,std::string&,int64_t layer_lo=0,int64_t layer_hi=-1);
    bool enable_inventory_capture(const NativeInventoryLimits&,
        std::shared_ptr<const ReservationLease>,std::shared_ptr<const void>,std::string&);
    // Borrow only while retaining this loader/stage; no view after failure or
    // retirement. SYCL has no producer yet and returns nullptr.
    const NativeLoadInventory* loaded_inventory() const noexcept {
        return inventory_complete_&&!release_started_?inventory_.get():nullptr;
    }
    ~NativeDense();
    // Opt-in owner retirement. Caller proves quiescence and selects the owning
    // device first. Failure is sticky: retain this object and its accounting in
    // worker quarantine; do not execute, reload, or retry uncertain frees.
    bool release_checked(std::string& error);
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// With layer_hi >= 0, only the `blk.<l>.` matrices with layer_lo <= l < layer_hi are uploaded (a layer
    /// split's stage holds its own layers' projections, not the whole model's); the others keep data == nullptr.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t layer_lo = 0, int64_t layer_hi = -1);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// Layer split: load only blocks [lb, le) (every other `blk.N.` projection belongs to another GPU's stage; the
    /// PLE tensors are loaded everywhere).  Process-wide, read by the next `load`; (-1, -1) = all layers.
    static void set_layer_range(int lb, int le);
    /// Layer split: the device bytes `load` would allocate for the matrices of layers [lb, le) - the same walk and
    /// filters, each matrix rounded up to the granule cudaMalloc backs it with - reading the GGUF headers and `table`
    /// only, so it allocates nothing and needs no device (#1238).  `table` must hold every tensor the shards carry.
    static bool weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                 int64_t lb, int64_t le, uint64_t& out, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    /// STRATA_HC_Q8=1: the hyper-connection down / up projections of the layers live only as Q8_0 (the bf16 copy is not
    /// loaded: `add_hc_skip` puts their rows in the arena's skip set, `load` quantizes them from the pack file).
    /// Whether the GGUF shards carry the hyper-connection down / up projections as Q8_0 (an Unsloth UD-* file; the GSQ-RCO packs keep them
    /// BF16).  Reads the tensor directories only.  generate.cpp turns STRATA_HC_Q8 on by default on CUDA when this is true (owner yes 2026-11-07).
    static bool gguf_carries_q8_hc(const std::vector<std::string>& shards);
    static bool hc_q8_requested();
    static bool add_hc_skip(const std::string& pack_dir, std::set<std::string>& skip);
    /// A projection's bf16 values for a path that has no Q8 kernel (the prompt path's GEMMs, the single-token read): the
    /// resident copy, or - when the row was skipped - the Q8_0 copy dequantized into a per-stream scratch (`slot` 0 / 1:
    /// two matrices alive at once).  nullptr if there is neither.
    /// `cls` 0 = the prompt path (a scratch pair shared by its streams, ordered by events: `hc_bf16_release` after the
    /// last GEMM that reads the slot), 1 = the layer / session read (its own pair: it is captured into graphs, where an
    /// event wait on an outside event is illegal). The scratches are allocated by `load`, never on first use.
    static const uint16_t* hc_bf16_view(const WeightRef& w, void* stream, int slot, int cls);
    static void hc_bf16_release(void* stream, int slot, int cls);
    uint64_t weight_bytes() const { return bytes_; }
    /// What the device lost to `load`: each matrix rounded up to the granule `cudaMalloc` backs it with.
    uint64_t allocated_bytes() const { return allocated_; }
    // Only original native weight allocations published by load; excludes
    // mutable scratch and packed helper buffers. Retired objects prove no spans.
    bool owns_weight_span(const void* base, uint64_t bytes) const;
    // An untouched instance is valid for a canonical-only model. A failed or
    // retired native load must never be adopted as a usable model stage.
    bool ready_for_adoption() const {return !release_started_&&(!load_attempted_||weight_spans_valid_);}
    size_t tensor_count() const { return weights_.size() - packed_keys_.size(); }

private:
    friend struct NativeDenseReleaseTestAccess;
    // Declared first: destroyed after allocation/registration metadata. The
    // destructor body retires native allocations before releasing these leases.
    std::unique_ptr<NativeLoadInventory> inventory_;
    bool inventory_complete_=false;
    bool load_attempted_ = false;
    bool weight_spans_valid_ = false;
    bool release_started_ = false;
    bool release_failed_ = false;
    std::vector<void*> weights_;
    std::vector<std::pair<const void*,uint64_t>> weight_spans_;
    std::vector<const void*> packed_keys_;   // GGUF-layout pointers registered with STRATA_Q8_PACKED=1
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    uint64_t allocated_ = 0;
};
} // namespace strata::core
