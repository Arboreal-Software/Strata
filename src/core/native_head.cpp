#include "strata/core/native_head.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>

namespace strata::core {

namespace {
std::atomic<HeadEmbedLoadInventory*> failed_head_embed_inventory{nullptr};
void retain_failed(std::unique_ptr<HeadEmbedLoadInventory>& inventory) noexcept {
    if(auto* p=inventory.release()) {
        auto* head=failed_head_embed_inventory.load();
        do {p->quarantined_next=head;} while(!failed_head_embed_inventory.compare_exchange_weak(head,p));
        std::fprintf(stderr,"head/embed inventory retained after uncertain retirement\n");
    }
}
bool prepare_inventory(std::unique_ptr<HeadEmbedLoadInventory>& out,const NativeInventoryLimits& limits,
 std::shared_ptr<const ReservationLease> reservation,std::shared_ptr<const void> inputs,std::string& error) {
    uint64_t required=0;
    if(!head_embed_inventory_host_bound(limits,required,error))return false;
    if(!reservation||reservation->purpose()!=ReservationPurpose::foreground||reservation->resources().host_bytes<required||!inputs) {
        error="head/embed inventory requires host reservation and retained inputs";return false;
    }
    try {
        auto next=std::make_unique<HeadEmbedLoadInventory>();
        next->metadata_reservation=std::move(reservation);next->retained_input_bindings=std::move(inputs);next->limits=limits;
        next->sources.reserve(size_t(limits.max_sources));next->tensor.source_shape.reserve(size_t(limits.max_rank));
        next->tensor.name.reserve(size_t(limits.max_string_bytes));out=std::move(next);return true;
    }catch(const std::bad_alloc&){error="head/embed inventory allocation failed";return false;}
}
bool inventory_sources(HeadEmbedLoadInventory* p,const std::vector<std::string>& paths,std::string& error) {
    if(!p)return true;
    if(paths.size()>p->limits.max_sources){error="head/embed inventory source count exceeded";return false;}
    for(const auto& path:paths)if(path.size()>p->limits.max_string_bytes){error="head/embed inventory source string exceeded";return false;}
    p->sources=paths;return true;
}
bool inventory_tensor(HeadEmbedLoadInventory* p,const strata::TensorInfo& t,size_t ordinal,uint64_t start,uint64_t bytes,std::string& error) {
    if(!p)return true;
    if(t.name.size()>p->limits.max_string_bytes||t.shape.size()>p->limits.max_rank||t.offset>UINT64_MAX-start){error="head/embed inventory tensor bounds exceeded";return false;}
    p->tensor.name=t.name;p->tensor.source_ordinal=ordinal;p->tensor.source_offset=start+t.offset;
    p->tensor.source_bytes=bytes;p->tensor.source_type=t.type;p->tensor.source_shape=t.shape;p->tensor.runtime_bytes=bytes;return true;
}
}
NativeHead::~NativeHead() { unload(); if(release_failed_||weights_||scratch_)retain_failed(inventory_); }


uint64_t NativeHead::allocated_bytes() const {
    return (weights_ ? allocated_weight_bytes_ : 0) + (scratch_ ? scratch_bytes_ : 0) +
           (weights_ ? strata::kernels::native_head_packed_allocation_bytes(weights_) : 0);
}

bool NativeHead::release_checked(std::string& err) {
    release_started_ = true; load_complete_ = false;
    if (release_failed_) { err = "native head release previously failed"; return false; }
    if (!weights_ && !scratch_) return true;
    int d = -1;
    if (cudaGetDevice(&d) != cudaSuccess || d != device_) {
        err = "native head release requires owning device"; return false;
    }
    if (weights_ && !strata::kernels::native_head_packed_release_checked(weights_)) {
        release_failed_ = true; err = "native head packed release failed"; return false;
    }
    if (scratch_) {
        if (cudaFree(scratch_) != cudaSuccess) { release_failed_ = true; err = "native head scratch release failed"; return false; }
        scratch_ = nullptr;
    }
    if (weights_) {
        if (cudaFree(weights_) != cudaSuccess) { release_failed_ = true; err = "native head weight release failed"; return false; }
        weights_ = nullptr;
    }
    return true;
}

void NativeHead::unload() {
    if (release_failed_) return;
    if (!weights_ && !scratch_) { if(inventory_) {release_started_=true;load_complete_=false;} return; }
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess ||
        (previous != device_ && cudaSetDevice(device_) != cudaSuccess)) { release_failed_ = true; return; }
    try {
        std::string err;
        if (cudaDeviceSynchronize() != cudaSuccess) release_failed_ = true;
        else release_checked(err);
    } catch (...) { release_failed_ = true; }
    if (previous != device_ && cudaSetDevice(previous) != cudaSuccess) release_failed_ = true;
}

bool NativeHead::enable_inventory_capture(const NativeInventoryLimits& limits,
 std::shared_ptr<const ReservationLease> reservation,std::shared_ptr<const void> inputs,std::string& error) {
    if(weights_||scratch_||release_failed_||(inventory_&&!release_started_)) {error="head/embed inventory must be armed before load or after checked retirement";return false;}
    if(inventory_generation_==UINT64_MAX) {error="head/embed inventory generation exhausted";return false;}
    if(!prepare_inventory(inventory_,limits,std::move(reservation),std::move(inputs),error))return false;
    inventory_required_=true;inventory_attempted_=false;return true;
}
bool NativeHead::load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err) {
    if (weights_ || scratch_ || release_failed_) { err = "native head is loaded or has retained failed resources"; return false; }
    if(inventory_required_&&(!inventory_||inventory_attempted_)) {err="head/embed inventory requires fresh capture admission";return false;}
    inventory_attempted_=true;
    load_complete_ = false; release_started_ = false;
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        if(!inventory_sources(inventory_.get(),shards,err))return false;
        // The architecture is the metadata shard's; output.weight comes from whichever shard holds it (shard 2
        // of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor).  GgufModel refuses a duplicate across shards.
        const strata::GgufModel model(shards);
        err = strata::check_engine_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        if(!inventory_tensor(inventory_.get(),*tensor,at,gguf.data_start(),bytes,err))return false;
        scratch_bytes_ = strata::kernels::native_q8_1_bytes((int) n_in, 1);
        if (!bytes || scratch_bytes_ > UINT64_MAX - bytes || cudaGetDevice(&device_) != cudaSuccess) {
            err = "native head size or owning device invalid"; return false;
        }
        bytes_ = bytes; allocated_weight_bytes_ = bytes;
        cudaError_t status = cudaMalloc(&weights_, bytes_);
        if (status == cudaSuccess) status = cudaMalloc(&scratch_, scratch_bytes_);
        strata::platform::advise_willneed(gguf.tensor_data(*tensor), bytes_);
        if (status == cudaSuccess)
            status = cudaMemcpy(weights_, gguf.tensor_data(*tensor), bytes_, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            err = std::string("native head upload: ") + cudaGetErrorString(status);
            return false;
        }
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
        if (type_ == 14 && strata::kernels::native_q6_k_packed_enabled())   // STRATA_Q6_PACKED=1
            { const bool packed=strata::kernels::native_q6_k_pack(weights_, n_in_, n_out_, "output head");
              if(inventory_&&packed)inventory_->packed=HeadPackedForm::q6_planes_256; }
        if (type_ == 8)   // STRATA_Q8_SM60=1 (a no-op otherwise); the MTP draft head reads the same matrix
            { const bool packed=strata::kernels::native_q8_0_sm60_pack(weights_, n_in_, n_out_, "output head");
              if(inventory_&&packed)inventory_->packed=HeadPackedForm::q8_sm60_qs_d; }
        if (cudaDeviceSynchronize() != cudaSuccess) { err = "native head load drain failed"; return false; }
        if(inventory_) {inventory_->generation=++inventory_generation_;inventory_->device=device_;
            inventory_->main_bytes=bytes_;inventory_->scratch_bytes=scratch_bytes_;
            inventory_->helper_bytes=strata::kernels::native_head_packed_allocation_bytes(weights_);inventory_->row_bytes=bytes_/n_out_;}
        load_complete_ = true;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}

bool NativeHead::run(const float* mixed, float* logits, void* stream, std::string& err) const {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        err = std::string("native head launch: ") + cudaGetErrorString(status);
        return false;
    }
    return true;
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

bool NativeEmbed::release_checked(std::string& err) {
    release_started_ = true; load_complete_ = false;
    if (g_embed == this) g_embed = nullptr;
    if (release_failed_) { err = "native embedding release previously failed"; return false; }
    if (!host_ && !dev_) return true;
    int d = -1;
    if (cudaGetDevice(&d) != cudaSuccess || d != device_) {
        err = "native embedding release requires owning device"; return false;
    }
    if (device_owned_ && dev_) {
        if (cudaFree(const_cast<void*>(dev_)) != cudaSuccess) { release_failed_ = true; err = "native embedding device release failed"; return false; }
        dev_ = nullptr; device_owned_ = false;
    }
    if (host_) {
        if (cudaFreeHost(host_) != cudaSuccess) { release_failed_ = true; err = "native embedding pinned release failed"; return false; }
        host_ = nullptr; dev_ = nullptr;
    }
    return true;
}

NativeEmbed::~NativeEmbed() {
    if (g_embed == this) g_embed = nullptr;
    if(inventory_) {
        if(!release_failed_&&(host_||dev_)) {
            int previous=-1;
            if(cudaGetDevice(&previous)!=cudaSuccess||(previous!=device_&&cudaSetDevice(device_)!=cudaSuccess))release_failed_=true;
            else {
                try {std::string error;if(cudaDeviceSynchronize()!=cudaSuccess||!release_checked(error))release_failed_=true;}
                catch(...) {release_failed_=true;}
                if(previous!=device_&&cudaSetDevice(previous)!=cudaSuccess)release_failed_=true;
            }
        }
        if(release_failed_||host_||dev_)retain_failed(inventory_);
        return;
    }
    if (release_failed_ || (!host_ && !dev_)) return;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess ||
        (previous != device_ && cudaSetDevice(device_) != cudaSuccess)) return;
    try {
        std::string err;
        if (cudaDeviceSynchronize() == cudaSuccess) release_checked(err);
    } catch (...) {}
    if (previous != device_) cudaSetDevice(previous);
}

bool NativeEmbed::enable_inventory_capture(const NativeInventoryLimits& limits,
 std::shared_ptr<const ReservationLease> reservation,std::shared_ptr<const void> inputs,std::string& error) {
    if(host_||dev_||release_failed_||(inventory_&&!release_started_)) {error="head/embed inventory must be armed before load or after checked retirement";return false;}
    if(inventory_generation_==UINT64_MAX) {error="head/embed inventory generation exhausted";return false;}
    if(!prepare_inventory(inventory_,limits,std::move(reservation),std::move(inputs),error))return false;
    inventory_required_=true;inventory_attempted_=false;return true;
}
bool NativeEmbed::load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err) {
    if (host_ || dev_ || release_failed_) { err = "native embedding is loaded or has retained failed resources"; return false; }
    if(inventory_required_&&(!inventory_||inventory_attempted_)) {err="head/embed inventory requires fresh capture admission";return false;}
    inventory_attempted_=true;
    load_complete_ = false; release_started_ = false;
    if (n_embd <= 0 || n_vocab <= 0 || n_embd > INT_MAX || n_vocab > INT_MAX || n_embd % 256) {
        err = "native embedding requires positive int32 dimensions and whole 256-value rows"; return false;
    }
    try {
        if(!inventory_sources(inventory_.get(),shards,err))return false;
        const strata::GgufModel model(shards);
        // --embd-gguf's one-tensor file (tools/embd_bf16_pack.py) says "strata-embd": only its tensor is checked
        const strata::MetaValue* arch = model.meta().get("general.architecture");
        err = arch != nullptr && arch->s == "strata-embd" ? std::string() : strata::check_engine_architecture(model.meta());
        if (!err.empty()) { err = "native embedding: " + err; return false; }
        size_t at = 0;
        const strata::TensorInfo* t = model.find("token_embd.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        if (!row_ || uint64_t(n_vocab) > UINT64_MAX / row_) { err = "native embedding size overflow"; return false; }
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        // the table is copied out of the mapping below: a truncated shard must be an error, not a read past EOF
        if (!model.in_bounds(*t, at) || strata::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        if(!inventory_tensor(inventory_.get(),*t,at,gguf.data_start(),bytes_,err))return false;
        if(inventory_) {inventory_->role=HeadEmbedRole::embedding;inventory_->embedding_override=arch&&arch->s=="strata-embd";}
        strata::platform::advise_willneed(gguf.tensor_data(*t), bytes_);   // copied out of the mapping below
        if (cudaGetDevice(&device_) != cudaSuccess) { err = "native embedding owning device unavailable"; return false; }
        cudaError_t host_err = cudaHostAlloc(&host_, bytes_, cudaHostAllocMapped | cudaHostAllocPortable);
        if (host_err != cudaSuccess) {
            if(inventory_)inventory_->placement=EmbedPlacement::device_pin_fallback;
            // Under WSL2 the driver's pinned/mapped host budget (~1 GiB) can be spent by the GPU contexts
            // themselves (three cards). The table is only gathered from, so keep it in the current device's VRAM
            // instead: it costs its size there and reads faster than over PCIe.
            cudaGetLastError();
            host_ = nullptr;
            void* d = nullptr;
            const auto alloc_status = cudaMalloc(&d, bytes_);
            if (d) { dev_ = d; device_owned_ = true; }
            if (alloc_status != cudaSuccess ||
                cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaGetLastError();
                err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB: " +
                      cudaGetErrorString(host_err) + ", nor place it in VRAM";
                return false;
            }
            std::fprintf(stderr, "strata: native embedding: cannot pin %llu MiB, kept in VRAM instead\n",
                         (unsigned long long) (bytes_ >> 20));
            dev_ = d;
        } else {
            if(inventory_)inventory_->placement=EmbedPlacement::mapped_portable;
            std::memcpy(host_, gguf.tensor_data(*t), bytes_);
            void* d = nullptr;
#if defined(STRATA_USE_HIP)
            // #325: the Windows HIP stack can refuse the device alias of a mapped allocation (and, when it gives
            // one, it is the host address itself - unified addressing; kernels read it correctly there, a
            // device-to-device copy into it does not land: tests/hip/mapped_alias). The table is only gathered from,
            // so without an alias it goes into VRAM like the unpinnable case above, instead of failing the start.
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess || d == nullptr) {
                if(inventory_)inventory_->placement=EmbedPlacement::device_alias_fallback;
                cudaGetLastError();
                d = nullptr;
                const auto alloc_status = cudaMalloc(&d, bytes_);
                if (d) { dev_ = d; device_owned_ = true; }
                if (alloc_status != cudaSuccess ||
                    cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                    cudaGetLastError();
                    err = "native embedding: no device alias for the mapped table, and no VRAM to copy it into";
                    return false;
                }
                if (cudaFreeHost(host_) != cudaSuccess) {
                    release_failed_ = true; err = "native embedding fallback pinned release failed"; return false;
                }
                host_ = nullptr;
                std::fprintf(stderr, "strata: native embedding: no device alias for the mapped table, kept in VRAM\n");
            }
#else
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess || d == nullptr) {
                err = "native embedding: no device alias for the mapped table";
                return false;
            }
#endif
            dev_ = d;
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        if (cudaDeviceSynchronize() != cudaSuccess) { err = "native embedding load drain failed"; return false; }
        if(inventory_) {inventory_->generation=++inventory_generation_;inventory_->device=device_;inventory_->main_bytes=bytes_;inventory_->row_bytes=row_;}
        load_complete_ = true;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    strata::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) dev_ + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace strata::core
