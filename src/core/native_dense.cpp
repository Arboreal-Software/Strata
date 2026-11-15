#include "strata/core/native_dense.hpp"
#include "strata/core/run_manifest.hpp"
#include "strata/core/verified_input_manifest.hpp"
#include <charconv>
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <memory>
#include <set>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core {
namespace {
// `load` makes one cudaMalloc per projection matrix, and the driver backs each call with 2 MiB granules
// (measured on the RTX 5060: 300 buffers of 6.73 MiB asked 2019 MiB and cost 2400).  So a stage's real
// footprint is the granule-rounded sum, not the payload sum.  This rule - a granule for every matrix of a
// MiB or more, the payload for the rest - prices the iq3_s pack's 300 matrices at 2255 MiB where they
// really take 2238 (+0.8%).  It over-prices, which is the safe side.  (Pricing every matrix at a granule is
// far worse, 2378 MiB: the driver packs the sub-MiB matrices into the slack of the rounded ones.)
constexpr uint64_t kAllocGranule = 2ull << 20;
constexpr uint64_t kSmallAlloc = 1ull << 20;
uint64_t alloc_bytes(uint64_t bytes) {
    if(bytes>=kSmallAlloc && bytes>UINT64_MAX-(kAllocGranule-1))
        throw std::overflow_error("native allocation rounding overflow");
    return bytes < kSmallAlloc ? bytes : (bytes + kAllocGranule - 1) / kAllocGranule * kAllocGranule;
}
void add_allocation(uint64_t& total,uint64_t bytes) {
    const auto charge=alloc_bytes(bytes);
    if(charge>UINT64_MAX-total)throw std::overflow_error("native allocation total overflow");
    total+=charge;
}
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(std::string_view name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.data() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
// S23 experiment (STRATA_HC_Q8=1): the hyper-connection projections' Q8_0 bytes for the verify read
bool hc_q8_env() {
    // STRATA_HC_Q8: 1 = on (the GGUF's own Q8_0 projections, or - a bf16 pack - a Q8_0 copy quantized at load); 0 = off;
    // unset / auto = on for gfx12 (RDNA4: the R9700 decode +2.5 to +4%, KL 0.0003-0.0010 over its noise floor) unless
    // STRATA_GFX12_DEFAULTS=0; also on for gfx1151 (Strix Halo, STRATA_HC_Q8=0 turns it off), off elsewhere
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HC_Q8");
        if (v != nullptr && v[0] == '1') return true;
        if (v != nullptr && v[0] != '\0' && std::strcmp(v, "auto") != 0) return false;
#if defined(STRATA_USE_HIP)
        {   // Strix Halo (gfx1151): on by default (decode +9.5%, json/code/tool/chat/agent +7.9..+10.7%); the answers differ in
            // near-tie tokens by about the size of the engine's own CPU-vs-GPU expert rounding (docs/AMD_HIP.md);
            // STRATA_HC_Q8=0 keeps the bf16 read
            int dev1151 = 0;
            cudaDeviceProp p1151;
            const bool g1151 = cudaGetDevice(&dev1151) == cudaSuccess && cudaGetDeviceProperties(&p1151, dev1151) == cudaSuccess &&
                               strata::kernels::gfx_arch_is_gfx1151(p1151.gcnArchName);
            (void) cudaGetLastError();
            if (g1151) {
                std::fprintf(stderr, "strata: Strix Halo (gfx1151): 8-bit hyper-connection weights are on (decode about 9%% faster, "
                                     "answers can differ in near-tie tokens); STRATA_HC_Q8=0 keeps the bf16 read\n");
                return true;
            }
        }
        const char* d = std::getenv("STRATA_GFX12_DEFAULTS");
        if (d != nullptr && d[0] == '0') return false;
        int dev = 0;
        cudaDeviceProp p;
        const bool g12 = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                         strata::kernels::gfx_arch_is_gfx12(p.gcnArchName);
        (void) cudaGetLastError();
        if (g12) std::fprintf(stderr, "strata: gfx12 (RDNA4): the hyper-connection projections are read as Q8_0 (quantized at load, "
                                      "decode faster, answers differ slightly); STRATA_HC_Q8=0 keeps the bf16 read\n");
        return g12;
#else
        return false;
#endif
    }();
    return on;
}
// STRATA_HC_Q8_ALL_STAGES=1: the Q8_0 hyper-connection rows of every layer are copied to every stage (the behaviour before #3559)
bool hc_q8_stage_local() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8_ALL_STAGES"); return !(v != nullptr && v[0] == '1'); }();
    return on;
}
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;   // S25: the final mixer too
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                                     ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// S25: the GGUF keeps the inject rows as F32 - quantized here to Q8_0 (ggml's reference rounding) so the Q8_0 read
// covers every hc projection
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}
// The same reference rounding worker for ordinary threaded conversion and
// the paid serial inventory path. All per-block workspace is explicit control.
void q8_0_bf16_blocks(NativeBf16Q8Control& w) {
    for (;w.block<w.end;++w.block) {
        w.amax=0.0f;
        for(w.column=0;w.column<32;++w.column) {
            w.bits=(uint32_t)w.source[w.block*32+w.column]<<16;
            std::memcpy(&w.values[w.column],&w.bits,4);
            w.amax=(std::max)(w.amax,std::fabs(w.values[w.column]));
        }
        w.d=w.amax/127.0f;w.id=w.d!=0.0f?1.0f/w.d:0.0f;
        w.block_output=w.output+w.block*34;
        w.scale=half_bits(w.d);
        std::memcpy(w.block_output,&w.scale,2);
        for(w.column=0;w.column<32;++w.column)
            w.block_output[2+w.column]=(uint8_t)(int8_t)std::lround(w.values[w.column]*w.id);
    }
}
// Ordinary conversion keeps its threaded execution and output vector. Only the
// real rounding worker is shared; no verified-mode thread resources are added.
std::vector<uint8_t> q8_0_of_bf16(const uint16_t* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    const uint64_t nb = n / 32;
    auto work = [&](uint64_t b0, uint64_t b1) {
        NativeBf16Q8Control w;w.source=x;w.output=out.data();w.block=b0;w.end=b1;
        q8_0_bf16_blocks(w);
    };
    const unsigned nt = (std::min)(8u, (std::max)(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < nt; ++i) pool.emplace_back(work, nb * i / nt, nb * (i + 1) / nt);
    for (auto& th : pool) th.join();
    return out;
}
bool eligible_name(std::string_view name,uint32_t type,bool include_ple_key) {
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (type == 42 || type == 18 || type == 23 || type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
template<class Tensor> bool eligible(const Tensor& tensor,bool include_ple_key) {
    return eligible_name(tensor.name,tensor.type,include_ple_key);
}
// DevicePtr exists only within load. Nestable thread-local scope keeps its
// deleter stateless (no larger default Pending allocations) and charges an
// unwind failure only to the actual opt-in load on this thread.
thread_local bool* inventory_unwind_uncertain=nullptr;
struct InventoryUnwindScope {
    bool* previous;
    explicit InventoryUnwindScope(bool* current):previous(inventory_unwind_uncertain){inventory_unwind_uncertain=current;}
    ~InventoryUnwindScope(){inventory_unwind_uncertain=previous;}
};
struct DeviceFree {
    void operator()(void* p) const {
        if(!p)return;
        const auto status=cudaFree(p);
        if(inventory_unwind_uncertain&&status!=cudaSuccess) {
            *inventory_unwind_uncertain=true;std::fprintf(stderr,"native inventory load cleanup uncertain; input authority retained\n");
        }
    }
};
// #4741: a native-eligible projection a pack keeps unquantized (BF16, GGUF type 30; Heretic v2 IQ4_NL has its layer-0 attn_gate so)
// has no native kernel.  Its siblings are Q8_0, so it is served as Q8_0 made from the file's BF16 values (ggml's rounding), and
// the dispatchers that need a native projection find one.  The pack's BF16 copy stays where it is.
template<class Tensor> bool bf16_as_q8_0(const Tensor& tensor, const WeightRef& ref) {
    return tensor.type == 30 && tensor.name != "blk.1.ple_key.weight" && tensor.shape.size() == 2 &&
           !ref.quantized() && ref.kind == WeightKind::Bf16InF32 &&
           ref.ne0 > 0 && ref.ne0 <= INT_MAX && ref.ne1 > 0 && ref.ne1 <= INT_MAX && ref.ne0 % 32 == 0 &&
           tensor.shape[0] == (uint64_t) ref.ne0 && tensor.shape[1] == (uint64_t) ref.ne1;
}
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    DevicePtr packed;   // STRATA_Q8_PACKED=1 copy, or null
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }

uint64_t NativeDense::inventory_bf16_control_bytes() noexcept {
    // Actual types, not a guessed ABI: conversion object + worker reference,
    // n/bytes, owner/current devices, upload status, allocation pointer and its
    // real stateless DeviceFree unique_ptr. No new std::thread/task/vector.
    return sizeof(NativeBf16Q8Control)+sizeof(NativeBf16Q8Control*)+
        2*sizeof(uint64_t)+2*sizeof(int)+sizeof(cudaError_t)+sizeof(void*)+sizeof(DevicePtr);
}

namespace {
#if defined(__linux__)
struct InventoryPreflightFrame {
    strata::GgufWorkspaceQuote quote;
    struct stat status{};
    uint64_t maximum=0,elements=0,payload=0;
    size_t source=0;
    long layer=0;
    std::map<std::string,WeightRef>::const_iterator found;
};
struct InventoryPreflightReportFrame {
    char bytes[4096];
    char number[std::numeric_limits<uint64_t>::digits10+3];
    uint64_t offset=0,remaining=0;
    bool stopped=false;
};
#endif
}
NativeInventoryPreflight::~NativeInventoryPreflight(){
    std::free(arena_); // actual parser/error storage dies while charge_ is live
}
NativeInventoryPreflight::NativeInventoryPreflight(NativeInventoryPreflight&& other) noexcept
    :charge_(std::move(other.charge_)),bindings_(std::move(other.bindings_)),
     arena_(std::exchange(other.arena_,nullptr)),path_(other.path_),limits_(other.limits_),
     requested_(other.requested_),allocation_attempts_(other.allocation_attempts_),
     parser_constructions_(other.parser_constructions_),argument_(other.argument_),
     name_offset_(other.name_offset_),name_bytes_(other.name_bytes_),error_(other.error_),
     fd_(other.fd_),okay_(other.okay_){}
uint64_t NativeDense::inventory_preflight_setup_bytes() noexcept {
#if defined(__linux__)
    // Concrete, serial startup source objects. This is not a physical stack,
    // libc allocator-rounding, kernel-page or RSS certificate. Both caller and
    // callee result objects may overlap during a move; reporting has its own
    // frame. The input quota dedicates this baseline BEFORE any exists.
    return 2*sizeof(NativeInventoryPreflight)+sizeof(InventoryPreflightFrame)+
        2*sizeof(strata::GgufDirectoryScan)+3*sizeof(strata::GgufDirectoryScan::MetaFrame)+
        sizeof(strata::GgufDirectoryScan::ScanFrame)+sizeof(strata::GgufDirectoryScan::ReplayFrame)+
        sizeof(strata::GgufWorkspaceDirectory)+
        sizeof(strata::GgufWorkspaceDirectory::SortFrame)+sizeof(InventoryPreflightReportFrame)+
        // Exact matcher buffer/result/view/local representations.
        (std::numeric_limits<int>::digits10+3)*sizeof(char)+2*sizeof(std::to_chars_result)+
        3*sizeof(std::string_view)+sizeof(const VerifiedInputLease*)+
        // Admission/reset/setup-check lock guards and explicit forwarding
        // arguments/iterators/loop state on the finite nonrecursive call path.
        4*sizeof(std::lock_guard<std::mutex>)+3*sizeof(WindowBudget*)+
        sizeof(GgufHostWorkspace*)+2*sizeof(GgufHostWorkspace*)+4*sizeof(uint64_t)+
        2*sizeof(const std::vector<std::string>*)+2*sizeof(WeightTable*)+
        2*sizeof(std::shared_ptr<const VerifiedDriverBindings>)+
        2*sizeof(const NativeInventoryLimits*)+2*sizeof(int64_t)+
        3*sizeof(size_t)+3*sizeof(bool)+sizeof(std::string_view)+
        2*sizeof(std::map<std::string,WeightRef>::const_iterator)+
        // Actual scan/value/read/text/constructor forwarding representations,
        // with three value frames plus the rejected depth-three entry.
        sizeof(strata::GgufDirectoryScan*)+sizeof(strata::GgufWorkspaceQuote*)+sizeof(void*)+sizeof(uint64_t)+
        4*(sizeof(strata::GgufDirectoryScan*)+sizeof(uint32_t)+sizeof(unsigned)+sizeof(uint64_t*))+
        sizeof(strata::GgufDirectoryScan*)+sizeof(void*)+sizeof(uint64_t)+2*sizeof(size_t)+sizeof(ssize_t)+
        sizeof(strata::GgufDirectoryScan*)+sizeof(void*)+
        sizeof(strata::GgufDirectoryScan*)+sizeof(strata::GgufWorkspaceSlice*)+sizeof(uint64_t)+
        sizeof(strata::GgufDirectoryScan*)+sizeof(strata::GgufWorkspaceSlice)+sizeof(std::string_view)+
        sizeof(strata::GgufWorkspaceDirectory*)+sizeof(strata::GgufDirectoryScan*)+sizeof(void*)+sizeof(uint64_t)+
        // Scan locals and concrete sort functors (not guessed closure extents).
        4*sizeof(uint64_t)+4*sizeof(void*)+sizeof(bool)+
        // duplicate replay forwarding, bounded chunk length and comparison flag.
        sizeof(strata::GgufDirectoryScan*)+2*sizeof(uint64_t)+sizeof(strata::GgufWorkspaceSlice)+
        sizeof(size_t)+sizeof(bool)+
        sizeof(strata::GgufWorkspaceDirectory::SortLess)+sizeof(strata::GgufWorkspaceDirectory::SortSift)+
        sizeof(strata::GgufWorkspaceDirectory::SortSift*)+3*sizeof(uint64_t)+
        sizeof(strata::GgufWorkspaceDirectory::SortLess*)+2*sizeof(uint64_t)+sizeof(int)+
        // Error reporter captures are checked against these actual pointers
        // and native result/record forwarding requires no payload allocation.
        sizeof(void*)+2*sizeof(void*)+3*sizeof(void*)+sizeof(NativeInventoryPreflight*)+
        2*sizeof(uint64_t)+sizeof(ssize_t)+
        // Reporter raw/emit/name views, ordinary name-helper arguments and
        // the driver's added diagnostic-prefix argument are separately named.
        3*sizeof(std::string_view)+sizeof(std::string_view)+sizeof(uint32_t)+
        sizeof(bool)+sizeof(int)+sizeof(const char*)+
        // Two nested range-for loops have their own spans and begin/end
        // iterators even though both borrow the same immutable arena.
        2*sizeof(std::span<const strata::GgufWorkspaceTensor>)+
        4*sizeof(const strata::GgufWorkspaceTensor*);
#else
    return 0;
#endif
}
void NativeInventoryPreflight::report_error(const char* prefix) const noexcept {
#if defined(__linux__)
    if(okay_)return;
    using Code=strata::GgufWorkspaceCode;
    InventoryPreflightReportFrame f;
    auto raw=[](std::string_view text){
        while(!text.empty()){
            const auto n=::write(STDERR_FILENO,text.data(),text.size());
            if(n<0&&errno==EINTR)continue;
            if(n<=0)return;
            text.remove_prefix(size_t(n));
        }
    };
    auto emit=[&](std::string_view text){
        if(f.stopped)return;
        const auto zero=text.find('\0');
        if(zero!=std::string_view::npos){text=text.substr(0,zero);f.stopped=true;}
        raw(text);
    };
    auto name=[&]{
        f.offset=name_offset_;f.remaining=name_bytes_;
        while(f.remaining&&!f.stopped){
            const size_t n=size_t((std::min)(f.remaining,uint64_t(sizeof f.bytes)));
            ssize_t got;do{got=::pread(fd_,f.bytes,n,off_t(f.offset));}while(got<0&&errno==EINTR);
            if(got<=0)break;
            emit({f.bytes,size_t(got)});f.offset+=uint64_t(got);f.remaining-=uint64_t(got);
        }
    };
    static_assert(sizeof(raw)<=sizeof(void*));
    static_assert(sizeof(emit)<=2*sizeof(void*));
    static_assert(sizeof(name)<=3*sizeof(void*));
    if(prefix)raw(prefix);
    const auto code=Code(error_);
    if(code==Code::authority)emit("native inventory requires matching verified input bindings");
    else if(code==Code::quota)emit("verified input quota cannot admit GGUF preflight workspace");
    else if(code==Code::missing){emit("native inventory preflight missing canonical matrix: ");name();}
    else if(code==Code::conversion)emit("native BF16 conversion extent overflow");
    else if(code==Code::truncated)emit("native BF16 conversion truncated payload");
    else if(code==Code::overlap)emit("native BF16 conversion overlapping payload");
    else {
        emit("native inventory BF16 preflight: ");
        switch(code){
        case Code::io:emit("GGUF: header read failed");break;
        case Code::stat:emit("fstat failed");break;
        case Code::eof:emit("GGUF: unexpected end of file in header");break;
        case Code::magic:emit("not a GGUF file (bad magic)");break;
        case Code::version:{
            emit("GGUF v");const auto number=std::to_chars(f.number,f.number+sizeof f.number,argument_);
            emit({f.number,size_t(number.ptr-f.number)});emit(", this reader handles v3");break;}
        case Code::depth:emit("GGUF: array nesting too deep");break;
        case Code::array:emit("GGUF: implausible array length");break;
        case Code::type:emit("GGUF: unknown metadata value type");break;
        case Code::dimensions:emit("GGUF: bad n_dims for ");name();break;
        case Code::duplicate:emit("GGUF: duplicate tensor name '");name();emit("' in ");emit(path_);break;
        case Code::data:emit("GGUF: data section starts past EOF");break;
        case Code::overflow:emit("GGUF: preflight allocation extent overflow");break;
        case Code::allocation:emit("std::bad_alloc");break;
        default:emit("GGUF: invalid preflight result");break;
        }
    }
    raw("\n");
#else
    (void)prefix;
#endif
}
NativeInventoryPreflight NativeDense::inventory_conversion_limits_admitted(
    const std::vector<std::string>& paths,WeightTable& table,bool include_ple_key,
    const NativeInventoryLimits& limits,WindowBudget& budget,
    std::shared_ptr<const VerifiedDriverBindings> bindings,int64_t layer_lo,int64_t layer_hi) {
    NativeInventoryPreflight result;
#if defined(__linux__) && defined(STRATA_WINDOW_STATE_AVAILABLE)
    using Code=strata::GgufWorkspaceCode;
    // The actual driver dedicates the fixed baseline at verified-input setup,
    // before this result, scan or argument-matcher storage is constructed.
    if(!budget.gguf_setup_paid(inventory_preflight_setup_bytes())||!bindings||paths.size()!=bindings->dense.size()){
        result.error_=uint32_t(Code::authority);return result;
    }
    result.bindings_=std::move(bindings);result.limits_=limits;
    InventoryPreflightFrame frame;
    for(frame.source=0;frame.source<paths.size();++frame.source){
        const auto& input=result.bindings_->dense[frame.source];
        if(!input||input.readonly_fd()<0||!input.matches_loader_path(paths[frame.source])){
            result.error_=uint32_t(Code::authority);return result;
        }
    }
    auto failure=[&](const strata::GgufWorkspaceError& error){
        result.error_=uint32_t(error.code);result.argument_=error.argument;
        result.name_offset_=error.name.offset;result.name_bytes_=error.name.bytes;
    };
    static_assert(sizeof(failure)==sizeof(NativeInventoryPreflight*));
    // Actual synchronous header IO also uses the existing input-ledger IO slot.
    // It stays held through diagnostics; even a zero-byte directory has IO.
    if(budget.begin_gguf_scan(inventory_preflight_setup_bytes(),result.charge_)!=ReservationError::none){
        result.error_=uint32_t(Code::quota);return result;
    }
    // Count every shard before payload allocation; fixed scan objects use S,
    // never a future reservation or an allocating resource snapshot.
    for(frame.source=0;frame.source<paths.size();++frame.source){
        const auto& input=result.bindings_->dense[frame.source];
        result.fd_=input.readonly_fd();result.path_=paths[frame.source];
        if(::fstat(result.fd_,&frame.status)||frame.status.st_size<0||uint64_t(frame.status.st_size)!=input.size()){
            result.error_=uint32_t(Code::stat);return result;
        }
        strata::GgufDirectoryScan scan(result.fd_,input.size());
        if(!scan.scan(frame.quote)){failure(scan.error());return result;}
        result.requested_=(std::max)(result.requested_,frame.quote.bytes);
    }
    if(budget.admit_gguf_workspace(result.requested_,inventory_preflight_setup_bytes(),result.charge_)!=ReservationError::none){
        result.error_=uint32_t(Code::quota);return result;
    }
    if(result.requested_){
        ++result.allocation_attempts_;
        result.arena_=std::malloc(size_t(result.requested_));
        if(!result.arena_){result.error_=uint32_t(Code::allocation);return result;}
    }
    for(frame.source=0;frame.source<paths.size();++frame.source){
        const auto& input=result.bindings_->dense[frame.source];
        result.fd_=input.readonly_fd();result.path_=paths[frame.source];
        strata::GgufDirectoryScan scan(result.fd_,input.size());
        // Previous directory and every name/shape view end before arena reuse.
        ++result.parser_constructions_;
        strata::GgufWorkspaceDirectory directory(scan,result.arena_,result.requested_);
        if(!directory.valid()){failure(scan.error());return result;}
        for(const auto& tensor:directory.tensors()){
            if(!eligible(tensor,include_ple_key))continue;
            frame.layer=std::strtol(tensor.name.data()+4,nullptr,10);
            if((layer_hi>=0&&(frame.layer<layer_lo||frame.layer>=layer_hi))||
               (!in_range(tensor.name)&&tensor.name.find("ple")==std::string_view::npos))continue;
            for(frame.found=table.all().begin();frame.found!=table.all().end()&&
                std::string_view(frame.found->first)!=tensor.name;++frame.found){}
            if(frame.found==table.all().end()){
                result.error_=uint32_t(Code::missing);result.name_offset_=tensor.source_name.offset;
                result.name_bytes_=tensor.source_name.bytes;return result;
            }
            if(!bf16_as_q8_0(tensor,frame.found->second))continue;
            frame.elements=uint64_t(frame.found->second.ne0)*uint64_t(frame.found->second.ne1);
            if(frame.elements>UINT64_MAX/2||frame.elements/32>UINT64_MAX/34||
               frame.elements/32*34>SIZE_MAX||directory.data_start()>input.size()){
                result.error_=uint32_t(Code::conversion);return result;
            }
            frame.payload=input.size()-directory.data_start();
            if(tensor.offset>frame.payload||frame.elements*2>frame.payload-tensor.offset){
                result.error_=uint32_t(Code::truncated);return result;
            }
            for(const auto& other:directory.tensors())
                if(other.offset>tensor.offset&&frame.elements*2>other.offset-tensor.offset){
                    result.error_=uint32_t(Code::overlap);return result;
                }
            frame.maximum=(std::max)(frame.maximum,frame.elements/32*34);
        }
    }
    result.limits_.max_bf16_q8_bytes=frame.maximum;
    result.limits_.bf16_control_bytes=frame.maximum?inventory_bf16_control_bytes():0;
    // Administrative inventory bounds are still validated by the unchanged
    // final native_inventory_host_bound AFTER this owner is destroyed.
    result.okay_=true;
#else
    (void)paths;(void)table;(void)include_ple_key;(void)limits;(void)budget;
    (void)bindings;(void)layer_lo;(void)layer_hi;
#endif
    return result;
}

bool NativeDense::inventory_conversion_limits(const std::vector<std::string>& shards,WeightTable& table,
    bool include_ple_key,NativeInventoryLimits& limits,std::string& error,int64_t layer_lo,int64_t layer_hi) {
    try {
        uint64_t maximum=0;
        for(const auto& path:shards) {
            strata::GgufFile file(path);
            for(const auto& tensor:file.tensors()) {
                if(!eligible(tensor,include_ple_key))continue;
                const long layer=std::strtol(tensor.name.c_str()+4,nullptr,10);
                if((layer_hi>=0&&(layer<layer_lo||layer>=layer_hi)) ||
                   (!in_range(tensor.name)&&tensor.name.find("ple")==std::string::npos))continue;
                const auto found=table.table_.find(tensor.name);
                if(found==table.table_.end()) {error="native inventory preflight missing canonical matrix: "+tensor.name;return false;}
                if(!bf16_as_q8_0(tensor,found->second))continue;
                const uint64_t n=uint64_t(found->second.ne0)*uint64_t(found->second.ne1);
                if(n>UINT64_MAX/2 || n/32>UINT64_MAX/34 || n/32*34>SIZE_MAX ||
                   file.data_start()>file.file_size()) {error="native BF16 conversion extent overflow";return false;}
                const uint64_t payload=file.file_size()-file.data_start();
                if(tensor.offset>payload||n*2>payload-tensor.offset) {error="native BF16 conversion truncated payload";return false;}
                for(const auto& other:file.tensors())
                    if(other.offset>tensor.offset&&n*2>other.offset-tensor.offset) {
                        error="native BF16 conversion overlapping payload";return false;
                    }
                maximum=(std::max)(maximum,n/32*34);
            }
        }
        auto candidate=limits;candidate.max_bf16_q8_bytes=maximum;
        candidate.bf16_control_bytes=maximum?inventory_bf16_control_bytes():0;
        uint64_t bound=0;if(!native_inventory_host_bound(candidate,bound,error))return false;
        limits=candidate;return true;
    } catch(const std::exception& e) {error=std::string("native inventory BF16 preflight: ")+e.what();return false;}
}

// The byte walk `load` does for the layers [lb, le), without the allocations: the same filters in the same order,
// reading the GGUF headers and the canonical table only, so it needs no device.  A later stage's projections are not
// on the card when the split search prices that stage, and `load` runs one cudaMalloc per matrix (#1238).  The two
// have to stay in step.  A repeated name is skipped rather than refused (`load` refuses it).
bool NativeDense::weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                   int64_t lb, int64_t le, uint64_t& out, std::string& err) {
    out = 0;
    try {
        std::set<std::string> seen;
        uint64_t total = 0;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key)) continue;
                if (tensor.name.rfind("blk.", 0) == 0) {
                    const long l = std::strtol(tensor.name.c_str() + 4, nullptr, 10);
                    if (l < lb || l >= le) continue;
                }
                if (!seen.insert(tensor.name).second) continue;
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name;
                    return false;
                }
                const auto& ref = found->second;
                if (bf16_as_q8_0(tensor, ref)) {   // #4741
                    total += alloc_bytes((uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1);
                    continue;
                }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name;
                    return false;
                }
                total += alloc_bytes(strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1));
            }
        }
        out = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

// STRATA_HC_Q8 without a bf16 copy: the two dequantization scratches (one bf16 matrix each) of `hc_bf16_view`, per device,
// allocated by `load` - never on first use, so a graph capture cannot meet the allocation
static void* g_hc_scratch[64][2][2] = {};     // [device][class][slot]
static cudaEvent_t g_hc_ev[64][2] = {};        // [device][slot]: the prompt path's last reader of the slot
static bool g_hc_ev_set[64][2] = {};

static std::atomic<NativeLoadInventory*> failed_inventory{nullptr};
bool NativeDense::enable_inventory_capture(const NativeInventoryLimits& limits,
    std::shared_ptr<const ReservationLease> reservation,std::shared_ptr<const void> inputs,std::string& error) {
    uint64_t required=0;
    if(load_attempted_||release_started_||inventory_) {error="native inventory must be enabled once before load";return false;}
    if((limits.max_bf16_q8_bytes&&limits.bf16_control_bytes!=inventory_bf16_control_bytes()) ||
       (!limits.max_bf16_q8_bytes&&limits.bf16_control_bytes)) {
        error="native inventory BF16 control quote mismatch";return false;
    }
    if(!native_inventory_host_bound(limits,required,error))return false;
    if(!reservation||reservation->resources().host_bytes<required||!inputs||
       reservation->purpose()!=ReservationPurpose::foreground) {error="native inventory requires admitted host envelope and retained input owner";return false;}
    try {
        auto candidate=std::make_unique<NativeLoadInventory>();
        candidate->metadata_reservation=std::move(reservation);candidate->retained_input_bindings=std::move(inputs);
        candidate->limits=limits;candidate->sources.reserve(size_t(limits.max_sources));candidate->records.reserve(size_t(limits.max_records));
        inventory_=std::move(candidate);return true;
    }catch(const std::bad_alloc&){error="native inventory allocation failed";return false;}
}
NativeDense::~NativeDense() {
    if(inventory_&&!release_started_&&!release_failed_) {
        try {std::string error;if(!release_checked(error))std::fprintf(stderr,"native inventory retirement: %s\n",error.c_str());}
        catch(...) {release_started_=release_failed_=true;std::fprintf(stderr,"native inventory retirement uncertain\n");}
    }
    if(release_failed_) {
        // Explicit failed retirement may never release its input authority. No
        // allocation/owner destruction on this uncertainty path.
        if(auto* p=inventory_.release()) {
            auto* head=failed_inventory.load();do {p->quarantined_next=head;}while(!failed_inventory.compare_exchange_weak(head,p));
        }
        return;
    } // uncertain explicit retirement must never retry frees
    for (auto& dev : g_hc_scratch)
        for (auto& cl : dev)
            for (void*& b : cl)
                if (b != nullptr && std::find(weights_.begin(), weights_.end(), b) != weights_.end()) b = nullptr;
    for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::owns_weight_span(const void* base,uint64_t bytes) const {
    if(release_started_||!weight_spans_valid_||!base||!bytes)return false;
    for(const auto& span:weight_spans_)if(span.first==base&&bytes<=span.second)return true;
    return false;
}

bool NativeDense::release_checked(std::string& error) {
    if(release_failed_&&!release_started_) {
        release_started_=true;error="native inventory load cleanup failed; worker quarantine required";return false;
    }
    if(release_started_) {
        if(release_failed_)error="native model retirement failed; worker quarantine required";
        return !release_failed_;
    }
    release_started_=true;release_failed_=true;
    // From here the object cannot execute again, including after partial release.
    for(const void* p:packed_keys_)strata::kernels::native_q8_0_packed_unregister(p);
    packed_keys_.clear();
    auto free_one=[&](void*& p) {
        if(!p)return true;
        const auto status=cudaFree(p);
        if(status!=cudaSuccess) {
            error=std::string("native model release failed: ")+cudaGetErrorString(status);return false;
        }
        for(auto& dev:g_hc_scratch)for(auto& cl:dev)for(void*& b:cl)if(b==p)b=nullptr;
        p=nullptr;return true;
    };
    if(!free_one(scratch_))return false;
    for(void*& p:weights_)if(!free_one(p))return false;
    weights_.clear();bytes_=0;allocated_=0;release_failed_=false;return true;
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi) {
    if(release_started_){err="retired native model cannot be loaded again";return false;}
    if(load_attempted_){err="native model load already attempted; use a fresh owner";return false;}
    load_attempted_=true;
    weight_spans_valid_=false;
    auto outside = [&](const std::string& name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        InventoryUnwindScope unwind_scope(inventory_?&release_failed_:nullptr);
        if(inventory_) {
            const auto& limits=inventory_->limits;
            if(shards.size()>limits.max_sources){err="native inventory source count bound";return false;}
            for(const auto& path:shards)if(path.size()>limits.max_string_bytes){err="native inventory source string bound";return false;}
            for(const auto& path:shards)inventory_->sources.push_back(path);
            inventory_->layer_lo=layer_lo;inventory_->layer_hi=layer_hi;
            inventory_->global_layer_lo=g_layer_lb;inventory_->global_layer_hi=g_layer_le;
            inventory_->include_ple_key=include_ple_key;
        }
        auto append_inventory=[&](const strata::TensorInfo* tensor,const strata::GgufFile* file,
            size_t ordinal,const std::string& name,const WeightRef& ref,uint64_t runtime_bytes,
            NativeLoadLayout layout,NativeLoadConversion conversion)->size_t {
            if(!inventory_)return SIZE_MAX;
            const auto& limits=inventory_->limits;
            if(inventory_->records.size()>=limits.max_records||name.size()>limits.max_string_bytes||
               (tensor?tensor->shape.size():size_t(2))>limits.max_rank)
                throw std::runtime_error("native inventory record/string/shape bound");
            NativeTensorLoadRecord record;record.name=name;record.runtime_bytes=runtime_bytes;
            record.runtime_layout=layout;record.conversion=conversion;
            if(tensor) {
                record.source_ordinal=ordinal;record.source_type=tensor->type;record.source_shape=tensor->shape;
                if(tensor->offset>UINT64_MAX-file->data_start())throw std::runtime_error("native inventory absolute offset overflow");
                record.source_offset=file->data_start()+tensor->offset;record.source_bytes=strata::tensor_payload_bytes(*tensor);
            } else {
                record.source=NativeLoadSource::unresolved_canonical;record.source_shape={uint64_t(ref.ne0),uint64_t(ref.ne1)};
                record.source_offset=ref.src_off;record.source_bytes=ref.src_bytes;
                record.canonical={ref.file_id,int64_t(ref.kind),ref.code_bits,ref.code_bias,ref.group_elems,ref.act_kind,
                    ref.src_off,ref.src_bytes,ref.codes_bytes,ref.scales_bytes,ref.offset_bytes,
                    ref.scales_fp16,ref.has_offset,ref.codebook_iq4nl};
                inventory_->has_unresolved_canonical=true;
            }
            inventory_->records.push_back(std::move(record));return inventory_->records.size()-1;
        };
        std::vector<Pending> pending;
        // At most one HC allocation per table entry, plus four shared HC scratch
        // buffers. Reserve ownership metadata before any such device allocation.
        if(table.table_.size()>weights_.max_size()-4)
            throw std::length_error("native HC ownership metadata overflow");
        weights_.reserve(table.table_.size()+4);
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0, allocated = 0, hc_q8_skipped_bytes = 0, hc_q8_skipped = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (size_t source_ordinal=0;source_ordinal<shards.size();++source_ordinal) {
            const auto& path=shards[source_ordinal];
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture") && (!number || number->u == 0)) {
                // splitter may copy general.architecture into every shard (Huihui abliterated re-split)
                err = strata::check_engine_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key) && !outside(tensor.name) &&
                    strata::kernels::native_mmvq_supported(tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (bf16_as_q8_0(tensor, ref)) {   // #4741: a BF16 projection of the pack, served as Q8_0
                    const uint64_t n = (uint64_t) ref.ne0 * (uint64_t) ref.ne1;
                    if(inventory_) {
                        const uint64_t bytes=n/32*34;
                        if(bytes>SIZE_MAX||bytes>inventory_->limits.max_bf16_q8_bytes) {
                            err="native BF16 conversion exceeds preadmitted stage peak";return false;
                        }
                        // Actual source/runtime receipt precedes conversion and
                        // device allocation. Partial receipts are never public.
                        append_inventory(&tensor,&gguf,source_ordinal,tensor.name,ref,bytes,
                            NativeLoadLayout::transformed_q8_0,NativeLoadConversion::bf16_to_q8_0_host);
                        int owner=-1;
                        if(cudaGetDevice(&owner)!=cudaSuccess) {err="native BF16 upload owning device unavailable";return false;}
                        NativeBf16Q8Control conversion;
                        conversion.storage.reset(static_cast<uint8_t*>(::operator new(size_t(bytes))));
                        conversion.source=(const uint16_t*)gguf.tensor_data(tensor);
                        conversion.output=conversion.storage.get();conversion.end=n/32;
                        q8_0_bf16_blocks(conversion);
                        void* allocation=nullptr;
                        auto status=cudaMalloc(&allocation,size_t(bytes));
                        DevicePtr data(allocation); // freed before the paid conversion buffer
                        if(status==cudaSuccess) {
                            status=cudaMemcpy(data.get(),conversion.output,size_t(bytes),cudaMemcpyHostToDevice);
                            // Even an attempted failed copy can have submitted
                            // work. Never unwind its host source without a real
                            // checked drain on exactly the original device.
                            int current=-1;
                            if(cudaGetDevice(&current)!=cudaSuccess||current!=owner||
                               cudaDeviceSynchronize()!=cudaSuccess) {
                                release_failed_=true;std::fputs("native BF16 upload retirement uncertain; ownership retained\n",stderr);
                                std::_Exit(1);
                            }
                        }
                        if(status!=cudaSuccess) {
                            err="native dense BF16 upload "+tensor.name+": "+cudaGetErrorString(status);return false;
                        }
                        max_in=(std::max)(max_in,(int)ref.ne0);total+=bytes;
                        add_allocation(allocated,bytes);
                        pending.push_back(Pending{&ref,8,bytes,std::move(data),DevicePtr()});
                        continue;
                    }
                    const std::vector<uint8_t> q8 = q8_0_of_bf16((const uint16_t*) gguf.tensor_data(tensor), n);
                    void* allocation = nullptr;
                    auto status = cudaMalloc(&allocation, q8.size());
                    DevicePtr data(allocation);
                    if (status == cudaSuccess) status = cudaMemcpy(data.get(), q8.data(), q8.size(), cudaMemcpyHostToDevice);
                    if (status != cudaSuccess) {
                        err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                    }
                    max_in = (std::max)(max_in, (int) ref.ne0);
                    total += q8.size();
                    allocated += alloc_bytes(q8.size());
                    std::fprintf(stderr, "native dense: %s is BF16 in the file: served as Q8_0 like its siblings (#4741)\n",
                                 tensor.name.c_str());
                    pending.push_back(Pending{&ref, 8, (uint64_t) q8.size(), std::move(data), DevicePtr()});
                    continue;
                }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                const auto receipt=append_inventory(&tensor,&gguf,source_ordinal,tensor.name,ref,bytes,
                    NativeLoadLayout::gguf_native,NativeLoadConversion::identity);
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), gguf.tensor_data(tensor), bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                add_allocation(allocated,bytes);
                // STRATA_Q8_PACKED=1: a second, packed copy of an eligible Q8_0 matrix for the decode MMVQ (the GGUF
                // copy stays: the prompt path's GEMMs read it).
                DevicePtr packed;
                if (tensor.type == 8 && tensor.name != "blk.1.ple_key.weight" &&   // the PLE has its own kernel
                    strata::kernels::native_q8_0_packed_enabled() &&
                    strata::kernels::native_q8_0_packed_eligible((int) ref.ne0, (int) ref.ne1)) {
                    std::vector<uint8_t> host(bytes);
                    strata::kernels::native_q8_0_pack_host(gguf.tensor_data(tensor), host.data(), (int) ref.ne0,
                                                           (int) ref.ne1);
                    void* packed_allocation = nullptr;
                    auto packed_status = cudaMalloc(&packed_allocation, bytes);
                    packed.reset(packed_allocation);
                    if (packed_status == cudaSuccess)
                        packed_status = cudaMemcpy(packed.get(), host.data(), bytes, cudaMemcpyHostToDevice);
                    if (packed_status != cudaSuccess) {
                        err = "native dense packed upload " + tensor.name + ": " + cudaGetErrorString(packed_status);
                        return false;
                    }
                    add_allocation(allocated,bytes);
                }
                if(inventory_&&packed)inventory_->records[receipt].q8_packed_helper_bytes=bytes;
                if (strata::core::manifest::enabled()) {   // STRATA_MANIFEST: the dense formats as held, and the packed copies
                    strata::core::manifest::instance().tensors("dense", strata::ggml_type_name((uint32_t) tensor.type), 1, bytes);
                    if (packed) strata::core::manifest::instance().tensors("dense_packed_copy", "Q8_0", 1, bytes);
                }
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data), std::move(packed)});
            }
            if (strata::core::manifest::enabled()) {
                const char* v = std::getenv("STRATA_HC_Q8");
                strata::core::manifest::instance().path("dense", "hc_q8", hc_q8_env(),
                    hc_q8_env() ? (v != nullptr && v[0] == '1' ? "STRATA_HC_Q8=1" : "architecture default (gfx12 / gfx1151 HIP)")
                                : (v != nullptr && v[0] != '\0' && std::strcmp(v, "auto") != 0 ? "STRATA_HC_Q8 turned it off"
                                                                                               : "off by default on this backend"));
            }
            if (hc_q8_env())
                for (const auto& tensor : gguf.tensors()) {
                    // S25: the F32 inject rows hold BF16-exact values: the read takes the pack's BF16 rows unless
                    // STRATA_HC_Q8_INJECT=1 asks for a Q8_0 copy
                    static const bool q8_inject = [] { const char* v = std::getenv("STRATA_HC_Q8_INJECT"); return v && v[0] == '1'; }();
                    const bool f32_inject = q8_inject && tensor.type == 0 && tensor.name.ends_with("_inject.weight");
                    if ((tensor.type != 8 && !f32_inject) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2) continue;
                    // a trimmed layer split: another stage's layers are never read here, so their rows are not copied (the upload
                    // loop above and the bf16 pack path below already filter this way; PR #3559, havok2-htwo).
                    // STRATA_HC_Q8_ALL_STAGES=1 copies them as before.
                    if (hc_q8_stage_local() && (outside(tensor.name) || !in_range(tensor.name))) {
                        hc_q8_skipped_bytes += (uint64_t) tensor.shape[0] / 32 * 34 * (uint64_t) tensor.shape[1];
                        ++hc_q8_skipped;
                        continue;
                    }
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 != nullptr) continue;
                    auto& ref = found->second;
                    if (tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 % 32 != 0) {
                        err = "native dense (STRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                    }
                    const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                    append_inventory(&tensor,&gguf,source_ordinal,tensor.name,ref,bytes,NativeLoadLayout::hc_q8_0,
                        f32_inject?NativeLoadConversion::f32_to_q8_0_host:NativeLoadConversion::identity);
                    void* p = nullptr;
                    std::vector<uint8_t> q8;
                    if (f32_inject) q8 = q8_0_of((const float*) gguf.tensor_data(tensor), (uint64_t) ref.ne0 * (uint64_t) ref.ne1);
                    if(cudaMalloc(&p,bytes)!=cudaSuccess) {
                        err="native dense (STRATA_HC_Q8) allocation "+tensor.name;return false;
                    }
                    weights_.push_back(p); // pre-reserved; owns p even if upload fails
                    add_allocation(allocated,bytes);
                    if(cudaMemcpy(p,f32_inject?static_cast<const void*>(q8.data()):gguf.tensor_data(tensor),
                                  bytes,cudaMemcpyHostToDevice)!=cudaSuccess) {
                        err="native dense (STRATA_HC_Q8) upload "+tensor.name;return false;
                    }
                    ref.hc_q8=p;
                    hc_q8_bytes+=bytes;
                    strata::core::manifest::instance().tensors("dense_hc_q8_copy", f32_inject ? "F32->Q8_0" : "Q8_0", 1, bytes);
                }
        }
        if (hc_q8_env()) {
            // a bf16 pack stores no Q8_0 copy: quantize the layers' down / up projections. A row the arena kept
            // (--keep-canonical, a layer split) is quantized on the GPU from the arena's bf16; a skipped row (the
            // single-GPU default: add_hc_skip) from the pack file on the host, and has no bf16 copy at all. The
            // inject rows stay bf16 (82 KB), the final mixer stays bf16 (it runs once a window).
            uint64_t made = 0;
            for (auto& [name, ref] : table.table_) {
                if (ref.hc_q8 != nullptr || name.rfind("blk.", 0) != 0 || !in_range(name) ||
                    !hc_q8_name(name) || !(name.ends_with("_down.weight") || name.ends_with("_up.weight"))) continue;
                const bool skipped = ref.data == nullptr;
                if (skipped && ref.resident) continue;
                if (ref.ne0 % 32 != 0 || ref.elements != ref.ne0 * ref.ne1 || (!skipped && ref.bytes != (uint64_t) ref.elements * 2)) {
                    err = "native dense (STRATA_HC_Q8): " + name + " is not a bf16 matrix"; return false;
                }
                const uint64_t nb = (uint64_t) ref.elements / 32, bytes = nb * 34;
                append_inventory(nullptr,nullptr,0,name,ref,bytes,NativeLoadLayout::hc_q8_0,
                    skipped?NativeLoadConversion::bf16_to_q8_0_host:NativeLoadConversion::bf16_to_q8_0_device);
                void* p = nullptr;
                if (cudaMalloc(&p, bytes) != cudaSuccess) { err = "native dense (STRATA_HC_Q8) alloc " + name; return false; }
                weights_.push_back(p);
                if (skipped) {
                    std::vector<uint16_t> host;
                    if (!table.read_bf16_host(name, host, err)) return false;
                    const std::vector<uint8_t> q8 = q8_0_of_bf16(host.data(), (uint64_t) host.size());
                    if (cudaMemcpy(p, q8.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = "native dense (STRATA_HC_Q8) upload " + name; return false;
                    }
                } else {
                    strata::kernels::fused_gr_quantize_q8_0((const uint16_t*) ref.data, (uint8_t*) p, (int64_t) nb, nullptr);
                }
                ref.hc_q8 = p;
                hc_q8_bytes += bytes;
                add_allocation(allocated,bytes);
                ++made;
            }
            if (made && (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)) {
                err = "native dense (STRATA_HC_Q8): quantizing the projections failed"; return false;
            }
        }
        if (hc_q8_env()) {   // the scratches of hc_bf16_view (rows without a bf16 copy), here and not on first use
            uint64_t elems = 0;
            for (const auto& [name, ref] : table.table_)
                if (ref.hc_q8 != nullptr && ref.data == nullptr) elems = (std::max)(elems, (uint64_t) ref.elements);
            int dev = 0;
            if (elems != 0 && cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && g_hc_scratch[dev][0][0] == nullptr) {
                for (int c = 0; c < 2; ++c)
                    for (int i = 0; i < 2; ++i) {
                        void* b = nullptr;
                        if (cudaMalloc(&b, (size_t) elems * 2) != cudaSuccess) { err = "native dense (STRATA_HC_Q8): scratch alloc"; return false; }
                        weights_.push_back(b);
                        add_allocation(allocated,(uint64_t)elems*2);
                        g_hc_scratch[dev][c][i] = b;
                    }
                for (int i = 0; i < 2; ++i)
                    if (g_hc_ev[dev][i] == nullptr && cudaEventCreateWithFlags(&g_hc_ev[dev][i], cudaEventDisableTiming) != cudaSuccess) {
                        err = "native dense (STRATA_HC_Q8): scratch event"; return false;
                    }
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        add_allocation(allocated,strata::kernels::native_q8_1_bytes(max_in));
        // All checks and allocations finish before publishing any reference.
        const size_t packed_pending=static_cast<size_t>(std::count_if(pending.begin(),pending.end(),
            [](const Pending& item){return bool(item.packed);}));
        if(pending.size()>weights_.max_size()-weights_.size() ||
           packed_pending>weights_.max_size()-weights_.size()-pending.size() ||
           pending.size()>weight_spans_.max_size()-weight_spans_.size() ||
           packed_pending>packed_keys_.max_size()-packed_keys_.size())
            throw std::length_error("native dense ownership metadata overflow");
        weights_.reserve(weights_.size()+pending.size()+packed_pending);
        packed_keys_.reserve(packed_keys_.size()+packed_pending);
        weight_spans_.reserve(weight_spans_.size()+pending.size());
        uint64_t packed_bytes = 0;
        size_t packed_count = 0;
        for (auto& item : pending) {
            const void* data=item.data.get();
            const void* packed=item.packed.get();
            // Reserve above means these inserts cannot allocate. Keep RAII until
            // each pointer is recorded, even if this code later changes capacity.
            if(packed) {
                weights_.push_back(item.packed.get());
                (void)item.packed.release();
            }
            weights_.push_back(item.data.get());
            (void)item.data.release();
            weight_spans_.emplace_back(data,item.bytes);
            if(packed) {
                // Registry insertion may allocate. Its cleanup identity and both
                // device allocations must already belong to this object.
                packed_keys_.push_back(data);
                strata::kernels::native_q8_0_packed_register(data,packed,(int)item.ref->ne0,(int)item.ref->ne1);
                packed_bytes+=item.bytes;
                ++packed_count;
            }
            item.ref->native_data=data;
            item.ref->native_type=item.type;
            item.ref->native_q8_1=scratch.get();
        }
        if (packed_count)
            std::fprintf(stderr, "native dense: STRATA_Q8_PACKED=1 packed %zu Q8_0 matrices for decode (+%.1f MiB)\n",
                         packed_count, packed_bytes / 1048576.0);
        scratch_ = scratch.release();
        bytes_ = total + hc_q8_bytes;
        allocated_ = allocated;
        weight_spans_valid_=true;
        inventory_complete_=bool(inventory_);
        if (hc_q8_skipped != 0 && strata::core::manifest::enabled())   // STRATA_MANIFEST: what the stage filter did not copy
            strata::core::manifest::instance().tensors("dense_hc_q8_out_of_stage_not_copied", "Q8_0", hc_q8_skipped, hc_q8_skipped_bytes);
        if (hc_q8_env())
            std::fprintf(stderr, "strata: STRATA_HC_Q8=1: %.2f GiB of Q8_0 hyper-connection projections for the verify read\n",
                         (double) hc_q8_bytes / 1073741824.0);
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::gguf_carries_q8_hc(const std::vector<std::string>& shards) {
    try {
        for (const auto& path : shards) {
            const strata::GgufFile g(path);
            for (const auto& tensor : g.tensors())
                if (tensor.type == 8 && tensor.shape.size() == 2 && hc_q8_name(tensor.name) && !tensor.name.ends_with("_inject.weight")) return true;
        }
    } catch (const std::exception&) {}
    return false;
}

bool NativeDense::hc_q8_requested() { return hc_q8_env(); }

bool NativeDense::add_hc_skip(const std::string& pack_dir, std::set<std::string>& skip) {
    std::FILE* f = std::fopen((pack_dir + "/index.txt").c_str(), "rb");
    if (!f) return false;
    char line[1024], name[256];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
        const std::string n = name;
        if (n.rfind("blk.", 0) == 0 && hc_q8_name(n) && (n.ends_with("_down.weight") || n.ends_with("_up.weight"))) skip.insert(n);
    }
    std::fclose(f);
    return true;
}

const uint16_t* NativeDense::hc_bf16_view(const WeightRef& w, void* stream, int slot, int cls) {
    if (w.data != nullptr) return (const uint16_t*) w.data;
    if (w.hc_q8 == nullptr) return nullptr;
    // the device's scratch (allocated at load, one pair per device): the prompt path and the single-token read run one
    // after the other on a device, and each use is stream-ordered after the previous GEMM that read the slot
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || slot < 0 || slot > 1 || cls < 0 || cls > 1) return nullptr;
    void* buf = g_hc_scratch[dev][cls][slot];
    if (buf == nullptr) return nullptr;
    // the prompt path: the scratch may still be read by a GEMM another stream enqueued (batch serving: several slots' prompts)
    if (cls == 0 && g_hc_ev_set[dev][slot]) cudaStreamWaitEvent((cudaStream_t) stream, g_hc_ev[dev][slot], 0);
    strata::kernels::fused_gr_dequant_q8_0((const uint8_t*) w.hc_q8, (uint16_t*) buf, w.elements / 32, stream);
    return (const uint16_t*) buf;
}

void NativeDense::hc_bf16_release(void* stream, int slot, int cls) {
    int dev = 0;
    if (cls != 0 || slot < 0 || slot > 1 || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || g_hc_ev[dev][slot] == nullptr) return;
    cudaEventRecord(g_hc_ev[dev][slot], (cudaStream_t) stream);
    g_hc_ev_set[dev][slot] = true;
}
} // namespace strata::core
