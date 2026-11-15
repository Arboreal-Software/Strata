// src/core/weights.cpp - the dense-weight loader.  See the header for the engine-vs-pack distinction.
#include "strata/core/weights.hpp"
#include <vector>
#include <cmath>

#include "strata/kernels/f16_bits.hpp"
#include "strata/platform/memory.hpp"

#include <cinttypes>
#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <atomic>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {

constexpr uint64_t CHUNK = 8ull << 20;   // 8 MiB of SOURCE per staging round

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct IndexRow {
    std::string name;
    int file = 0;
    WeightKind kind = WeightKind::Verbatim;
    uint64_t src_off = 0, src_bytes = 0, dst_off = 0, dst_bytes = 0;
    int64_t ne0 = 0, ne1 = 0;
    int code_bits = 0, code_bias = 0, group_elems = 0, codebook = 0, has_offset = 0;
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    int scales_fp16 = 0;
    int act_kind = 0;
    bool raw16 = false;   ///< plan v0.3 P6: already 16-bit in the file (kinds 4/5)
};

/// How a segment's bytes must be transformed on the way from the pack to the arena.
enum class Conv : int {
    Copy = 0,      ///< the bytes are already the engine form
    Bf16High = 1,  ///< f32 container holding a bf16 value: take the high 16 bits
    ToF16 = 2,     ///< f32 container holding an f16 value: a real round-to-nearest-even conversion
    WidenF16 = 3,  ///< fp16 bits to be held as f32: EXACT, and the reason this path exists at all
};

/// One contiguous run of a tensor's source span with a single destination range and one transform.
///
/// THE SEGMENT LIST IS WHY THIS LOADER CAN CHANGE WIDTHS PLANE BY PLANE.  A quantized tensor used to be one
/// verbatim span, which is only correct when every plane is already in engine form - and for 90 of the 303
/// quantized tensors the scale plane is fp16 in the pack and must be f32 in the arena.
struct Seg {
    uint64_t src_off = 0;   ///< relative to the row's src_off
    uint64_t src_bytes = 0;
    uint64_t dst_off = 0;   ///< relative to the row's dst_off
    uint64_t dst_bytes = 0;
    Conv conv = Conv::Copy;
};

const char* file_name(int id) {
    switch (id) {
        case 0: return "dense.bin";
        case 1: return "embd.bin";
        case 2: return "experts.bin";
        case 3: return "extra.bin";   // plan v0.3 P6: a native pack's float tensors that the base pack quantized
        default: return nullptr;
    }
}

/// UTF-16 path for CreateFile-free portability: the loader uses `std::fopen`, which on Windows takes an ANSI
/// path.  The pack lives beside the executable in practice, but a pack under a user directory with a
/// non-ASCII name would silently fail to open, so the caller gets the errno rather than a null pointer.
bool read_at(std::FILE* f, uint64_t off, void* dst, size_t n, std::string& err, const char* what) {
#if defined(_WIN32)
    if (_fseeki64(f, (long long) off, SEEK_SET) != 0) { err = std::string(what) + ": seek failed"; return false; }
#else
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) { err = std::string(what) + ": seek failed"; return false; }
#endif
    if (std::fread(dst, 1, n, f) != n) { err = std::string(what) + ": short read"; return false; }
    return true;
}

}  // namespace

bool WeightTable::pool_bytes(const std::string& pack_dir, uint64_t& out, std::string& err,
                             const std::set<std::string>* skip) {
    const std::string path = pack_dir + "/index.txt";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char line[1024];
    out = 0;
    uint64_t pool = 0, compact = 0;
    int align = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, tensors = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &tensors) == 3) { pool = p; align = a; }
            continue;
        }
        if (skip == nullptr) continue;
        char name[256] = {0};
        unsigned long long dst_bytes = 0, dummy = 0;
        int i1 = 0, i2 = 0;
        if (std::sscanf(line, "%255s %d %d %llu %llu %llu %llu", name, &i1, &i2, &dummy, &dummy, &dummy, &dst_bytes) != 7)
            continue;
        if (skip->count(name)) continue;
        const uint64_t a = align > 0 ? (uint64_t) align : 256;
        compact += (dst_bytes + a - 1) / a * a;
    }
    std::fclose(f);
    if (pool == 0) { err = "no '# align ... pool ...' header in " + path; return false; }
    out = skip ? compact : pool;
    return true;
}

bool WeightTable::index_code_bits(const std::string& pack_dir, const std::string& name, int& code_bits,
                                  std::string& err) {
    const std::string path = pack_dir + "/index.txt";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char line[1024];
    code_bits = -1;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        char row[256] = {0};
        unsigned long long dummy = 0;
        long long ne = 0;
        int i1 = 0, i2 = 0, bits = 0;
        // the first ten fields of the row `load` parses: name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits
        if (std::sscanf(line, "%255s %d %d %llu %llu %llu %llu %lld %lld %d", row, &i1, &i2, &dummy, &dummy, &dummy,
                        &dummy, &ne, &ne, &bits) != 10)
            continue;
        if (name == row) { code_bits = bits; break; }
    }
    std::fclose(f);
    return true;
}

bool WeightTable::read_bf16_host(const std::string& name, std::vector<uint16_t>& out, std::string& err) const {
    const auto it = table_.find(name);
    if (it == table_.end()) { err = name + ": no such tensor"; return false; }
    const WeightRef& r = it->second;
    const char* fn = file_name(r.file_id);
    if (fn == nullptr || r.kind != WeightKind::Bf16InF32) { err = name + ": not a bf16 row of a pack file"; return false; }
    const uint64_t elems = (uint64_t) r.elements;
    const bool raw16 = r.src_bytes == elems * 2;
    if (!raw16 && r.src_bytes != elems * 4) { err = name + ": unexpected source size"; return false; }
    std::FILE* f = std::fopen((pack_dir_ + "/" + fn).c_str(), "rb");
    if (!f) { err = "cannot open " + pack_dir_ + "/" + fn; return false; }
    out.resize((size_t) elems);
    bool ok = true;
    if (raw16) {
        ok = read_at(f, r.src_off, out.data(), (size_t) (elems * 2), err, name.c_str());
    } else {
        std::vector<uint32_t> tmp((size_t) std::min<uint64_t>(elems, 1u << 22));
        for (uint64_t done = 0; ok && done < elems;) {
            const uint64_t n = std::min<uint64_t>(tmp.size(), elems - done);
            ok = read_at(f, r.src_off + done * 4, tmp.data(), (size_t) (n * 4), err, name.c_str());
            for (uint64_t i = 0; ok && i < n; ++i) out[(size_t) (done + i)] = (uint16_t) (tmp[(size_t) i] >> 16);
            done += n;
        }
    }
    std::fclose(f);
    return ok;
}

namespace { std::atomic<CanonicalLoadInventory*> failed_canonical_inventory{nullptr}; }
bool WeightTable::enable_inventory_capture(const CanonicalInventoryLimits& limits,
    std::shared_ptr<const ReservationLease> lease,std::shared_ptr<const VerifiedDriverBindings> inputs,
    const std::string& expected_path,std::string& error) {
    uint64_t required=0;
    if(inventory_load_attempted_||retired_||hc_attempted_||inventory_){error="canonical inventory must be enabled once before load";return false;}
    if(!canonical_inventory_host_bound(limits,required,error))return false;
    if(!lease||lease->purpose()!=ReservationPurpose::foreground||lease->resources().host_bytes<required||
       !inputs||expected_path.empty()||expected_path.size()>limits.max_string_bytes){error="canonical inventory requires reservation, bindings and bounded pack path";return false;}
    try {
        auto candidate=std::make_unique<CanonicalLoadInventory>();
        candidate->metadata_reservation=std::move(lease);candidate->retained_input_bindings=std::move(inputs);
        candidate->limits=limits;candidate->expected_pack_loader_path=expected_path;
        candidate->records.reserve(size_t(limits.max_records));candidate->hc_records.reserve(size_t(limits.max_records));
        inventory_=std::move(candidate);return true;
    }catch(const std::bad_alloc&){error="canonical inventory allocation failed";return false;}
}
void WeightTable::quarantine_inventory() noexcept {
    if(auto* p=inventory_.release()){
        auto* head=failed_canonical_inventory.load();
        do{p->quarantined_next=head;}while(!failed_canonical_inventory.compare_exchange_weak(head,p));
    }
}

bool WeightTable::load(const std::string& pack_dir, void* arena_base, uint64_t arena_bytes, std::string& err,
                       const std::set<std::string>* skip) {
    if(inventory_)load_complete_ = false;
    inventory_load_attempted_ = true;
    if (retired_ || hc_attempted_) { err = "cannot reload retired or HC-owned weight table"; return false; }
    if (inventory_uncertain_) {err="cannot reload uncertain canonical inventory";return false;}
    load_complete_ = false;
    if(inventory_){
        inventory_->records.clear();inventory_->hc_records.clear();inventory_->hc_completed=false;
        inventory_->index_pool_bytes=inventory_->canonical_arena_bytes=inventory_->hc_allocation_bytes=0;
        if(pack_dir!=inventory_->expected_pack_loader_path){err="canonical inventory pack argument differs from retained binding claim";return false;}
    }
    const std::string path = pack_dir + "/index.txt";
    std::FILE* idx = std::fopen(path.c_str(), "rb");
    if (!idx) { err = "cannot open " + path + " (run tools/pack_index.py)"; return false; }

    uint64_t pool = 0;
    int align = 0;
    std::vector<IndexRow> rows;
    char line[1024];
    while (std::fgets(line, sizeof line, idx)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, t = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &t) == 3) { pool = p; align = a; }
            continue;
        }
        IndexRow r;
        char name[256] = {0};
        int kind = 0;
        // The format is written by tools/pack_index.py and is space-separated with no quoting; a name with a
        // space in it would break the parse, which is why the parser REFUSES rather than taking what it got.
        const int n = std::sscanf(line,
                                  "%255s %d %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNd64 " %" SCNd64 " %d %d %d %d %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %d %d",
                                  name, &r.file, &kind, &r.src_off, &r.src_bytes, &r.dst_off, &r.dst_bytes,
                                  &r.ne0, &r.ne1, &r.code_bits, &r.code_bias, &r.group_elems, &r.codebook,
                                  &r.has_offset, &r.codes_bytes, &r.scales_bytes, &r.offset_bytes,
                                  &r.scales_fp16, &r.act_kind);
        if (n != 19) {
            // A malformed index line is a WRONG BYTE OFFSET waiting to happen, and a wrong offset decodes to
            // a plausible weight.  Refuse loudly instead of loading the rows that happened to parse.
            err = "index.txt: could not parse a row (" + std::to_string(n) + " of 19 fields)";
            std::fclose(idx);
            return false;
        }
        if(inventory_){
            auto& inv=*inventory_;
            if(inv.records.size()>=inv.limits.max_records||std::strlen(name)>inv.limits.max_string_bytes||
               kind<0||kind>5||!file_name(r.file)||r.ne0<=0||r.ne1<0||
               uint64_t(r.ne0)>uint64_t(INT64_MAX)/uint64_t(r.ne1?r.ne1:1)||
               r.src_bytes>UINT64_MAX-r.src_off||r.dst_bytes>UINT64_MAX-r.dst_off||
               std::any_of(inv.records.begin(),inv.records.end(),[&](const auto& x){return x.name==name;})){
                err="canonical inventory row bounds or duplicate identity refused";std::fclose(idx);return false;
            }
            try{
                CanonicalTensorLoadRecord record;record.name=name;record.source_index_kind=kind;
                record.normalized_kind=kind==4?1:(kind==5?3:kind);record.raw16=kind==4||kind==5;
                record.file_id=r.file;record.source_offset=r.src_off;record.source_bytes=r.src_bytes;
                record.index_destination_offset=r.dst_off;record.runtime_bytes=r.dst_bytes;
                record.ne0=r.ne0;record.ne1=r.ne1;record.code_bits=r.code_bits;record.code_bias=r.code_bias;
                record.group_elems=r.group_elems;record.activation_kind=r.act_kind;record.index_codebook=r.codebook;
                record.index_has_offset=r.has_offset;record.index_scales_fp16=r.scales_fp16;
                record.index_codes_bytes=r.codes_bytes;record.index_scales_bytes=r.scales_bytes;record.index_offsets_bytes=r.offset_bytes;
                inv.records.push_back(std::move(record));
            }catch(const std::bad_alloc&){err="canonical inventory row allocation failed";std::fclose(idx);return false;}
        }
        r.name = name;
        // plan v0.3 P6: a standalone native pack (tools/iq_pack.py) stores 16-bit floats as they are in the GGUF:
        // index kinds 4 (BF16) and 5 (F16) are copied and take the engine forms of kinds 1 and 3
        if (kind == 4 || kind == 5) {
            r.raw16 = true;
            kind = kind == 4 ? (int) WeightKind::Bf16InF32 : (int) WeightKind::F16InF32;
        }
        r.kind = (WeightKind) kind;
        rows.push_back(r);
    }
    std::fclose(idx);

    if (pool == 0 || rows.empty()) { err = "index.txt has no header or no rows"; return false; }
    if(inventory_){inventory_->index_pool_bytes=pool;inventory_->index_alignment=align;}
    // Plan v0.3 P1: a skip set compacts the arena.  Kept rows are re-placed in index order at the index's
    // alignment; skipped rows keep their metadata and get no bytes.
    std::vector<bool> skipped(rows.size(), false);
    if (skip != nullptr) {
        const uint64_t a = align > 0 ? (uint64_t) align : 256;
        uint64_t at = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (skip->count(rows[i].name)) { skipped[i] = true; continue; }
            if(inventory_&&(rows[i].dst_bytes>UINT64_MAX-(a-1)||
               ((rows[i].dst_bytes+a-1)/a)*a>UINT64_MAX-at)) {err="canonical inventory compacted arena overflow";return false;}
            rows[i].dst_off = at;
            at += (rows[i].dst_bytes + a - 1) / a * a;
        }
        pool = at;
    }
    if (arena_bytes < pool) {
        err = "arena is " + std::to_string(arena_bytes) + " B but the index needs " + std::to_string(pool);
        return false;
    }

    if(inventory_){
        inventory_->canonical_arena_bytes=pool;
        for(size_t i=0;i<rows.size();++i){auto& record=inventory_->records[i];record.chosen=!skipped[i];record.runtime_offset=rows[i].dst_off;
            if(record.chosen&&(rows[i].dst_off>arena_bytes||rows[i].dst_bytes>arena_bytes-rows[i].dst_off)){err="canonical inventory destination outside arena";return false;}}
    }
    // Two pinned staging buffers, both reused.  Pinned because a 5 GB pageable upload spends its time in the
    // driver copying through a bounce buffer, and bounded by CHUNK rather than by the largest tensor.
    //
    // STAGE_OUT IS NOT OPTIONAL.  The first version of this loop converted the values straight into the
    // destination pointer - which is DEVICE memory - from host code.  A host write to a device address is
    // undefined behaviour, and on this driver it does not fault: it corrupts whatever the address maps to in
    // the host's own space, which is nothing at all, so the tensor would have loaded as the UNCONVERTED pack
    // bytes and every BF16 weight would have been read as 32-bit garbage.  Converted data goes to a host
    // buffer and crosses the bus once.
    //
    // STAGE_OUT IS TWICE STAGE_IN because a conversion can now GROW: widening an fp16 scale plane to f32
    // doubles it.  Sized at CHUNK it would have written 8 MiB past the end of a pinned allocation for every
    // one of the 90 widened tensors - a heap corruption that would have been blamed on whatever ran next.
    bool inventory_cleanup_failed=false;
    auto release_stage=[&](void* p){const auto rc=cudaFreeHost(p);if(inventory_&&rc!=cudaSuccess)inventory_cleanup_failed=true;return rc;};
    if(inventory_)inventory_uncertain_=true;
    void* stage_in = nullptr;
    void* stage_out = nullptr;
    if (cudaHostAlloc(&stage_in, CHUNK, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&stage_out, CHUNK * 2, cudaHostAllocDefault) != cudaSuccess) {
        err = "cudaHostAlloc for the staging buffers failed";
        if (stage_in) release_stage(stage_in);
        if (stage_out) release_stage(stage_out);
        return false;
    }

    uint8_t* dst_base = (uint8_t*) arena_base;
    std::FILE* cur = nullptr;
    int cur_file = -1;
    table_.clear();
    report_ = LoadReport{};
    pack_dir_ = pack_dir;

    const double t_read0 = now_ms();
    double upload_ms = 0;
    for (size_t row_i = 0; row_i < rows.size(); ++row_i) {
        const IndexRow& r = rows[row_i];
        if (skipped[row_i]) {
            WeightRef wr;
            wr.data = nullptr;
            wr.resident = false;
            wr.bytes = r.dst_bytes;
            wr.ne0 = r.ne0;
            wr.ne1 = r.ne1;
            wr.elements = r.ne0 * (r.ne1 > 0 ? r.ne1 : 1);
            wr.kind = r.kind;
            wr.code_bits = r.code_bits;
            wr.code_bias = r.code_bias;
            wr.group_elems = r.group_elems;
            wr.codebook_iq4nl = r.codebook != 0;
            wr.has_offset = r.has_offset != 0;
            wr.codes_bytes = r.codes_bytes;
            wr.scales_bytes = r.scales_bytes;
            wr.offset_bytes = r.offset_bytes;
            wr.src_off = r.src_off;
            wr.src_bytes = r.src_bytes;
            wr.file_id = r.file;
            wr.scales_fp16 = r.scales_fp16 != 0;
            wr.act_kind = r.act_kind;
            table_[r.name] = wr;
            ++report_.tensors;
            continue;
        }
        if (r.code_bits != 0 && r.dst_bytes == 0) {
            // plan v0.3 P6: a native pack's row that carries a shape only - the GGUF form must serve it
            err = r.name + ": this pack holds the tensor only in its GGUF form (run with --native SHARD1)";
            release_stage(stage_in); release_stage(stage_out);
            return false;
        }
        if (r.file != cur_file) {
            if (cur) std::fclose(cur);
            const char* fn = file_name(r.file);
            const std::string p = pack_dir + "/" + (fn ? fn : "?");
            cur = std::fopen(p.c_str(), "rb");
            if (!cur) { err = "cannot open " + p; release_stage(stage_in); release_stage(stage_out); return false; }
            cur_file = r.file;
#if !defined(_WIN32)
            // ask for this file's remaining rows up front so their reads overlap
            for (size_t j = row_i; j < rows.size(); ++j)
                if (rows[j].file == r.file && !skipped[j])
                    strata::platform::advise_willneed(fileno(cur), rows[j].src_off, rows[j].src_bytes);
#endif
        }

        // ---- the segment list: what this tensor's bytes are, plane by plane, in both forms
        Seg segs[3];
        int n_segs = 0;
        bool bad = false;
        uint64_t sum_src = 0, sum_dst = 0;
        if (r.codes_bytes != 0) {
            // A quantized tensor: codes, then scales, then the optional offset plane, each contiguous in the
            // source span in that order (pack_index.py verifies the contiguity and refuses otherwise).
            segs[n_segs++] = Seg{0, r.codes_bytes, 0, r.codes_bytes, Conv::Copy};
            if (r.scales_bytes) {
                const uint64_t src_scale_bytes = r.scales_fp16 ? r.scales_bytes / 2 : r.scales_bytes;
                if (r.scales_fp16 && (r.scales_bytes % 2 || src_scale_bytes * 2 != r.scales_bytes)) {
                    err = r.name + ": an fp16 scale plane of an odd byte count";
                    bad = true;
                }
                segs[n_segs++] = Seg{r.codes_bytes, src_scale_bytes, r.codes_bytes, r.scales_bytes,
                                     r.scales_fp16 ? Conv::WidenF16 : Conv::Copy};
            }
            if (r.offset_bytes) {
                const uint64_t after = r.codes_bytes + (r.scales_fp16 ? r.scales_bytes / 2 : r.scales_bytes);
                segs[n_segs++] = Seg{after, r.offset_bytes, r.codes_bytes + r.scales_bytes, r.offset_bytes,
                                     Conv::Copy};
            }
        } else {
            // Everything else is one run whose transform is the row's kind.  The two promoted types shrink
            // 4 B/elem to 2; the rest are already the engine form.
            Conv c = Conv::Copy;
            if (r.raw16) c = Conv::Copy;
            else if (r.kind == WeightKind::Bf16InF32) c = Conv::Bf16High;
            else if (r.kind == WeightKind::F16InF32) c = Conv::ToF16;
            const uint64_t elems = (uint64_t) r.ne0 * (uint64_t) (r.ne1 > 0 ? r.ne1 : 1);
            if (c != Conv::Copy && (r.src_bytes != elems * 4 || r.dst_bytes != elems * 2)) {
                // A promoted type is 4 B/elem in and 2 B/elem out, and nothing else.  If those two numbers do
                // not say so, the shape and the byte counts disagree and the element count this loop would
                // use to walk the source is not the element count the pack wrote.
                char buf[320];
                std::snprintf(buf, sizeof buf,
                              "%s: a promoted tensor of %llu elements is %llu B in and %llu B out, not %llu/%llu",
                              r.name.c_str(), (unsigned long long) elems, (unsigned long long) r.src_bytes,
                              (unsigned long long) r.dst_bytes, (unsigned long long) (elems * 4),
                              (unsigned long long) (elems * 2));
                err = buf;
                bad = true;
            }
            segs[n_segs++] = Seg{0, r.src_bytes, 0, r.dst_bytes, c};
        }
        if(inventory_)for(int i=0;i<n_segs;++i){const auto& seg=segs[i];
            if(seg.src_off>r.src_bytes||seg.src_bytes>r.src_bytes-seg.src_off||seg.dst_off>r.dst_bytes||seg.dst_bytes>r.dst_bytes-seg.dst_off||
               seg.src_bytes>UINT64_MAX-sum_src||seg.dst_bytes>UINT64_MAX-sum_dst){bad=true;err="canonical inventory plane extent overflow";break;}
            sum_src+=seg.src_bytes;sum_dst+=seg.dst_bytes;
        }
        else for (int i = 0; i < n_segs; ++i) { sum_src += segs[i].src_bytes; sum_dst += segs[i].dst_bytes; }
        // The segment list is this loop's own idea of the layout, and it is the ONLY thing standing between a
        // mis-sized plane and a wrong byte offset.  It is checked against both recorded totals, not trusted.
        if (!bad && (sum_src != r.src_bytes || sum_dst != r.dst_bytes)) {
            char buf[360];
            std::snprintf(buf, sizeof buf,
                          "%s: the segments cover %llu/%llu B but the index says %llu/%llu - the plane layout "
                          "this loader built is not the one the index describes",
                          r.name.c_str(), (unsigned long long) sum_src, (unsigned long long) sum_dst,
                          (unsigned long long) r.src_bytes, (unsigned long long) r.dst_bytes);
            err = buf;
            bad = true;
        }
        if (bad) { release_stage(stage_in); release_stage(stage_out); return false; }

        if(inventory_){auto& record=inventory_->records[row_i];record.plane_count=uint32_t(n_segs);
            for(int j=0;j<n_segs;++j){const auto& seg=segs[j];record.planes[size_t(j)]={seg.src_off,seg.src_bytes,seg.dst_off,seg.dst_bytes,static_cast<CanonicalPlaneConversion>(seg.conv)};}}
        for (int si = 0; si < n_segs; ++si) {
            const Seg& s = segs[si];
            const bool widening = (s.conv == Conv::WidenF16);
            if (widening && (s.src_bytes & 1ull)) {
                err = r.name + ": an fp16 plane with an odd source byte count";
                release_stage(stage_in); release_stage(stage_out);
                return false;
            }
            uint64_t done = 0;
            while (done < s.src_bytes) {
                uint64_t n = std::min(CHUNK, s.src_bytes - done);
                // WIDENING NEVER SPLITS AN ELEMENT.  CHUNK is even, so a full chunk cannot, but the tail of a
                // plane whose length is odd would - and half an fp16 is a plausible-looking scale.
                if (widening && (n & 1ull)) {
                    if (n == 1) { err = r.name + ": an fp16 plane ending on a half element"; release_stage(stage_in); release_stage(stage_out); return false; }
                    n -= 1;
                }
                if (!read_at(cur, r.src_off + s.src_off + done, stage_in, (size_t) n, err, r.name.c_str())) {
                    release_stage(stage_in); release_stage(stage_out);
                    return false;
                }
                const uint8_t* src = (const uint8_t*) stage_in;
                const void* host_src = src;
                uint64_t out_bytes = n;
                uint64_t out_at = s.dst_off + done;

                switch (s.conv) {
                    case Conv::Copy:
                        break;
                    case Conv::Bf16High: {
                        // THE HIGH 16 BITS ARE THE BF16 ENCODING, exactly - not a rounding.  A bf16 value is
                        // the top half of its f32 image, and the pack promoted it without changing it, so
                        // this is a copy with a stride.
                        const uint64_t elems = n / 4;
                        out_bytes = elems * 2;
                        uint8_t* o = (uint8_t*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            uint32_t v;
                            std::memcpy(&v, src + i * 4, 4);
                            const uint16_t h = (uint16_t) (v >> 16);
                            std::memcpy(o + i * 2, &h, 2);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done / 2;
                        break;
                    }
                    case Conv::ToF16: {
                        const uint64_t elems = n / 4;
                        out_bytes = elems * 2;
                        uint8_t* o = (uint8_t*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            float v;
                            std::memcpy(&v, src + i * 4, 4);
                            const uint16_t h = strata::kernels::f16_from_f32(v);
                            std::memcpy(o + i * 2, &h, 2);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done / 2;
                        break;
                    }
                    case Conv::WidenF16: {
                        // fp16 bits -> f32.  EXACT: every fp16 value is an f32 value, including the
                        // subnormals, so this cannot change a single weight - which is why the engine may do
                        // it at all, and why doing it costs 10.88 MiB rather than a second kernel family.
                        const uint64_t elems = n / 2;
                        out_bytes = elems * 4;
                        float* o = (float*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            uint16_t h;
                            std::memcpy(&h, src + i * 2, 2);
                            o[i] = strata::kernels::f32_from_f16(h);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done * 2;
                        break;
                    }
                }

                const double u0 = now_ms();
                if (cudaMemcpy(dst_base + r.dst_off + out_at, host_src, out_bytes, cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                    err = "cudaMemcpy failed for " + r.name;
                    release_stage(stage_in); release_stage(stage_out);
                    return false;
                }
                upload_ms += now_ms() - u0;
                done += n;
            }
        }

        WeightRef wr;
        wr.data = dst_base + r.dst_off;
        wr.bytes = r.dst_bytes;
        wr.ne0 = r.ne0;
        wr.ne1 = r.ne1;
        wr.elements = r.ne0 * (r.ne1 > 0 ? r.ne1 : 1);
        wr.kind = r.kind;
        wr.code_bits = r.code_bits;
        wr.code_bias = r.code_bias;
        wr.group_elems = r.group_elems;
        wr.codebook_iq4nl = r.codebook != 0;
        wr.has_offset = r.has_offset != 0;
        wr.codes_bytes = r.codes_bytes;
        wr.scales_bytes = r.scales_bytes;
        wr.offset_bytes = r.offset_bytes;
        wr.src_off = r.src_off;
        wr.src_bytes = r.src_bytes;
        wr.file_id = r.file;
        wr.scales_fp16 = r.scales_fp16 != 0;
        wr.act_kind = r.act_kind;
        table_[r.name] = wr;

        ++report_.tensors;
        if ((r.kind == WeightKind::Bf16InF32 || r.kind == WeightKind::F16InF32) && !r.raw16) {
            ++report_.re_rounded;
            report_.bytes_saved += r.src_bytes - r.dst_bytes;
        }
    }
    if (cur) std::fclose(cur);
    release_stage(stage_in);
    release_stage(stage_out);

    report_.arena_bytes = pool;
    report_.read_ms = now_ms() - t_read0 - upload_ms;
    report_.upload_ms = upload_ms;
    if (cudaDeviceSynchronize() != cudaSuccess) { err = "cudaDeviceSynchronize after the load failed"; return false; }
    if(inventory_cleanup_failed){err="canonical inventory staging retirement uncertain";return false;}
    inventory_uncertain_=false;
    load_complete_ = true;
    return true;
}

const WeightRef* WeightTable::find(const std::string& name) const {
    const auto it = table_.find(name);
    return it == table_.end() ? nullptr : &it->second;
}

void WeightTable::hc_q8_names(int64_t n_layers, std::set<std::string>& out) {
    for (int64_t l = 0; l < n_layers; ++l)
        for (const char* h : {"hc_attn_", "hc_ffn_"})
            for (const char* m : {"down", "up"})
                out.insert("blk." + std::to_string(l) + "." + h + m + ".weight");
}

bool WeightTable::load_hc_q8(const std::string& pack_dir, uint64_t& bytes, std::string& err) {
    bytes = 0;
    const bool canonical_ready=load_complete_;
    if(inventory_)load_complete_=false;
    if (!canonical_ready || retired_ || inventory_uncertain_) { err = "HC load requires a completed canonical load"; return false; }
    if(inventory_&&pack_dir!=inventory_->expected_pack_loader_path){err="HC pack argument differs from canonical binding claim";return false;}
    if (hc_attempted_) { bytes = hc_bytes_; load_complete_=canonical_ready; return true; }
    hc_attempted_ = true;
    load_complete_ = false;
    std::vector<WeightRef*> todo;
    for (auto& [name, w] : table_) {
        const bool hc = name.find(".hc_attn_down.") != std::string::npos || name.find(".hc_attn_up.") != std::string::npos ||
                        name.find(".hc_ffn_down.") != std::string::npos || name.find(".hc_ffn_up.") != std::string::npos;
        if (!hc || w.resident) continue;
        // the pack holds BF16 either promoted to f32 (4 B/elem) or as it is in the GGUF (a native pack, 2 B/elem)
        if (w.kind != WeightKind::Bf16InF32 || w.ne0 <= 0 || w.ne0 % 32 != 0 || w.elements <= 0 || w.elements % 32 != 0 ||
            uint64_t(w.elements) > UINT64_MAX / 4 ||
            (w.src_bytes != (uint64_t) w.elements * 4 && w.src_bytes != (uint64_t) w.elements * 2)) {
            err = "STRATA_HC_REQ8: " + name + " is not a BF16 tensor of whole 32-value blocks";
            return false;
        }
        todo.push_back(&w);
    }
    if (todo.empty()) { if(inventory_)inventory_->hc_completed=true; load_complete_ = true; return true; }
    auto al = [](uint64_t b) { return (b + 255) & ~(uint64_t) 255; };
    uint64_t total = 0;
    for (const WeightRef* w : todo) {
        const auto codes = al(uint64_t(w->elements)), scales = al(uint64_t(w->elements) / 32 * 4);
        if (scales > UINT64_MAX - codes || codes + scales > UINT64_MAX - total) {
            err = "HC allocation size overflow"; return false;
        }
        total += codes + scales;
    }
    std::vector<WeightRef> staged;
    staged.reserve(todo.size());
    if(inventory_)inventory_uncertain_=true;
    if (cudaGetDevice(&hc_device_) != cudaSuccess) { err = "HC owning device unavailable"; return false; }
    uint8_t* dev = nullptr;
    if (cudaMalloc((void**) &dev, total) != cudaSuccess) { err = "STRATA_HC_REQ8: cudaMalloc failed"; return false; }
    hc_allocation_ = dev;
    hc_bytes_ = total;
    std::vector<float> src;
    std::vector<int8_t> q;
    std::vector<float> sc;
    uint64_t at = 0;
    for (WeightRef* w : todo) {
        const CanonicalTensorLoadRecord* canonical_row=nullptr;
        if(inventory_){
            const auto row=std::find_if(inventory_->records.begin(),inventory_->records.end(),[&](const auto& r){auto it=table_.find(r.name);return it!=table_.end() && &it->second==w;});
            if(row==inventory_->records.end()||row->chosen||row->file_id!=w->file_id||row->source_offset!=w->src_off||row->source_bytes!=w->src_bytes||
               row->normalized_kind!=int(w->kind)||row->ne0!=w->ne0||row->ne1!=w->ne1||row->code_bits!=w->code_bits||row->code_bias!=w->code_bias||
               row->group_elems!=w->group_elems||row->activation_kind!=w->act_kind||bool(row->index_codebook)!=w->codebook_iq4nl||
               bool(row->index_has_offset)!=w->has_offset||bool(row->index_scales_fp16)!=w->scales_fp16||row->index_codes_bytes!=w->codes_bytes||
               row->index_scales_bytes!=w->scales_bytes||row->index_offsets_bytes!=w->offset_bytes){err="HC dependency differs from actual canonical row";return false;}
            canonical_row=&*row;
        }
        src.resize((size_t) w->elements);
        const char* fn = file_name(w->file_id);
        const std::string path = pack_dir + "/" + (fn ? fn : "?");
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "STRATA_HC_REQ8: cannot open " + path; return false; }
        std::unique_ptr<std::FILE, decltype(&std::fclose)> source_file(f, &std::fclose);
        const bool raw16 = w->src_bytes == (uint64_t) w->elements * 2;
        if(canonical_row&&canonical_row->raw16!=raw16){err="HC source encoding differs from original index kind";return false;}
#if defined(_WIN32)
        bool ok = _fseeki64(f, (long long) w->src_off, SEEK_SET) == 0;
#else
        bool ok = fseeko(f, (off_t) w->src_off, SEEK_SET) == 0;   // 64-bit: a pack's dense file can pass 2 GiB
#endif
        if (ok && raw16) {
            std::vector<uint16_t> h(src.size());
            ok = std::fread(h.data(), 2, h.size(), f) == h.size();
            for (size_t i = 0; ok && i < h.size(); ++i) {
                const uint32_t b = (uint32_t) h[i] << 16;   // bf16 -> f32 is exact
                std::memcpy(&src[i], &b, 4);
            }
        } else if (ok) {
            ok = std::fread(src.data(), 4, src.size(), f) == src.size();
        }
        source_file.reset();
        if (!ok) { err = "STRATA_HC_REQ8: cannot read the source bytes"; return false; }
        const size_t nb = src.size() / 32;
        q.resize(src.size());
        sc.resize(nb);
        for (size_t b = 0; b < nb; ++b) {   // Q8_0's rule: d = amax / 127, q = round-to-nearest-even(x / d)
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(src[b * 32 + j]));
            const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
            sc[b] = d;
            for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t) std::nearbyint(src[b * 32 + j] * id);
        }
        CanonicalHcLoadRecord receipt;
        if(inventory_){
            receipt.canonical_row=uint64_t(canonical_row-inventory_->records.data());
            receipt.conversion=raw16?CanonicalHcConversion::bf16_raw16_to_q8_host:CanonicalHcConversion::bf16_in_f32_to_q8_host;
            receipt.codes_offset=at;receipt.codes_bytes=q.size();receipt.scales_offset=at+al(q.size());receipt.scales_bytes=sc.size()*4;
        }
        uint8_t* codes = dev + at;
        at += al(q.size());
        float* scales = (float*) (dev + at);
        at += al(sc.size() * 4);
        if (cudaMemcpy(codes, q.data(), q.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(scales, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "STRATA_HC_REQ8: upload failed";
            return false;
        }
        WeightRef converted = *w;
        converted.data = codes;
        converted.q8_scale = scales;
        converted.resident = true;
        converted.bytes = q.size() + sc.size() * 4;
        staged.push_back(converted);
        if(inventory_)inventory_->hc_records.push_back(receipt);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { err = "HC load drain failed"; return false; }
    for (size_t i = 0; i < todo.size(); ++i) *todo[i] = staged[i];
    if(inventory_){inventory_->hc_completed=true;inventory_->hc_allocation_bytes=total;}
    inventory_uncertain_=false;
    load_complete_ = true;
    bytes = total;
    return true;
}

bool WeightTable::owns_auxiliary_span(const void* pointer, uint64_t bytes) const {
    if (!load_complete_ || retired_ || !hc_allocation_ || !pointer || !bytes) return false;
    const auto base = reinterpret_cast<uintptr_t>(hc_allocation_), p = reinterpret_cast<uintptr_t>(pointer);
    return p >= base && p - base <= hc_bytes_ && bytes <= hc_bytes_ - (p - base);
}

bool WeightTable::release_auxiliary_checked(std::string& err) {
    retired_ = true;
    load_complete_ = false;
    if (release_failed_) { err = "HC release previously failed"; return false; }
    if (!hc_allocation_) return true;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != hc_device_) {
        err = "HC release requires owning device"; return false;
    }
    if (cudaFree(hc_allocation_) != cudaSuccess) {
        release_failed_ = true; err = "HC release failed; retain allocation"; return false;
    }
    hc_allocation_ = nullptr; hc_bytes_ = 0;
    return true;
}

WeightTable::~WeightTable() {
    struct ReceiptGuard {WeightTable* table;~ReceiptGuard(){if(table->inventory_uncertain_||table->release_failed_||table->hc_allocation_)table->quarantine_inventory();}} guard{this};
    if (!hc_allocation_ || release_failed_) return;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess) return;
    if (previous != hc_device_ && cudaSetDevice(hc_device_) != cudaSuccess) return;
    try {
        std::string err;
        if (cudaDeviceSynchronize() == cudaSuccess) release_auxiliary_checked(err);
    } catch (...) {}
    if (previous != hc_device_ && cudaSetDevice(previous) != cudaSuccess && inventory_) inventory_uncertain_=true;
}

}  // namespace strata::core
