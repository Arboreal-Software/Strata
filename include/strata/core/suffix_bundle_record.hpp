// Qualification-only bundle RECORD codec (EXECUTION-PLAN step 5 evidence seam).
//
// Purpose: make a captured all-stage RAW state exportable and re-comparable OFFLINE so the EXISTING strict
// comparator (`src/program/resident_repair_compare.inc: clean_recompute_qualification::compare`) can be run
// against a matched clean recomputation.  Decoding rebuilds a plain SnapshotDescriptor; the caller then freezes it
// with `core::BundleHandle::freeze(descriptor, key, actual_layout, out)` using the layout the RUNTIME produced -
// metadata never supplies its own expected layout, and the comparator's own `validate_window_state` pass stays in
// force.  Nothing here weakens the comparator; it only moves bytes across a file boundary.
//
// A record is self-describing and self-digested: the trailing digest covers the header AND every payload byte, so a
// truncated file, a rewritten tensor, a swapped stage or a changed cut/history is refused on load rather than
// silently compared.  Producer identity (binary sha256), request ordinal and phase are part of the header and are
// checked by the caller against the run that produced them.
#pragma once

#include "strata/core/conversation_buffer.hpp"
#include "strata/core/window_hash.hpp"
#include "strata/core/window_state.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core::suffix_record {

inline constexpr uint32_t kSchema = 0x53545242;   // 'STRB'
inline constexpr uint32_t kVersion = 1;
enum class Kind : uint32_t { snapshot = 1 };
// NOTE: there is deliberately no layout record. The expected WindowStateLayout must come from the RUNTIME that
// captured the state (native_window_layout over the actual configuration); a self-described layout from a file is
// not trusted evidence and would let a capture define its own expectations.
enum class Phase : uint32_t { unknown = 0, certified_frontier = 1, clean_reference = 2 };

// Export/load ceilings.  A capture for a large context is many megabytes per stage and the encode/decode path
// overlaps the original bundle, the encoded record and the decoded copy, so the caller MUST hold a real
// WindowBudget/host credit for the peak (2x body + record) and pass these limits down: nothing here allocates
// uncharged multiples.
struct Limits {
    uint64_t max_record_bytes = 0;     // whole file
    uint64_t max_body_bytes = 0;       // encoded snapshot/layout body
    uint64_t max_tensor_bytes = 0;     // one tensor payload
    uint64_t max_total_payload = 0;    // all tensor payloads in one record
};
inline bool within(const Limits& limits, uint64_t body_bytes, uint64_t payload_bytes, uint64_t record_bytes) noexcept {
    if (!limits.max_record_bytes || !limits.max_body_bytes || !limits.max_total_payload) return false;
    return body_bytes <= limits.max_body_bytes && payload_bytes <= limits.max_total_payload &&
           record_bytes <= limits.max_record_bytes;
}

struct Header {
    Kind kind = Kind::snapshot;
    Phase phase = Phase::unknown;
    uint64_t request_ordinal = 0;
    std::string producer_binary;          // sha256 of the engine binary that captured this record
    std::string producer_source_manifest; // input-manifest sha256 of the frozen build that produced it
};

inline void put_u32(std::vector<uint8_t>& o, uint32_t v) {
    for (int i = 0; i < 4; ++i) o.push_back(uint8_t((v >> (8 * i)) & 0xff));
}
inline void put_u64(std::vector<uint8_t>& o, uint64_t v) {
    for (int i = 0; i < 8; ++i) o.push_back(uint8_t((v >> (8 * i)) & 0xff));
}
inline void put_str(std::vector<uint8_t>& o, const std::string& s) {
    put_u64(o, s.size());
    o.insert(o.end(), s.begin(), s.end());
}
inline void put_digest(std::vector<uint8_t>& o, const StateDigest& d) {
    o.insert(o.end(), d.begin(), d.end());
}
inline bool get_u32(const std::vector<uint8_t>& b, size_t& at, uint32_t& v) {
    if (at + 4 > b.size()) return false;
    v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t(b[at + i]) << (8 * i);
    at += 4;
    return true;
}
inline bool get_u64(const std::vector<uint8_t>& b, size_t& at, uint64_t& v) {
    if (at + 8 > b.size()) return false;
    v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(b[at + i]) << (8 * i);
    at += 8;
    return true;
}
inline bool get_str(const std::vector<uint8_t>& b, size_t& at, std::string& s) {
    uint64_t n = 0;
    if (!get_u64(b, at, n) || n > b.size() - at) return false;
    s.assign(b.begin() + long(at), b.begin() + long(at + n));
    at += size_t(n);
    return true;
}
inline bool get_digest(const std::vector<uint8_t>& b, size_t& at, StateDigest& d) {
    if (at + d.size() > b.size()) return false;
    std::memcpy(d.data(), b.data() + at, d.size());
    at += d.size();
    return true;
}

inline void put_key(std::vector<uint8_t>& o, const WindowKey& k) {
    put_u64(o, k.engine_generation); put_u64(o, k.window_generation);
    put_str(o, k.session_id); put_str(o, k.thread_id); put_str(o, k.window_id);
    put_u64(o, k.revision); put_u64(o, k.append_epoch); put_u64(o, k.selection_epoch);
    put_u64(o, k.provenance_epoch); put_digest(o, k.model_profile); put_digest(o, k.topology);
}
inline bool get_key(const std::vector<uint8_t>& b, size_t& at, WindowKey& k) {
    return get_u64(b, at, k.engine_generation) && get_u64(b, at, k.window_generation) &&
           get_str(b, at, k.session_id) && get_str(b, at, k.thread_id) && get_str(b, at, k.window_id) &&
           get_u64(b, at, k.revision) && get_u64(b, at, k.append_epoch) && get_u64(b, at, k.selection_epoch) &&
           get_u64(b, at, k.provenance_epoch) && get_digest(b, at, k.model_profile) && get_digest(b, at, k.topology);
}
inline void put_cut(std::vector<uint8_t>& o, const StateCut& c) {
    put_u64(o, c.id); put_u64(o, c.accepted); put_u64(o, c.forwarded); put_u64(o, c.exact_frontier);
    put_u64(o, c.output_watermark); put_digest(o, c.history);
}
inline bool get_cut(const std::vector<uint8_t>& b, size_t& at, StateCut& c) {
    return get_u64(b, at, c.id) && get_u64(b, at, c.accepted) && get_u64(b, at, c.forwarded) &&
           get_u64(b, at, c.exact_frontier) && get_u64(b, at, c.output_watermark) && get_digest(b, at, c.history);
}
inline void put_tokens(std::vector<uint8_t>& o, const std::vector<int32_t>& t) {
    put_u64(o, t.size());
    for (int32_t v : t) put_u32(o, uint32_t(v));
}
inline bool get_tokens(const std::vector<uint8_t>& b, size_t& at, std::vector<int32_t>& t) {
    uint64_t n = 0;
    if (!get_u64(b, at, n) || n > (b.size() - at) / 4) return false;
    t.resize(size_t(n));
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t v = 0;
        if (!get_u32(b, at, v)) return false;
        t[size_t(i)] = int32_t(v);
    }
    return true;
}
inline void put_tensor_layout(std::vector<uint8_t>& o, const TensorLayout& l) {
    put_u32(o, uint32_t(l.kind)); put_u64(o, uint64_t(l.layer)); put_u32(o, l.device_part);
    put_str(o, l.dtype); put_str(o, l.native_layout);
    put_u64(o, l.shape.size());
    for (uint64_t v : l.shape) put_u64(o, v);
    put_u64(o, l.strides.size());
    for (uint64_t v : l.strides) put_u64(o, v);
    put_u64(o, l.logical_begin); put_u64(o, l.valid_elements); put_u64(o, l.bytes);
}
inline bool get_tensor_layout(const std::vector<uint8_t>& b, size_t& at, TensorLayout& l) {
    uint32_t kind = 0, part = 0;
    uint64_t layer = 0;
    if (!get_u32(b, at, kind) || !get_u64(b, at, layer) || !get_u32(b, at, part) || !get_str(b, at, l.dtype) ||
        !get_str(b, at, l.native_layout))
        return false;
    l.kind = StatePartKind(kind);
    l.layer = int64_t(layer);
    l.device_part = part;
    uint64_t n = 0;
    if (!get_u64(b, at, n) || n > (b.size() - at) / 8) return false;
    l.shape.resize(size_t(n));
    for (uint64_t i = 0; i < n; ++i) if (!get_u64(b, at, l.shape[size_t(i)])) return false;
    if (!get_u64(b, at, n) || n > (b.size() - at) / 8) return false;
    l.strides.resize(size_t(n));
    for (uint64_t i = 0; i < n; ++i) if (!get_u64(b, at, l.strides[size_t(i)])) return false;
    return get_u64(b, at, l.logical_begin) && get_u64(b, at, l.valid_elements) && get_u64(b, at, l.bytes);
}

inline void put_buffer(std::vector<uint8_t>& o, const ConversationBuffer& p) {
    put_u64(o, p.size());
    p.visit(0, p.size(), [&](const uint8_t* bytes, size_t n, size_t) {
        o.insert(o.end(), bytes, bytes + n);
        return true;
    });
}
inline bool get_buffer(const std::vector<uint8_t>& b, size_t& at, ConversationBuffer& p, const Limits& limits,
                       uint64_t& payload_total) {
    uint64_t n = 0;
    if (!get_u64(b, at, n) || n > b.size() - at) return false;
    // every allocation is bounded BEFORE it happens: a corrupted count must not reach resize()
    if (!limits.max_tensor_bytes || n > limits.max_tensor_bytes) return false;
    if (n > limits.max_total_payload - payload_total) return false;
    payload_total += n;
    p.resize(size_t(n));
    size_t copied = 0;
    p.visit(0, size_t(n), [&](uint8_t* dst, size_t chunk, size_t) {
        std::memcpy(dst, b.data() + at + copied, chunk);
        copied += chunk;
        return true;
    });
    at += size_t(n);
    return copied == n && p.size() == size_t(n);
}

inline void put_snapshot(std::vector<uint8_t>& o, const SnapshotDescriptor& s) {
    put_key(o, s.key);
    put_cut(o, s.cut);
    put_str(o, s.native_layout);
    put_tokens(o, s.accepted_tokens);
    put_tokens(o, s.pending_tokens);
    put_u64(o, s.devices.size());
    for (const auto& d : s.devices) {
        put_u32(o, d.id); put_u64(o, uint64_t(d.layer_begin)); put_u64(o, uint64_t(d.layer_end)); put_cut(o, d.cut);
        put_u64(o, d.tensors.size());
        for (const auto& t : d.tensors) {
            put_tensor_layout(o, t.layout);
            put_buffer(o, t.payload);
        }
    }
    put_u64(o, s.approximation.size());
    for (const auto& a : s.approximation) {
        put_u64(o, a.begin); put_u64(o, a.end); put_u64(o, a.first_unresolved_unix_ms); put_digest(o, a.ancestor);
    }
    put_u32(o, uint32_t(s.continuation.mode));
    put_str(o, s.continuation.adapter);
    put_u64(o, s.continuation.payload.size());
    o.insert(o.end(), s.continuation.payload.begin(), s.continuation.payload.end());
    put_u32(o, s.speculation_disabled ? 1u : 0u);
    put_u32(o, s.steering_enabled ? 1u : 0u);
}
inline bool get_snapshot(const std::vector<uint8_t>& b, size_t& at, SnapshotDescriptor& s,
                         const Limits& limits) {
    if (!get_key(b, at, s.key) || !get_cut(b, at, s.cut) || !get_str(b, at, s.native_layout) ||
        !get_tokens(b, at, s.accepted_tokens) || !get_tokens(b, at, s.pending_tokens))
        return false;
    // Inventory completeness is a LOAD-time property, not something the comparator is asked to discover after a
    // silently shortened record: a snapshot with no stages, or a stage with no tensors, is refused here.
    uint64_t devices = 0;
    if (!get_u64(b, at, devices) || devices == 0 || devices > 8) return false;
    uint64_t payload_total = 0;
    s.devices.resize(size_t(devices));
    for (uint64_t i = 0; i < devices; ++i) {
        auto& d = s.devices[size_t(i)];
        uint32_t id = 0;
        uint64_t begin = 0, end = 0, tensors = 0;
        if (!get_u32(b, at, id) || !get_u64(b, at, begin) || !get_u64(b, at, end) || !get_cut(b, at, d.cut) ||
            !get_u64(b, at, tensors) || tensors == 0 || tensors > 65536)
            return false;
        d.id = id;
        d.layer_begin = int64_t(begin);
        d.layer_end = int64_t(end);
        d.tensors.resize(size_t(tensors));
        for (uint64_t j = 0; j < tensors; ++j) {
            auto& t = d.tensors[size_t(j)];
            if (!get_tensor_layout(b, at, t.layout) || !get_buffer(b, at, t.payload, limits, payload_total))
                return false;
        }
    }
    uint64_t regions = 0;
    if (!get_u64(b, at, regions) || regions > 4096) return false;
    s.approximation.resize(size_t(regions));
    for (uint64_t i = 0; i < regions; ++i) {
        auto& a = s.approximation[size_t(i)];
        if (!get_u64(b, at, a.begin) || !get_u64(b, at, a.end) ||
            !get_u64(b, at, a.first_unresolved_unix_ms) || !get_digest(b, at, a.ancestor))
            return false;
    }
    uint32_t mode = 0, spec = 0, steer = 0;
    uint64_t payload = 0;
    if (!get_u32(b, at, mode) || !get_str(b, at, s.continuation.adapter) || !get_u64(b, at, payload) ||
        payload > b.size() - at)
        return false;
    s.continuation.mode = ContinuationMode(mode);
    s.continuation.payload.assign(b.begin() + long(at), b.begin() + long(at + payload));
    at += size_t(payload);
    if (!get_u32(b, at, spec) || !get_u32(b, at, steer)) return false;
    s.speculation_disabled = spec != 0;
    s.steering_enabled = steer != 0;
    return at == b.size();   // the whole body must be consumed: trailing bytes mean a foreign or padded record
}

inline StateDigest digest_of(const std::vector<uint8_t>& bytes) {
    WindowHasher h;
    h.update(bytes.data(), bytes.size());
    return h.digest();
}

inline std::vector<uint8_t> encode(const Header& header, const std::vector<uint8_t>& body) {
    std::vector<uint8_t> o;
    put_u32(o, kSchema); put_u32(o, kVersion);
    put_u32(o, uint32_t(header.kind)); put_u32(o, uint32_t(header.phase));
    put_u64(o, header.request_ordinal); put_str(o, header.producer_binary);
    put_str(o, header.producer_source_manifest);
    put_u64(o, body.size());
    o.insert(o.end(), body.begin(), body.end());
    put_digest(o, digest_of(o));
    return o;
}

/// Refuses a foreign schema/version, an over-limit record, a truncated body, a tampered digest or trailing
/// garbage. The digest is verified BEFORE any output is committed, so `header`/`body` stay untouched on refusal.
inline bool decode(const std::vector<uint8_t>& bytes, Header& header, std::vector<uint8_t>& body,
                   const Limits& limits) {
    if (!limits.max_record_bytes || bytes.size() > limits.max_record_bytes) return false;
    size_t at = 0;
    uint32_t schema = 0, version = 0, kind = 0, phase = 0;
    uint64_t ordinal = 0, body_size = 0;
    std::string producer, manifest;
    if (!get_u32(bytes, at, schema) || !get_u32(bytes, at, version) || schema != kSchema || version != kVersion ||
        !get_u32(bytes, at, kind) || !get_u32(bytes, at, phase) || !get_u64(bytes, at, ordinal) ||
        !get_str(bytes, at, producer) || !get_str(bytes, at, manifest) || !get_u64(bytes, at, body_size))
        return false;
    if (body_size > bytes.size() - at) return false;
    if (!limits.max_body_bytes || body_size > limits.max_body_bytes) return false;
    const size_t body_at = at;
    at += size_t(body_size);
    StateDigest want{};
    if (!get_digest(bytes, at, want) || at != bytes.size()) return false;
    // The digest covers the header AND the body and is verified IN PLACE (no whole-record copy) before anything
    // is committed to the caller's header/body.
    const size_t prefix_end = at - want.size();
    WindowHasher prefix_hash;
    prefix_hash.update(bytes.data(), prefix_end);
    if (prefix_hash.digest() != want) return false;
    if (kind != uint32_t(Kind::snapshot) || phase > uint32_t(Phase::clean_reference)) return false;
    if (phase == uint32_t(Phase::unknown) || ordinal == 0) return false;
    header.kind = Kind(kind);
    header.phase = Phase(phase);
    header.request_ordinal = ordinal;
    header.producer_binary = std::move(producer);
    header.producer_source_manifest = std::move(manifest);
    body.assign(bytes.begin() + long(body_at), bytes.begin() + long(body_at + body_size));
    return true;
}

}   // namespace strata::core::suffix_record
