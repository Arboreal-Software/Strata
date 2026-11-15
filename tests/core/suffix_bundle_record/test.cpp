// CPU gate for the qualification bundle RECORD codec (include/strata/core/suffix_bundle_record.hpp).
// Proves a captured all-stage snapshot/layout survives export+load exactly, and that a truncated, tampered,
// shortened or foreign record is REFUSED rather than compared. It proves nothing about the engine's captures:
// the codec is the transport for the strict comparator, not a substitute for it.
#include "strata/core/suffix_bundle_record.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace strata::core;
using namespace strata::core::suffix_record;

const Limits LIMITS{1u << 24, 1u << 23, 1u << 20, 1u << 22};

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++failures; }
}

ConversationBuffer buffer_of(std::vector<uint8_t> bytes) {
    ConversationBuffer b;
    b.resize(bytes.size());
    size_t at = 0;
    b.visit(0, bytes.size(), [&](uint8_t* p, size_t n, size_t) {
        for (size_t i = 0; i < n; ++i) p[i] = bytes[at + i];
        at += n;
        return true;
    });
    return b;
}

SnapshotDescriptor snapshot(size_t stages, size_t tensors_per_stage, uint64_t bytes_each) {
    SnapshotDescriptor s;
    s.key.engine_generation = 7; s.key.window_generation = 1;
    s.key.session_id = "serve"; s.key.thread_id = "main"; s.key.window_id = "one";
    s.key.revision = s.key.append_epoch = s.key.selection_epoch = s.key.provenance_epoch = 1;
    s.key.model_profile[3] = 0xa1; s.key.topology[5] = 0xb2;
    s.cut = StateCut{9, 130, 128, 128, 0, {}};
    s.cut.history[11] = 0xcd;
    s.native_layout = "native-int8-v1";
    for (int32_t i = 0; i < 130; ++i) s.accepted_tokens.push_back(i);
    s.pending_tokens.push_back(128);
    for (size_t d = 0; d < stages; ++d) {
        DevicePart part;
        part.id = uint32_t(d);
        part.layer_begin = int64_t(d * 8);
        part.layer_end = int64_t((d + 1) * 8);
        part.cut = s.cut;
        for (size_t j = 0; j < tensors_per_stage; ++j) {
            TensorPart t;
            t.layout.kind = j ? StatePartKind::qsa_v : StatePartKind::qsa_k;
            t.layout.layer = int64_t(part.layer_begin + int64_t(j));
            t.layout.dtype = "int8"; t.layout.native_layout = s.native_layout;
            t.layout.shape = {64, 128}; t.layout.strides = {128, 1};
            t.layout.logical_begin = 0; t.layout.valid_elements = 128 * 64; t.layout.bytes = bytes_each;
            const size_t wide = static_cast<size_t>(bytes_each);
            std::vector<uint8_t> payload(wide);
            for (size_t k = 0; k < payload.size(); ++k) payload[k] = uint8_t((d * 31 + j * 7 + k) & 0xff);
            t.payload = buffer_of(payload);
            part.tensors.push_back(std::move(t));
        }
        s.devices.push_back(std::move(part));
    }
    return s;
}

bool same(const SnapshotDescriptor& a, const SnapshotDescriptor& b) {
    if (!(a.key == b.key) || !(a.cut == b.cut) || a.native_layout != b.native_layout ||
        a.accepted_tokens != b.accepted_tokens || a.pending_tokens != b.pending_tokens ||
        a.devices.size() != b.devices.size() || a.speculation_disabled != b.speculation_disabled ||
        a.steering_enabled != b.steering_enabled)
        return false;
    for (size_t i = 0; i < a.devices.size(); ++i) {
        const auto& x = a.devices[i]; const auto& y = b.devices[i];
        if (x.id != y.id || x.layer_begin != y.layer_begin || x.layer_end != y.layer_end || !(x.cut == y.cut) ||
            x.tensors.size() != y.tensors.size())
            return false;
        for (size_t j = 0; j < x.tensors.size(); ++j) {
            if (!(x.tensors[j].layout == y.tensors[j].layout)) return false;
            const auto& p = x.tensors[j].payload; const auto& q = y.tensors[j].payload;
            if (p.size() != q.size()) return false;
            std::vector<uint8_t> left(p.size()), right(q.size());
            if (!p.read(left.data(), 0, left.size()) || !q.read(right.data(), 0, right.size())) return false;
            if (left != right) return false;
        }
    }
    return true;
}
}   // namespace

int main() {
    const auto original = snapshot(2, 2, 4096);
    std::vector<uint8_t> body;
    put_snapshot(body, original);
    Header header{Kind::snapshot, Phase::certified_frontier, 2, "binary-sha", "manifest-sha"};
    const auto record = encode(header, body);

    Header back{};
    std::vector<uint8_t> loaded_body;
    check(decode(record, back, loaded_body, LIMITS), "a complete record decodes");
    check(back.kind == Kind::snapshot && back.phase == Phase::certified_frontier && back.request_ordinal == 2 &&
          back.producer_binary == "binary-sha" && back.producer_source_manifest == "manifest-sha",
          "the header round-trips: kind, phase, ordinal and producer identity");
    size_t at = 0;
    SnapshotDescriptor restored;
    check(get_snapshot(loaded_body, at, restored, LIMITS), "the snapshot body decodes fully");
    check(same(original, restored), "every stage, tensor layout and payload byte survives export and load");

    // truncation anywhere in the record must be refused
    check(!decode(std::vector<uint8_t>(record.begin(), record.end() - 1), back, loaded_body, LIMITS),
          "a truncated record is refused");
    check(!decode(std::vector<uint8_t>(record.begin(), record.begin() + 40), back, loaded_body, LIMITS),
          "a header-only fragment is refused");
    // tampering: change one payload byte inside the body
    auto tampered = record;
    tampered[record.size() / 2] ^= 0xff;
    check(!decode(tampered, back, loaded_body, LIMITS), "a tampered payload byte is refused by the record digest");
    // trailing garbage
    auto extended = record;
    extended.push_back(0);
    check(!decode(extended, back, loaded_body, LIMITS), "trailing bytes after the digest are refused");
    // foreign schema / version
    auto foreign = record;
    foreign[0] ^= 0x01;
    check(!decode(foreign, back, loaded_body, LIMITS), "a foreign schema is refused");
    auto future = record;
    future[4] = 9;
    check(!decode(future, back, loaded_body, LIMITS), "an unknown record version is refused");

    // inventory completeness is enforced at load, not discovered after the fact
    std::vector<uint8_t> no_stage;
    put_snapshot(no_stage, snapshot(0, 2, 16));
    size_t probe = 0;
    SnapshotDescriptor empty;
    check(!get_snapshot(no_stage, probe, empty, LIMITS), "a snapshot with zero stages is refused");
    std::vector<uint8_t> no_tensor;
    put_snapshot(no_tensor, snapshot(2, 0, 16));
    probe = 0;
    check(!get_snapshot(no_tensor, probe, empty, LIMITS), "a stage with zero tensors is refused");
    std::vector<uint8_t> nine_stages;
    put_snapshot(nine_stages, snapshot(9, 1, 8));
    probe = 0;
    check(!get_snapshot(nine_stages, probe, empty, LIMITS), "more stages than the layout allows are refused");

    // a wrong cut / history is a byte-level change: the digest refuses the record, and the restored value differs
    auto moved = original;
    moved.cut.exact_frontier = 127;
    std::vector<uint8_t> moved_body;
    put_snapshot(moved_body, moved);
    const auto moved_record = encode(header, moved_body);
    Header moved_header{};
    std::vector<uint8_t> moved_loaded;
    check(decode(moved_record, moved_header, moved_loaded, LIMITS), "an altered cut still forms a valid record");
    probe = 0;
    SnapshotDescriptor moved_snapshot;
    check(get_snapshot(moved_loaded, probe, moved_snapshot, LIMITS) && !(moved_snapshot.cut == original.cut),
          "the altered semantic cut is visible as a difference, not silently equal");

    // ceilings and refusal-without-mutation
    auto big = record;
    big.insert(big.end(), size_t(4096), 0);
    Header untouched{};
    std::vector<uint8_t> untouched_body;
    check(!decode(big, untouched, untouched_body, LIMITS), "a record over the byte ceiling is refused");
    check(untouched.request_ordinal == 0 && untouched_body.empty(),
          "a refused decode leaves the caller's header and body untouched (digest checked before commit)");
    Limits tight{1u << 24, 1u << 23, 32, 1u << 22};
    Header part_header{};
    std::vector<uint8_t> part_body;
    check(decode(record, part_header, part_body, tight), "the outer record still decodes under a small tensor cap");
    size_t probe2 = 0;
    SnapshotDescriptor too_big;
    check(!get_snapshot(part_body, probe2, too_big, tight), "a tensor payload over the per-tensor cap is refused");
    Limits small_total{1u << 24, 1u << 23, 1u << 20, 64};
    probe2 = 0;
    check(!get_snapshot(part_body, probe2, too_big, small_total), "a record over the total payload cap is refused");
    std::vector<uint8_t> padded_body = body;
    padded_body.push_back(0);
    probe2 = 0;
    check(!get_snapshot(padded_body, probe2, too_big, LIMITS), "trailing bytes inside the body are refused");
    Header bad_phase{};
    std::vector<uint8_t> bad_body;
    const auto no_phase = encode(Header{Kind::snapshot, Phase::unknown, 2, "b", "m"}, body);
    check(!decode(no_phase, bad_phase, bad_body, LIMITS), "a record without an explicit phase is refused");
    const auto no_ordinal = encode(Header{Kind::snapshot, Phase::certified_frontier, 0, "b", "m"}, body);
    check(!decode(no_ordinal, bad_phase, bad_body, LIMITS), "a record without a request ordinal is refused");

    {   // an unknown/zero kind is refused: only the snapshot schema is accepted here
        auto zero_kind = record;
        zero_kind[8] = 0;                      // Kind field in the header
        Header bad{};
        std::vector<uint8_t> bad_body;
        check(!decode(zero_kind, bad, bad_body, LIMITS), "a zero/unknown record kind is refused");
        auto foreign_kind = record;
        foreign_kind[8] = 7;
        check(!decode(foreign_kind, bad, bad_body, LIMITS), "an unknown record kind is refused");
    }

    if (failures) { std::printf("suffix bundle record gate FAILED: %d checks\n", failures); return 1; }
    std::printf("suffix bundle record gate PASS (transport only: no engine capture, no comparison verdict)\n");
    return 0;
}
