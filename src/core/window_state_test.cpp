#include "strata/core/window_state.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;
namespace {
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
struct Fixture {
    SnapshotDescriptor snapshot;
    WindowStateLayout layout;
    Fixture() {
        snapshot.key.engine_generation = 1;
        snapshot.key.window_generation = 1;
        snapshot.key.session_id = "session";
        snapshot.key.thread_id = "thread";
        snapshot.key.window_id = "window";
        snapshot.key.model_profile[0] = 1;
        snapshot.key.topology[0] = 2;
        snapshot.cut = {1, 3, 2, 2, 9, {3}};
        snapshot.accepted_tokens = {7, 8, 9};
        snapshot.cut.history = window_history_digest(snapshot.accepted_tokens);
        snapshot.pending_tokens = {9};
        snapshot.native_layout = "fixture-v1";
        layout.native_layout = snapshot.native_layout;
        layout.model_profile = snapshot.key.model_profile;
        layout.topology = snapshot.key.topology;
        layout.layers = {WindowStateLayout::LayerKind::qsa, WindowStateLayout::LayerKind::gdn};
        layout.max_tokens = 16;
        layout.max_payload_bytes = 1024;
        layout.stages.push_back({0, 0, 2, true, {}});
        snapshot.devices.push_back({0, 0, 2, snapshot.cut, {}});
        for (auto kind : {StatePartKind::qsa_k, StatePartKind::qsa_v, StatePartKind::qsa_pooled,
                          StatePartKind::qsa_tail, StatePartKind::qsa_spare, StatePartKind::qsa_block_position})
            add(kind, 0);
        add(StatePartKind::gdn_recurrence_and_convolution, 1);
        add(StatePartKind::ple_history, -1);
        add(StatePartKind::ple_previous_tokens, -1);
        add(StatePartKind::ple_current_token, -1);
    }
    void add(StatePartKind kind, int64_t layer) {
        TensorLayout spec{kind, layer, 0, "u8", "fixture-v1", {4}, {1}, 0, 4, 4};
        layout.stages[0].tensors.push_back(spec);
        TensorPart part;
        part.layout = spec;
        part.payload.resize(4, 23);
        snapshot.devices[0].tensors.push_back(std::move(part));
    }
    WindowStateResult validate() { return validate_window_state(snapshot, snapshot.key, layout); }
};
}
int main() {
    Fixture original;
    check(bool(original.validate()), "complete hybrid cut with pending token accepted");
    // All frontier orderings at small lengths, independently of the validator.
    for (uint64_t l = 0; l < 4; ++l)
        for (uint64_t c = 0; c < 5; ++c)
            for (uint64_t f = 0; f < 5; ++f) {
                Fixture x;
                x.snapshot.cut.accepted = l;
                x.snapshot.cut.forwarded = c;
                x.snapshot.cut.exact_frontier = f;
                x.snapshot.accepted_tokens.resize(l, 7);
                x.snapshot.cut.history = window_history_digest(x.snapshot.accepted_tokens);
                x.snapshot.pending_tokens.clear();
                if (c <= l)
                    x.snapshot.pending_tokens.assign(x.snapshot.accepted_tokens.begin() + c, x.snapshot.accepted_tokens.end());
                if (f < c) x.snapshot.approximation.push_back({f, c, 100, {4}});
                x.snapshot.devices[0].cut = x.snapshot.cut;
                check(bool(x.validate()) == (f <= c && c <= l), "frontier ordering matrix");
            }
    {
        auto expected = original.snapshot.key;
        ++expected.window_generation;
        check(validate_window_state(original.snapshot, expected, original.layout).error == WindowStateError::stale_generation,
              "reused window ID rejected");
        expected = original.snapshot.key; ++expected.append_epoch;
        check(validate_window_state(original.snapshot, expected, original.layout).error == WindowStateError::stale_append,
              "append race rejected");
        expected = original.snapshot.key; ++expected.selection_epoch;
        check(validate_window_state(original.snapshot, expected, original.layout).error == WindowStateError::stale_selection,
              "selection race rejected");
    }
    for (size_t i = 0; i < original.snapshot.devices[0].tensors.size(); ++i) {
        Fixture x;
        x.snapshot.devices[0].tensors.erase(x.snapshot.devices[0].tensors.begin() + i);
        check(x.validate().error == WindowStateError::missing_state_part, "every missing hybrid component rejected");
        // Even an expected inventory that omits a mandatory component cannot certify completeness.
        x.layout.stages[0].tensors.erase(x.layout.stages[0].tensors.begin() + i);
        check(x.validate().error == WindowStateError::missing_state_part, "incomplete runtime inventory rejected");
    }
    {
        Fixture x;
        x.layout.stages[0].tensors.push_back(x.layout.stages[0].tensors[0]);
        x.snapshot.devices[0].tensors.push_back(x.snapshot.devices[0].tensors[0]);
        check(x.validate().error == WindowStateError::incompatible_layout, "duplicate layer part rejected");
    }
    {
        Fixture x; x.snapshot.pending_tokens[0] = 8;
        check(x.validate().error == WindowStateError::invalid_cut, "pending value mismatch rejected");
        x.snapshot.pending_tokens.clear();
        check(x.validate().error == WindowStateError::invalid_cut, "absent pending token rejected");
    }
    {
        Fixture x; x.snapshot.accepted_tokens[0] = 42;
        check(x.validate().error == WindowStateError::invalid_cut, "history mutation cannot keep an old digest");
    }
    {
        Fixture x; x.snapshot.devices[0].cut.id++;
        check(x.validate().error == WindowStateError::invalid_cut, "mixed stage cut rejected");
    }
    {
        Fixture x; x.snapshot.continuation.mode = ContinuationMode::exact_continuation;
        check(x.validate().error == WindowStateError::unsupported_continuation, "unsupported continuation rejected");
        x.snapshot.continuation.mode = ContinuationMode::new_generation_policy;
        x.snapshot.continuation.adapter = "unknown-parser";
        check(x.validate().error == WindowStateError::unsupported_continuation, "parser cannot be silently dropped");
    }
    {
        Fixture x; x.layout.max_payload_bytes = 39;
        check(x.validate().error == WindowStateError::limit_exceeded, "whole bundle payload budget enforced");
    }
    {
        Fixture x; x.snapshot.devices[0].tensors[0].payload.resize(3);
        check(x.validate().error == WindowStateError::missing_state_part, "truncated native payload rejected");
    }
    {
        Fixture x; x.snapshot.speculation_disabled = false;
        check(x.validate().error == WindowStateError::unsupported_speculation, "unsupported speculative state rejected");
    }
    {
        BundleHandle bundle;
        check(bool(BundleHandle::freeze(original.snapshot, original.snapshot.key, original.layout, bundle)), "freeze valid cut");
        original.snapshot.devices[0].tensors[0].payload.visit(0, 4, [](uint8_t* p, size_t n, size_t) {
            for (size_t i = 0; i < n; ++i) p[i] = 99;
            return true;
        });
        uint8_t byte = 0;
        bundle.get()->devices[0].tensors[0].payload.read(&byte, 0, 1);
        check(byte == 23, "published bundle owns independent immutable bytes");
        const auto* before = bundle.get();
        original.snapshot.pending_tokens.clear();
        check(!BundleHandle::freeze(original.snapshot, original.snapshot.key, original.layout, bundle), "invalid freeze fails");
        check(bundle.get() == before, "failed freeze preserves previous bundle");
    }
    std::puts("window_state_test passed");
}
