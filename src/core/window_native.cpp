#include "strata/core/window_native.hpp"
#include "strata/kernels/ngram.hpp"
#include "conversation_checked.hpp"

#include <cstring>
#include <bit>
#include <limits>
#include <new>

namespace strata::core {
namespace {
constexpr uint64_t page_size = 4, index_block = 4;
constexpr const char* schema = "qwen4exp-native-window-v1";
WindowStateResult fail(WindowStateError e, const char* s) { return {e, s}; }
bool multiply(uint64_t& result, std::initializer_list<uint64_t> factors) {
    size_t n = 0;
    if (!conversation_detail::product(n, factors)) return false;
    result = n;
    return true;
}
std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}
bool supported_geometry(const ModelGeometry& g) {
    if (std::endian::native != std::endian::little || g.family != ModelFamily::Qwen4Exp || g.n_layers <= 0 || g.n_layers > 4096 ||
        g.qsa_interval <= 0 || g.idx_block != int64_t(index_block)) return false;
    for (auto n : geometry_key(g)) if (n <= 0) return false;
    return g.ssm_d_conv > 1 && g.head_dim <= INT32_MAX;
}
bool format_sizes(int format, const ModelGeometry& g, uint64_t cells,
                  uint64_t& k, uint64_t& v, uint64_t& ks, uint64_t& vs) {
    if (format != 0 && format != 1 && format != 2 && format != 3 && format != 16 && format != 17) return false;
    const bool q4 = format == 2, int8 = format == 1 || format == 3 || format == 17;
    if ((q4 || format == 3) && g.head_dim % 32) return false;
    if (int8 && g.head_dim % 64) return false;
    const uint64_t per = q4 ? uint64_t(g.head_dim / 32) * 18 : uint64_t(g.head_dim) * (int8 ? 1 : 2);
    if (!multiply(k, {cells, uint64_t(g.n_head_kv), per}) ||
        !multiply(ks, {cells, uint64_t(g.n_head_kv), int8 ? uint64_t(g.head_dim / 64) * 2 : 0})) return false;
    v = k; vs = ks;
    if (format == 3) {
        if (!multiply(v, {cells, uint64_t(g.n_head_kv), uint64_t(g.head_dim / 32) * 18})) return false;
        vs = 0;
    }
    return true;
}
void add(WindowStateLayout::Stage& stage, StatePartKind kind, int64_t layer,
         uint64_t bytes, uint64_t valid, const std::string& layout, const char* dtype) {
    // Serialized native spans are byte-addressed; valid_elements is the logical
    // token extent for KV and byte extent for endpoint state. No raw pointer or
    // allocator-capacity padding is part of the exported span.
    stage.tensors.push_back({kind, layer, stage.id, dtype, layout, {bytes}, {1}, 0, valid, bytes});
}
bool copy_bytes(ConversationBuffer& out, const uint8_t* bytes, size_t n) {
    out.resize(n);
    return out.visit(0, n, [&](uint8_t* p, size_t count, size_t at) {
        std::memcpy(p, bytes + at, count);
        return true;
    });
}
}

WindowStateResult native_window_layout(const ModelGeometry& g, const std::vector<NativeWindowStage>& stages,
                                      const WindowKey& key, const StateCut& cut, uint64_t max_tokens,
                                      uint64_t max_bytes, WindowStateLayout& out, bool steering_enabled) {
    if (!supported_geometry(g) || stages.empty() || stages.size() > size_t(g.n_layers))
        return fail(WindowStateError::incompatible_layout, "unsupported native model geometry or stage count");
    if (cut.exact_frontier > cut.forwarded || cut.forwarded > cut.accepted || cut.accepted > max_tokens ||
        cut.forwarded > uint64_t(INT64_MAX) - (page_size - 1))
        return fail(WindowStateError::invalid_cut, "native layout cut exceeds bounds");
    const uint64_t cells = (cut.forwarded + page_size - 1) / page_size * page_size;
    const uint64_t rows = cut.forwarded ? cut.forwarded / index_block + 1 : 0;
    uint64_t recurrence, convolution, gdn, pooled, tail, spare, ple;
    if (!multiply(recurrence, {uint64_t(g.ssm_state_size), uint64_t(g.ssm_v_heads), uint64_t(g.ssm_state_size), 4}) ||
        !multiply(convolution, {uint64_t(g.ssm_conv_channels), uint64_t(g.ssm_d_conv - 1), 4}) ||
        recurrence > UINT64_MAX - convolution ||
        !multiply(pooled, {rows, uint64_t(g.idx_key_dim), 4}) ||
        !multiply(tail, {index_block - 1, uint64_t(g.idx_key_dim), 4}) ||
        !multiply(spare, {uint64_t(g.idx_key_dim), 4}) ||
        !multiply(ple, {strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, 4}))
        return fail(WindowStateError::limit_exceeded, "native state byte arithmetic overflow");
    gdn = recurrence + convolution;
    WindowStateLayout result;
    result.native_layout = schema; result.model_profile = key.model_profile; result.topology = key.topology;
    result.max_tokens = max_tokens; result.max_payload_bytes = max_bytes;
    result.steering_enabled = steering_enabled;
    for (int64_t layer = 0; layer < g.n_layers; ++layer)
        result.layers.push_back(is_qsa_layer(g, layer) ? WindowStateLayout::LayerKind::qsa : WindowStateLayout::LayerKind::gdn);
    int64_t previous_end = 0;
    uint64_t total = 0;
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto& config = stages[i];
        if (config.layer_begin != previous_end || config.layer_end <= config.layer_begin || config.layer_end > g.n_layers)
            return fail(WindowStateError::incompatible_layout, "native stages are not a complete contiguous partition");
        previous_end = config.layer_end;
        WindowStateLayout::Stage stage{uint32_t(i), config.layer_begin, config.layer_end, config.ple, {}};
        std::vector<int64_t> qsa;
        for (int64_t layer = config.layer_begin; layer < config.layer_end; ++layer)
            if (is_qsa_layer(g, layer)) qsa.push_back(layer);
            else add(stage, StatePartKind::gdn_recurrence_and_convolution, layer, gdn, gdn,
                     "gdn-recurrence-then-convolution-f32-v1", "f32");
        // SessionState allocates a support QSA state even when the stage executes
        // only GDN layers. Preserve it explicitly under layer=-1.
        if (qsa.empty() && g.n_qsa_layers() > 0) qsa.push_back(-1);
        if (config.qsa_formats.size() != qsa.size())
            return fail(WindowStateError::incompatible_layout, "resolved native QSA allocation count differs");
        for (size_t j = 0; j < qsa.size(); ++j) {
            uint64_t k, v, ks, vs;
            const int format = config.qsa_formats[j];
            if (!format_sizes(format, g, cells, k, v, ks, vs))
                return fail(WindowStateError::incompatible_layout, "unsupported native KV format or byte geometry");
            const auto tag = "qsa-native-format-" + std::to_string(format) + "-page4-head" +
                             std::to_string(g.n_head_kv) + "-dim" + std::to_string(g.head_dim);
            add(stage, StatePartKind::qsa_k, qsa[j], k, cut.forwarded, tag, "native-k");
            add(stage, StatePartKind::qsa_v, qsa[j], v, cut.forwarded, tag, "native-v");
            if (ks) add(stage, StatePartKind::qsa_k_scale, qsa[j], ks, cut.forwarded, tag, "f16");
            if (vs) add(stage, StatePartKind::qsa_v_scale, qsa[j], vs, cut.forwarded, tag, "f16");
            add(stage, StatePartKind::qsa_pooled, qsa[j], pooled, rows, "index-pooled-with-spare-v1", "f32");
            add(stage, StatePartKind::qsa_tail, qsa[j], tail, tail, "index-raw-tail-v1", "f32");
            add(stage, StatePartKind::qsa_spare, qsa[j], spare, spare, "index-spare-v1", "f32");
            add(stage, StatePartKind::qsa_block_position, qsa[j], 4, 1, "index-block-position-v1", "i32");
        }
        if (config.ple) {
            add(stage, StatePartKind::ple_history, -1, ple, ple, "ple-row-fastest-v1", "f32");
            add(stage, StatePartKind::ple_previous_tokens, -1, 8, 2, "ple-oldest-first-v1", "i32");
            add(stage, StatePartKind::ple_current_token, -1, 4, 1, "ple-current-v1", "i32");
        }
        for (const auto& spec : stage.tensors) {
            if (total > max_bytes || spec.bytes > max_bytes - total)
                return fail(WindowStateError::limit_exceeded, "native payload exceeds configured budget");
            total += spec.bytes;
        }
        result.stages.push_back(std::move(stage));
    }
    if (previous_end != g.n_layers)
        return fail(WindowStateError::missing_state_part, "native stages omit model layers");
    out = std::move(result);
    return {};
}

WindowStateResult describe_native_window(const SavedConversation& image, const ModelGeometry& g,
                                        const WindowStateLayout& layout, const SnapshotDescriptor& metadata,
                                        const std::vector<NativePleTokens>& tokens, SnapshotDescriptor& out) {
    if (!supported_geometry(g) || layout.native_layout != schema || layout.stages.empty() ||
        image.stage_images.size() + 1 != layout.stages.size() || tokens.size() != layout.stages.size())
        return fail(WindowStateError::incompatible_layout, "native stage metadata does not match layout");
    if (!metadata.devices.empty() || metadata.cut.forwarded > metadata.accepted_tokens.size())
        return fail(WindowStateError::invalid_cut, "native conversion requires metadata without payloads");
    try {
        SnapshotDescriptor result = metadata;
        result.native_layout = schema;
        for (size_t i = 0; i < layout.stages.size(); ++i) {
            const auto& saved = i == 0 ? image : image.stage_images[i - 1];
            const auto& stage = layout.stages[i];
            if (saved.geometry != geometry_key(g) || saved.cvec != layout.steering_enabled ||
                saved.layer_lo != stage.layer_begin || saved.layer_hi != stage.layer_end ||
                (i && !saved.stage_images.empty()) || !saved.live.stage_parts.empty() || !saved.live.imgs.empty() ||
                saved.live.ids.size() != metadata.cut.forwarded ||
                !std::equal(saved.live.ids.begin(), saved.live.ids.end(), metadata.accepted_tokens.begin()))
                return fail(WindowStateError::incompatible_layout, "native capture identity or forwarded history differs");
            DevicePart part{stage.id, stage.layer_begin, stage.layer_end, metadata.cut, {}};
            std::vector<int64_t> qsa, gdn_layers;
            for (int64_t layer = stage.layer_begin; layer < stage.layer_end; ++layer)
                (is_qsa_layer(g, layer) ? qsa : gdn_layers).push_back(layer);
            if (qsa.empty() && g.n_qsa_layers() > 0) qsa.push_back(-1);
            if (saved.kv.size() != qsa.size())
                return fail(WindowStateError::unsupported_speculation, "native capture must omit draft KV");
            const uint64_t c = metadata.cut.forwarded;
            if (c > uint64_t(INT64_MAX) - (page_size - 1))
                return fail(WindowStateError::invalid_cut, "native cut arithmetic overflow");
            for (size_t j = 0; j < qsa.size(); ++j) {
                const auto& kv = saved.kv[j];
                const auto tag = "qsa-native-format-" + std::to_string(kv.format) + "-page4-head" +
                                 std::to_string(g.n_head_kv) + "-dim" + std::to_string(g.head_dim);
                const auto spec = std::find_if(stage.tensors.begin(), stage.tensors.end(), [&](const auto& t) {
                    return t.kind == StatePartKind::qsa_k && t.layer == qsa[j];
                });
                if (spec == stage.tensors.end() || spec->native_layout != tag || kv.page_size != int64_t(page_size) ||
                    kv.cells != int64_t((c + page_size - 1) / page_size * page_size) || kv.heads != g.n_head_kv ||
                    kv.head_dim != g.head_dim || kv.idx_dim != g.idx_key_dim ||
                    kv.pooled_rows != int64_t(c ? c / index_block + 1 : 0))
                    return fail(WindowStateError::incompatible_layout, "native KV format or position geometry differs");
                uint64_t k, v, ks, vs;
                if (!format_sizes(kv.format, g, uint64_t(kv.cells), k, v, ks, vs) ||
                    kv.k.size() != k || kv.v.size() != v || kv.k_scale.size() != ks || kv.v_scale.size() != vs)
                    return fail(WindowStateError::missing_state_part, "native KV or scale extent differs");
            }
            size_t expected_gdn = 0, expected_tail = 0, expected_spare = 0, expected_pos = 0, expected_ple = 0;
            for (const auto& spec : stage.tensors) {
                TensorPart tensor; tensor.layout = spec;
                const auto q = std::find(qsa.begin(), qsa.end(), spec.layer);
                const size_t ordinal = size_t(q - qsa.begin());
                const ConversationKv* kv = q == qsa.end() ? nullptr : &saved.kv[ordinal];
                const ConversationBuffer* source = nullptr;
                const uint8_t* raw = nullptr;
                const std::vector<uint8_t>* state = nullptr;
                size_t offset = 0;
                switch (spec.kind) {
                case StatePartKind::qsa_k: source = kv ? &kv->k : nullptr; break;
                case StatePartKind::qsa_v: source = kv ? &kv->v : nullptr; break;
                case StatePartKind::qsa_k_scale: source = kv ? &kv->k_scale : nullptr; break;
                case StatePartKind::qsa_v_scale: source = kv ? &kv->v_scale : nullptr; break;
                case StatePartKind::qsa_pooled: source = kv ? &kv->pooled : nullptr; break;
                case StatePartKind::qsa_tail: state = &saved.live.tails; offset = expected_tail; expected_tail += spec.bytes; break;
                case StatePartKind::qsa_spare: state = &saved.live.dead; offset = expected_spare; expected_spare += spec.bytes; break;
                case StatePartKind::qsa_block_position: state = &saved.live.block_pos; offset = expected_pos; expected_pos += spec.bytes; break;
                case StatePartKind::gdn_recurrence_and_convolution:
                    state = &saved.live.gdn; offset = expected_gdn; expected_gdn += spec.bytes; break;
                case StatePartKind::ple_history: state = &saved.live.ple; expected_ple += spec.bytes; break;
                case StatePartKind::ple_previous_tokens: raw = reinterpret_cast<const uint8_t*>(tokens[i].previous.data()); break;
                case StatePartKind::ple_current_token: raw = reinterpret_cast<const uint8_t*>(&tokens[i].current); break;
                }
                if (source) {
                    if (source->size() != spec.bytes)
                        return fail(WindowStateError::missing_state_part, "native span length differs");
                    tensor.payload = *source;
                } else if (state) {
                    if (offset > state->size() || spec.bytes > state->size() - offset)
                        return fail(WindowStateError::missing_state_part, "native endpoint state truncated");
                    copy_bytes(tensor.payload, state->data() + offset, size_t(spec.bytes));
                } else if (raw) {
                    const uint64_t expected = spec.kind == StatePartKind::ple_previous_tokens ? 8 : 4;
                    if (spec.bytes != expected)
                        return fail(WindowStateError::incompatible_layout, "invalid native PLE token span");
                    copy_bytes(tensor.payload, raw, size_t(spec.bytes));
                } else return fail(WindowStateError::missing_state_part, "missing native span source");
                part.tensors.push_back(std::move(tensor));
            }
            if (expected_gdn != saved.live.gdn.size() || expected_tail != saved.live.tails.size() ||
                expected_spare != saved.live.dead.size() || expected_pos != saved.live.block_pos.size() ||
                expected_ple != saved.live.ple.size())
                return fail(WindowStateError::incompatible_layout, "unexpected native endpoint bytes");
            result.devices.push_back(std::move(part));
        }
        auto valid = validate_window_state(result, metadata.key, layout);
        if (!valid) return valid;
        out = std::move(result);
        return {};
    } catch (const std::bad_alloc&) {
        return fail(WindowStateError::limit_exceeded, "native descriptor allocation failed");
    }
}

WindowStateResult materialize_native_window(const SnapshotDescriptor& snapshot, const WindowKey& expected,
                                           const ModelGeometry& g, const std::vector<NativeWindowStage>& stages,
                                           uint64_t max_tokens, uint64_t max_bytes, bool steering_enabled,
                                           SavedConversation& out, std::vector<NativePleTokens>& tokens) {
    WindowStateLayout layout;
    auto status = native_window_layout(g,stages,expected,snapshot.cut,max_tokens,max_bytes,layout,steering_enabled);
    if (!status) return status;
    status = validate_window_state(snapshot,expected,layout);
    if (!status) return status;
    try {
        SavedConversation result;
        std::vector<NativePleTokens> token_result(stages.size());
        result.stage_images.resize(stages.size() - 1);
        for (size_t i=0; i<stages.size(); ++i) {
            auto& saved = i == 0 ? result : result.stage_images[i-1];
            const auto& stage = snapshot.devices[i];
            saved.geometry = geometry_key(g); saved.layer_lo = stage.layer_begin; saved.layer_hi = stage.layer_end;
            saved.cvec = steering_enabled;
            saved.live.ids.assign(snapshot.accepted_tokens.begin(),snapshot.accepted_tokens.begin()+snapshot.cut.forwarded);
            std::vector<int64_t> qsa;
            for (int64_t layer=stage.layer_begin; layer<stage.layer_end; ++layer)
                if (is_qsa_layer(g,layer)) qsa.push_back(layer);
            if (qsa.empty() && g.n_qsa_layers()>0) qsa.push_back(-1);
            saved.kv.resize(qsa.size());
            for (size_t j=0; j<qsa.size(); ++j) {
                auto& kv = saved.kv[j];
                kv.format = stages[i].qsa_formats[j];
                kv.cells = int64_t((snapshot.cut.forwarded + page_size-1)/page_size*page_size);
                kv.heads = g.n_head_kv; kv.head_dim = g.head_dim; kv.page_size = page_size;
                kv.pooled_rows = snapshot.cut.forwarded ? snapshot.cut.forwarded/index_block+1 : 0;
                kv.idx_dim = g.idx_key_dim;
            }
            for (const auto& tensor : stage.tensors) {
                const auto q = std::find(qsa.begin(),qsa.end(),tensor.layout.layer);
                ConversationKv* kv = q == qsa.end() ? nullptr : &saved.kv[size_t(q-qsa.begin())];
                ConversationBuffer* native = nullptr;
                std::vector<uint8_t>* state = nullptr;
                void* scalar = nullptr;
                switch (tensor.layout.kind) {
                case StatePartKind::qsa_k: native = kv ? &kv->k : nullptr; break;
                case StatePartKind::qsa_v: native = kv ? &kv->v : nullptr; break;
                case StatePartKind::qsa_k_scale: native = kv ? &kv->k_scale : nullptr; break;
                case StatePartKind::qsa_v_scale: native = kv ? &kv->v_scale : nullptr; break;
                case StatePartKind::qsa_pooled: native = kv ? &kv->pooled : nullptr; break;
                case StatePartKind::qsa_tail: state = &saved.live.tails; break;
                case StatePartKind::qsa_spare: state = &saved.live.dead; break;
                case StatePartKind::qsa_block_position: state = &saved.live.block_pos; break;
                case StatePartKind::gdn_recurrence_and_convolution: state = &saved.live.gdn; break;
                case StatePartKind::ple_history: state = &saved.live.ple; break;
                case StatePartKind::ple_previous_tokens: scalar = token_result[i].previous.data(); break;
                case StatePartKind::ple_current_token: scalar = &token_result[i].current; break;
                }
                if (native) *native = tensor.payload;
                else if (state) {
                    const size_t old = state->size();
                    if (tensor.payload.size() > state->max_size() - old)
                        return fail(WindowStateError::limit_exceeded,"native reconstruction size overflow");
                    state->resize(old+tensor.payload.size());
                    tensor.payload.read(state->data()+old,0,tensor.payload.size());
                } else if (scalar) tensor.payload.read(scalar,0,tensor.payload.size());
                else return fail(WindowStateError::missing_state_part,"native reconstruction has no destination");
            }
        }
        out = std::move(result);
        tokens = std::move(token_result);
        return {};
    } catch (const std::bad_alloc&) {
        return fail(WindowStateError::limit_exceeded,"private native reconstruction allocation failed");
    }
}
} // namespace strata::core
