#pragma once
#include "strata/core/geometry.hpp"
#include "strata/core/window_hash.hpp"
#include <memory>
#include <string>
#include <vector>
namespace strata::core {
enum class ProfileRole : uint32_t { dense, embedding, head, experts, ple, tokenizer, chat_template, engine };
// Caller/verifier attestation, not proof manufactured by this composer. Owner must
// retain the actual immutable bytes consumed by loaders, never a no-op alias.
struct ProfileArtifact {
    ProfileRole role=ProfileRole::dense;
    std::string name;
    uint64_t bytes=0;
    StateDigest content{}, verification_evidence{};
    bool verified_full_content=false;
    std::shared_ptr<const void> immutable_owner;
};
struct ProfileTensor {
    ProfileRole role=ProfileRole::dense;
    std::string name, artifact, source_type, runtime_layout, conversion;
    uint64_t offset=0, bytes=0;
    std::vector<uint64_t> shape;
    uint64_t codes_bytes=0, scales_bytes=0, offsets_bytes=0;
    int64_t weight_kind=0,code_bits=0,code_bias=0,group_elems=0,activation_kind=0,native_type=-1;
    bool codebook_iq4nl=false,has_offset=false,source_scales_fp16=false;
};
struct ProfilePle {
    bool enabled=false;
    std::vector<uint64_t> layers,multipliers,vocab,offsets;
    uint64_t ngram_size=0,heads_per_ngram=0,row_width=0,conv_kernel=0;
    int64_t null_token=-1,eos_token=-1,image_token=-1;
    std::string format;
    double scale=0;
};
enum class ProfileFrontend : uint32_t {rendered_text,raw_token_ids};
struct ProfileRuntime {
    ProfileFrontend frontend=ProfileFrontend::rendered_text;
    std::string raw_token_abi; // raw mode requires model-vocabulary-index-v1
    std::string engine_version,backend,compiler,toolkit,driver,native_state_layout;
    StateDigest tokenizer_implementation{},template_overrides{};
    std::string kv;
    uint64_t max_context=0,kv_resident=0,batch_groups=0;
    // type, base, factor, scale, original context, extension, attention, beta fast/slow
    int64_t rope_type=0;
    double rope_base=0,rope_factor=0,rope_scale=0,rope_orig_ctx=0,rope_ext=0,rope_attn=0,rope_beta_fast=0,rope_beta_slow=0;
    // Every supported switch must be supplied exactly once; unknown keys refused.
    std::vector<std::pair<std::string,int64_t>> switches;
    bool mtp=false,suffix=false,lookup=false,batch_mtp=false,coupled_draft=false,pipeline_speculation=false,vision=false,steering=false;
};
struct ProfileStage {
    uint32_t id=0;
    uint64_t layer_begin=0,layer_end=0,capacity_bytes=0;
    std::string device_uuid,architecture,transport;
    std::vector<uint32_t> qsa_formats;
    bool ple_owner=false;
};
struct ProfileLimits {uint64_t max_artifacts=0,max_tensors=0,max_string_bytes=0,max_rank=0,max_stages=0,max_layers=0;};
struct WindowProfileInput {
    ModelGeometry geometry;
    std::vector<ProfileArtifact> artifacts;
    std::vector<ProfileTensor> tensors;
    ProfilePle ple;
    ProfileRuntime runtime;
    std::vector<ProfileStage> stages;
};
struct WindowProfileIdentity {
    StateDigest model_profile{},topology{};
    // Retention only. This does not prove loader identity or authorize execution.
    std::vector<std::shared_ptr<const void>> artifact_owners;
};
const std::vector<std::string>& window_profile_switches();
// CPU-only. No path reads/GPU. Complete resolved inputs from trusted loaders are
// required; self-reported digests/flags cannot establish provenance. Output is
// unchanged on refusal. Version2 supports initial text/no-speculation capability.
bool compose_window_profile(const WindowProfileInput&,const ProfileLimits&,WindowProfileIdentity&,std::string& error);
} // namespace strata::core
