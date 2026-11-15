#include "strata/core/window_native.hpp"
#include "window_native_test_fixture.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace strata::core;
namespace {
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
using namespace strata::core::window_test;
void fixture(int format, int split) {
    auto m = metadata();
    std::vector<NativeWindowStage> stages;
    SavedConversation s;
    if (!split) {
        stages = {{0,8,{format,format},true}};
        s = saved(0,8,format,true);
    } else {
        stages = {{0,split,std::vector<int>(1,format),true},
                  {split,8,std::vector<int>(split == 1 ? 2 : 1,format),false}};
        s = saved(0,split,format,true);
        s.stage_images.push_back(saved(split,8,format,false));
    }
    WindowStateLayout layout;
    check(bool(native_window_layout(geometry(),stages,m.key,m.cut,1024,1<<24,layout)), "native layout built");
    const std::vector<NativePleTokens> ple(stages.size(), NativePleTokens{{8,9},10});
    SnapshotDescriptor out;
    auto result = describe_native_window(s,geometry(),layout,m,ple,out);
    if (!result) std::fprintf(stderr, "detail: %s\n",result.detail.c_str());
    check(bool(result), "native capture converted");
    check(bool(validate_window_state(out,m.key,layout)), "converted native bundle validates");
    check(out.pending_tokens == std::vector<int32_t>{10}, "pending accepted token preserved");
    SavedConversation restored;
    std::vector<NativePleTokens> restored_tokens;
    check(bool(materialize_native_window(out,m.key,geometry(),stages,1024,1<<24,true,restored,restored_tokens)),
          "typed bundle reconstructs private native image");
    auto same_stage = [&](const SavedConversation& a, const SavedConversation& b) {
        check(a.geometry == b.geometry && a.cvec == b.cvec && a.layer_lo == b.layer_lo && a.layer_hi == b.layer_hi,
              "roundtrip native geometry and steering");
        check(a.live.ids == b.live.ids && a.live.gdn == b.live.gdn && a.live.ple == b.live.ple &&
              a.live.tails == b.live.tails && a.live.dead == b.live.dead && a.live.block_pos == b.live.block_pos,
              "roundtrip every hybrid endpoint byte");
        check(a.kv.size() == b.kv.size(),"roundtrip native KV layer count");
        for (size_t j=0;j<a.kv.size();++j)
            check(a.kv[j].format == b.kv[j].format && a.kv[j].cells == b.kv[j].cells &&
                  a.kv[j].pooled_rows == b.kv[j].pooled_rows && a.kv[j].k == b.kv[j].k && a.kv[j].v == b.kv[j].v &&
                  a.kv[j].k_scale == b.kv[j].k_scale && a.kv[j].v_scale == b.kv[j].v_scale &&
                  a.kv[j].pooled == b.kv[j].pooled,"roundtrip every native KV/code/scale/index byte");
    };
    same_stage(s,restored);
    for (size_t i=0;i<s.stage_images.size();++i) same_stage(s.stage_images[i],restored.stage_images[i]);
    check(restored_tokens[0].previous == ple[0].previous && restored_tokens[0].current == ple[0].current,
          "roundtrip PLE token metadata");
    check(!materialize_native_window(out,m.key,geometry(),stages,1024,1<<24,false,restored,restored_tokens),
          "steering mismatch cannot silently change state meaning");
    for (const auto& stage : out.devices) for (const auto& tensor : stage.tensors) {
        if (tensor.layout.kind == StatePartKind::gdn_recurrence_and_convolution) {
            uint8_t byte = 255;
            tensor.payload.read(&byte,0,1);
            check(byte == tensor.layout.layer, "each GDN row keeps global layer identity");
            check(tensor.payload.size() == 128, "native GDN recurrence plus convolution byte count");
        }
        if (tensor.layout.kind == StatePartKind::ple_previous_tokens) {
            std::array<int32_t,2> previous{};
            tensor.payload.read(previous.data(),0,8);
            check(previous == ple[0].previous, "actual PLE previous tokens retained");
        }
    }
    const auto original_output = out.cut;
    auto accounting = std::make_shared<int>(1);
    std::weak_ptr<int> accounted = accounting;
    BundleHandle frozen;
    check(bool(BundleHandle::freeze(out,m.key,layout,frozen,accounting)), "native bundle accepts accounting owner");
    accounting.reset();
    auto retained = frozen; frozen = {};
    check(!accounted.expired(), "copied bundle retains memory accounting");
    retained = {};
    check(accounted.expired(), "last bundle releases accounting after payload");
    auto bad = s; bad.kv[0].format = format == 0 ? 16 : 0;
    check(!describe_native_window(bad,geometry(),layout,m,ple,out), "wrong native format rejected even if same byte size");
    check(out.cut == original_output, "failed native conversion preserves destination");
    bad = s; bad.kv[0].pooled_rows++;
    check(!describe_native_window(bad,geometry(),layout,m,ple,out), "pooled geometry rejected");
    bad = s; bad.live.tails.pop_back();
    check(!describe_native_window(bad,geometry(),layout,m,ple,out), "missing raw index tail rejected");
    bad = s; bad.kv.push_back(bad.kv[0]);
    check(!describe_native_window(bad,geometry(),layout,m,ple,out), "undeclared draft state rejected");
    if (split) {
        bad = s; bad.stage_images[0].live.ids.back() = 88;
        check(!describe_native_window(bad,geometry(),layout,m,ple,out), "stage history skew rejected");
    }
    auto huge = geometry(); huge.ssm_state_size = std::numeric_limits<int64_t>::max();
    check(!native_window_layout(huge,stages,m.key,m.cut,1024,UINT64_MAX,layout), "native byte overflow rejected");
    check(!native_window_layout(geometry(),stages,m.key,m.cut,1024,10,layout), "native payload admission refused before allocation");
}
}
int main() {
    for (int format : {0,1,2,3,16,17})
        for (int split : {0,4,1}) fixture(format,split);
    std::puts("window_native_test passed");
}
