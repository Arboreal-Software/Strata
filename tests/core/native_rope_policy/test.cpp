// Compile separately per extracted backend and mode, never run all cached modes
// in one process. No host getenv: selectors see only these literal test values.
#include "strata/kernels/native_rope.hpp"
#include <atomic>
#include <cassert>
#include <cstring>
#ifndef SELECTOR_FILE
#error "SELECTOR_FILE must name fresh extracted backend source"
#endif
static int mode=0;static bool changed=false;
static const char* fake_getenv(const char* key) {
 if(changed)return "0";
 if(!std::strcmp(key,"STRATA_ROPE_TABLE"))return mode==1?"1":nullptr;
 if(!std::strcmp(key,"STRATA_NO_NORM_ROPE"))return mode==2?"1":nullptr;
 if(!std::strcmp(key,"STRATA_NORM_ROPE"))return mode==3?"1":nullptr;
 return nullptr;
}
// Select only the extracted HIP policy branch; this is not HIP compilation.
#ifdef TEST_HIP_SELECTOR
#define __HIPCC__ 1
#endif
namespace strata::kernels {
static std::atomic<bool> enabled{false};
#include SELECTOR_FILE
}
int main(int argc,char** argv) {
 assert(argc==2);mode=argv[1][0]-'0';assert(mode>=0&&mode<=3);
 using namespace strata::kernels;
 const bool fused=
#ifdef __HIPCC__
 mode==3;
#else
 mode!=2;
#endif
 auto a=native_rope_policy_observation(128,64);
 assert(a.supported&&a.head_dim==128&&a.n_rot==64&&!a.native_enabled);
 assert(a.table_policy_enabled==(mode==1)&&a.fused_norm_geometry_usable==fused);
 changed=true;native_rope_set_enabled(true);
 auto b=native_rope_policy_observation(256,64);
 assert(b.native_enabled&&b.table_policy_enabled==a.table_policy_enabled&&b.fused_norm_geometry_usable==fused);
 assert(!native_rope_policy_observation(127,64).fused_norm_geometry_usable);
 assert(!native_rope_policy_observation(128,63).fused_norm_geometry_usable);
 native_rope_set_enabled(false);assert(!native_rope_policy_observation(128,64).native_enabled);
}
