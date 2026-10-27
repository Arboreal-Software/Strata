#include "strata/core/session.hpp"
#include <cstdio>
#include <cstdlib>
using namespace strata::core;
static void check(bool ok,const char* why) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}
}
int main() {
    ModelGeometry g;
    qsa_set_kv_resident(0);qsa_set_kv_int8(true);
    constexpr int64_t cells=262144;
    QsaPrivateHostPlan primary,borrowed;
    check(qsa_private_host_plan(g,cells,true,primary)&&qsa_private_host_plan(g,cells,false,borrowed),"resident host pricing");
    check(primary.pinned_bytes==131072&&borrowed.pinned_bytes==primary.pinned_bytes,"two staging allocations reserve rounded pin capacity");
    check(primary.upload_bytes>borrowed.upload_bytes&&borrowed.upload_bytes>0,"shared RoPE excludes only its upload, not the page table");
    SessionPrivatePlan first,second,whole;
    check(session_private_plan(g,cells,10,0,24,first)&&session_private_plan(g,cells,10,24,48,second)&&
          session_private_plan(g,cells,10,0,48,whole),"resolved split and whole session plans");
    check(first.pinned_bytes==6*primary.pinned_bytes&&whole.pinned_bytes==12*primary.pinned_bytes,"each allocated QSA state charges its host pins");
    check(first.device_bytes==session_bytes(g,cells,10,0,24),"device plan uses actual session carve");
    check(first.host_bytes==second.host_bytes&&first.host_bytes>first.pinned_bytes,"split metadata and uploads charged in physical host total");
    check(whole.host_bytes<first.host_bytes+second.host_bytes,"two stages each reserve their own primary RoPE upload and metadata");
    qsa_set_kv_resident(20480);
    QsaPrivateHostPlan streamed;
    check(qsa_private_host_plan(g,cells,true,streamed)&&streamed.pinned_bytes>primary.pinned_bytes,"streamed authoritative KV host copy priced");
    check(streamed.upload_bytes<primary.upload_bytes,"streamed page-table kernel needs no identity upload buffer");
    auto untouched=first;
    check(!session_private_plan(g,cells,10,24,24,first)&&first.device_bytes==untouched.device_bytes,"bad stage range leaves output intact");
    check(!qsa_private_host_plan(g,0,true,primary),"invalid context refused");
    std::puts("session_private_plan_test passed");
}
