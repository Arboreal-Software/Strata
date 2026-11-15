#pragma once
#include "strata/core/window_hash.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>

namespace strata::program {
// Qualification observation only. Called AFTER actual GEN parse/context/vocab validation,
// before native serving work. Never authorizes execution, changes tokens, or emits a cut.
inline void report_http_native_accept(uint64_t ordinal,std::span<const int64_t> ids,uint64_t budget,
    bool supported,float temperature,float top_p,int top_k,float min_p,int penalty_last_n,
    float penalty_repeat,float penalty_freq,float penalty_present,uint64_t seed,bool bias,bool cvec_loaded) {
    const char* enabled=std::getenv("STRATA_HTTP_NATIVE_RECEIPTS");
    if(!enabled||std::strcmp(enabled,"1")!=0)return;
    // Same domain/count/int32-LE SHA-256 as window_history_digest, without an
    // extra full prompt vector. Bounded4KiB stack buffer; no prompt text/IDs logged.
    core::WindowHasher hash;
    constexpr char domain[]="strata-window-history-v1";
    hash.update(domain,sizeof(domain));
    uint8_t count[8];for(unsigned j=0;j<8;++j)count[j]=uint8_t(uint64_t(ids.size())>>(8*j));
    hash.update(count,sizeof(count));
    std::array<uint8_t,4096> buffer{};size_t used=0;
    for(const int64_t id:ids){
        if(id<0||uint64_t(id)>uint64_t(INT32_MAX)){supported=false;break;}
        for(unsigned j=0;j<4;++j)buffer[used++]=uint8_t(uint32_t(id)>>(8*j));
        if(used==buffer.size()){hash.update(buffer.data(),used);used=0;}
    }
    hash.update(buffer.data(),used);
    const auto sum=hash.digest();char hex[65];constexpr char digits[]="0123456789abcdef";
    for(size_t i=0;i<sum.size();++i){hex[i*2]=digits[sum[i]>>4];hex[i*2+1]=digits[sum[i]&15];}hex[64]=0;
    std::fprintf(stdout,"HTTP_ACCEPT {\"schema\":1,\"request\":%llu,\"input_count\":%llu,"
        "\"input_sha256\":\"%s\",\"max_new\":%llu,\"supported_serial_profile\":%s,"
        "\"temperature\":%.9g,\"top_p\":%.9g,\"top_k\":%d,\"min_p\":%.9g,\"penalty_last_n\":%d,"
        "\"penalty_repeat\":%.9g,\"penalty_freq\":%.9g,\"penalty_present\":%.9g,\"seed\":%llu,"
        "\"bias\":%s,\"cvec_loaded\":%s}\n",
        (unsigned long long)ordinal,(unsigned long long)ids.size(),hex,(unsigned long long)budget,
        supported?"true":"false",double(temperature),double(top_p),top_k,double(min_p),penalty_last_n,
        double(penalty_repeat),double(penalty_freq),double(penalty_present),(unsigned long long)seed,
        bias?"true":"false",cvec_loaded?"true":"false");
    std::fflush(stdout);
}
} // namespace strata::program
