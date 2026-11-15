#pragma once
#include "strata/core/native_batch_group_completion.hpp"
#include "strata/core/window_hash.hpp"
#include "strata/core/window_state.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>

namespace strata::program {
// Observations only: actual actor/receipt/cut issuance remains elsewhere. These
// functions cannot construct a group receipt or authorize work/stop/retirement.
inline bool http2_request_key(std::string_view key) noexcept {
    if(key.size()!=12)return false;
    for(char c:key)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    return true;
}
inline bool http2_receipts_enabled() noexcept {
    const auto* value=std::getenv("STRATA_HTTP_TWO_SLOT_RECEIPTS");return value&&std::strcmp(value,"1")==0;
}
inline void http2_hex(const core::StateDigest& digest,char (&out)[65]) noexcept {
    constexpr char hex[]="0123456789abcdef";
    for(size_t i=0;i<digest.size();++i){out[2*i]=hex[digest[i]>>4];out[2*i+1]=hex[digest[i]&15];}out[64]=0;
}
inline core::StateDigest http2_input_digest(std::span<const int64_t> input) {
    core::WindowHasher h;constexpr char domain[]="strata-window-history-v1";h.update(domain,sizeof(domain));
    uint8_t count[8];for(unsigned j=0;j<8;++j)count[j]=uint8_t(uint64_t(input.size())>>(8*j));h.update(count,sizeof(count));
    std::array<uint8_t,4096> buffer{};size_t used=0;
    for(auto token:input){for(unsigned j=0;j<4;++j)buffer[used++]=uint8_t(uint32_t(token)>>(8*j));
        if(used==buffer.size()){h.update(buffer.data(),used);used=0;}}
    h.update(buffer.data(),used);return h.digest();
}
inline void report_http2_accept(std::string_view request,uint64_t leg,size_t slot,uint64_t occupancy,
    uint64_t full_budget,std::span<const int64_t> validated_input,const core::StateCut& actual_cut) {
    // Required owned-slot lifecycle protocol, independent of optional tracing.
    if(!http2_request_key(request)||!leg||slot>1||!occupancy||!full_budget){std::fputs("BOWN_ERROR invalid_accept\n",stdout);std::fflush(stdout);return;}
    char input[65],history[65];http2_hex(http2_input_digest(validated_input),input);http2_hex(actual_cut.history,history);
    std::fprintf(stdout,"BOWN_ACCEPT {\"request\":\"%.*s\",\"leg\":%llu,\"slot\":%zu,\"occupancy\":%llu,\"full_budget\":%llu,\"input_count\":%zu,\"input_sha256\":\"%s\",\"cut\":%llu,\"forwarded\":%llu,\"accepted\":%llu,\"watermark\":%llu,\"history\":\"%s\"}\n",
        int(request.size()),request.data(),(unsigned long long)leg,slot,(unsigned long long)occupancy,(unsigned long long)full_budget,validated_input.size(),input,
        (unsigned long long)actual_cut.id,(unsigned long long)actual_cut.forwarded,(unsigned long long)actual_cut.accepted,(unsigned long long)actual_cut.output_watermark,history);std::fflush(stdout);
}
inline void report_http2_group(const core::NativeBatchGroupCompletion& receipt) {
    if(!http2_receipts_enabled())return;
    std::fprintf(stdout,"BOWN_GROUP {\"op\":%llu,\"stages\":%zu,\"active_rows\":%zu,\"rows\":[",
        (unsigned long long)receipt.operation(),receipt.stages(),receipt.rows());
    for(size_t row=0;row<receipt.rows();++row){const int slot=receipt.slot(row);
        std::fprintf(stdout,"%s{\"slot\":%d,\"occupancy\":%llu,\"position\":%lld}",row?",":"",slot,
            (unsigned long long)receipt.occupancy_incarnation(size_t(slot)),(long long)receipt.position(row));}
    std::fputs("]}\n",stdout);std::fflush(stdout);
}
inline void report_http2_commit(std::string_view request,const core::NativeBatchGroupCompletion& receipt,size_t slot,const core::StateCut& actual_cut) {
    if(!http2_receipts_enabled())return;
    size_t row=receipt.rows();for(size_t i=0;i<receipt.rows();++i)if(receipt.slot(i)==int(slot)){row=i;break;}
    if(!http2_request_key(request)||row==receipt.rows()){std::fputs("BOWN_ERROR invalid_commit\n",stdout);std::fflush(stdout);return;}
    char history[65];http2_hex(actual_cut.history,history);
    std::fprintf(stdout,"BOWN_COMMIT {\"request\":\"%.*s\",\"slot\":%zu,\"occupancy\":%llu,\"op\":%llu,\"cut\":%llu,\"forwarded\":%llu,\"accepted\":%llu,\"watermark\":%llu,\"history\":\"%s\",\"sample\":%d}\n",
        int(request.size()),request.data(),slot,(unsigned long long)receipt.occupancy_incarnation(slot),(unsigned long long)receipt.operation(),
        (unsigned long long)actual_cut.id,(unsigned long long)actual_cut.forwarded,(unsigned long long)actual_cut.accepted,(unsigned long long)actual_cut.output_watermark,history,int(receipt.output(row)));std::fflush(stdout);
}
inline void report_http2_known(std::string_view request,const core::NativeBatchGroupCompletion& receipt,size_t slot,const core::StateCut& actual_cut,int32_t next_known) {
    if(!http2_receipts_enabled())return;
    size_t row=receipt.rows();for(size_t i=0;i<receipt.rows();++i)if(receipt.slot(i)==int(slot)){row=i;break;}
    if(!http2_request_key(request)||row==receipt.rows()){std::fputs("BOWN_ERROR invalid_commit\n",stdout);std::fflush(stdout);return;}
    char history[65];http2_hex(actual_cut.history,history);
    std::fprintf(stdout,"BOWN_KNOWN {\"request\":\"%.*s\",\"slot\":%zu,\"occupancy\":%llu,\"op\":%llu,\"cut\":%llu,\"forwarded\":%llu,\"accepted\":%llu,\"watermark\":%llu,\"history\":\"%s\",\"next_known\":%d}\n",
        int(request.size()),request.data(),slot,(unsigned long long)receipt.occupancy_incarnation(slot),(unsigned long long)receipt.operation(),
        (unsigned long long)actual_cut.id,(unsigned long long)actual_cut.forwarded,(unsigned long long)actual_cut.accepted,(unsigned long long)actual_cut.output_watermark,history,int(next_known));std::fflush(stdout);
}
inline void report_http2_stop(std::string_view request,size_t slot,uint64_t occupancy,uint64_t last_op) {
    if(!http2_receipts_enabled())return;
    if(!http2_request_key(request)){std::fputs("BOWN_ERROR invalid_stop\n",stdout);std::fflush(stdout);return;}
    std::fprintf(stdout,"BOWN_STOP {\"request\":\"%.*s\",\"slot\":%zu,\"occupancy\":%llu,\"last_op\":%llu}\n",
        int(request.size()),request.data(),slot,(unsigned long long)occupancy,(unsigned long long)last_op);std::fflush(stdout);
}
inline void report_http2_done(std::string_view request,size_t slot,uint64_t occupancy,const char* reason,const core::StateCut& retired_cut) {
    // Required retirement acknowledgement: the server cannot release the slot before this.
    if(!http2_request_key(request)||(!reason||(std::strcmp(reason,"cancel")&&std::strcmp(reason,"length")&&std::strcmp(reason,"stop")&&std::strcmp(reason,"shutdown")&&std::strcmp(reason,"nonfinite")))){
        std::fputs("BOWN_ERROR invalid_done\n",stdout);std::fflush(stdout);return;}
    char history[65];http2_hex(retired_cut.history,history);
    std::fprintf(stdout,"BOWN_DONE {\"request\":\"%.*s\",\"slot\":%zu,\"occupancy\":%llu,\"reason\":\"%s\",\"watermark\":%llu,\"history\":\"%s\"}\n",
        int(request.size()),request.data(),slot,(unsigned long long)occupancy,reason,(unsigned long long)retired_cut.output_watermark,history);std::fflush(stdout);
}
} // namespace strata::program
