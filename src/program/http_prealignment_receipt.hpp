#pragma once
#include "http_two_slot_receipt.hpp"
namespace strata::program {
inline void prealignment_cut(const char* field,const core::StateCut& cut) noexcept {
    char history[65];http2_hex(cut.history,history);
    std::fprintf(stdout,"\"%s\":{\"cut\":%llu,\"forwarded\":%llu,\"accepted\":%llu,\"exact\":%llu,\"watermark\":%llu,\"history\":\"%s\"}",field,
        (unsigned long long)cut.id,(unsigned long long)cut.forwarded,(unsigned long long)cut.accepted,
        (unsigned long long)cut.exact_frontier,(unsigned long long)cut.output_watermark,history);
}
inline void report_prealignment_begin(std::string_view request,uint64_t leg,size_t slot,uint64_t occupancy,
        uint64_t budget,std::span<const int64_t> input,const core::StateCut& original_source,
        const core::StateCut& actual_installed,const core::StateCut& actual_prefix,uint64_t planned_u) {
    if(!http2_receipts_enabled())return;
    if(!http2_request_key(request)||!leg||slot>1||!occupancy){std::fputs("BPREALIGN_ERROR begin\n",stdout);std::fflush(stdout);return;}
    char hash[65];http2_hex(http2_input_digest(input),hash);
    std::fprintf(stdout,"BPREALIGN_BEGIN {\"request\":\"%.*s\",\"leg\":%llu,\"slot\":%zu,\"occupancy\":%llu,\"full_budget\":%llu,\"input_count\":%zu,\"input_sha256\":\"%s\",\"page\":4,\"planned_u\":%llu,",
        int(request.size()),request.data(),(unsigned long long)leg,slot,(unsigned long long)occupancy,
        (unsigned long long)budget,input.size(),hash,(unsigned long long)planned_u);
    prealignment_cut("source",original_source);std::fputc(',',stdout);prealignment_cut("installed",actual_installed);
    std::fputc(',',stdout);prealignment_cut("prefix",actual_prefix);
    std::fputs("}\n",stdout);std::fflush(stdout);
}
inline void report_prealignment_known(std::string_view request,const core::NativeBatchGroupCompletion& group,
        size_t slot,const core::StateCut& actual,int32_t next_known) noexcept {
    if(!http2_receipts_enabled())return;
    size_t row=group.rows();for(size_t i=0;i<group.rows();++i)if(group.slot(i)==int(slot)){row=i;break;}
    if(!http2_request_key(request)||row==group.rows()){std::fputs("BPREALIGN_ERROR known\n",stdout);std::fflush(stdout);return;}
    std::fprintf(stdout,"BPREALIGN_KNOWN {\"request\":\"%.*s\",\"slot\":%zu,\"occupancy\":%llu,\"op\":%llu,\"position\":%lld,\"input\":%d,\"next_known\":%d,",
        int(request.size()),request.data(),slot,(unsigned long long)group.occupancy_incarnation(slot),
        (unsigned long long)group.operation(),(long long)group.position(row),int(group.input(row)),int(next_known));
    prealignment_cut("actual",actual);std::fputs("}\n",stdout);std::fflush(stdout);
}
inline void report_prealignment_seal(std::string_view request,size_t slot,uint64_t occupancy,const core::StateCut& captured) noexcept {
    if(!http2_receipts_enabled())return;
    if(!http2_request_key(request)){std::fputs("BPREALIGN_ERROR seal\n",stdout);std::fflush(stdout);return;}
    std::fprintf(stdout,"BPREALIGN_SEAL {\"request\":\"%.*s\",\"slot\":%zu,\"occupancy\":%llu,",int(request.size()),request.data(),slot,(unsigned long long)occupancy);
    prealignment_cut("captured",captured);std::fputs("}\n",stdout);std::fflush(stdout);
}
inline void report_prealignment_abort(std::string_view request,size_t slot,uint64_t occupancy,uint64_t op,
        const char* reason,const char* phase,const core::StateCut& retired) noexcept {
    if(!http2_receipts_enabled())return;
    if(!http2_request_key(request)||!phase||(std::strcmp(phase,"donor")&&std::strcmp(phase,"prefix")&&std::strcmp(phase,"imported"))||!reason||(std::strcmp(reason,"cancel")&&std::strcmp(reason,"refusal")&&std::strcmp(reason,"shutdown"))){
        std::fputs("BPREALIGN_ERROR abort\n",stdout);std::fflush(stdout);return;}
    std::fprintf(stdout,"BPREALIGN_ABORT {\"request\":\"%.*s\",\"slot\":%zu,\"occupancy\":%llu,\"last_op\":%llu,\"reason\":\"%s\",\"phase\":\"%s\",",
        int(request.size()),request.data(),slot,(unsigned long long)occupancy,(unsigned long long)op,reason,phase);
    prealignment_cut("retired",retired);std::fputs("}\n",stdout);std::fflush(stdout);
}
} // namespace strata::program
