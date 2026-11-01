#include "suffix_experiment_input.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace strata::program;
static void check(bool ok,const char* why) {
    if(!ok) { std::fprintf(stderr,"FAIL %s\n",why); std::exit(1); }
}
int main() {
    const std::string valid="STRATA_SUFFIX_EXPERIMENT_V1 boundary 4 old 9 1 2 3 4 5 6 7 8 9 edited 9 1 20 30 4 5 6 7 8 9";
    SuffixExperimentInput input;
    check(parse_suffix_experiment_input(valid,9,valid.size(),input),"bounded input parses");
    check(suffix_experiment_fallback_reason(input).empty(),"one changed run and aligned suffix");
    const auto preserved=input.edited_tokens;
    for(const auto& bad : {valid+" junk",valid+" 0",std::string("STRATA_SUFFIX_EXPERIMENT_V1 boundary -1"),
                          std::string("STRATA_SUFFIX_EXPERIMENT_V1 boundary 4 old 18446744073709551616"),
                          std::string("STRATA_SUFFIX_EXPERIMENT_V1 boundary 4 old 1 2147483648 edited 1 0"),
                          std::string("STRATA_SUFFIX_EXPERIMENT_V1 boundary 4 old 2 1 edited 1 0")}) {
        check(!parse_suffix_experiment_input(bad,100,1024,input),"malformed input refuses");
        check(input.edited_tokens==preserved&&input.boundary==4,"parse failure leaves output unchanged");
    }
    check(!parse_suffix_experiment_input(valid,8,valid.size(),input),"token count bound");
    check(!parse_suffix_experiment_input(valid,9,valid.size()-1,input),"text byte bound");
    for(uint64_t remainder=1;remainder<4;++remainder) {
        auto changed=input;changed.boundary=4+remainder;
        check(!suffix_experiment_fallback_reason(changed).empty(),"every boundary remainder falls back");
        changed=input;changed.old_tokens.resize(9+remainder,9);changed.edited_tokens.resize(9+remainder,9);
        check(!suffix_experiment_fallback_reason(changed).empty(),"every endpoint remainder falls back");
    }
    auto changed=input;changed.edited_tokens.pop_back();
    check(suffix_experiment_fallback_reason(changed)=="unequal-length edit","unequal edit is exact fallback");
    changed=input;changed.edited_tokens[8]=10;
    check(suffix_experiment_fallback_reason(changed)=="suffix differs at original positions","fresh tail identity preserved");
    changed=input;changed.edited_tokens[2]=3;changed.edited_tokens[3]=40;
    check(suffix_experiment_fallback_reason(changed)=="multiple changed runs","separated edits fall back");
    changed=input;changed.edited_tokens=changed.old_tokens;
    check(suffix_experiment_fallback_reason(changed)=="no changed middle","unchanged input is not approximate reuse");
    changed=input;changed.boundary=UINT64_MAX;
    check(!suffix_experiment_fallback_reason(changed).empty(),"huge boundary never indexes input");
    check(parse_suffix_experiment_input(valid+"\n\t",9,valid.size()+2,input),"trailing ASCII whitespace");
#if defined(__linux__)
    char path[]="/tmp/strata-suffix-input-XXXXXX";
    const int fd=::mkstemp(path);
    check(fd>=0,"private regular fixture file");
    check(::write(fd,valid.data(),valid.size())==static_cast<ssize_t>(valid.size()),"write fixture");
    check(::close(fd)==0,"close fixture writer");
    check(read_suffix_experiment_input(path,9,valid.size(),input)==SuffixInputReadStatus::ok,"bounded regular file read");
    check(read_suffix_experiment_input(path,9,valid.size()-1,input)==SuffixInputReadStatus::too_large,"file extent cap before parsing");
    check(input.edited_tokens==preserved,"read refusal preserves prior parsed configuration");
    check(::unlink(path)==0,"remove regular fixture");
    check(::mkfifo(path,0600)==0,"create no-writer FIFO fixture");
    check(read_suffix_experiment_input(path,9,valid.size(),input)==SuffixInputReadStatus::not_regular,"FIFO refuses without waiting for writer");
    check(::unlink(path)==0,"remove FIFO fixture");
    check(read_suffix_experiment_input(path,9,valid.size(),input)==SuffixInputReadStatus::open_failed,"missing source refuses");
    check(read_suffix_experiment_input("/tmp",9,valid.size(),input)==SuffixInputReadStatus::not_regular,"directory is not token input");
#endif
    std::puts("suffix_experiment_input_test passed");
}
