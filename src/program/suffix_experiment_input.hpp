#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::program {
// Standalone experiment input, deliberately separate from GEN/GENI and the
// shared exact conversation cache. Token IDs avoid tokenizer/template ambiguity.
struct SuffixExperimentInput {
    uint64_t boundary = 0;
    std::vector<int32_t> old_tokens, edited_tokens;
};

// Grammar (ASCII whitespace between fields):
// STRATA_SUFFIX_EXPERIMENT_V1 boundary B old N <N IDs> edited M <M IDs>
// Caller bounds file reading before constructing text. Parsing is transactional;
// unsupported edit shapes remain valid input for explicit exact fallback.
inline bool parse_suffix_experiment_input(std::string_view text, uint64_t max_tokens,
                                         uint64_t max_bytes, SuffixExperimentInput& out) {
    if (!max_tokens || text.size() > max_bytes) return false;
    auto space=[](char c) { return c==' ' || c=='\t' || c=='\n' || c=='\r'; };
    size_t at=0;
    auto word=[&]() {
        while(at<text.size() && space(text[at])) ++at;
        const size_t begin=at;
        while(at<text.size() && !space(text[at])) ++at;
        return text.substr(begin,at-begin);
    };
    auto number=[&](uint64_t& value) {
        const auto token=word();
        if(token.empty()) return false;
        const auto result=std::from_chars(token.data(),token.data()+token.size(),value);
        return result.ec==std::errc{} && result.ptr==token.data()+token.size();
    };
    try {
        SuffixExperimentInput candidate;
        if(word()!="STRATA_SUFFIX_EXPERIMENT_V1" || word()!="boundary" ||
           !number(candidate.boundary)) return false;
        auto tokens=[&](std::string_view label,std::vector<int32_t>& ids) {
            uint64_t count=0;
            if(word()!=label || !number(count) || !count || count>max_tokens ||
               count>ids.max_size() || count>text.size()) return false;
            ids.reserve(static_cast<size_t>(count));
            for(uint64_t i=0;i<count;++i) {
                uint64_t token=0;
                if(!number(token) || token>uint64_t(INT32_MAX)) return false;
                ids.push_back(static_cast<int32_t>(token));
            }
            return true;
        };
        if(!tokens("old",candidate.old_tokens) || !tokens("edited",candidate.edited_tokens) ||
           !word().empty()) return false;
        out=std::move(candidate);
        return true;
    } catch(const std::bad_alloc&) { return false; }
}

enum class SuffixInputReadStatus { ok, unsupported, invalid_limits, open_failed,
                                 not_regular, too_large, read_failed, malformed, allocation_failed };
// The first experiment is Linux-only. Open nonblocking BEFORE inspecting the
// descriptor so a FIFO/device cannot stall input preflight. Size is also checked
// while reading: a growing file cannot bypass the caller's explicit byte cap.
// This owns one bounded token configuration, not the later RAW image budget.
inline SuffixInputReadStatus read_suffix_experiment_input(const char* path,
        uint64_t max_tokens,uint64_t max_bytes,SuffixExperimentInput& out) {
#if !defined(__linux__)
    (void)path;(void)max_tokens;(void)max_bytes;(void)out;
    return SuffixInputReadStatus::unsupported;
#else
    if(!path || !*path || !max_tokens || !max_bytes ||
       max_bytes>uint64_t(std::string{}.max_size())) return SuffixInputReadStatus::invalid_limits;
    const int fd=::open(path,O_RDONLY|O_CLOEXEC|O_NONBLOCK);
    if(fd<0) return SuffixInputReadStatus::open_failed;
    struct Close { int fd; ~Close(){::close(fd);} } close{fd};
    struct stat info{};
    if(::fstat(fd,&info)!=0) return SuffixInputReadStatus::read_failed;
    if(!S_ISREG(info.st_mode)) return SuffixInputReadStatus::not_regular;
    if(info.st_size<0 || uint64_t(info.st_size)>max_bytes) return SuffixInputReadStatus::too_large;
    try {
        std::string text;
        // Reserve only the observed regular-file extent, not an arbitrary cap.
        text.reserve(static_cast<size_t>(info.st_size));
        char chunk[4096];
        for(;;) {
            const auto n=::read(fd,chunk,sizeof(chunk));
            if(n<0) {
                if(errno==EINTR) continue;
                return SuffixInputReadStatus::read_failed;
            }
            if(!n) break;
            if(uint64_t(n)>max_bytes-text.size()) return SuffixInputReadStatus::too_large;
            text.append(chunk,static_cast<size_t>(n));
        }
        return parse_suffix_experiment_input(text,max_tokens,max_bytes,out) ?
            SuffixInputReadStatus::ok : SuffixInputReadStatus::malformed;
    } catch(const std::bad_alloc&) { return SuffixInputReadStatus::allocation_failed; }
#endif
}

// Empty reason admits only the TEXT SHAPE, never a native format, state cut,
// resource reservation, or execution permission. Runtime must validate all IDs
// against its actual vocabulary and independently certify those prerequisites.
inline std::string_view suffix_experiment_fallback_reason(const SuffixExperimentInput& input) {
    const auto& old=input.old_tokens;
    const auto& edited=input.edited_tokens;
    if(old.size()!=edited.size()) return "unequal-length edit";
    if(edited.empty()) return "empty prompt";
    const uint64_t endpoint=edited.size()-1; // final token is forwarded freshly
    const uint64_t boundary=input.boundary;
    if(boundary%4 || endpoint%4 || boundary>=endpoint || endpoint-boundary<4)
        return "unaligned or empty imported suffix";
    for(uint64_t i=boundary;i<edited.size();++i)
        if(old[size_t(i)]!=edited[size_t(i)]) return "suffix differs at original positions";
    bool changed=false, ended=false;
    for(uint64_t i=0;i<boundary;++i) {
        if(old[size_t(i)]!=edited[size_t(i)]) {
            if(ended) return "multiple changed runs";
            changed=true;
        } else if(changed) ended=true;
    }
    return changed ? std::string_view{} : std::string_view{"no changed middle"};
}
} // namespace strata::program
