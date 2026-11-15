#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

namespace strata::program {
struct SuffixLogitComparison {
    int32_t exact_argmax=-1, approximate_argmax=-1;
    double max_absolute_error=0, root_mean_square_error=0;
    double exact_top_probability=0, approximate_probability_of_exact_top=0;
    double kl_exact_to_approximate=0;
};
// Allocation-free endpoint diagnostics on identical vocabulary/order. These
// describe one next-token distribution; they are not a task-quality certificate.
// Reject nonfinite input and preserve out on failure. Ties choose the first ID.
inline bool compare_suffix_logits(std::span<const float> exact,
                                  std::span<const float> approximate,
                                  SuffixLogitComparison& out) noexcept {
    if(exact.empty() || exact.size()!=approximate.size() || exact.size()>size_t(INT32_MAX)) return false;
    SuffixLogitComparison result;
    double amax=exact.front(),bmax=approximate.front(),square=0;
    result.exact_argmax=result.approximate_argmax=0;
    for(size_t i=0;i<exact.size();++i) {
        const double a=exact[i],b=approximate[i];
        if(!std::isfinite(a)||!std::isfinite(b)) return false;
        if(a>amax){amax=a;result.exact_argmax=static_cast<int32_t>(i);}
        if(b>bmax){bmax=b;result.approximate_argmax=static_cast<int32_t>(i);}
        const double difference=a-b;
        result.max_absolute_error=std::max(result.max_absolute_error,std::abs(difference));
        square+=difference*difference;
    }
    double asum=0,bsum=0;
    for(size_t i=0;i<exact.size();++i) {
        asum+=std::exp(double(exact[i])-amax);
        bsum+=std::exp(double(approximate[i])-bmax);
    }
    const double alog=std::log(asum),blog=std::log(bsum);
    double divergence=0;
    for(size_t i=0;i<exact.size();++i) {
        const double a=double(exact[i])-amax,b=double(approximate[i])-bmax;
        divergence+=(std::exp(a)/asum)*((a-alog)-(b-blog));
    }
    // Only roundoff-sized negative divergence is clamped; unexpected arithmetic
    // failure must not be reported as an exact distribution match.
    if(!std::isfinite(divergence)||divergence < -1e-10) return false;
    result.kl_exact_to_approximate=std::max(0.0,divergence);
    result.root_mean_square_error=std::sqrt(square/double(exact.size()));
    result.exact_top_probability=1.0/asum;
    result.approximate_probability_of_exact_top=
        std::exp(double(approximate[size_t(result.exact_argmax)])-bmax)/bsum;
    out=result;
    return true;
}
} // namespace strata::program
