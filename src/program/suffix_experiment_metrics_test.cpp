#include "suffix_experiment_metrics.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
using namespace strata::program;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
int main(){
    const float logarithm=std::log(3.0f);
    const std::array<float,2> a{0,logarithm},b{logarithm,0};
    SuffixLogitComparison result;
    check(compare_suffix_logits(a,b,result),"finite comparison");
    check(result.exact_argmax==1&&result.approximate_argmax==0,"top token disagreement");
    check(std::abs(result.kl_exact_to_approximate-0.5*std::log(3.0))<1e-6,"known KL divergence");
    check(std::abs(result.exact_top_probability-0.75)<1e-6&&
          std::abs(result.approximate_probability_of_exact_top-0.25)<1e-6,"known normalized probabilities");
    check(std::abs(result.root_mean_square_error-logarithm)<1e-7,"known RMS error");
    check(compare_suffix_logits(a,a,result)&&result.kl_exact_to_approximate==0&&result.max_absolute_error==0,"identical logits");
    const std::array<float,2> flat{10000,10000},shifted{-10000,-10000};
    check(compare_suffix_logits(flat,shifted,result)&&result.kl_exact_to_approximate==0&&
          result.exact_argmax==0&&result.approximate_argmax==0,"stable normalization and deterministic ties");
    for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
        const std::array<float,2> bad{0,invalid};const auto prior=result;
        check(!compare_suffix_logits(a,bad,result)&&result.max_absolute_error==prior.max_absolute_error,"nonfinite refuses unchanged");
    }
    check(!compare_suffix_logits({},a,result)&&!compare_suffix_logits(a,std::span<const float>(b).first(1),result),"vocabulary dimensions required");
    std::puts("suffix_experiment_metrics_test passed");
}
