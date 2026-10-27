#include "strata/core/window_ple_weights.hpp"
#include "strata/kernels/ngram.hpp"

namespace strata::core {
namespace {
bool shape(const WeightRef& w,int64_t columns,int64_t rows) {
    return w.ne0==columns && (w.ne1==rows || (rows==1 && w.ne1==0)) && w.elements==columns*rows;
}
bool aligned(const void* p,uint64_t width) {return p&&reinterpret_cast<uintptr_t>(p)%width==0;}
bool plain(const WeightRef& w,WeightKind kind,uint64_t width) {
    return aligned(w.data,width) && w.resident && !w.quantized() && w.kind==kind &&
        !w.codes_bytes && !w.scales_bytes && !w.offset_bytes && !w.has_offset &&
        // Native pack rows preserve neutral scalar group1 for unquantized
        // weights. Both0 (unspecified) and1 describe the same plain layout.
        (w.group_elems==0 || w.group_elems==1) && !w.code_bias && !w.codebook_iq4nl &&
        w.elements>0 && w.bytes==uint64_t(w.elements)*width && !w.q8_scale;
}
}
bool private_ple_weights(const std::array<const WeightRef*,6>& refs,bool native_key,
                         strata::kernels::PleWeights& out,std::string& error) {
    using namespace strata::kernels;
    for(auto* ref:refs)if(!ref){error="private PLE requires all six weight tensors";return false;}
    const auto& key=*refs[0];const auto& value=*refs[1];const auto& conv=*refs[5];
    if(!shape(key,NG_N_EMBD,NG_HC_DIM)||!shape(value,NG_N_EMBD,NG_N_EMBD)||
       !shape(conv,PLE_CONV_KERNEL,NG_HC_DIM)) {
        error="private PLE projection/convolution geometry mismatch";return false;
    }
    if(!plain(value,WeightKind::Bf16InF32,2)||
       !(plain(conv,WeightKind::F16InF32,2)||plain(conv,WeightKind::Verbatim,2))) {
        error="private PLE value must be BF16 and convolution F16";return false;
    }
    for(size_t i=2;i<5;++i)if((!shape(*refs[i],NG_HC_DIM,1)&&!shape(*refs[i],NG_N_EMBD,NG_HC))||!plain(*refs[i],WeightKind::F32,4)) {
        error="private PLE norms must be complete F32 vectors";return false;
    }
    PleWeights result;
    if(native_key) {
        if(!aligned(key.native_data,4) || (key.native_type!=42 && key.native_type!=18 && key.native_type!=23 && key.native_type!=8)) {
            error="private PLE native key missing or unsupported";return false;
        }
        result.key_native_data=key.native_data;result.key_native_type=key.native_type;
        // Never borrow WeightRef::native_q8_1: the private allocator owns scratch.
    } else if(!key.quantized()) {
        if(!plain(key,WeightKind::Bf16InF32,2)){error="private PLE key must be BF16";return false;}
        result.key_bf16=static_cast<const uint16_t*>(key.data);
    } else {
        const uint64_t codes=uint64_t(NG_N_EMBD)*NG_HC_DIM/4;
        const uint64_t scales=uint64_t(NG_N_EMBD)*NG_HC_DIM/64*sizeof(float);
        if(!aligned(key.data,4) || key.q8_scale || !key.resident || key.kind!=WeightKind::Verbatim || key.code_bits!=2 ||
           key.code_bias!=-1 || key.group_elems!=64 || key.codebook_iq4nl || key.has_offset ||
           key.codes_bytes!=codes || key.scales_bytes!=scales || key.offset_bytes || key.bytes!=codes+scales) {
            error="private PLE canonical key requires complete S2 planes";return false;
        }
        result.key_codes=static_cast<const uint8_t*>(key.data);
        result.key_scales=reinterpret_cast<const float*>(static_cast<const uint8_t*>(key.data)+codes);
    }
    result.value_bf16=static_cast<const uint16_t*>(value.data);
    result.norm_key=static_cast<const float*>(refs[2]->data);
    result.norm_query=static_cast<const float*>(refs[3]->data);
    result.norm_conv=static_cast<const float*>(refs[4]->data);
    result.conv1d_f16=static_cast<const uint16_t*>(conv.data);
    out=result;return true;
}
} // namespace strata::core
