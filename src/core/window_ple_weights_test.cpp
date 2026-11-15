#include "strata/core/window_ple_weights.hpp"
#include "strata/kernels/ngram.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <tuple>
using namespace strata::core;
using namespace strata::kernels;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static auto fields(const PleWeights& w) {
    return std::tie(w.key_codes,w.key_scales,w.value_bf16,w.norm_key,w.norm_query,
        w.norm_conv,w.conv1d_f16,w.key_native_data,w.key_native_type,w.key_native_q8_1,
        w.key_bf16,w.pre_key,w.pre_value);
}
int main() {
    float sentinel=0;
    auto tensor=[&](int64_t n,int64_t m,WeightKind kind,uint64_t width) {
        WeightRef w;w.ne0=n;w.ne1=m;w.elements=n*m;w.kind=kind;w.bytes=uint64_t(n*m)*width;w.data=&sentinel;return w;
    };
    std::array<WeightRef,6> weights{tensor(NG_N_EMBD,NG_HC_DIM,WeightKind::Bf16InF32,2),
        tensor(NG_N_EMBD,NG_N_EMBD,WeightKind::Bf16InF32,2),
        tensor(NG_HC_DIM,1,WeightKind::F32,4),tensor(NG_HC_DIM,1,WeightKind::F32,4),
        tensor(NG_HC_DIM,1,WeightKind::F32,4),tensor(PLE_CONV_KERNEL,NG_HC_DIM,WeightKind::F16InF32,2)};
    std::array<const WeightRef*,6> refs;for(size_t i=0;i<6;++i)refs[i]=&weights[i];
    PleWeights out;std::string error;
    check(private_ple_weights(refs,false,out,error)&&out.key_bf16&&out.value_bf16,"BF16 tensors extracted");
    // Actual IQ3_S pack index metadata (no model payload). WeightTable::load
    // maps raw index kinds4/5 to engine kinds1/3 and preserves group_elems=1
    // even when code_bits=0; it is a neutral scalar group, not quantization.
    const auto synthetic=weights;
    const char* index_rows[]={
        "blk.1.ple_key.weight 0 4 72059776 52428800 72059776 52428800 2560 10240 0 0 1 0 0 0 0 0 0 0",
        "blk.1.ple_value.weight 0 4 124611456 13107200 124611456 13107200 2560 2560 0 0 1 0 0 0 0 0 0 0",
        "blk.1.ple_norm_key.weight 0 2 124529536 40960 124529536 40960 10240 0 0 0 1 0 0 0 0 0 0 0",
        "blk.1.ple_norm_query.weight 0 2 124570496 40960 124570496 40960 10240 0 0 0 1 0 0 0 0 0 0 0",
        "blk.1.ple_norm_conv.weight 0 2 124488576 40960 124488576 40960 10240 0 0 0 1 0 0 0 0 0 0 0",
        "blk.1.ple_conv1d.weight 0 5 71977856 81920 71977856 81920 4 10240 0 0 1 0 0 0 0 0 0 0",
    };
    for(size_t i=0;i<weights.size();++i) {
        WeightRef w;std::string name;int kind=0,codebook=0,offset=0,scales16=0;
        uint64_t destination=0;
        std::istringstream row(index_rows[i]);
        row>>name>>w.file_id>>kind>>w.src_off>>w.src_bytes>>destination>>w.bytes
           >>w.ne0>>w.ne1>>w.code_bits>>w.code_bias>>w.group_elems>>codebook>>offset
           >>w.codes_bytes>>w.scales_bytes>>w.offset_bytes>>scales16>>w.act_kind;
        check(!row.fail(),"actual pack metadata fixture parses all19 fields");
        w.kind=kind==4?WeightKind::Bf16InF32:kind==5?WeightKind::F16InF32:static_cast<WeightKind>(kind);
        w.elements=w.ne0*(w.ne1>0?w.ne1:1);w.codebook_iq4nl=codebook!=0;
        w.has_offset=offset!=0;w.scales_fp16=scales16!=0;w.data=&sentinel;
        weights[i]=w;
    }
    check(private_ple_weights(refs,false,out,error)&&out.key_bf16==weights[0].data&&
          out.value_bf16==weights[1].data&&out.conv1d_f16==weights[5].data,
          "actual raw BF16/F16 pack neutral group1 accepted without conversion");
    for(size_t i=0;i<weights.size();++i) {
        for(int group:{-1,2,32,64}) {
            weights[i].group_elems=group;out.pre_key=&sentinel;const auto before=out;
            check(!private_ple_weights(refs,false,out,error)&&fields(out)==fields(before),
                  "plain non-neutral group refuses without publishing");
        }
        weights[i].group_elems=1;
    }
    weights=synthetic;
    weights[2].ne0=NG_N_EMBD;weights[2].ne1=NG_HC;
    check(private_ple_weights(refs,false,out,error),"artifact two-dimensional norm shape accepted");
    auto refuses=[&] {out.pre_key=&sentinel;out.pre_value=&sentinel;const auto before=out;check(!private_ple_weights(refs,false,out,error)&&fields(out)==fields(before),"malformed weights refuse without partial publication");};
    for(size_t i=0;i<6;++i){refs[i]=nullptr;refuses();refs[i]=&weights[i];}
    for(size_t i=0;i<6;++i){--weights[i].bytes;refuses();++weights[i].bytes;}
    for(size_t i=0;i<6;++i) {
        weights[i].codes_bytes=4;refuses();weights[i].codes_bytes=0;
        weights[i].has_offset=true;refuses();weights[i].has_offset=false;
        weights[i].group_elems=32;refuses();weights[i].group_elems=0;
        weights[i].data=reinterpret_cast<const char*>(&sentinel)+1;refuses();weights[i].data=&sentinel;
    }
    weights[5].kind=WeightKind::Bf16InF32;refuses();weights[5].kind=WeightKind::F16InF32;
    --weights[0].ne0;refuses();++weights[0].ne0;
    for(int type:{42,18,23,8}) {
        weights[0].native_type=type;weights[0].native_data=&sentinel;weights[0].native_q8_1=&sentinel;
        check(private_ple_weights(refs,true,out,error)&&out.key_native_data&&!out.key_native_q8_1&&!out.pre_key&&!out.key_bf16&&!out.key_codes&&!out.key_scales&&!out.pre_value,
              "native key extraction never borrows mutable loader scratch");
    }
    weights[0].native_data=reinterpret_cast<const char*>(&sentinel)+1;
    check(!private_ple_weights(refs,true,out,error),"unaligned native pointer refused");
    weights[0].native_data=&sentinel;
    weights[0].native_type=-1;check(!private_ple_weights(refs,true,out,error),"unknown native format rejected");
    auto& key=weights[0];key.kind=WeightKind::Verbatim;key.code_bits=2;key.code_bias=-1;key.group_elems=64;
    key.codes_bytes=uint64_t(NG_N_EMBD)*NG_HC_DIM/4;
    key.scales_bytes=uint64_t(NG_N_EMBD)*NG_HC_DIM/64*sizeof(float);key.bytes=key.codes_bytes+key.scales_bytes;
    std::vector<float> storage(size_t((key.bytes+3)/4));key.data=storage.data();
    check(private_ple_weights(refs,false,out,error)&&out.key_codes==reinterpret_cast<uint8_t*>(storage.data())&&
          out.key_scales==reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(storage.data())+key.codes_bytes),"canonical planes use validated recorded sizes");
    // scales_fp16 describes PACK provenance, not engine storage: WeightTable
    // widens the scale plane while retaining this flag (weights.cpp segment list).
    key.scales_fp16=true;
    check(private_ple_weights(refs,false,out,error)&&out.key_scales==
          reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(storage.data())+key.codes_bytes)&&
          !out.key_bf16&&!out.key_native_data&&out.key_native_type==-1&&
          !out.key_native_q8_1&&!out.pre_key&&!out.pre_value,
          "widened engine scales retain source FP16 provenance and clear inactive modes");
    key.scales_fp16=false;
    key.q8_scale=&sentinel;refuses();key.q8_scale=nullptr;
    key.data=reinterpret_cast<const char*>(storage.data())+1;refuses();key.data=storage.data();
    key.group_elems=32;refuses();key.group_elems=64;
    key.code_bias=0;refuses();key.code_bias=-1;
    key.has_offset=true;refuses();key.has_offset=false;
    --key.scales_bytes;refuses();++key.scales_bytes;
    key.resident=false;refuses();
    std::puts("window_ple_weights_test passed");
}
