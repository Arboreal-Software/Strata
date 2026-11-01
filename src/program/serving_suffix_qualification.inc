// Included inside SuffixExperimentDriver's public section. Qualification only;
// reuses its private authenticated execution primitives, never standalone validate().
struct ServingFirstOutputLogits {
    std::unique_ptr<float[]> values;
    size_t vocabulary=0;
    uint64_t request=0,operation=0,publication=0,cut=0,forwarded=0,exact_frontier=0,watermark=0;
    int32_t emitted=-1;
    core::StateDigest conditioning{};
    double capture_ms=0;
    bool captured=false;
};
struct ServingQualification {
    std::shared_ptr<const void> credit; // outlives every oracle buffer
    std::array<char,65> binary{}, source{};
    uint64_t token_limit=4096;
    ServingFirstOutputLogits first; // storage dies BEFORE its pre-admitted credit above
    PrefixReplayPlan prefix; // lazily allocated only for actual batched qualified prefixes
    std::shared_ptr<PrefixReplayPlan> durable_prefix; // preallocated before origin mutation; no callbacks
    uint64_t required_origin_boundary=0;
    bool admitted=false,retained=false;
    // True for the RETAINED-YIELD oracle admission (slice 4, D3). It is a separate instance from the strict
    // final_*/live_* admission: the two never share credit, and `admitted` (which gates the existing
    // qualify_serving_frontier) keeps exactly its old meaning, so the existing oracle's preconditions are
    // unchanged byte for byte.
    bool owned_slot_oracle=false;
    bool yield_oracle=false;
    bool activation_oracle=false;
    uint64_t activation_request=0; // immutable admission target; never a relabelled request ordinal
};
static bool serving_qualification_enabled() noexcept {
    const char* value=std::getenv("STRATA_SUFFIX_REPAIR_QUALIFY");
    return value&&std::strcmp(value,"1")==0;
}
static bool serving_qualification_limit(uint64_t& limit) noexcept {
    limit=4096;
    if(const char* value=std::getenv("STRATA_SUFFIX_REPAIR_MAX_TOKENS")) {
        const size_t n=std::strlen(value);
        const auto result=std::from_chars(value,value+n,limit);
        if(result.ec!=std::errc{}||result.ptr!=value+n)return false;
    }
    return limit>0&&limit<=65536;
}
// Qualification-only target: the historical A3 request remains the default. A later target leaves
// earlier requests on the ordinary retained-yield path. Accepted values are intentionally bounded.
static bool activation_qualification_request(uint64_t& ordinal) noexcept {
    ordinal=4;
    if(const char* value=std::getenv("STRATA_SUFFIX_ACTIVATION_REQUEST")) {
        if(!std::getenv("STRATA_SUFFIX_ACTIVATION_QUALIFY"))return false;
        if(std::strcmp(value,"4")==0)ordinal=4;
        else if(std::strcmp(value,"6")==0)ordinal=6;
        else return false;
    }
    return true;
}
static bool activation_qualification_selects(uint64_t ordinal) noexcept {
    uint64_t target=0;
    return std::getenv("STRATA_SUFFIX_ACTIVATION_QUALIFY")&&
        activation_qualification_request(target)&&ordinal==target;
}
template<class Policy>
static bool serving_qualification_policy(const Policy& o,std::string& error) {
    const char* enabled=std::getenv("STRATA_SUFFIX_REPAIR_QUALIFY");
    const char* activation=std::getenv("STRATA_SUFFIX_ACTIVATION_QUALIFY");
    uint64_t activation_request=0;
    if(!activation_qualification_request(activation_request)){error="invalid activation qualification request (expected 4 or 6 with activation mode)";return false;}
    if(std::getenv("STRATA_SUFFIX_ACTIVATION_REQUEST") &&
       (std::getenv("STRATA_SUFFIX_REPAIR_FAULT_AT")||std::getenv("STRATA_SUFFIX_REPAIR_CANCEL_AT"))){error="activation request target cannot mix with fault/cancel qualification";return false;}
    if(!enabled&&!activation)return true;
    if(activation&&std::strcmp(activation,"success")&&std::strcmp(activation,"rollback")) {
        error="unknown activation qualification mode";return false;
    }
    if(activation&&enabled){error="activation and strict first-output qualification need separate campaigns";return false;}
    uint64_t tokens=0;
    const char* source=std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST");
    const char* lookahead=std::getenv("STRATA_LOOKAHEAD");
    if((enabled&&std::strcmp(enabled,"1"))||!o.serve||!o.suffix_cache||o.suffix_cache_unrepaired_experiment||
       !o.owned_session_lifetime||!o.owned_model_lifetime||o.prefill_head_alias||o.vision||
       o.batch||o.pipeline_windows||o.adapt_every||o.adapt_async||o.kv_grow||
       !source||!*source||std::strlen(source)>4096||!serving_qualification_limit(tokens)||
       !lookahead||std::strcmp(lookahead,"0")) {
        error="suffix qualification requires repaired owned serial serving, available head, disabled lookahead, source manifest and bounded token ceiling";
        return false;
    }
    for(const char* diagnostic:{"STRATA_DBG_NAN","STRATA_TAP_FILE","STRATA_NAN_INJECT",
                                "STRATA_VERIFY_PROFILE","STRATA_VERIFY_NODES","STRATA_COMMIT_TIMING"})
        if(std::getenv(diagnostic)){error="unadmitted verifier diagnostic in suffix qualification";return false;}
    return true;
}
static std::array<char,65> qualification_hex(const core::StateDigest& digest) noexcept {
    constexpr char digits[]="0123456789abcdef";
    std::array<char,65> result{};
    for(size_t i=0;i<digest.size();++i){result[2*i]=digits[digest[i]>>4];result[2*i+1]=digits[digest[i]&15];}
    return result;
}
static bool qualification_file_hash(const char* path,uint64_t ceiling,std::array<char,65>& out) {
    const int fd=::open(path,O_RDONLY|O_CLOEXEC|O_NONBLOCK);
    if(fd<0)return false;
    struct stat before{},after{};
    bool ok=::fstat(fd,&before)==0&&S_ISREG(before.st_mode)&&before.st_size>0&&
        uint64_t(before.st_size)<=ceiling;
    try {
        core::WindowHasher hash;
        std::array<uint8_t,65536> block{};
        uint64_t read_bytes=0,calls=0;
        const auto start=Clock::now();
        while(ok) {
            if(++calls>65536||elapsed(start)>30000){ok=false;break;}
            const ssize_t n=::read(fd,block.data(),block.size());
            if(n<0){ok=false;break;}
            if(!n)break;
            if(uint64_t(n)>ceiling-read_bytes){ok=false;break;}
            read_bytes+=uint64_t(n);hash.update(block.data(),size_t(n));
        }
        ok=ok&&read_bytes==uint64_t(before.st_size)&&::fstat(fd,&after)==0&&
            before.st_dev==after.st_dev&&before.st_ino==after.st_ino&&before.st_size==after.st_size&&
            before.st_mtim.tv_sec==after.st_mtim.tv_sec&&before.st_mtim.tv_nsec==after.st_mtim.tv_nsec&&
            before.st_ctim.tv_sec==after.st_ctim.tv_sec&&before.st_ctim.tv_nsec==after.st_ctim.tv_nsec;
        if(ok)out=qualification_hex(hash.digest());
    }catch(...){ok=false;}
    if(::close(fd)!=0)failstop("qualification identity descriptor close uncertain");
    return ok;
}
template<class Context>
static bool admit_serving_qualification(Context& c,ServingRawSession& session,uint64_t accepted,
                                        ServingQualification& out,std::string& error,uint64_t prefix_quote=0) {
    if(!serving_qualification_enabled())return true;
    if(out.admitted||out.credit||!serving_qualification_policy(c.options,error)||
       !serving_qualification_limit(out.token_limit)||accepted>out.token_limit||
       accepted>session.max_tokens()||c.geometry.n_vocab<=0||c.geometry.n_vocab>INT32_MAX||
       !validate_serving_qualification_execution(c,error)) {
        if(error.empty())error="suffix qualification token/admission bounds";
        return false;
    }
    // Retained qualification holds THREE simultaneous FP32 vocabulary rows: saved real first output,
    // clean replay scratch, and the existing repaired continuation scratch. No fourth row is allocated.
    // Exact-first keeps its two-row envelope. RAW captures/restores separately reserve transient peaks.
    const char* exact_first=std::getenv("STRATA_SUFFIX_REPAIR_EXACT_FIRST");
    out.retained=!(exact_first&&std::strcmp(exact_first,"1")==0);
    uint64_t bytes=1ULL<<20;
    if(!add_product(bytes,session.max_tokens(),128)||
       !add_product(bytes,uint64_t(c.geometry.n_vocab),(out.retained?3u:2u)*sizeof(float))) {
        error="suffix qualification scratch quote overflow";return false;
    }
#if defined(STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE)
    // One original admission owner is shared with the durable FD owner; do not consume another fixed H slot.
    if(std::getenv("STRATA_SUFFIX_REPAIR_FAULT_AT")&&
       !add_product(bytes,1,ServingSuffixFaultQualification::host_bytes())){
        error="serving fault metadata quote overflow";return false;
    }
#endif
    if(!add_product(bytes,1,prefix_quote)){error="prefix transcript quote overflow";return false;}
    const auto admitted=session.admit_host(bytes,out.credit);
    if(!admitted){error=admitted.detail;return false;}
    if(out.retained){
        try {out.first.values=std::make_unique<float[]>(size_t(c.geometry.n_vocab));}
        catch(...){error="qualification first-output row allocation refused before mutation";return false;}
        out.first.vocabulary=size_t(c.geometry.n_vocab);
    }
    const char* source=std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST");
    if(!qualification_file_hash("/proc/self/exe",2ULL<<30,out.binary)||
       !qualification_file_hash(source,4ULL<<20,out.source)) {
        error="suffix qualification actual executable/source manifest hash refused";return false;
    }
    out.admitted=true;return true;
}
// Independent admission for the bounded activation-only numerical oracle. No global strict selector
// is toggled and no strict first-output evidence is invented.
static const char* activation_qualification_mode() noexcept {
    return std::getenv("STRATA_SUFFIX_ACTIVATION_QUALIFY");
}
template<class Context>
static bool admit_activation_qualification(Context& c,ServingRawSession& session,uint64_t accepted,
    ServingQualification& out,std::string& error) {
    if(!activation_qualification_mode()||out.credit||!serving_qualification_policy(c.options,error)||
       !serving_qualification_limit(out.token_limit)||accepted>out.token_limit||accepted>session.max_tokens()||
       c.geometry.n_vocab<=0||c.geometry.n_vocab>INT32_MAX||!validate_serving_qualification_execution(c,error)) {
        if(error.empty())error="activation oracle admission bounds";return false;
    }
    uint64_t bytes=1ULL<<20;
    if(!add_product(bytes,session.max_tokens(),128)||
       !add_product(bytes,uint64_t(c.geometry.n_vocab),2u*sizeof(float))) {
        error="activation oracle scratch quote overflow";return false;
    }
    if(auto credit=session.admit_host(bytes,out.credit);!credit){error=credit.detail;return false;}
    const char* source=std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST");
    if(!qualification_file_hash("/proc/self/exe",2ULL<<30,out.binary)||
       !qualification_file_hash(source,4ULL<<20,out.source)) {
        error="activation oracle executable/source identity refused";return false;
    }
    if(!activation_qualification_request(out.activation_request)){error="activation request target refused";return false;}
    out.activation_oracle=true;out.admitted=true;out.retained=true;return true;
}
// Origin recording has its own metadata-only admission. It does not enable strict
// first-output qualification or borrow the eventual A3 oracle's scratch/credit.
template<class Context>
static bool admit_activation_prefix_origin(Context& c,ServingRawSession& session,uint64_t accepted,
    uint64_t boundary,uint64_t ordinal,ServingQualification& out,std::string& error) {
    if(!activation_qualification_mode()||ordinal!=2||boundary<=64||out.credit||out.admitted||
       !serving_qualification_policy(c.options,error)||!serving_qualification_limit(out.token_limit)||
       accepted>out.token_limit||accepted>session.max_tokens()||c.geometry.n_layers<=0||c.geometry.n_expert<=0||
       uint64_t(c.geometry.n_layers)>UINT64_MAX/uint64_t(c.geometry.n_expert)||
       !validate_serving_qualification_execution(c,error))return false;
    uint64_t bytes=0;
    if(!PrefixReplayPlan::quote(boundary,uint64_t(c.geometry.n_layers)*uint64_t(c.geometry.n_expert),bytes)||
       !add_product(bytes,1,sizeof(PrefixReplayPlan)+16384)){
        error="durable origin prefix quote overflow";return false;
    }
    if(auto credit=session.admit_host(bytes,out.credit);!credit){error=credit.detail;return false;}
    try {out.durable_prefix=std::make_shared<PrefixReplayPlan>();}
    catch(...){error="pre-admitted durable prefix owner allocation refused";return false;}
    const char* manifest=std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST");
    if(!qualification_file_hash("/proc/self/exe",2ULL<<30,out.binary)||
       !qualification_file_hash(manifest,4ULL<<20,out.source)){
        error="origin prefix executable/source identity refused";return false;
    }
    out.prefix.boundary=boundary;out.prefix.request=ordinal;out.admitted=true;return true;
}
// Canonical hashes bind semantic state and the runtime-generated layout. Cut IDs
// are reported separately because clean comparison deliberately permits them to differ.
static void qualification_u64(core::WindowHasher& hash,uint64_t value) {
    std::array<uint8_t,8> bytes{};
    for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(value>>(8*i));
    hash.update(bytes.data(),bytes.size());
}
static void qualification_string(core::WindowHasher& hash,const std::string& value) {
    qualification_u64(hash,value.size());hash.update(value.data(),value.size());
}
// Distinct diagnostic identity for the conditioning tokens, excluding the newly emitted pending token.
// This is deliberately not qualification_state_hash: approximation/frontier tags are not numerical error.
static bool qualification_conditioning_digest(const std::vector<int32_t>& tokens,uint64_t count,
                                               core::StateDigest& digest) {
    if(!count||count>tokens.size())return false;
    core::WindowHasher hash;qualification_string(hash,"serving-first-output-conditioning-v1");
    qualification_u64(hash,count);
    for(size_t i=0;i<size_t(count);++i)qualification_u64(hash,uint32_t(tokens[i]));
    digest=hash.digest();return true;
}
static bool qualification_first_output_matches(const ServingFirstOutputLogits& first,uint64_t ordinal,
    const core::SnapshotDescriptor& final,uint64_t generated) {
    if(!first.captured||!first.values||!first.vocabulary||first.request!=ordinal||!first.operation||
       !first.publication||!first.cut||!first.forwarded||first.exact_frontier>=first.forwarded||
       !generated||generated>64||generated-1>UINT64_MAX-first.forwarded||
       generated-1>UINT64_MAX-first.watermark||first.forwarded+generated-1!=final.cut.forwarded||
       first.watermark+generated-1!=final.cut.output_watermark||
       final.accepted_tokens.size()!=final.cut.accepted||first.forwarded>=final.accepted_tokens.size()||
       final.accepted_tokens[size_t(first.forwarded)]!=first.emitted)return false;
    core::StateDigest digest;
    return qualification_conditioning_digest(final.accepted_tokens,first.forwarded,digest)&&digest==first.conditioning;
}
template<class Context,class Receipt>
static void capture_serving_first_output(Context& c,ServingRawSession& session,uint64_t ordinal,
    ServingQualification& admission,const core::SnapshotDescriptor& selected,const Receipt& receipt) {
    const auto started=Clock::now();
    auto& first=admission.first;std::string error;
    if(!serving_qualification_enabled()||!admission.admitted||!admission.retained||!admission.credit||
       !first.values||first.vocabulary!=size_t(c.geometry.n_vocab)||first.captured||!ordinal||
       !session.request_open()||!session.capture_allowed()||selected.cut!=session.cut()||
       selected.cut.accepted!=selected.cut.forwarded+1||selected.cut.accepted>admission.token_limit||
       selected.accepted_tokens.size()!=selected.cut.accepted||selected.pending_tokens.size()!=1||
       selected.approximation.empty()||selected.cut.exact_frontier>=selected.cut.forwarded||
       !receipt.operation||!receipt.publication||receipt.cut!=selected.cut.id||
       receipt.token!=selected.pending_tokens.front()||receipt.token!=selected.accepted_tokens.back()||
       !validate_serving_qualification_execution(c,error))
        failstop("actual first-output receipt/qualification binding refused");
    try {
        if(!qualification_conditioning_digest(selected.accepted_tokens,selected.cut.forwarded,first.conditioning))
            failstop("actual first-output conditioning history refused");
        // Public readback delegates to the last verifier. Bind and select its real device explicitly.
        const core::Verifier* head=&c.primary_verifier;
        if(c.stage_count<1||c.stage_count>8)failstop("first-output head stage bound");
        for(int i=1;i<c.stage_count;++i){head=head->serial_routing_binding().next;
            if(!head)failstop("first-output head chain incomplete");}
        if(head->serial_routing_binding().next)failstop("first-output head chain exceeds actual stages");
        core::WindowDeviceScope device(head->device());
        if(!device.ok()||!c.primary_verifier.copy_logits(0,first.values.get())||
           cudaDeviceSynchronize()!=cudaSuccess||!device.finish())
            failstop("actual first-output head readback/drain/device restoration failed");
        int32_t argmax=0;
        for(size_t i=0;i<first.vocabulary;++i){
            if(!std::isfinite(first.values[i]))failstop("nonfinite actual first-output logits");
            if(first.values[i]>first.values[size_t(argmax)])argmax=int32_t(i);
        }
        if(argmax!=receipt.token)failstop("actual first-output logits differ from emitted greedy receipt");
        first.request=ordinal;first.operation=receipt.operation;first.publication=receipt.publication;
        first.cut=receipt.cut;first.forwarded=selected.cut.forwarded;first.exact_frontier=selected.cut.exact_frontier;
        first.watermark=selected.cut.output_watermark;first.emitted=receipt.token;
        first.capture_ms=elapsed(started);first.captured=true;
    }catch(...){failstop("actual first-output capture exception retains qualification credit");}
}
static bool report_serving_approximation(const ServingQualification& admission,uint64_t ordinal,
    uint64_t clean_forwarded,int32_t clean_pick,std::span<const float> exact) {
    const auto& first=admission.first;SuffixLogitComparison metrics;
    if(!first.captured||first.request!=ordinal||first.forwarded!=clean_forwarded||
       exact.size()!=first.vocabulary||!compare_suffix_logits(exact,{first.values.get(),first.vocabulary},metrics)||
       metrics.approximate_argmax!=first.emitted||metrics.exact_argmax!=clean_pick)return false;
    const auto conditioning=qualification_hex(first.conditioning);
    std::fprintf(stderr,"suffix_repair_approximation req=%llu phase=first_output exact_certificate=0 operation=%llu publication=%llu cut=%llu forwarded=%llu exact_frontier=%llu watermark=%llu conditioning=%s emitted_token=%d vocabulary=%zu row_bytes=%llu simultaneous_rows=3 kl_exact_to_approximate=%.17g rms=%.17g max_abs=%.17g exact_argmax=%d approximate_argmax=%d emitted_matches_exact=%u finite=1 reference=reset_replay_same_conditioning capture_ms=%.6f binary=%s source=%s\n",
        (unsigned long long)ordinal,(unsigned long long)first.operation,(unsigned long long)first.publication,
        (unsigned long long)first.cut,(unsigned long long)first.forwarded,(unsigned long long)first.exact_frontier,
        (unsigned long long)first.watermark,conditioning.data(),first.emitted,first.vocabulary,
        (unsigned long long)(first.vocabulary*sizeof(float)),metrics.kl_exact_to_approximate,
        metrics.root_mean_square_error,metrics.max_absolute_error,metrics.exact_argmax,metrics.approximate_argmax,
        unsigned(first.emitted==metrics.exact_argmax),first.capture_ms,admission.binary.data(),admission.source.data());
    std::fflush(stderr);return true; // divergence is observational: zero and nonzero are both valid
}
static core::StateDigest qualification_layout_hash(const core::WindowStateLayout& layout) {
    core::WindowHasher hash;
    qualification_string(hash,"serving-suffix-trusted-layout-v1");
    qualification_string(hash,layout.native_layout);
    hash.update(layout.model_profile.data(),layout.model_profile.size());
    hash.update(layout.topology.data(),layout.topology.size());
    qualification_u64(hash,layout.max_tokens);qualification_u64(hash,layout.max_payload_bytes);
    qualification_u64(hash,layout.steering_enabled);qualification_u64(hash,layout.layers.size());
    for(auto kind:layout.layers)qualification_u64(hash,uint64_t(kind));
    qualification_u64(hash,layout.stages.size());
    for(const auto& stage:layout.stages) {
        qualification_u64(hash,stage.id);qualification_u64(hash,uint64_t(stage.layer_begin));
        qualification_u64(hash,uint64_t(stage.layer_end));qualification_u64(hash,stage.ple);
        qualification_u64(hash,stage.tensors.size());
        for(const auto& t:stage.tensors) {
            qualification_u64(hash,uint64_t(t.kind));qualification_u64(hash,uint64_t(t.layer));
            qualification_u64(hash,t.device_part);qualification_string(hash,t.dtype);
            qualification_string(hash,t.native_layout);qualification_u64(hash,t.shape.size());
            for(auto n:t.shape)qualification_u64(hash,n);
            qualification_u64(hash,t.strides.size());for(auto n:t.strides)qualification_u64(hash,n);
            qualification_u64(hash,t.logical_begin);qualification_u64(hash,t.valid_elements);qualification_u64(hash,t.bytes);
        }
    }
    return hash.digest();
}
static core::StateDigest qualification_state_hash(const core::SnapshotDescriptor& state,
                                                   const core::StateDigest& layout) {
    core::WindowHasher hash;
    qualification_string(hash,"serving-suffix-semantic-state-v1");hash.update(layout.data(),layout.size());
    qualification_u64(hash,state.cut.accepted);qualification_u64(hash,state.cut.forwarded);
    qualification_u64(hash,state.cut.exact_frontier);qualification_u64(hash,state.cut.output_watermark);
    hash.update(state.cut.history.data(),state.cut.history.size());
    qualification_string(hash,state.native_layout);
    qualification_u64(hash,state.speculation_disabled);qualification_u64(hash,state.steering_enabled);
    qualification_u64(hash,uint64_t(state.continuation.mode));
    qualification_string(hash,state.continuation.adapter);
    qualification_u64(hash,state.continuation.payload.size());
    hash.update(state.continuation.payload.data(),state.continuation.payload.size());
    qualification_u64(hash,state.approximation.size());
    for(const auto& region:state.approximation){
        qualification_u64(hash,region.begin);qualification_u64(hash,region.end);
        qualification_u64(hash,region.first_unresolved_unix_ms);
        hash.update(region.ancestor.data(),region.ancestor.size());
    }
    qualification_u64(hash,state.accepted_tokens.size());
    for(auto token:state.accepted_tokens)qualification_u64(hash,uint32_t(token));
    qualification_u64(hash,state.pending_tokens.size());
    for(auto token:state.pending_tokens)qualification_u64(hash,uint32_t(token));
    qualification_u64(hash,state.devices.size());
    for(const auto& stage:state.devices){qualification_u64(hash,stage.id);
        qualification_u64(hash,stage.tensors.size());
        for(const auto& tensor:stage.tensors){qualification_u64(hash,tensor.payload.size());
            if(!tensor.payload.visit(0,tensor.payload.size(),[&](const uint8_t* p,size_t n,size_t){hash.update(p,n);return true;}))
                failstop("qualification payload hashing incomplete");
        }
    }
    return hash.digest();
}
// What the comparison emitter actually observed, reported out to the RETAINED-YIELD caller so the yield decision
// (`serving_yield_record_decision`) reads the same facts the record line prints. Optional: the existing
// final_*/live_* call sites pass nothing and their behaviour, records and verdicts are unchanged.
struct ServingComparisonWitness {
    bool completed=false,equal=false;
    uint64_t mismatches=0;
    uint64_t forwarded=0,accepted=0,exact_frontier=0;
};
static bool report_serving_comparison(ServingRawSession& session,const ServingQualification& admission,
    uint64_t ordinal,const char* phase,const core::BundleHandle& repaired,const core::BundleHandle& clean,
    bool logits_equal=true,bool picks_equal=true,double logit_error=0,
    ServingComparisonWitness* witness=nullptr,const char* oracle_kind="reset_replay",const core::WindowStateLayout* clean_actual=nullptr) {
    const bool bridge_phase=std::strcmp(phase,"owned_slot_bridge")==0;
    const bool bridge_oracle=std::strcmp(oracle_kind,"physical_bridge")==0;
    if(bridge_phase!=bridge_oracle||bridge_phase!=(clean_actual!=nullptr))
        failstop("cross-owner comparison requires explicit physical bridge phase and both actual layouts");
    core::WindowStateLayout layout;
    const auto valid=core::native_window_layout(session.geometry(),session.initial_config(),repaired.get()->key,
        repaired.get()->cut,session.max_tokens(),session.max_payload_bytes(),layout,false);
    if(!valid)failstop("serving oracle trusted layout refused");
    uint64_t tensors=0,bytes=0;
    for(const auto& stage:layout.stages)for(const auto& tensor:stage.tensors){++tensors;bytes+=tensor.bytes;}
    clean_recompute_qualification::Report report;
    const bool equal=clean_actual?clean_recompute_qualification::compare_physical_bridge(repaired,clean,layout,*clean_actual,report):
        clean_recompute_qualification::compare(repaired,clean,layout,report);
    const bool pass=equal&&report.completed&&report.equal&&!report.mismatches&&
        report.tensors_compared==tensors&&report.bytes_compared==bytes&&logits_equal&&picks_equal;
    const auto layout_digest=qualification_layout_hash(layout);
    const auto layout_hex=qualification_hex(layout_digest),history=qualification_hex(repaired.get()->cut.history);
    const auto left=qualification_hex(qualification_state_hash(*repaired.get(),layout_digest));
    const auto right=qualification_hex(qualification_state_hash(*clean.get(),layout_digest));
    if(clean_actual){
        const auto destination_layout=qualification_layout_hash(*clean_actual);
        const auto destination_state=qualification_hex(qualification_state_hash(*clean.get(),destination_layout));
        const auto destination_hex=qualification_hex(destination_layout);
        std::fprintf(stderr,"owned_slot_bridge_binding req=%llu source_cut=%llu destination_cut=%llu source_layout=%s destination_layout=%s source_state=%s destination_state=%s binary=%s source=%s\n",
            (unsigned long long)ordinal,(unsigned long long)repaired.get()->cut.id,(unsigned long long)clean.get()->cut.id,
            layout_hex.data(),destination_hex.data(),left.data(),destination_state.data(),admission.binary.data(),admission.source.data());
    }
    const auto& cut=repaired.get()->cut;
    std::fprintf(stderr,"%s req=%llu phase=%s repaired_cut=%llu clean_cut=%llu accepted=%llu forwarded=%llu exact_frontier=%llu watermark=%llu history=%s layout=%s repaired=%s clean=%s binary=%s source=%s stages=%zu tensors=%llu bytes=%llu completed=%u equal=%u mismatches=%llu oracle=%s emitted=0 logits_equal=%u picks_equal=%u logits_max_absolute_error=%.17g\n",
        admission.activation_oracle?"suffix_activation_qualification":"suffix_repair_qualification",
        (unsigned long long)ordinal,phase,(unsigned long long)cut.id,(unsigned long long)clean.get()->cut.id,
        (unsigned long long)cut.accepted,(unsigned long long)cut.forwarded,(unsigned long long)cut.exact_frontier,
        (unsigned long long)cut.output_watermark,history.data(),layout_hex.data(),left.data(),right.data(),
        admission.binary.data(),admission.source.data(),layout.stages.size(),
        (unsigned long long)report.tensors_compared,(unsigned long long)report.bytes_compared,
        unsigned(report.completed),unsigned(pass),(unsigned long long)report.mismatches,
        oracle_kind,unsigned(logits_equal),unsigned(picks_equal),logit_error);
    for(size_t i=0;i<std::min<uint64_t>(report.mismatches,report.first.size());++i){const auto& m=report.first[i];
        std::fprintf(stderr,"suffix_repair_qualification_difference req=%llu phase=%s reason=%u stage=%u layer=%lld kind=%u offset=%llu left=%llu right=%llu validation_error=%u\n",
            (unsigned long long)ordinal,phase,unsigned(m.reason),m.stage,(long long)m.layer,unsigned(m.kind),
            (unsigned long long)m.offset,(unsigned long long)m.left,(unsigned long long)m.right,unsigned(report.validation_error));
    }
    if(witness) *witness={report.completed,report.equal,report.mismatches,
        cut.forwarded,cut.accepted,cut.exact_frontier};
    std::fflush(stderr);return pass;
}
// Compare the real captured B state before displacement against the real post-rollback image.
// Both operands remain charged through the checked final restore. This is a restore-image oracle,
// never a claim that an approximate B equals a clean replay.
template<class Context,class SerialUnit>
static bool qualify_activation_rollback(Context& c,SerialUnit& serial,ServingRawSession& session,
    uint64_t ordinal,const ServingQualification& admission,const core::BundleHandle& before,
    uint64_t& next_cut) {
    if(!admission.activation_oracle||!admission.credit||ordinal!=admission.activation_request||!before||!session.request_open()||
       !session.capture_allowed()||session.cut()!=before.get()->cut||
       before.get()->cut.accepted>admission.token_limit||before.get()->pending_tokens.size()!=1||
       before.get()->cut.accepted!=before.get()->cut.forwarded+1)
        failstop("activation rollback oracle endpoint/admission refused");
    auto& tx=*session.transaction();const auto& original=*before.get();
    if(next_cut==UINT64_MAX||tx.cut_.id==UINT64_MAX)failstop("activation rollback cut overflow");
    next_cut=std::max(next_cut,tx.cut_.id+1);
    try {
        core::BundleHandle after;
        const auto metadata=ServingRawSession::endpoint_metadata(original.key,original.cut,original.accepted_tokens);
        // endpoint_metadata is exact-only. This first qualification deliberately admits exact ordinary B;
        // approximate rollback needs its tracked descriptor passed through, never relabelled as exact.
        if(!original.approximation.empty()||original.cut.exact_frontier!=original.cut.forwarded)
            failstop("activation rollback initial qualification requires exact B");
        if(!raw_call([&]{return session.capture(metadata,after);}))failstop("activation rollback postimage capture refused");
        bool pass=report_serving_comparison(session,admission,ordinal,"rollback_frontier",after,before,
            true,true,0,nullptr,"restore_image");
        Buffers buffers;buffers.history=original.accepted_tokens;
        buffers.exact_logits.resize(size_t(c.geometry.n_vocab));
        buffers.approximate_logits.resize(size_t(c.geometry.n_vocab));
        struct SamplingScope {core::Verifier& v;explicit SamplingScope(core::Verifier& x):v(x){v.set_head_sampling(false);}
            ~SamplingScope(){v.set_head_sampling(true);}} sampling(c.primary_verifier);
        auto step=[&](const core::BundleHandle& image,std::vector<float>& logits,core::BundleHandle& out){
            if(!raw_call([&]{return session.restore(image,original.key);}))failstop("activation rollback operand restore refused");
            const auto result=forward(c,serial,tx,next_cut,original.pending_tokens.front(),original.cut.forwarded,
                original.cut.accepted,original.cut.history,original.cut.accepted,logits);
            const auto meta=ServingRawSession::endpoint_metadata(original.key,result.cut,buffers.history);
            if(!raw_call([&]{return session.capture(meta,out);}))failstop("activation rollback continuation capture refused");
            return result.next;
        };
        core::BundleHandle left,right;
        const auto a=step(before,buffers.exact_logits,left);
        const auto b=step(after,buffers.approximate_logits,right);
        SuffixLogitComparison logits;const bool valid=compare_suffix_logits(buffers.exact_logits,buffers.approximate_logits,logits);
        const bool picks=valid&&a==b&&a==logits.exact_argmax&&b==logits.approximate_argmax;
        pass=report_serving_comparison(session,admission,ordinal,"rollback_continuation",right,left,
            valid&&logits.max_absolute_error==0,picks,valid?logits.max_absolute_error:-1,nullptr,"restore_image")&&pass;
        left={};right={};after={};
        if(!raw_call([&]{return session.restore(before,original.key);}))failstop("activation rollback final restore refused");
        core::BundleHandle restored;
        if(!raw_call([&]{return session.capture(metadata,restored);}))failstop("activation rollback restoration capture refused");
        pass=report_serving_comparison(session,admission,ordinal,"rollback_restoration",restored,before,
            true,true,0,nullptr,"restore_image")&&pass;
        std::fprintf(stderr,"suffix_activation_qualification_done req=%llu mode=rollback pass=%u restored=1 binary=%s source=%s\n",
            (unsigned long long)ordinal,unsigned(pass),admission.binary.data(),admission.source.data());
        std::fflush(stderr);return pass;
    }catch(...){failstop("activation rollback exception retains execution buffers");}
}
// ---- the RETAINED-YIELD oracle (slice 4, D3) ---------------------------------------------------------
// The fixed vocabulary lives in one place: `strata::program::kServingYieldProgressPhase` /
// `kServingYieldFrontierPhase` / `kServingYieldContinuationPhase` (serving_repair_owners.hpp), which match
// harness/retained_yield_oracle.py's YIELD_PROGRESS_PHASE and YIELD_PHASE_NAMES exactly.
// Admission for the yield comparator. It reuses the strict oracle's own policy, token ceiling and identity
// binding, and asks for only the two FP32 vocabulary rows the comparison holds at once (exact and approximate
// logits): no first-output row, because a yield certifies a private cut, not the response's first emission.
// It is deliberately NOT gated on STRATA_SUFFIX_REPAIR_QUALIFY: the D3 defect is that the yield existed only in
// the arm where the strict oracle was OFF, so the yield's own certificate cannot depend on that selector. What
// it does require is a pinned source manifest, i.e. the same host-measurable producer identity the harness
// joins the records to; without it there is no honest `binary=`/`source=` to print and the yield stays refused.
template<class Context>
static bool admit_serving_yield_oracle(Context& c,ServingRawSession& session,ServingQualification& out,
                                       std::string& error) {
    if(out.yield_oracle)return true;                       // already admitted for this request
    if(out.credit){error="a retained-yield oracle needs its own admission record";return false;}
    const char* manifest=std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST");
    if(!manifest||!*manifest||std::strlen(manifest)>4096) {
        error="retained-yield oracle requires a pinned source manifest for its producer identity";return false;
    }
    if(!serving_qualification_policy(c.options,error)||!serving_qualification_limit(out.token_limit)||
       c.geometry.n_vocab<=0||c.geometry.n_vocab>INT32_MAX||!validate_serving_qualification_execution(c,error)) {
        if(error.empty())error="yield oracle admission bounds";
        return false;
    }
    uint64_t bytes=1ULL<<20;
    if(!add_product(bytes,session.max_tokens(),128)||
       !add_product(bytes,uint64_t(c.geometry.n_vocab),2u*sizeof(float))) {
        error="yield oracle scratch quote overflow";return false;
    }
    const auto credit=session.admit_host(bytes,out.credit);
    if(!credit){error=credit.detail;return false;}
    const char* source=std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST");
    if(!qualification_file_hash("/proc/self/exe",2ULL<<30,out.binary)||
       !qualification_file_hash(source,4ULL<<20,out.source)) {
        error="yield oracle actual executable/source manifest hash refused";return false;
    }
    out.yield_oracle=true;return true;
}
static bool serving_qualification_live_phases(const char* frontier,const char* continuation) noexcept {
    return std::strcmp(frontier,"live_frontier")==0&&std::strcmp(continuation,"live_continuation")==0;
}
static bool serving_qualification_retained_phases(const char* frontier,const char* continuation) noexcept {
    return serving_qualification_live_phases(frontier,continuation)||
        (std::strcmp(frontier,"final_frontier")==0&&std::strcmp(continuation,"final_continuation")==0);
}
template<class Context,class SerialUnit,class Cancelled>
static bool qualify_serving_frontier(Context& c,SerialUnit& serial,ServingRawSession& session,
    uint64_t ordinal,const ServingQualification& admission,const core::BundleHandle& repaired,
    uint64_t& next_cut,Cancelled&& cancelled,const char* frontier_phase="certified_frontier",
    const char* continuation_phase="continuation",uint64_t generated_receipts=0
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
    ,const ServingSuffixCancelQualification::Witness* stopped=nullptr
#endif
    ,PrefixReplayView prefix={}
    ) {
    const auto oracle_started=Clock::now();
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
    const bool post_stop=std::strcmp(frontier_phase,"cancelled_frontier")==0&&
        std::strcmp(continuation_phase,"cancelled_continuation")==0&&stopped&&stopped->request()==ordinal;
    if(stopped&&!post_stop)failstop("post-STOP oracle witness/phase mismatch");
#else
    constexpr bool post_stop=false;
#endif
    const bool live=serving_qualification_live_phases(frontier_phase,continuation_phase);
    const bool activation=admission.activation_oracle&&
        std::strcmp(frontier_phase,"activation_frontier")==0&&
        std::strcmp(continuation_phase,"activation_continuation")==0;
    const bool slot_oracle=admission.owned_slot_oracle&&std::strcmp(frontier_phase,"owned_slot_frontier")==0&&std::strcmp(continuation_phase,"owned_slot_continuation")==0;
    const bool retained=post_stop||activation||slot_oracle||serving_qualification_retained_phases(frontier_phase,continuation_phase);
    const bool exact_first=std::strcmp(frontier_phase,"certified_frontier")==0&&
        std::strcmp(continuation_phase,"continuation")==0;
    if((admission.activation_oracle&&!activation)||!admission.admitted||!repaired||!session.request_open()||!session.capture_allowed()||
       (!retained&&!exact_first)||
       repaired.get()->cut!=session.cut()||repaired.get()->cut.forwarded==0||
       repaired.get()->cut.accepted>admission.token_limit||
       repaired.get()->cut.accepted!=repaired.get()->cut.forwarded+1||
       repaired.get()->pending_tokens.size()!=1||!repaired.get()->approximation.empty()||
       repaired.get()->cut.exact_frontier!=repaired.get()->cut.forwarded||
       (exact_first&&repaired.get()->cut.output_watermark))
        failstop("serving oracle actual publication/admission differs");
    auto& tx=*session.transaction();
    const auto& original=*repaired.get();
    const uint64_t accepted=original.cut.accepted,forwarded=original.cut.forwarded;
    const bool needs_first=retained&&!activation&&!slot_oracle&&generated_receipts!=0;
    if((retained&&!admission.retained)||(!retained&&generated_receipts)||
       (activation&&generated_receipts)||(needs_first&&!qualification_first_output_matches(admission.first,ordinal,original,generated_receipts))||
       (!needs_first&&admission.first.captured))
        failstop("first-output evidence differs from actual final accepted/emitted history");
    if(tx.cut_.id==UINT64_MAX||next_cut==UINT64_MAX)failstop("serving oracle cut overflow");
    next_cut=std::max(next_cut,tx.cut_.id+1);
    if(admission.prefix.active&&(!prefix.plan||prefix.plan!=&admission.prefix||!prefix.calls||
       !prefix.calls->check||!prefix.plan->matches(original.accepted_tokens,prefix.origin_request?prefix.origin_request:ordinal,forwarded)))
        failstop("actual batched prefix requires its sealed request transcript");
    if(prefix.plan&&(!prefix.calls||!prefix.calls->check||
       !prefix.plan->matches(original.accepted_tokens,prefix.origin_request?prefix.origin_request:ordinal,forwarded)))
        failstop("prefix replay view differs from actual accepted history");
    // Every owner/buffer remains in scope until checked execution/restore drains.
    try {
        Buffers buffers;buffers.history=original.accepted_tokens;
        buffers.exact_logits.resize(size_t(c.geometry.n_vocab));
        buffers.approximate_logits.resize(size_t(c.geometry.n_vocab));
        struct SamplingScope {core::Verifier& verifier;
            explicit SamplingScope(core::Verifier& v):verifier(v){verifier.set_head_sampling(false);}
            ~SamplingScope(){verifier.set_head_sampling(true);}
        } sampling(c.primary_verifier);
        bool pass=false,was_cancelled=cancelled(),first_compared=!needs_first;
        if(!was_cancelled) {
            std::string prefix_error;
            if(prefix.plan&&!prefix.calls->check(prefix.calls->user,*prefix.plan,prefix_error))
                failstop("prefix base mapping/owned numeric policy changed before reset");
            reset(c,tx,next_cut,original.cut.output_watermark);
            Step step;uint64_t prefix_at=0;
            if(prefix.plan){
                const auto prefix_started=Clock::now();
                if(needs_first&&admission.first.forwarded<=prefix.plan->boundary)
                    failstop("first-output conditioning precedes the recorded exact prefix");
                auto cancelled_thunk=[&](){return cancelled();};
                PrefixReplayAbort abort{&cancelled_thunk,[](void* p){return (*static_cast<decltype(cancelled_thunk)*>(p))();}};
                uint64_t batches=0,windows=0;
                for(const auto& event:prefix.plan->events){
                    if(cancelled()){was_cancelled=true;break;}
                    if(event.kind==PrefixReplayEvent::Kind::window){
                        PrefixReplayEvent refill;refill.from=refill.to=event.from;
                        step=replay_prefill_piece(c,tx,next_cut,refill,*prefix.calls,buffers.history,
                            accepted,original.cut.history,abort,was_cancelled);
                        for(uint64_t at=event.from;at<event.to;++at){
                            if(cancelled()){was_cancelled=true;break;}
                            step=forward(c,serial,tx,next_cut,buffers.history[size_t(at)],at,accepted,
                                original.cut.history,at+1,{});
                        }
                        ++windows;
                    }else{
                        step=replay_prefill_piece(c,tx,next_cut,event,*prefix.calls,buffers.history,
                            accepted,original.cut.history,abort,was_cancelled);++batches;
                    }
                    if(was_cancelled)break;
                }
                PrefixReplayEvent refill;refill.from=refill.to=tx.cut_.forwarded;
                step=replay_prefill_piece(c,tx,next_cut,refill,*prefix.calls,buffers.history,
                    accepted,original.cut.history,abort,was_cancelled);
                if(!prefix.calls->check(prefix.calls->user,*prefix.plan,prefix_error))
                    failstop("prefix replay final refill/base mapping differs");
                prefix_at=step.cut.forwarded;
                if(!was_cancelled&&prefix_at!=prefix.plan->boundary)failstop("prefix replay incomplete boundary");
                const auto schedule=qualification_hex(prefix.plan->digest);
                const auto& replay_policy=prefix.plan->policy;
                const char* policy_name=!replay_policy.automatic&&replay_policy.configured_max==16?
                    "fixed_prefill16_short64":"resolved_prefill_short64_v1";
                std::fprintf(stderr,"suffix_repair_prefix_replay req=%llu origin_req=%llu boundary=%llu reached=%llu schedule=%s events=%zu batches=%llu windows=%llu ancestry=reset_from0 mapping_equal=1 completed=%u stages=%d policy=%s configured_max=%llu allocated_max=%llu auto_mode=%u auto_max=%llu elapsed_ms=%.6f binary=%s source=%s\n",
                    (unsigned long long)ordinal,(unsigned long long)prefix.plan->request,(unsigned long long)prefix.plan->boundary,
                    (unsigned long long)prefix_at,schedule.data(),prefix.plan->events.size(),
                    (unsigned long long)batches,(unsigned long long)windows,unsigned(!was_cancelled),c.stage_count,policy_name,
                    (unsigned long long)replay_policy.configured_max,(unsigned long long)replay_policy.allocated_max,
                    unsigned(replay_policy.automatic),(unsigned long long)replay_policy.auto_requested_max,elapsed(prefix_started),
                    admission.binary.data(),admission.source.data());std::fflush(stderr);
            }
            for(uint64_t at=prefix_at;!was_cancelled&&at<forwarded;++at) {
                if(cancelled()){was_cancelled=true;break;}
                const bool first_position=needs_first&&at+1==admission.first.forwarded;
                // Reuse the already admitted exact-oracle scratch row; never allocate a fourth row.
                const auto row=first_position?std::span<float>(buffers.exact_logits):std::span<float>{};
                step=forward(c,serial,tx,next_cut,buffers.history[size_t(at)],at,accepted,
                    original.cut.history,at+1,row);
                if(first_position){
                    first_compared=report_serving_approximation(admission,ordinal,step.cut.forwarded,step.next,row);
                    if(!first_compared)failstop("first-output distribution diagnostic is nonfinite or unbound");
                }
            }
            if(!was_cancelled) {
                core::BundleHandle clean;
                const auto metadata=ServingRawSession::endpoint_metadata(original.key,step.cut,buffers.history);
                if(!raw_call([&]{return session.capture(metadata,clean);}))failstop("serving cold capture refused");
                const bool frontier=report_serving_comparison(session,admission,ordinal,frontier_phase,repaired,clean);
                clean={};
                if(!cancelled()) {
                    const int32_t pending=original.pending_tokens.front();
                    const auto cold_next=forward(c,serial,tx,next_cut,pending,forwarded,accepted,
                        original.cut.history,accepted,buffers.exact_logits);
                    core::BundleHandle cold_continuation;
                    const auto cold_meta=ServingRawSession::endpoint_metadata(original.key,cold_next.cut,buffers.history);
                    if(!raw_call([&]{return session.capture(cold_meta,cold_continuation);}))failstop("serving cold continuation capture refused");
                    if(!raw_call([&]{return session.restore(repaired,original.key);}))failstop("serving repaired restore before continuation refused");
                    const auto repaired_next=forward(c,serial,tx,next_cut,pending,forwarded,accepted,
                        original.cut.history,accepted,buffers.approximate_logits);
                    core::BundleHandle repaired_continuation;
                    const auto next_meta=ServingRawSession::endpoint_metadata(original.key,repaired_next.cut,buffers.history);
                    if(!raw_call([&]{return session.capture(next_meta,repaired_continuation);}))failstop("serving repaired continuation capture refused");
                    SuffixLogitComparison logits;
                    const bool valid=compare_suffix_logits(buffers.exact_logits,buffers.approximate_logits,logits);
                    const bool picks=valid&&cold_next.next==repaired_next.next&&
                        cold_next.next==logits.exact_argmax&&repaired_next.next==logits.approximate_argmax;
                    const bool continuation=report_serving_comparison(session,admission,ordinal,continuation_phase,
                        repaired_continuation,cold_continuation,valid&&logits.max_absolute_error==0,picks,
                        valid?logits.max_absolute_error:-1);
                    pass=frontier&&continuation&&first_compared;
                }else was_cancelled=true;
            }
        }
        // Always return the actual repaired frontier, including on completed mismatch/cancel.
        if(!raw_call([&]{return session.restore(repaired,original.key);}))failstop("serving oracle final repaired restoration refused");
        was_cancelled=was_cancelled||cancelled();pass=pass&&!was_cancelled;
        std::fprintf(stderr,activation?
            "suffix_activation_replay_done req=%llu pass=%u restored=1 cancelled=%u next_cut=%llu binary=%s source=%s elapsed_ms=%.6f\n":live?
            "suffix_repair_live_qualification_done req=%llu pass=%u restored=1 cancelled=%u next_cut=%llu binary=%s source=%s elapsed_ms=%.6f\n":
            "suffix_repair_qualification_done req=%llu pass=%u restored=1 cancelled=%u next_cut=%llu binary=%s source=%s elapsed_ms=%.6f\n",
            (unsigned long long)ordinal,unsigned(pass),unsigned(was_cancelled),(unsigned long long)next_cut,
            admission.binary.data(),admission.source.data(),elapsed(oracle_started));
        std::fflush(stderr);return pass;
    }catch(...){failstop("serving qualification exception retains execution buffers");}
}
// A3's actual final image, numerical next-token comparison, and independently captured restoration.
// Admission is separate from the strict first-output oracle, and phases are accepted only with that admission.
template<class Context,class SerialUnit,class Cancelled>
static bool qualify_activation_final(Context& c,SerialUnit& serial,ServingRawSession& session,
    uint64_t ordinal,const ServingQualification& admission,const core::BundleHandle& actual,
    uint64_t& next_cut,Cancelled&& cancelled,PrefixReplayView prefix) {
    if(ordinal!=admission.activation_request||!admission.activation_oracle||!actual)failstop("activation final request/admission mismatch");
    if(!activation_prefix_view_matches(admission.required_origin_boundary,prefix,*actual.get(),
                                      c.owner->remote_policy_incarnation,0))
        failstop("activation final requires durable authentic batched origin, never T1 substitution");
    if(prefix.plan && (prefix.plan->origin_binary!=admission.binary || prefix.plan->origin_source!=admission.source))
        failstop("activation origin executable/source provenance differs");
    bool pass=qualify_serving_frontier(c,serial,session,ordinal,admission,actual,next_cut,cancelled,
        "activation_frontier","activation_continuation",0
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
        ,nullptr
#endif
        ,prefix);
    core::BundleHandle restored;
    const auto& original=*actual.get();
    const auto metadata=ServingRawSession::endpoint_metadata(original.key,original.cut,original.accepted_tokens);
    if(!raw_call([&]{return session.capture(metadata,restored);}))failstop("activation final restoration capture refused");
    pass=report_serving_comparison(session,admission,ordinal,"activation_restoration",restored,actual,
        true,true,0,nullptr,"restore_image")&&pass;
    std::fprintf(stderr,"suffix_activation_qualification_done req=%llu mode=success pass=%u restored=1 binary=%s source=%s\n",
        (unsigned long long)ordinal,unsigned(pass),admission.binary.data(),admission.source.data());
    std::fflush(stderr);return pass;
}
// The RETAINED-YIELD oracle cell: compare the yielded owner's certified private cut R (the all-stage image
// `yield_retained` captured through the private transaction's own drained-phase capture) against an independent
// clean recomputation of the same accepted history [0,R+1), then compare that state advanced by exactly one
// ordinary-loop window against the clean recomputation's own next window. The record format is the existing one
// (`report_serving_comparison`), the phase names are fixed, and the restoration record is the existing
// `suffix_repair_qualification_done` form. Emission order is progress `retained_yield` -> frontier -> continuation
// -> restoration, which is the order harness/retained_yield_oracle.py `yield_marker_order` requires.
//
// Why this is a SEPARATE comparator and not `qualify_serving_frontier` with new names: that function's own
// preconditions (repaired.cut == session.cut(), an approximation-free bundle at the CURRENT cut, a final/live/
// cancelled phase name) are exactly what a debt-retaining yield cannot satisfy - the certified cut sits strictly
// BELOW the session's current cut, and the session must be left standing on the current cut. Its preconditions
// are not relaxed here; they are unchanged, and this entry point adds its own, stricter for a yield:
//   * the certified image must be exact at its own frontier (capture_exact's own rule);
//   * the session's CURRENT cut D must sit strictly above R, must NOT be exact at D (that is pp_exact=false /
//     live_approximate=true: an exact published cut is a zero-debt claim), and must still carry approximation
//     regions - the debt the yield reports rather than hides;
//   * the published certified frontier p must not exceed R (the controller's own p <= r < d bound);
//   * the live session's own CURRENT (approximate) state is captured before the replay, restored afterwards, and
//     verified against the publication it must preserve, so the comparison can neither roll the conversation back
//     to the certified cut nor leave it on anything other than the foreground cut it stood on. That is also what
//     keeps every OTHER parked owner's recorded session cut true: the registry re-validates those records
//     (`entry_record_matches`, `parked_owner_publishable`) and refuses `refused_stale_publication`, and this
//     comparator leaves nothing for it to refuse.
// Nothing is emitted when a precondition refuses: no device or session state has been touched at that point, the
// caller takes today's drain path unchanged, and no half record reaches the transcript. Once the comparison HAS
// run, a bad verdict is the established failstop, never a recovery.
template<class Context,class SerialUnit,class Cancelled>
static bool qualify_serving_yield_frontier(Context& c,SerialUnit& serial,ServingRawSession& session,
    uint64_t ordinal,const ServingQualification& admission,const core::BundleHandle& certified,
    const core::SnapshotDescriptor& published,uint64_t& next_cut,Cancelled&& cancelled,
    strata::program::ServingYieldRecordFacts& report,PrefixReplayView prefix={}) {
    const auto oracle_started=Clock::now();
    report={};
    const auto* image=certified.get();
    if(!admission.yield_oracle||!admission.credit||!image||!ordinal||!session.request_open()||
       !session.capture_allowed()||image->cut.id==0||image->cut.forwarded==0||
       image->cut.accepted!=image->cut.forwarded||
       image->cut.exact_frontier!=image->cut.forwarded||!image->pending_tokens.empty()||
       !image->approximation.empty()||image->cut.accepted>admission.token_limit||
       published.cut.id==0||image->cut.accepted>published.cut.accepted||
       published.cut.forwarded<=image->cut.forwarded||
       published.cut.exact_frontier>=published.cut.forwarded||
       published.cut.exact_frontier>image->cut.forwarded||published.approximation.empty()||
       image->accepted_tokens.size()!=image->cut.accepted||
       image->accepted_tokens.size()>published.accepted_tokens.size()||
       !std::equal(image->accepted_tokens.begin(),image->accepted_tokens.end(),
                   published.accepted_tokens.begin())) {
        // Structured diagnostic: name each failing conjunct without exposing tensors
        std::fprintf(stderr,
            "yield_oracle_precondition: "
            "oracle=%d credit=%d image=%d ordinal=%d req_open=%d capture_ok=%d "
            "img_id=%llu img_fwd=%llu img_acc=%llu img_ef=%llu img_pend=%zu "
            "img_approx_empty=%d img_acc>%d tok_lim=%d "
            "pub_id=%llu img_acc>pub_acc=%d pub_fwd=%llu pub_fwd<=img_fwd=%d "
            "pub_ef=%llu pub_ef>=pub_fwd=%d pub_ef>img_fwd=%d pub_approx_empty=%d "
            "img_tok_sz=%zu img_tok!=acc=%d img_tok>pub_tok=%d prefix_eq=%d\n",
            (int)admission.yield_oracle, (int)(bool)admission.credit, (int)(bool)image, (int)(bool)ordinal,
            (int)session.request_open(), (int)session.capture_allowed(),
            image?(unsigned long long)image->cut.id:0, image?(unsigned long long)image->cut.forwarded:0,
            image?(unsigned long long)image->cut.accepted:0, image?(unsigned long long)image->cut.exact_frontier:0,
            image?image->pending_tokens.size():0, image?(int)image->approximation.empty():0,
            (int)(image&&image->cut.accepted>admission.token_limit), (int)admission.token_limit,
            (unsigned long long)published.cut.id, (int)(image&&image->cut.accepted>published.cut.accepted),
            (unsigned long long)published.cut.forwarded,
            (int)(image&&published.cut.forwarded<=image->cut.forwarded),
            (unsigned long long)published.cut.exact_frontier,
            (int)(published.cut.exact_frontier>=published.cut.forwarded),
            (int)(image&&published.cut.exact_frontier>image->cut.forwarded),
            (int)published.approximation.empty(),
            image?image->accepted_tokens.size():0,
            (int)(image&&image->accepted_tokens.size()!=image->cut.accepted),
            (int)(image&&image->accepted_tokens.size()>published.accepted_tokens.size()),
            image?(int)std::equal(image->accepted_tokens.begin(),image->accepted_tokens.end(),
                                  published.accepted_tokens.begin()):-1);
        failstop("yield oracle actual publication/admission differs");
    }
    // The certified cut must be a prefix of the same accepted history the session stands on, so the recomputation
    // and the yielded image are two readings of ONE conversation, not two conversations. Epochs legitimately
    // differ (the private gang bumps its own selection epoch), so identity is bound through the digest and the
    // token prefix above rather than through a whole-key comparison.
    auto& tx_ref=*session.transaction();
    if(tx_ref.cut_!=published.cut)failstop("yield oracle session state differs from its publication");
    const auto& original=*image;
    const uint64_t accepted=original.cut.accepted,forwarded=original.cut.forwarded;
    if(tx_ref.cut_.id==UINT64_MAX||next_cut==UINT64_MAX)failstop("yield oracle cut overflow");
    next_cut=std::max(next_cut,tx_ref.cut_.id+1);
    // The two cuts the records will report, taken from the bundles themselves before any of this runs, so a
    // refusal part-way through still leaves the caller an honest picture of what was and was not compared.
    report.published_cut=published.cut.forwarded;report.certified_cut=forwarded;
    report.frontier_forwarded=forwarded;report.frontier_accepted=accepted;
    report.ordinal_bound=ordinal!=0;
    // The live session stands on D. Freeze it first: every comparison below dirties the live transaction, and a
    // yield must not roll the conversation back to its certified cut.
    core::BundleHandle live_image;
    if(!raw_call([&]{return session.capture(published,live_image);}))
        failstop("yield oracle live publication capture refused");
    if(!live_image||!live_image.get()||live_image.get()->cut!=published.cut)
        failstop("yield oracle live capture returned an image of another cut");
    bool pass=false,was_cancelled=cancelled(),frontier=false,continuation=false;
    bool frontier_written=false,continuation_written=false;
    try {
        Buffers buffers;buffers.history=original.accepted_tokens;
        buffers.exact_logits.resize(size_t(c.geometry.n_vocab));
        buffers.approximate_logits.resize(size_t(c.geometry.n_vocab));
        struct SamplingScope {core::Verifier& verifier;
            explicit SamplingScope(core::Verifier& v):verifier(v){verifier.set_head_sampling(false);}
            ~SamplingScope(){verifier.set_head_sampling(true);}
        } sampling(c.primary_verifier);
        ServingComparisonWitness frontier_witness{},continuation_witness{};
        if(!was_cancelled) {
            std::string prefix_error;
            if(prefix.plan && (!prefix.calls || !prefix.calls->check ||
               !prefix.plan->matches(original.accepted_tokens,prefix.origin_request?prefix.origin_request:ordinal,forwarded) ||
               !prefix.calls->check(prefix.calls->user,*prefix.plan,prefix_error)))
                failstop("yield oracle recorded prefix binding differs before reset");
            reset(c,tx_ref,next_cut,original.cut.output_watermark);
            Step step;uint64_t prefix_at=0;
            if(prefix.plan){
                auto cancelled_thunk=[&](){return cancelled();};
                PrefixReplayAbort abort{&cancelled_thunk,[](void* p){return (*static_cast<decltype(cancelled_thunk)*>(p))();}};
                uint64_t batches=0;
                for(const auto& event:prefix.plan->events){
                    if(cancelled()){was_cancelled=true;break;}
                    if(event.kind==PrefixReplayEvent::Kind::window){
                        PrefixReplayEvent refill;refill.from=refill.to=event.from;
                        step=replay_prefill_piece(c,tx_ref,next_cut,refill,*prefix.calls,buffers.history,
                            accepted,original.cut.history,abort,was_cancelled);
                        for(uint64_t at=event.from;at<event.to;++at){
                            if(cancelled()){was_cancelled=true;break;}
                            step=forward(c,serial,tx_ref,next_cut,buffers.history[size_t(at)],at,accepted,
                                original.cut.history,at+1,{});
                        }
                    }else{
                        step=replay_prefill_piece(c,tx_ref,next_cut,event,*prefix.calls,buffers.history,
                            accepted,original.cut.history,abort,was_cancelled);++batches;
                    }
                    if(was_cancelled)break;
                }
                PrefixReplayEvent refill;refill.from=refill.to=tx_ref.cut_.forwarded;
                step=replay_prefill_piece(c,tx_ref,next_cut,refill,*prefix.calls,buffers.history,
                    accepted,original.cut.history,abort,was_cancelled);
                if(!prefix.calls->check(prefix.calls->user,*prefix.plan,prefix_error))
                    failstop("yield oracle prefix final base mapping differs");
                prefix_at=step.cut.forwarded;
                if(!was_cancelled&&prefix_at!=prefix.plan->boundary)failstop("yield prefix replay boundary incomplete");
                const auto schedule=qualification_hex(prefix.plan->digest);
                std::fprintf(stderr,"suffix_repair_prefix_replay req=%llu origin_req=%llu phase=retained_yield boundary=%llu reached=%llu schedule=%s batches=%llu completed=%u\n",
                    (unsigned long long)ordinal,(unsigned long long)prefix.plan->request,
                    (unsigned long long)prefix.plan->boundary,(unsigned long long)prefix_at,schedule.data(),
                    (unsigned long long)batches,unsigned(!was_cancelled));
            }
            for(uint64_t at=prefix_at;!was_cancelled&&at<forwarded;++at) {
                if(cancelled()){was_cancelled=true;break;}
                step=forward(c,serial,tx_ref,next_cut,buffers.history[size_t(at)],at,accepted,
                    original.cut.history,at+1,{});
            }
            if(!was_cancelled) {
                core::BundleHandle clean;
                const auto metadata=ServingRawSession::endpoint_metadata(original.key,step.cut,buffers.history);
                if(!raw_call([&]{return session.capture(metadata,clean);}))
                    failstop("yield oracle cold capture refused");
                frontier=report_serving_comparison(session,admission,ordinal,
                    strata::program::kServingYieldFrontierPhase,certified,
                    clean,true,true,0,&frontier_witness);
                frontier_written=true;
                clean={};
                if(!cancelled()) {
                    // The private checkpoint is a closed consumed prefix (no pending token).
                    // The next real token comes from the published accepted sequence at position R.
                    const uint64_t R = forwarded;  // private image's forwarded (= accepted, closed prefix)
                    if (R >= published.accepted_tokens.size())
                        failstop("yield oracle published history exhausted at certified cut");
                    const int32_t pending=published.accepted_tokens[size_t(R)];
                    buffers.history.push_back(pending);  // extend consumed prefix to R+1
                    const auto extended_history_digest = core::window_history_digest(buffers.history);
                    const uint64_t cont_accepted = accepted + 1;  // R+1
                    const uint64_t cont_exact_frontier = R + 1;   // private state is exact at R+1
                    const auto cold_next=forward(c,serial,tx_ref,next_cut,pending,forwarded,cont_accepted,
                        extended_history_digest,cont_exact_frontier,buffers.exact_logits);
                    core::BundleHandle cold_continuation;
                    const auto cold_meta=ServingRawSession::endpoint_metadata(original.key,cold_next.cut,buffers.history);
                    if(!raw_call([&]{return session.capture(cold_meta,cold_continuation);}))
                        failstop("yield oracle cold continuation capture refused");
                    if(!raw_call([&]{return session.restore(certified,original.key);}))
                        failstop("yield oracle repaired restore before continuation refused");
                    const auto repaired_next=forward(c,serial,tx_ref,next_cut,pending,forwarded,cont_accepted,
                        extended_history_digest,cont_exact_frontier,buffers.approximate_logits);
                    core::BundleHandle repaired_continuation;
                    const auto next_meta=ServingRawSession::endpoint_metadata(original.key,repaired_next.cut,buffers.history);
                    if(!raw_call([&]{return session.capture(next_meta,repaired_continuation);}))
                        failstop("yield oracle repaired continuation capture refused");
                    SuffixLogitComparison logits;
                    const bool valid=compare_suffix_logits(buffers.exact_logits,buffers.approximate_logits,logits);
                    const bool picks=valid&&cold_next.next==repaired_next.next&&
                        cold_next.next==logits.exact_argmax&&repaired_next.next==logits.approximate_argmax;
                    continuation=report_serving_comparison(session,admission,ordinal,
                        strata::program::kServingYieldContinuationPhase,
                        repaired_continuation,cold_continuation,valid&&logits.max_absolute_error==0,picks,
                        valid?logits.max_absolute_error:-1,&continuation_witness);
                    continuation_written=true;
                    pass=frontier&&continuation;
                    report.continuation_forwarded=cold_next.cut.forwarded;
                }
            }
            report.frontier_forwarded=frontier_witness.forwarded;
            report.frontier_accepted=frontier_witness.accepted;
            report.frontier_exact=frontier_witness.exact_frontier==frontier_witness.forwarded;
            report.continuation_exact=continuation_witness.exact_frontier==continuation_witness.forwarded;
            report.completed=frontier_witness.completed&&continuation_witness.completed;
            report.payload_equal=frontier_witness.equal&&continuation_witness.equal;
            report.mismatches=frontier_witness.mismatches+continuation_witness.mismatches;
            report.pair_emitted=frontier_written&&continuation_written;
        }
        // Always return the session to its own CURRENT publication, including on a completed mismatch/cancel,
        // and verify it: the foreground state this response continues from is the approximate publication at D,
        // with its own cut id, position, certified frontier, output watermark and history digest. A restore that
        // landed anywhere else - including on the shorter certified cut - is the established failstop, never a
        // silently accepted difference.
        if(!raw_call([&]{return session.restore(live_image,published.key);}))
            failstop("yield oracle live publication restoration refused");
        if(tx_ref.cut_!=published.cut||tx_ref.cut_.id!=published.cut.id||
           tx_ref.cut_.forwarded!=published.cut.forwarded||
           tx_ref.cut_.accepted!=published.cut.accepted||
           tx_ref.cut_.exact_frontier!=published.cut.exact_frontier||
           tx_ref.cut_.output_watermark!=published.cut.output_watermark||
           tx_ref.cut_.history!=published.cut.history||
           tx_ref.cut_.id==original.cut.id||session.cut()!=published.cut)
            failstop("yield oracle left the live session on a state other than the foreground publication it must preserve");
        was_cancelled=was_cancelled||cancelled();pass=pass&&!was_cancelled;
        report.restoration_emitted=true;report.restored=true;report.cancelled=was_cancelled;
        report.identity_pinned=admission.binary.front()!='\0'&&admission.binary.back()=='\0'&&
            admission.source.front()!='\0'&&admission.source.back()=='\0';
        report.ordinal_bound=ordinal!=0;
        std::fprintf(stderr,"suffix_repair_qualification_done req=%llu pass=%u restored=1 cancelled=%u "
            "next_cut=%llu binary=%s source=%s elapsed_ms=%.6f\n",
            (unsigned long long)ordinal,unsigned(pass),unsigned(was_cancelled),(unsigned long long)next_cut,
            admission.binary.data(),admission.source.data(),elapsed(oracle_started));
        std::fflush(stderr);
        return pass;
    }catch(...){failstop("yield oracle exception retains execution buffers");}
}

#include "serving_slot_qualification.inc"
