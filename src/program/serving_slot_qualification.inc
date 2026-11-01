// Included inside SuffixExperimentDriver. Disposable opt-in native oracle only.
static bool owned_slot_qualification_target(uint64_t& ordinal) noexcept {
    ordinal=0;const char* value=std::getenv("STRATA_OWNED_BATCH_QUALIFY");
    if(!value)return true;
    const auto n=std::strlen(value);const auto parsed=std::from_chars(value,value+n,ordinal);
    return parsed.ec==std::errc{}&&parsed.ptr==value+n&&ordinal>0&&ordinal<=65536;
}
static bool owned_slot_qualification_selected(uint64_t ordinal) {
    uint64_t target=0;if(!owned_slot_qualification_target(target))failstop("invalid owned-slot qualification request");
    return target&&target==ordinal;
}
template<class Policy>
static bool owned_slot_qualification_policy(const Policy& options,std::string& error) {
    uint64_t target=0;if(!owned_slot_qualification_target(target)){error="invalid positive owned-slot qualification request";return false;}
    if(!target)return true;
    uint64_t tokens=0;const char* source=std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST");
    const char* lookahead=std::getenv("STRATA_LOOKAHEAD");
    if(!core::native_owned_batch_serving::selected(std::getenv("STRATA_OWNED_BATCH_ALLOCATION"))||
       !core::native_owned_batch_serving::eligible(true,options,true)||!options.suffix_cache||
       options.suffix_cache_unrepaired_experiment||!source||!*source||std::strlen(source)>4096||
       !serving_qualification_limit(tokens)||!lookahead||std::strcmp(lookahead,"0")||
       std::getenv("STRATA_SUFFIX_REPAIR_QUALIFY")||activation_qualification_mode()){
        error="owned-slot qualification requires its strict owned profile, separate request target, source identity and disabled lookahead";return false;
    }
    return true;
}
template<class Context>
static bool admit_owned_slot_qualification(Context& c,ServingRawSession& session,uint64_t accepted,
        uint64_t boundary,uint64_t ordinal,ServingQualification& out,std::string& error) {
    if(!owned_slot_qualification_selected(ordinal)||!c.options.native_owned_batch_serving||
       out.credit||out.admitted||!serving_qualification_limit(out.token_limit)||accepted>out.token_limit||
       accepted>session.max_tokens()||c.geometry.n_vocab<=0||c.geometry.n_expert<=0||c.geometry.n_layers<=0||
       !validate_serving_qualification_execution(c,error))return false;
    const char* manifest=std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST");
    const char* lookahead=std::getenv("STRATA_LOOKAHEAD");
    if(!manifest||!*manifest||!lookahead||std::strcmp(lookahead,"0")||
       std::getenv("STRATA_SUFFIX_REPAIR_QUALIFY")||activation_qualification_mode()){
        error="owned-slot oracle needs explicit identity/disabled lookahead and a separate qualification selector";return false;
    }
    uint64_t bytes=1ULL<<20;
    if(uint64_t(c.geometry.n_layers)>UINT64_MAX/uint64_t(c.geometry.n_expert)||
       !PrefixReplayPlan::quote(boundary,uint64_t(c.geometry.n_layers)*uint64_t(c.geometry.n_expert),bytes)||
       !add_product(bytes,session.max_tokens(),128)||
       !add_product(bytes,uint64_t(c.geometry.n_vocab),3*sizeof(float))){error="owned-slot oracle admission overflow";return false;}
    const auto admitted=session.admit_host(bytes,out.credit);if(!admitted){error=admitted.detail;return false;}
    if(!qualification_file_hash("/proc/self/exe",2ULL<<30,out.binary)||
       !qualification_file_hash(manifest,4ULL<<20,out.source)){error="owned-slot oracle actual identity refused";return false;}
    if(boundary>64)try{out.durable_prefix=std::make_shared<PrefixReplayPlan>();}catch(...){return false;}
    out.prefix.boundary=boundary;out.prefix.request=ordinal;
    out.admitted=true;out.retained=true;out.owned_slot_oracle=true;return true;
}
static bool report_owned_slot_preserved(ServingRawSession& session,const ServingQualification& admission,
        uint64_t ordinal,const char* role,int64_t slot,uint64_t occupancy,const core::BundleHandle& before,const core::BundleHandle& after) {
    core::WindowStateLayout layout;
    if(!before||!after||!core::native_window_layout(session.geometry(),session.initial_config(),before.get()->key,before.get()->cut,
        session.max_tokens(),session.max_payload_bytes(),layout,false))failstop("owned-slot preservation actual trusted layout refused");
    clean_recompute_qualification::Report report;
    const bool equal=clean_recompute_qualification::compare_preserved_selection(before,after,layout,report);
    const auto layout_digest=qualification_layout_hash(layout);
    const auto left=qualification_hex(qualification_state_hash(*before.get(),layout_digest));
    const auto right=qualification_hex(qualification_state_hash(*after.get(),layout_digest));
    const auto history=qualification_hex(before.get()->cut.history);
    std::fprintf(stderr,"owned_slot_preservation req=%llu role=%s slot=%lld occupancy=%llu completed=%u equal=%u mismatches=%llu stages=%zu tensors=%llu bytes=%llu before_cut=%llu after_cut=%llu history=%s before=%s after=%s binary=%s source=%s oracle=%s\n",
        (unsigned long long)ordinal,role,(long long)slot,(unsigned long long)occupancy,unsigned(report.completed),unsigned(equal),
        (unsigned long long)report.mismatches,layout.stages.size(),(unsigned long long)report.tensors_compared,(unsigned long long)report.bytes_compared,
        (unsigned long long)before.get()->cut.id,(unsigned long long)after.get()->cut.id,history.data(),left.data(),right.data(),admission.binary.data(),admission.source.data(),std::strcmp(role,"main")==0?"restore_image":"actual_recapture");
    std::fflush(stderr);return equal&&report.completed&&report.equal&&!report.mismatches;
}
// Cross-owner physical transfer: source keeps its real key; MAIN derives a new
// common cut only after reset/copy/PLE/all-stage drains, then captures its OWN key.
static core::BundleHandle copy_owned_slot_oracle_to_main(ServingRawSession& main,
        const core::BundleHandle& source,const core::WindowKey& destination_key,uint64_t& next_cut) {
    if(!source||!main.request_open()||!main.capture_allowed()||main.slot_occupancy_incarnation()||
       !source.get()->approximation.empty()||source.get()->cut.exact_frontier!=source.get()->cut.forwarded)
        failstop("owned-slot oracle bridge actual source/main precondition");
    auto& tx=*main.transaction();std::string error;std::vector<core::NativeWindowStage> config;
    if(!tx.resolve(config,error)||!next_cut||next_cut==UINT64_MAX||next_cut<=tx.cut_.id||
       source.get()->key.model_profile!=tx.identity_.model_profile||!tx.identity_matches(destination_key))
        failstop("owned-slot oracle bridge actual identity/layout differs");
    core::WindowStateLayout layout;
    if(!core::native_window_layout(tx.geometry_,config,source.get()->key,source.get()->cut,
        tx.max_tokens_,tx.max_bytes_,layout,false))failstop("owned-slot oracle source layout refused");
    uint64_t bytes=0;std::shared_ptr<const void> credit;
    if(!tx.quote(layout,3,bytes)||!tx.admit_controller_host(bytes,credit))failstop("owned-slot oracle bridge host admission refused");
    core::SavedConversation native;std::vector<core::NativePleTokens> ple;
    if(!core::materialize_native_window(*source.get(),source.get()->key,tx.geometry_,config,tx.max_tokens_,tx.max_bytes_,false,native,ple)||
       ple.size()!=tx.allocation_.stages().size())failstop("owned-slot oracle source materialization refused");
    core::SnapshotDescriptor metadata=*source.get();metadata.devices.clear();metadata.key=destination_key;
    metadata.cut.id=next_cut++;std::vector<core::StateCut> cuts(config.size(),metadata.cut);
    for(size_t i=0;i<config.size();++i){const auto& image=i?native.stage_images[i-1]:native;
        if(!core::conversation_snapshot_validate(image,*tx.allocation_.stages()[i].session,tx.geometry_,
            static_cast<const core::QsaState*>(nullptr),error))failstop("owned-slot oracle actual destination shape refused");}
    if(!tx.poll_all()||!tx.begin_driver_work())failstop("owned-slot oracle bridge begin refused");
    for(const auto& stage:tx.allocation_.stages()){
        core::WindowDeviceScope device(stage.device);
        if(!device.ok()||!core::session_zero_checked(*stage.session,tx.geometry_,nullptr,nullptr,error)||
           cudaDeviceSynchronize()!=cudaSuccess||!device.finish())failstop("owned-slot oracle MAIN reset uncertain");
    }
    for(size_t i=0;i<config.size();++i){const auto& stage=tx.allocation_.stages()[i];const auto& image=i?native.stage_images[i-1]:native;
        core::WindowDeviceScope device(stage.device);
        if(!device.ok()||core::conversation_snapshot_restore(image,*stage.session,tx.geometry_,
           static_cast<const core::QsaState*>(nullptr),error)!=core::ConversationRestore::restored)
            failstop("owned-slot oracle MAIN copy uncertain");
        stage.session->ple_prev[0]=ple[i].previous[0];stage.session->ple_prev[1]=ple[i].previous[1];stage.session->ple_token=ple[i].current;
        if(cudaDeviceSynchronize()!=cudaSuccess||!device.finish())failstop("owned-slot oracle MAIN copy drain uncertain");
    }
    if(!tx.complete_driver_work(true,metadata.cut,cuts))failstop("owned-slot oracle MAIN common cut refused");
    core::BundleHandle captured;if(!main.capture(metadata,captured))failstop("owned-slot oracle true MAIN recapture refused");
    return captured;
}
template<class Context,class SerialUnit>
static bool qualify_owned_slot_final(Context& context,SerialUnit& serial,ServingRawSession& main,
        ServingRawSession& slot,uint64_t ordinal,ServingQualification& admission,const core::BundleHandle& actual,
        const core::SnapshotDescriptor& prior_metadata,uint64_t& next_cut,PrefixReplayView prefix) {
    if(!admission.owned_slot_oracle||!owned_slot_qualification_selected(ordinal)||!actual||
       actual.get()->cut!=slot.cut()||!slot.request_open()||main.request_open()||
       main.cut()!=prior_metadata.cut||!main.healthy())failstop("owned-slot oracle admission/prior differs");
    if(admission.prefix.boundary>64&&(!prefix.plan||!prefix.plan->matches(actual.get()->accepted_tokens,ordinal,actual.get()->cut.forwarded)))
        failstop("owned-slot oracle requires its real sealed prefix transcript");
    std::array<char,65> binary{},source{};
    if(!qualification_file_hash("/proc/self/exe",2ULL<<30,binary)||
       !qualification_file_hash(std::getenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST"),4ULL<<20,source)||
       binary!=admission.binary||source!=admission.source)failstop("owned-slot oracle actual executable/source changed");
    std::string error;if(!main.open_request(error))failstop("owned-slot oracle MAIN request refused");
    core::BundleHandle prior;if(!(prior_metadata.cut.forwarded?main.capture(prior_metadata,prior):main.transaction()->capture_empty_oracle(prior_metadata,prior)))failstop("owned-slot oracle MAIN prior capture refused");
    next_cut=std::max(next_cut,main.cut().id+1);
    const auto key=prefix.plan?prefix.plan->origin_key:main.key();
    const auto copy=copy_owned_slot_oracle_to_main(main,actual,key,next_cut);
    // Allocation identity is authenticated against EACH actual live owner; neither key is relabeled.
    auto& source_tx=*slot.transaction();auto& destination_tx=*main.transaction();
    if(!source_tx.identity_matches(actual.get()->key)||!destination_tx.identity_matches(copy.get()->key)||
       source_tx.geometry_identity_!=destination_tx.geometry_identity_||
       source_tx.identity_.model_profile!=destination_tx.identity_.model_profile||
       slot.cut()!=actual.get()->cut||main.cut()!=copy.get()->cut)
        failstop("owned-slot bridge actual owner key/cut/geometry binding differs");
    core::WindowStateLayout destination_layout;
    if(!core::native_window_layout(main.geometry(),main.initial_config(),copy.get()->key,copy.get()->cut,
        main.max_tokens(),main.max_payload_bytes(),destination_layout,false))
        failstop("owned-slot bridge actual MAIN layout refused");
    const bool bridge=report_serving_comparison(slot,admission,ordinal,"owned_slot_bridge",actual,copy,true,true,0,nullptr,"physical_bridge",&destination_layout);
    const auto never_cancel=[](){return false;};
    const bool numerical=qualify_serving_frontier(context,serial,main,ordinal,admission,copy,next_cut,
        never_cancel,"owned_slot_frontier","owned_slot_continuation",0
#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)
        ,nullptr
#endif
        ,prefix);
    if(!(prior.get()->cut.forwarded?main.restore(prior,prior.get()->key):main.transaction()->restore_empty_oracle(prior,prior.get()->key)))failstop("owned-slot oracle MAIN prior restoration uncertain");
    core::BundleHandle restored;if(!(prior_metadata.cut.forwarded?main.capture(prior_metadata,restored):main.transaction()->capture_empty_oracle(prior_metadata,restored)))failstop("owned-slot oracle restored MAIN capture refused");
    core::WindowStateLayout prior_layout;
    if(!core::native_window_layout(main.geometry(),main.initial_config(),prior.get()->key,prior.get()->cut,
        main.max_tokens(),main.max_payload_bytes(),prior_layout,false))failstop("owned-slot oracle prior trusted layout refused");
    const bool main_restored=report_owned_slot_preserved(main,admission,ordinal,"main",-1,0,prior,restored);
    if(!main.close_request(error))failstop("owned-slot oracle MAIN close refused");
    core::SnapshotDescriptor slot_metadata=*actual.get();slot_metadata.devices.clear();
    core::BundleHandle slot_after;if(!slot.capture(slot_metadata,slot_after))failstop("owned-slot oracle source recapture refused");
    core::WindowStateLayout slot_layout;
    if(!core::native_window_layout(slot.geometry(),slot.initial_config(),actual.get()->key,actual.get()->cut,
        slot.max_tokens(),slot.max_payload_bytes(),slot_layout,false))failstop("owned-slot oracle slot trusted layout refused");
    const bool slot_preserved=report_owned_slot_preserved(slot,admission,ordinal,"selected",
        int64_t(slot.transaction()->slot_occupancy_->slot()),slot.slot_occupancy_incarnation(),actual,slot_after);
    const bool pass=bridge&&numerical&&main_restored&&slot_preserved;
    std::fprintf(stderr,"owned_slot_qualification_main_done req=%llu pass=%u main_restored=%u slot_preserved=%u binary=%s source=%s\n",
        (unsigned long long)ordinal,unsigned(pass),unsigned(main_restored),unsigned(slot_preserved),admission.binary.data(),admission.source.data());
    std::fflush(stderr);return pass;
}
