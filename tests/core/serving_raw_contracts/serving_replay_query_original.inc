    void arm_serving_replay_fault(core::resident_deferred_fault::Scope& scope,const Pause& pause,
                                 const core::resident_serving_fault::Snapshot& snapshot){
        if(!busy_||cancel_requested_||serving_fault_fired_||!serving_fault_.valid()||
           serving_fault_.site()!=core::resident_serving_fault::Site::private_replay||!snapshot.generated||
           private_->phase_!=SuffixRawTransaction::Phase::running||live_.phase_!=SuffixRawTransaction::Phase::drained||
           runtime_->phase_!=RuntimePhase::drained||runtime_->service_->active_!=&pause||runtime_->service_->unhealthy_||
           runtime_->dispatch_->callback_depth_||!runtime_->allocation_.healthy()||!private_->allocation_.healthy()||
           !runtime_->dispatch_->last_completion_||!runtime_->dispatch_->last_completion_->committed()||
           runtime_->dispatch_->last_completed_operation_!=runtime_->operation_||
           private_history_.size()!=snapshot.repaired+1||snapshot.repaired>=snapshot.forwarded||
           runtime_->controller_!=std::this_thread::get_id())fatal("serving replay fault actual T1 boundary refused");
        auto& stage=runtime_->stages_[1];
        if(!stage.session||!stage.session_owner||!stage.verifier||!stage.stream||
           stage.service.device!=1||stage.verifier->stream()!=stage.stream||private_->verifiers_.size()!=2||
           private_->verifiers_[1]!=stage.verifier.get()||private_->allocation_.stages()[1].session!=stage.session)
            fatal("serving replay fault actual stage1 binding refused");
        core::WindowDeviceScope device(1);if(!device.ok())fatal("serving replay fault device prerequisite refused");
        core::resident_serving_fault::binding_record(serving_fault_,snapshot);
        core::resident_serving_fault::event(serving_fault_,snapshot,"private_work_completed","run_private_one",0);
        core::resident_fault_precondition::check(serving_fault_.fd(),core::resident_serving_fault::name(serving_fault_.site()),1,stage.stream);
        scope.arm_serving(serving_fault_,snapshot,stage.verifier.get(),stage.stream);serving_fault_fired_=true;
        scope.record("armed",0);
        const auto launched=core::resident_deferred_fault::launch_private_stage_trap(stage.stream);scope.record("launch",int(launched));
        if(launched!=cudaSuccess)fatal("serving replay fault trap launch refused");
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        for(unsigned polls=0;polls<100000&&std::chrono::steady_clock::now()<deadline;++polls){
            std::string error;const auto drained=runtime_->stages_[0].verifier->poll_snapshot_drain(error);
            if(drained==core::Verifier::SnapshotDrain::unhealthy){
                if(!scope.observed_||(scope.observed_status_!=cudaErrorIllegalInstruction&&scope.observed_status_!=cudaErrorLaunchFailure))
                    fatal("serving replay fault query attribution unavailable");
                return; // actual RAW complete_driver_work must poison; no state/status is manufactured here
            }
            if(drained==core::Verifier::SnapshotDrain::unsupported)fatal("serving replay fault drain unsupported");
        }
        fatal("serving replay fault query deadline or poll bound");
    }
