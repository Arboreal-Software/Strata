struct SplitDrive{static constexpr int kMax=8;};
int main(int argc,char**argv){
 int mode=argc>1?std::atoi(argv[1]):0;
 bool serving_threads=true,live_approximate=false,serving_raw_fence_running=true,serving_ordinary_fence=false;
 uint64_t serving_raw_cut=9;std::vector<int64_t>ids{1,2,3,4,5,6,7,8};std::vector<int>stages(2);
 struct{uint64_t request=7;}owned_admission_scope;
 auto serving_raw_session=std::make_shared<strata::program::ServingRawSession>();auto original=serving_raw_session;
 original->identity.engine_generation=1;original->current.id=3;original->current.output_watermark=2;
 int refill=0,dispatched=0,fallback=0;
 auto shared_prefix_refill=[&](bool,std::string&){++refill;return true;};
 // ACTUAL_HELPER
 prefill_joined_main=original;prefill_joined_key=original->key();prefill_joined_request=7;prefill_joined_end=4;prefill_joined_stage_end.fill(4);
 std::vector<int32_t>joined{1,2,3,4};prefill_joined_history=strata::core::window_history_digest(joined);
 if(mode==1)stages.resize(8);
 if(mode==2)prefill_joined_main=std::make_shared<strata::program::ServingRawSession>(*original);
 if(mode==3)ids[0]=999;
 if(mode==4)prefill_joined_request=6;
 if(mode==5)original->oom=true;
 if(mode==6)original->budget=true;
 if(mode==7)serving_raw_cut=2;
 if(mode==8)serving_raw_cut=UINT64_MAX;
 if(mode==16){prefill_joined_main.reset();prefill_joined_end=UINT64_MAX;}
 auto dispatch=[&](){++dispatched;
  if(mode==10)serving_raw_session.reset();
  if(mode==11)serving_raw_session=std::make_shared<strata::program::ServingRawSession>(*original);
  if(mode==12)++serving_raw_cut;
  if(mode==13)serving_ordinary_fence=true;
  if(mode==14)serving_raw_fence_running=true;
  if(mode==15)++prefill_turn_history[0];
  return true;
 };
 std::string error;const bool okay=serving_prefill_safe_turn(mode==16?0:4,dispatch,error);
 if(mode==5||mode==6){
  check(!okay&&prefill_turn_unavailable,"scratch refusal typed optional unavailable");check(refill==0&&dispatched==0,"refusal before loan return/dispatch");
  check(serving_raw_session.get()==original.get()&&original->request_open()&&original->phase()==strata::program::ServingRawSession::Phase::running&&serving_raw_fence_running,"scratch refusal preserves MAIN/fence");
  check(!prefill_turn_credit&&original->last_credit.expired(),"failed reserve releases temporary credit");
  const bool turned=okay;auto choose_and_service=[&](bool borrowed){check(!borrowed,"optional scratch fallback uses ordinary foreground path");++fallback;return true;};
  auto actual_fallback=[&]()->bool{ // ACTUAL_FALLBACK
   return true;
  };check(actual_fallback()&&fallback==1,"actual caller continues foreground on scratch refusal");
  original->budget=false;check(serving_prefill_safe_turn(4,dispatch,error),"retry re-admits paid scratch");check(original->admitted==2,"retry performs authentic re-admission");
 }else if((mode>=1&&mode<=4)||mode==7||mode==8||mode==16){check(!okay&&refill==0&&dispatched==0,"contract mismatch refused before refill");}
 else{check(okay&&refill==1&&dispatched==1,"normal exact helper ran once");check(serving_raw_session.get()==original.get()&&original->request_open()&&original->phase()==strata::program::ServingRawSession::Phase::running&&serving_raw_fence_running,"original MAIN reopened");check(serving_raw_cut==10,"allocator may exceed original MAIN cut");}
 std::puts("PASS actual safe-turn helper recording boundary");
}
