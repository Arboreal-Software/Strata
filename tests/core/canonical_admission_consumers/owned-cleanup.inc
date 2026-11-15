bool OwnedQsaElasticPool::retire_checked()noexcept{
    auto& s=*storage_;if(s.retired)return true;if(s.sealed)return false;if(s.attempted&&!s.ready)return false;
    if(!s.attempted){s.retired=true;return true;}
    s.ready=false;
    WindowDeviceScope device(s.plan.device);
    if(!device.ok()||cudaDeviceSynchronize()!=cudaSuccess){(void)device.finish();return false;}
    s.range.release();
    if(!device.finish())return false;
    s.retired=true;s.committed=0;return true;
}
OwnedQsaElasticPool::~OwnedQsaElasticPool(){
    if(!storage_)return;
    if(!retire_checked()){
        static std::atomic<Storage*> quarantine{nullptr};auto* p=storage_.release();
        p->next=quarantine.load();while(!quarantine.compare_exchange_weak(p->next,p)){}
    }
}
