// Fixture destructors only exercise stop/join, not production GPU frees.
RouterLookahead::~RouterLookahead(){assert(close_for_retirement());}
void RouterLookahead::run() {} // numeric worker loop is not this fixture's target
ForesightSwap::~ForesightSwap(){assert(close_for_retirement());}
} // namespace strata::core
#undef private
using namespace strata::core;
static void setup(ForesightSwap& s){
    s.slots=1;s.n_layers=1;s.n_expert=1;s.slot.reset(new ForesightSwap::Slot[1]);
    s.where={-1};s.miss_cnt={0};s.miss_at={0};s.layer_card={0};s.card_round={~0ull};s.card_used={0};
}
int main(){
    {
        // Actual start holds the lifecycle mutex while invoking virtual warms.
        // Close must reject reentry BEFORE touching that same mutex.
        RouterLookahead r;ExpertSource src;bool refused=false;std::string error;
        src.callback=[&]{refused=!r.close_for_retirement();};
        assert(!r.start({},8,1,1,&src,error));assert(refused);
        assert(!r.thread_.joinable()&&r.routers_.empty()&&r.x_.empty());
        assert(r.close_for_retirement());assert(r.close_for_retirement());
    }
    {
        // Nested starts must detect an owner below the top lifecycle frame.
        RouterLookahead a,b;ExpertSource as,bs;bool refused=false;std::string error;
        as.callback=[&]{assert(b.start({},8,1,1,&bs,error));};
        bs.callback=[&]{refused=!a.close_for_retirement();};
        assert(!a.start({},8,1,1,&as,error));assert(refused&&!a.thread_.joinable());
        assert(a.close_for_retirement());assert(b.close_for_retirement());
    }
    {
        RouterLookahead r;ExpertSource src;bool refused=false;std::string error;
        src.callback=[&]{refused=!r.start({},8,1,1,&src,error);};
        assert(r.start({},8,1,1,&src,error));assert(refused);
        assert(r.close_for_retirement());
    }

    {
        RouterLookahead r;r.routers_.resize(2);r.n_embd_=1;r.x_={19};
        auto* bytes=r.x_.data();float x=7;int32_t res=0;
        assert(r.close_for_retirement());assert(r.close_for_retirement());
        r.submit(0,&x,1,&res);r.set_depth(4);r.stats_window_start();r.enable_stats(2);
        r.set_foresight(reinterpret_cast<ForesightSwap*>(1));
        assert(r.x_.data()==bytes&&r.x_[0]==19&&!r.pending_&&r.depth_==1&&r.gen_==0&&!r.stats_on_&&r.fs_==nullptr);
    }
    {
        // Join must release mu_: the worker needs it to observe quit.
        RouterLookahead r;
        r.thread_=std::thread([&]{std::unique_lock<std::mutex> lock(r.mu_);r.cv_.wait(lock,[&]{return r.quit_;});});
        assert(r.close_for_retirement());assert(!r.thread_.joinable());
    }
    {
        RouterLookahead r;bool refused=false,ready=false,done=false;std::mutex mu;std::condition_variable cv;
        r.thread_=std::thread([&]{
            {std::unique_lock<std::mutex> lock(mu);cv.wait(lock,[&]{return ready;});}
            refused=!r.close_for_retirement();
            {std::lock_guard<std::mutex> lock(mu);done=true;}cv.notify_all();
        });
        {std::lock_guard<std::mutex> lock(mu);ready=true;}cv.notify_all();
        {std::unique_lock<std::mutex> lock(mu);cv.wait(lock,[&]{return done;});}
        r.thread_.join();assert(refused);assert(r.close_for_retirement());
    }
    {
        ForesightSwap s;setup(s);auto* slots=s.slot.get();s.card_mem={reinterpret_cast<void*>(17)};
        assert(s.close_for_retirement());unsigned long long out=91;
        assert(!s.take(0,0,out)&&out==91);s.hold(0,0);s.unhold(0,0);s.note_miss(nullptr,0,0);s.predict(nullptr,0,0);
        assert(s.slot.get()==slots&&s.card_mem[0]==reinterpret_cast<void*>(17)&&s.where[0]==-1&&s.copies==0);
        assert(s.close_for_retirement());
    }
    {
        // Source callback reentry must refuse before trying to relock smu_.
        ForesightSwap s;setup(s);ExpertSource src;bool refused=false;
        src.callback=[&]{refused=!s.close_for_retirement();};s.predict(&src,0,0);
        assert(refused&&src.blobs==0&&s.queue_.empty()&&s.copies==0);
        assert(s.close_for_retirement());
    }
    {
        // Detect this owner even below another active request frame.
        ForesightSwap a,b;setup(a);setup(b);ExpertSource as,bs;bool refused=false;
        as.callback=[&]{b.predict(&bs,0,0);};bs.callback=[&]{refused=!a.close_for_retirement();};
        a.predict(&as,0,0);assert(refused&&as.blobs==0&&a.copies==0);
        assert(a.close_for_retirement());assert(b.close_for_retirement());
    }
    {
        ForesightSwap s;setup(s);std::mutex gate;std::condition_variable cv;bool entered=false,proceed=false,closed=false;
        s.filler_=std::thread([&]{
            {std::unique_lock<std::mutex> lock(gate);entered=true;cv.notify_all();cv.wait(lock,[&]{return proceed;});}
            std::lock_guard<std::mutex> slots(s.smu_);std::lock_guard<std::mutex> worker(s.mu_);assert(s.stop_);
        });
        {std::unique_lock<std::mutex> lock(gate);cv.wait(lock,[&]{return entered;});}
        std::thread closer([&]{closed=s.close_for_retirement();});
        while(!s.retirement_started_.load())std::this_thread::yield();
        // First close owns lifecycle while joining; a contender must not block.
        while(true){std::lock_guard<std::mutex> lock(s.mu_);if(s.stop_)break;}
        assert(!s.close_for_retirement());
        {std::lock_guard<std::mutex> lock(gate);proceed=true;}cv.notify_all();closer.join();assert(closed);
    }
    {
        ForesightSwap s;setup(s);ExpertSource src;bool refused=false;std::string error;
        src.callback=[&]{refused=!s.set_strict_retirement(true,{0},error);};
        s.predict(&src,0,0);assert(refused&&!s.strict_retirement_);
        assert(s.close_for_retirement());
    }
    assert(device_calls==0); // close/closed mutators never synchronize/free GPU storage
}
