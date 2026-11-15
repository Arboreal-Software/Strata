#include <atomic>
#include <cassert>
#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>
struct Worker {
 int id; std::vector<int>* order; bool ok=true,throws=false;
 std::function<void()> hook;
 bool call(){order->push_back(id);if(hook)hook();if(throws)throw std::runtime_error("worker");return ok;}
 bool close_for_retirement(){return call();}
 bool stop_after_dispatch_callers_joined(){return call();}
 bool stop_background_after_callers_joined(){return call();}
};
struct Ple {std::shared_ptr<Worker> read_context;};
struct Session {Ple ple;};
struct Stage {std::shared_ptr<Session> session;};
struct Sessions {std::vector<Stage> data;const auto& stages()const{return data;}};
struct DriverWindowRuntime {
 std::atomic<unsigned> background_retirement{0};
 std::shared_ptr<Worker> lookahead,foresight,pool,source;
 Sessions sessions;
 bool close_background_after_callers_joined() noexcept;
};
