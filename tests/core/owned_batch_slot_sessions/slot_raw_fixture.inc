// Actual slot factory/RAW transaction binding and receipt consumer; only native
// layout admission and physical snapshot/drain boundaries are recording here.
namespace strata::core {
struct NativeWindowStage {int64_t layer_begin=0,layer_end=0;std::vector<int> qsa_formats;bool ple=false;};
}
namespace strata::program {
using namespace core;
enum class SuffixRawStatus {ok,invalid,budget_refused,poisoned};
struct SuffixRawResult {SuffixRawStatus status=SuffixRawStatus::invalid;std::string detail;std::shared_ptr<const void> retained_failure;explicit operator bool()const{return status==SuffixRawStatus::ok;}};
inline SuffixRawResult invalid(const char* why){return {SuffixRawStatus::invalid,why,{}};}
inline bool add(uint64_t& a,uint64_t b){if(b>UINT64_MAX-a)return false;a+=b;return true;}
inline bool product(uint64_t a,uint64_t b,uint64_t& value){if(a&&b>UINT64_MAX/a)return false;value=a*b;return true;}
inline bool present(const StateDigest& digest){return std::any_of(digest.begin(),digest.end(),[](auto byte){return byte!=0;});}
class SuffixRawTransaction {public:
    enum class Phase{idle,running,drained,capturing,restoring,poisoned};
    ModelGeometry geometry_;const ModelGeometry* const geometry_identity_;WindowKey identity_;NativeSessionAllocation allocation_;
    std::shared_ptr<const void> runtime_;std::vector<Verifier*> verifiers_;WindowBudget& budget_;
    uint64_t max_tokens_=0,max_bytes_=0,host_limit_=0;bool steering_enabled_=false,initialized_=false;
    std::thread::id controller_=std::this_thread::get_id();Phase phase_=Phase::idle;StateCut cut_{};
    std::vector<NativeBatchSlotInspection> slot_bindings_;std::shared_ptr<const NativeBatchSlotOccupancy> slot_occupancy_;
    std::vector<NativeWindowStage> initial_config_;std::shared_ptr<void> pending_;
    SuffixRawTransaction(const ModelGeometry&,const WindowKey&,NativeSessionAllocation,std::shared_ptr<const void>,std::vector<Verifier*>,WindowBudget&,uint64_t,uint64_t,uint64_t,bool);
    ~SuffixRawTransaction();
    struct HostCap;struct HostCredit;
    std::shared_ptr<HostCap> host_cap_;std::shared_ptr<HostCredit> setup_credit_;
    std::shared_ptr<const HostCredit> slot_host_root_credit_;
    bool resolve(std::vector<NativeWindowStage>&,std::string&)const;
    bool identity_matches(const WindowKey&)const;
    SuffixRawResult initialize();SuffixRawResult initialize_slot_shared_host(const SuffixRawTransaction&);
    SuffixRawResult initialize_resident_shared_host(const SuffixRawTransaction&);
    SuffixRawResult admit_host(uint64_t,bool,std::shared_ptr<HostCredit>&);
    SuffixRawResult admit_controller_host(uint64_t,std::shared_ptr<const void>&);
    SuffixRawResult begin_driver_work();SuffixRawResult poll_all();SuffixRawResult poison(const char*);
    SuffixRawResult complete_driver_work(bool,const StateCut&,std::span<const StateCut>);
};
ACTUAL_HOST_CAP;
ACTUAL_HOST_CREDIT;
ACTUAL_TX_INITIALIZE
ACTUAL_TX_SHARED_INITIALIZE
ACTUAL_TX_RESIDENT_SHARED_INITIALIZE
ACTUAL_TX_ADMIT_HOST
ACTUAL_TX_ADMIT_CONTROLLER
ACTUAL_TX_IDENTITY_MATCH
ACTUAL_TX_CONSTRUCTOR
ACTUAL_TX_DESTRUCTOR
ACTUAL_TX_RESOLVE
ACTUAL_TX_POISON
ACTUAL_TX_BEGIN
ACTUAL_TX_POLL
ACTUAL_TX_COMPLETE
class ServingRawSession {public:
    using Phase=serving_raw::Phase;
    ACTUAL_ALIGNMENT_KNOWN;
    std::unique_ptr<AlignmentKnown> alignment_known_;
    bool slot_controller_bound_=false;
    explicit ServingRawSession(std::shared_ptr<SuffixRawTransaction> tx):tx_(std::move(tx)){}
    std::shared_ptr<SuffixRawTransaction> tx_;std::shared_ptr<NativeBatchSlotOccupancy> slot_occupancy_;
    WindowKey slot_current_key_;
    std::vector<int32_t> slot_tokens_;uint64_t slot_last_group_=0;serving_raw::Lifecycle lifecycle_;
    static WindowKey serving_identity(const void*,const NativeSessionAllocation&);
    static std::shared_ptr<ServingRawSession> open(const ModelGeometry&,NativeSessionAllocation,std::shared_ptr<const void>,std::vector<Verifier*>,WindowBudget&,uint64_t,uint64_t,std::string&);
    static std::shared_ptr<ServingRawSession> open_owned_slot(const ModelGeometry&,NativeSessionAllocation,std::shared_ptr<const void>,std::vector<Verifier*>,std::vector<NativeBatchSlotInspection>,WindowBudget&,uint64_t,ServingRawSession&,std::string&);
    bool terminal()const noexcept;bool healthy()const noexcept;
    bool transfer_capture_reached=false;
    SuffixRawResult capture(const SnapshotDescriptor&,BundleHandle&){transfer_capture_reached=true;return invalid("recorded refusal at native capture boundary");}
    bool install_owned_slot_image(const BundleHandle&,const WindowKey&,bool,std::string&){std::terminate();}
    bool transfer_owned_slot_from(ServingRawSession&,const SnapshotDescriptor&,std::string&);

    bool queue_owned_slot_input(int32_t,std::string&);
    bool drain_owned_slot_group(const NativeBatchGroupCompletion&,std::FILE*,std::string&);
    bool retire_owned_slot(std::string&);
    bool open_request(std::string&);bool begin_work(std::string&);bool close_request(std::string&);
    bool request_open()const noexcept{return lifecycle_.request_open();}
    Phase phase()const noexcept;
    static Phase raw_phase(Phase phase){return phase;}
};
ACTUAL_SERVING_IDENTITY
ACTUAL_SERVING_FACTORY
ACTUAL_SERVING_TERMINAL
ACTUAL_SERVING_HEALTHY
ACTUAL_SLOT_FACTORY
ACTUAL_SLOT_QUEUE
ACTUAL_SLOT_CONSUME
ACTUAL_SLOT_RETIRE
ACTUAL_SERVING_PHASE
ACTUAL_SERVING_OPEN
ACTUAL_SERVING_BEGIN
ACTUAL_SERVING_CLOSE
ACTUAL_SLOT_TRANSFER
}
