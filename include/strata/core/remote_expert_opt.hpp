#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExperts;
class ExpertSource;

// Remote Expert Decode Optimization v1 on existing helper caches. One instance per serial engine.
class RemoteExpertOpt {
public:
    RemoteExpertOpt() = default;
    ~RemoteExpertOpt();
    RemoteExpertOpt(const RemoteExpertOpt&) = delete;
    RemoteExpertOpt& operator=(const RemoteExpertOpt&) = delete;

    // Serialized lifecycle: configure before the first attach/init attempt.
    // Caller prepays topology metadata and supplies EVERY device that can read
    // mapped output, including the primary device. Helpers outlive this owner.
    bool set_strict_retirement(bool enabled, const std::vector<int>& consumers,
                               std::string& error);
    bool attach_checked(RemoteExperts& remote, std::string& error);
    // Caller has joined all host producers and prevents any future numerical
    // calls. This is not a routing/controller fence. Disabled mode refuses.
    // Strict uncertainty fail-stops before borrowed owners can be destroyed.
    bool close_after_callers_joined() noexcept;
    // Quiescent actual attachment observations; controller excludes attach/close
    // across validation and the admitted unit. No ownership is minted here.
    size_t attached_peer_count() const noexcept { return peers_.size(); }
    const RemoteExperts* attached_peer(size_t i) const noexcept {
        return i<peers_.size()?peers_[i].remote:nullptr;
    }

    void attach(RemoteExperts& remote); // before RemoteExperts::open
    bool init(std::string& err);        // after all helper caches are open
    bool owns(int64_t layer, int32_t expert) const;
    bool adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source);

    enum class AdaptResult { complete, deferred, failed };
    struct AdaptPlanEntry {
        uint32_t peer = 0;
        int32_t slot = -1, layer = -1, out = -1, in = -1;
        uint64_t bytes = 0;
    };
    // Caller supplies actual tracker/source authority. No callback is made under
    // a helper mutex. prepare_all must admit every transaction before effects.
    // Publication is AFTER actual cache.replace and checked copy completion.
    struct AdaptPublication {
        virtual ~AdaptPublication() = default;
        virtual bool prepare_all(const AdaptPlanEntry*, size_t) = 0;
        virtual bool begin_attempt(uint32_t peer) = 0;
        virtual bool publish_after_copies(uint32_t peer, const AdaptPlanEntry*, size_t) = 0;
        virtual void abandon_unattempted() noexcept = 0;
        virtual void fail_unknown(uint32_t peer) noexcept = 0;
    };
    class OwnedAdaptWorkspace {
    public:
        OwnedAdaptWorkspace();
        ~OwnedAdaptWorkspace();
        OwnedAdaptWorkspace(const OwnedAdaptWorkspace&) = delete;
        OwnedAdaptWorkspace& operator=(const OwnedAdaptWorkspace&) = delete;
        // Exact fixed-array element bytes + object sizes + metadata allowance.
        // This is a raw logical allocation envelope, not allocator/RSS accounting.
        // Caller reserves genuine host quota BEFORE prepare and retains it until
        // this workspace dies. This API does not authenticate a quota marker.
        static bool required_bytes(size_t rows, size_t experts, size_t swaps,
                                   uint64_t& bytes) noexcept;
        bool prepare(size_t rows, size_t experts, size_t swaps, std::string& error);
    private:
        friend class RemoteExpertOpt;
        struct Storage;
        std::unique_ptr<Storage> storage_;
    };
    // Serialized controller only; hooks/source/workspace/runtime must outlive
    // the call. All helpers must already be bound. Busy means skip this entire
    // tick, never fall back to legacy mutation. Caller still owns primary mapped
    // optimizer-output routing guards through full primary consumption.
    // Not enabled by this API: driver quota/source/tracker joins remain required.
    AdaptResult adapt_owned(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source, OwnedAdaptWorkspace&, AdaptPublication&, std::string& error);

    // Bracket the existing pool callback; its signature and dispatch remain unchanged.
    void begin(const float* weights, int token_begin, int tokens);
    void end() { weights_ = nullptr; }
    void copy_rows(float* dst, const float* src, int token_begin, int tokens,
                   const int32_t* hit_rows, const int32_t* count, void* stream) const;
    void combine(float* dst, float* scratch, int token_begin, int tokens,
                 const uint32_t* skip, uint32_t ring, void* stream) const;

private:
    friend class RemoteExperts;
    static size_t metadata_bytes();
    bool active() const { return weights_ != nullptr; }
    void prepare(const RemoteExperts& remote, void* metadata) const;
    bool reduce(RemoteExperts& remote, const void* metadata, std::string& err);
    void accumulate(const RemoteExperts& remote);

    bool strict_retirement_ = false, lifecycle_started_ = false;
    bool init_attempted_ = false, retirement_closed_ = false;
    std::vector<int> retirement_consumers_;
    int allocation_device_ = -1;
    int peer_allocation_devices_[3] = {-1, -1, -1};

    struct Peer { RemoteExperts* remote; float* sum = nullptr; };
    std::vector<Peer> peers_;
    float* h_sum_ = nullptr;
    float* m_sum_ = nullptr;
    int32_t* h_mask_ = nullptr;
    int32_t* m_mask_ = nullptr;
    const float* weights_ = nullptr;
    int token_begin_ = 0, tokens_ = 0;
};
} // namespace strata::core
