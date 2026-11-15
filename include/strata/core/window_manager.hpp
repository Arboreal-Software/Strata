#pragma once
#include "strata/core/window_selection.hpp"
#include <optional>

namespace strata::core {
struct WindowManagerState;
struct WindowRegistrationId {
    uint64_t engine=0, manager=0, serial=0;
    bool operator==(const WindowRegistrationId&) const = default;
};
struct ManagedWindowToken {
    WindowRegistrationId registration;
    WindowSelectionToken selection;
    bool operator==(const ManagedWindowToken&) const = default;
};
struct WindowLaneToken {
    uint64_t engine=0, manager=0, publication=0;
    size_t lane=0;
    bool closed=true;
    std::optional<ManagedWindowToken> window;
    bool operator==(const WindowLaneToken&) const = default;
};
struct WindowLaneCompatibility {
    StateDigest model_profile{}, topology{};
    uint64_t runtime_class=0;
};
struct WindowManagerLimits {
    uint64_t engine_generation=0, manager_incarnation=0;
    size_t max_entries=0, max_identity_bytes=0;
    // Persisted high-water seeds; never reduce/reuse them within this incarnation.
    uint64_t last_serial=0, last_window_generation=0;
    std::vector<WindowLaneCompatibility> lanes;
};
enum class WindowManagerStatus {
    ok, stale, invalid, capacity, exhausted, duplicate, pending, busy,
    read_held, bound, incompatible, cancelled, unhealthy
};
struct ManagedWindowSnapshot {
    ManagedWindowToken token;
    WindowExecutionView execution;
    bool busy=false;
    std::optional<size_t> lane;
};
// A dropped pending ticket cancels registration. It does not own prepared GPU work.
class WindowRegistrationTicket {
public:
    WindowRegistrationTicket()=default;
    ~WindowRegistrationTicket();
    WindowRegistrationTicket(WindowRegistrationTicket&&) noexcept;
    WindowRegistrationTicket& operator=(WindowRegistrationTicket&&) noexcept;
    WindowRegistrationTicket(const WindowRegistrationTicket&)=delete;
    WindowRegistrationTicket& operator=(const WindowRegistrationTicket&)=delete;
    WindowRegistrationId id() const { return id_; }
private:
    friend class WindowManager;
    void reset() noexcept;
    WindowRegistrationId id_;
    std::weak_ptr<WindowManagerState> state_;
};
class ManagedWindowWork {
public:
    const SelectedWindowWork& selected() const { return work_; }
    const ManagedWindowToken& token() const { return token_; }
    WindowWork lease() const { return work_.lease(); }
private:
    friend class WindowManager;
    SelectedWindowWork work_;
    ManagedWindowToken token_;
    std::shared_ptr<const void> entry_;
};
// Internal routing mechanism, not an authentication or scheduling policy boundary.
// Driver supplies strictly increasing creation generations and compatible runtime
// classes. All setup must already have policy/budget admission. Synchronous borrow
// must be serialized with cancel/finish as required by SelectedWindowWork.
class WindowManager {
public:
    WindowManager(WindowManagerLimits, WindowBudget&);
    WindowManager(const WindowManager&)=delete;
    WindowManager& operator=(const WindowManager&)=delete;
    WindowManagerStatus reserve_registration(const WindowKey&, uint64_t runtime_class,
                                             WindowRegistrationTicket&);
    WindowManagerStatus publish_registration(WindowRegistrationTicket&, PreparedNativeWindow,
                                             ManagedWindowToken&);
    WindowManagerStatus cancel_registration(WindowRegistrationTicket&);
    // Concurrent cancellation uses the immutable id, not a shared mutable ticket.
    WindowManagerStatus cancel_registration(WindowRegistrationId);
    WindowManagerStatus entry_snapshot(WindowRegistrationId, ManagedWindowSnapshot&) const;
    WindowManagerStatus lane_snapshot(size_t, WindowLaneToken&) const;
    bool matches_prepared(const ManagedWindowToken&, const PreparedNativeWindow&) const;
    WindowManagerStatus request_park(const WindowLaneToken&);
    WindowManagerStatus switch_active(const WindowLaneToken&, const ManagedWindowToken&, WindowLaneToken&);
    WindowManagerStatus unbind(const WindowLaneToken&, WindowLaneToken&);
    WindowManagerStatus admit(const WindowLaneToken&, ManagedWindowWork&);
    WindowManagerStatus acquire_read(const ManagedWindowToken&, SelectedWindowRead&);
    WindowManagerStatus request_cancel(const ManagedWindowToken&);
    WindowManagerStatus retire(const ManagedWindowToken&);
    WindowManagerStatus replace_parked(const ManagedWindowToken&, const WindowRead&,
                                      const PreparedNativeWindow&, ManagedWindowToken&);
private:
    std::shared_ptr<WindowManagerState> state_;
};
} // namespace strata::core
