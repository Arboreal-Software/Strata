#pragma once
#include "strata/core/window_restore.hpp"
#include "strata/core/window_lease.hpp"

namespace strata::core {
struct WindowSelectionToken {
    uint64_t publication = 0;
    WindowKey key;
    StateCut cut;
    bool operator==(const WindowSelectionToken&) const = default;
};

// Internal synchronous borrow contract: callers serialize borrow with finish,
// unhealthy marking and replacement. Callbacks must not retain views/references,
// finish their lease, or replace selection while borrowing. Submitted async work
// must retain the matching lease/owner until joined and drained. Escaped raw
// pointers cannot be revoked; these APIs are not a security boundary.
class SelectedWindowWork {
public:
    const WindowSelectionToken& token() const { return token_; }
    // A copy shares completion state but cannot rebind this admission to another lease.
    WindowWork lease() const { return work_; }
    bool borrow(const std::function<void(const std::vector<NativeSessionView>&,
                                        const std::shared_ptr<NativeRuntimeAttachment>&)>&) const;
private:
    friend class WindowSelection;
    WindowSelectionToken token_;
    WindowWork work_;
    std::shared_ptr<WindowExecution> execution_;
    std::vector<NativeSessionView> stages_;
    std::weak_ptr<NativeRuntimeAttachment> runtime_;
};
class SelectedWindowRead {
public:
    const WindowSelectionToken& token() const { return token_; }
    const WindowRead& lease() const { return read_; }
    bool borrow(const std::function<void(const std::vector<NativeReadSessionView>&,
                                        const std::shared_ptr<const NativeRuntimeAttachment>&)>&) const;
private:
    friend class WindowSelection;
    WindowSelectionToken token_;
    WindowRead read_;
    std::vector<NativeReadSessionView> stages_;
    std::weak_ptr<const NativeRuntimeAttachment> runtime_;
};

// One selected gang. Scheduler admission and replacement go through this object;
// no independently cached writable SessionState may bypass its execution gate.
class WindowSelection {
public:
    explicit WindowSelection(PreparedNativeWindow initial);
    WindowSelectionToken token() const;
    bool matches_prepared(const WindowSelectionToken&, const PreparedNativeWindow&) const;
    bool admit(const WindowSelectionToken&, WindowWork&);
    bool admit(const WindowSelectionToken&, SelectedWindowWork&);
    bool quiesce(const WindowSelectionToken&);
    bool view(const WindowSelectionToken&, WindowExecutionView&) const;
    bool cancel(const WindowSelectionToken&);
    bool acquire_read(const WindowSelectionToken&, WindowRead&);
    bool acquire_read(const WindowSelectionToken&, SelectedWindowRead&);
    bool resume(const WindowSelectionToken&);
    bool replace(const WindowSelectionToken& expected, const WindowRead& old_read,
                 const PreparedNativeWindow& replacement);
private:
    bool matches(const WindowSelectionToken&) const;
    mutable std::mutex mutex_;
    uint64_t publication_ = 1;
    PreparedNativeWindow selected_;
    std::shared_ptr<WindowExecution> execution_;
};
} // namespace strata::core
