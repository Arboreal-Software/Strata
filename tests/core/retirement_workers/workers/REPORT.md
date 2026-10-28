# Lookahead and Foresight stop-only retirement

Source-only implementation and fixtures; no compiler/test/GPU/model execution.

APIs:

- `bool RouterLookahead::close_for_retirement() noexcept`
- `bool ForesightSwap::close_for_retirement() noexcept`

Files: RouterLookahead class in expert_source.hpp, ForesightSwap class in
foresight_swap.hpp, and only their methods/helper definition in expert_source.cpp.
No driver, pool, doorbell, source, CMake or numerical kernel changes.

Close atomically seals admission, uses a try-locked lifecycle mutex to refuse a
competing/reentrant closer, and refuses joining its own worker. It signals stop
under the worker mutex, releases worker/slot mutexes, then joins. A successful
second close is true. Join exceptions permanently refuse subsequent close.
Contention/self-callback refusal leaves the owner terminally sealed but permits
an outside caller to finish closing later. None of these false returns allows
freeing the owner. The caller must keep a strong owner alive throughout close;
concurrent destruction is not supported.

No stream/event/memory destruction or device drain happens here. Queued but not
started work is abandoned in retained queue/storage. Work already popped by a
worker completes its existing host activity before join returns; its submitted
GPU copies can still be pending. Every storage allocation, event, stream and
raw source/FS target reference remains intact for the caller's next gang-wide
drain. Pointee lifetime remains the caller's responsibility; these raw references
do not invent shared ownership.

Router submit/start/FS replacement/statistics/depth mutations refuse after seal.
Foresight take/hold/unhold/predict/note_miss/init/configuration methods refuse;
take leaves its output pointer unchanged. Request checks the seal under smu_
and again after source callbacks. Its thread-local active-request stack detects
reentry through source pinned/blob even under nested requests on other owners,
refusing close before relocking smu_. Lifecycle start/configuration also use
try-locks so a worker callback cannot block on a closer joining that worker.

Public Foresight data/configuration fields and completed atomic remain directly
accessible to existing driver code: no claim is made that arbitrary direct
writes are intercepted. The driver must serialize configuration and prevent new
units. Existing startup-only lookahead configuration assumptions remain; this
patch does not make every historical accessor/configuration combination a
concurrent interface.

Destructors reuse stop/join and fail-stop if safe closure cannot be established,
rather than continuing into std::thread/member/storage destruction. On successful
close the existing Foresight strict/default GPU cleanup path is unchanged. The
new APIs do not stop ExpertPool or doorbell completion services. Main must finish
active windows before shutting down those services and must retain targets until
all producer joins and GPU drains succeed.

Prepared fixture:

`prepare.py` extracts exact Router class and actual close/submit/update/request
method bodies; copies the exact Foresight header; combines prefix.inc/cases.inc
into test.cpp. Prefix supplies CPU source and CUDA-event seams. Fixture destructors
invoke close only, so this is NOT a production GPU-destruction test. It exercises
empty/idempotent close, sealed mutation/output preservation, worker mutex released
before join, worker self-close refusal, source callback and nested callback
reentry, competing close refusal, retained backing pointer identity, zero GPU
calls. There is no real source/model IO. Thread gates are test-owned CPU objects.

Future granted command (not run):

```
c++ -std=c++20 -O0 -g0 -UNDEBUG -pthread -fsanitize=address,undefined test.cpp -o test
./test
```

source-pins.json records current full file hashes; actual-methods.inc preserves
reviewed method content independently of concurrent edits to other classes in the
same file. git diff --check passed for scoped files. Real backend compilation,
actual worker numerical behavior, OS join failure injection and driver integration
remain open. Main owns P02/P03/P08/P09/P15 propagation and retirement orchestration.

## Superseding lifecycle reentry repair (unrun)

Independent review correctly identified that recursive try_lock on a nonrecursive
std::mutex is undefined behavior. WorkerLifecycleScope now records an allocation-free
thread-local linked owner stack BEFORE every Router/Foresight lifecycle mutex
acquisition. start/init/configure/close refuse same-owner reentry before touching
the mutex. Foresight request callbacks also publish this frame, so a source
callback cannot reenter lifecycle configuration. Nested owners are searched, not
just the top frame. No actual virtual callbacks exist in current Foresight
init/set_strict body, but their full scopes are fenced for allocation/platform
reentry as well. Existing ActiveRequest retains its narrower smu_ guard.

New fixture uses exact actual RouterLookahead::start and a virtual mocked warms
callback invoking close: start refuses after seal with no worker constructed.
Also covers nested starts closing an outer owner, recursive same-owner start,
and Foresight source callback invoking actual set_strict_retirement. Numeric run
loop is a fixture stub; no claim of its behavior. Earlier worker-close fixtures
remain immutable; this worker-close-v2 directory supersedes their source pins.
No compilation or test execution occurred.
