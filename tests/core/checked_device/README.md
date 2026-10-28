# Checked verifier device scopes

`OnDevice(device, true)` checks selection and restoration without synchronizing.
Verifier instances configured for strict retirement use it, including stage
capture and worker prelaunch. Legacy one-argument scopes keep their old calls
and error handling. Unused strict verifiers can still release host wait flags
without attempting a device selection.

The snapshot preserves the tested standalone fixture layout. It compiles the
actual modified header and retirement helper with CUDA mocks, plus the complete
actual `release_gpu_waits` method. Eleven supervised modes check get/select and
restore failures, nested restoration, worker selection, invalid device,
partial-init refusal, unused cleanup, success and legacy behavior. Diagnostics
must match exactly; subprocess output and lifetime are bounded.

Validation: the unchanged snapshot passed both CTests (source freshness and all
eleven modes) with assertions, ASan and UBSan in a CPU-only container. The whole
attempt took 1.3455 seconds. No GPU, model or network access was used.

The root CMake registration and `check_repository.py` wrapper were added during
publication and have **not been executed**. The wrapper binds all copied
production files to the repository before building. To run after configuring an
opted-in public conversation-test build, build `checked_device_fixture`, then
run `ctest -R '^checked_device_' --output-on-failure`.

The fixture is based on PR38. Its inherited release-query error loop is not
repaired here; that is a separate change. These tests do not execute full
Verifier graph/callback paths or establish CUDA, HIP, SYCL or Metal compatibility.
No backend build or hardware test has been performed for this patch. The
SYCL-specific OnDevice and verifier sources are unchanged.

Release assertion policy: configure this public directory, not the historical
snapshot, for standalone use: `cmake -S tests/core/checked_device -B build/checked-device -DCMAKE_BUILD_TYPE=Release`.
The public target explicitly undefines NDEBUG and compiles a guard that refuses
any configuration still defining it. This also applies when the repository adds
the directory. Historical snapshot inputs remain unchanged. This follow-up has
not yet been compiled or executed.
