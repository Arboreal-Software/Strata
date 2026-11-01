# Owned suffix-repair checkpoint

This is a recovery checkpoint and evidence index, not a qualified release. The source checkpoint is `75ef2747f615ea7c30b66818812c6e02dfd86079`. Required source fixtures and extraction runners are committed under `tests/`; operational GPU campaign tooling and full raw evidence remain local.

The index records original receipt hashes without copying credentials, live grants, model inputs, raw tensor captures or private runtime configuration. Hashes alone do not give a remote reviewer access to the evidence. An external evidence archive and relocation of the GPU fixture closure are still outstanding.

Current reported passes include the exact-history native target3/4 tests, ordinary authenticated concurrent HTTP, three ordinary cancellation cases and selected-repair cancellation. Their scopes differ: the native numerical fixtures use context512; they do not establish occupied32k capacity, deployment-context behavior, representative task quality or performance. Interrupted, invalid-fixture and unrun attempts remain in the index. The fresh retained-return attempt is running and has no result yet.

The deployment target remains393216 context with two lanes. Managed task-quality/performance tooling, broader lifecycle/fault coverage, target-profile qualification, installed launcher integration and actual deployment rollback remain incomplete. Broader state/persistence/statistics/codec/maintenance/long-context requirements also remain open. CUDA evidence does not qualify all touched HIP/SYCL/Metal paths. No deployment gates are promoted by this checkpoint.
