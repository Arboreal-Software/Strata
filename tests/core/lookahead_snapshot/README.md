# Full residency snapshot follow-up — prepared, UNRUN

Built by the sibling lookahead_drain standalone CMake fixture. This target links
real window_budget.cpp; the two inherited guard targets intentionally do NOT,
checking default link closure without optional budget implementation or GC tricks.
All use exact source freshness (class/method/worker) and the declared controlled
CV/source mocks. The original PR44 candidate remains immutable and independently
qualified; these updated combined fixtures have not run.

Six counted scenarios cover: original table mutated/freed while actual worker is
paused before reading; busy refusal keeps its snapshot; repeated accepted submit
without allocation; resident/null filtering; overflow quote output unchanged;
legacy empty startup; persistent allocation failure before worker/state publication;
real dedicated foreground lease transfer, geometry refusal, storage deallocation
before charge release; shared/wrong-size/empty/replaced lease refusal; and real
wrong-purpose/device/pinned resource refusals. Global allocation fault injection
is thread-local, avoiding spurious failures in the real worker's own allocations.
The delete observer checks the genuine lease remains alive when its array is freed.

These tests do not certify simultaneous mutation DURING submit, arbitrary forged
aliasing shared_ptr authority, downstream IO/foresight completion, or numerical GPU
behavior. Source/geometry and genuine ledger identity are trusted driver bindings.
No model, filesystem expert data or CUDA device is used.

Canonical admission follow-up SOURCE ONLY / UNRUN: seventh counted group uses two independent lifetime-preserving use_count1 reboxes of one genuine quota. Actual configure rejects before allocation/worker publication, preserves by-reference input and releases the original charge only when both wrappers die. Updated actual methods and precise notify-one/submit-seal counterfactuals remain sibling extracts. No claim about arbitrary forged owners or executing those controls.
