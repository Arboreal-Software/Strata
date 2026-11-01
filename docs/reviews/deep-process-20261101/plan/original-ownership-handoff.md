# Owner-authorized Strata implementation ownership

## Current owner request

Claire explicitly requested: “I would like you to direct it to work through all of the scoped out strata work. It's goal is to get to a point where all features are completed. It should maintain its own local todo list and follow the implementation plan.” The owner then stated that the coordinating session and Claire will focus on setting up Dax.

You, codex-strata, now own the scoped Strata engine feature implementation through completion and qualification. Set a persistent goal for this full scope if goal tools are available. Maintain your own local TODO.md (suggested: strata/.local/strata-implementation/TODO.md), with current step, dependencies, findings, evidence, blocked items and next actions. Do not stop at a plan, source build, or a single working configuration. Do not silently drop features when a prerequisite is harder than expected.

## Read first

All paths below are relative to LOCAL_WORKSPACE:

- dev-baseline/work-queue-humanized.md: conversational complete scope.
- dev-baseline/WORK-QUEUE.md: hierarchical index.
- dev-baseline/work-items/**/PLAN.md: separate workstream and work-item plans.
- dev-baseline/work-items/REVIEW-PROTOCOL.md: mandatory downstream plan-review process.
- dev-baseline/README.md, provenance.json and validation.json: completed build prerequisite and its limits.
- GitHub shadowndacorner/fig issues 24–31, especially 30 and detailed 25–29. Read complete issue artifacts, not just seed descriptions. Issues 32–38 concern the separate HyperQwen track and are not this implementation.
- Applicable AGENTS.md in any checkout before changes.

The local plans were extracted from the original research proposal. Reconcile their dependency summaries against their preserved detailed proposals; some summaries omit transitive/integration prerequisites. Their draft status means review and refine them with source evidence, not abandon them. Historical research-only restrictions quoted inside the source artifacts describe that research pass; the current owner instruction explicitly authorizes implementation. Existing safety/coordination boundaries still apply.

## Full scope

P00 target and compatibility; P01 complete hybrid state/identity/cuts; P02 safe cuts and asynchronous ownership; P03 per-device/host/pinned/disk reservations; P04 exact context edits and gap replay; P05 RAW transactional capture/restore; P06 disk persistence, receipts and crash recovery; P07 priorities and dependency inheritance; P08 bounded work/eligibility/server integration; P09 dual-GPU and concurrent-window transactions; P10 per-session expert accounting; P11 optional codec capability with RAW fallback; P12 bounded approximate suffix reuse; P13 live incremental repair; P14 idle saved-window maintenance; P15 qualification/migration/observability/enablement readiness.

Compression is an optional runtime capability, not permission to silently omit its implementation/evaluation from this assignment. If infeasible or unjustified, document evidence and surface the scope decision rather than claiming all features complete.

Owner addition: live repair urgency increases with the oldest unresolved region affected by suffix reuse. Preserve age across edits and ancestry. Distinguish age, earliest affected position and token debt. Explore explicit age/debt fallback under continuous foreground load; priority alone cannot guarantee convergence. Saved maintenance remains idle-only.

Scope is one engine/model instance with shared weights, on Defiant's 64 GB host RAM and two non-peered RTX3090s. Single/dual GPU and concurrency are required. Complete hybrid state includes recurrent and pending-token state, not only KV. The external context DAG owns its graph and window identities; distributed cluster/storage-graph architecture is excluded.

## Existing baseline and repository

Organization fork: https://github.com/Arboreal-Software/Strata
Isolated source: LOCAL_WORKSPACE/development, detached exact deployed engine commit 1aff6d14117b88171fcfcec8f2b55fdc47f2b165 (engine 0.1.50).
Terminal/design commit: 4b9e05a2a806a3fe24f7b2560d98626c37dade35 (v0.2.6).
Differences: dev-baseline/release-source-delta.patch. Choose/document the implementation base deliberately; do not mix pins silently. Preserve existing baseline evidence and use an implementation branch/isolated checkout. Build script deliberately rejects dirty/different source; create deliberate development tooling instead of weakening the baseline receipt.

CUDA13 builder and SM86 developer image exist. Ten selected CPU tests and GPU-free version/family/help checks passed. No GPU model correctness/performance qualification for that image. Developer image retains toolchain; it is not the live Terminal server image.

## Workflow and completion

Work through bounded testable checkpoints. Update individual plans and your todo as you go. A finding that changes a state contract, resource budget, topology, persistence or scheduling assumption triggers review of every affected dependent plan. Mark review-required and stop only dependent work; preserve old evidence with its original source/configuration. Parent items are incomplete until their child acceptance criteria are satisfied.

Completion requires actual feature implementation, appropriate failure/cancellation tests, source/configuration-linked runtime evidence, single/dual/concurrent qualification where specified, compatibility and rollback documentation, and an honest list of remaining limitations. Mock tests or a single-lane smoke test cannot qualify the full feature set. Keep deferred or failed cells visible; do not call the goal complete with mandatory features outstanding.

## Coordination and service boundaries

The parent session and owner now handle Dax (formerly Yeti), accessible at 10.0.0.131 / Tailscale 100.78.236.127. Do not modify Dax or take over voice-webui work. The voice follow-up group in the queue remains owned elsewhere.

Defiant's live Strata and voice sessions may be in use. Prepare source, tests, container builds and offline qualification autonomously. Before a disruptive GPU test, serving image change, or restart, inspect live activity and coordinate the concrete test window with the owner/coordinator. Never silently kill serving work or overwrite image-pin.json. Do not replay accepted voice input, rewrite journals, or assume an uncertain operation had no effect.

Reviews: owner preference is asynchronous/nonblocking ChatSwarm Extra High source review via Slack, with findings fixed when returned; integration review was requested via GPT-5.6 Pro. Exact routing was unresolved in prior work. Do not fabricate review completion or substitute expensive routes silently. Review waiting does not prevent independent implementation work.

## First actions

Acknowledge receipt, create your local todo and full-scope goal, inspect current source/instructions and baseline, reconcile P00/P01 and dependency links, then proceed with implementation checkpoints. Report the first concrete milestone and any coordination needs. Keep a restartable checkpoint before context compaction or stopping.
