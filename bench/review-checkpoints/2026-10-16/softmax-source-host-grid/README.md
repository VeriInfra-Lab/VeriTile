# Softmax original real source host/grid composition

Nine additive modules introduce two public exact-real-erased execution
specifications for the original softmax forward/backward producer-consumer chain.
Every original kernel invocation executes on the preceding memory, including
arbitrary finite autotune trial sequences and the final launch. The proofs cover
the four complete grids, full logical frames, saved-output/gradient aliasing,
actual Torch primitive-produced strides and source dispatch/rejection order.

The primary admitted contract preserves cold/cache-hit fp64 fallback rejection.
The mathematical helper also has real-erased frontend-rejected instances, which
are not source-admitted native executions. Undefined backward padding remains
arbitrary. K=0 unsafe behavior, M/N zero errors and the unused tl.cidv helper
rejection are recorded without inventing successful guards. Rounded typed
arithmetic and finite scalar/pointer/view domains remain structural obligations.
No numerical assumptions or source acceptance increment are introduced.

Root independently replayed 960 host, 352 heuristic, seven boundary, 80 frontend,
252 autotune, 24 alias and 24 layout cases, plus the helper probe. All 42 producer
compilation attempts, including 12 failures, are preserved. The final benign
HostGridTuning simpa linter warning remains visible. Historical external object
hashes and raw IR for every profile were not retained by the producer; the root
review records that limitation and a full active integration build is required.

Both accepted source scopes remain separate from the 181 pending programs.
All previous public statements, 41 numerical rules and the original metadata
baseline are preserved. Integration results will be recorded after actual checks.
No GPU execution, official comparator success or whole-corpus CI is claimed.

## Observed integration results

The eight-target build passed (7,383 jobs), establishing active import resolution
for all nine new modules. Catalog generation and all 25 sharding tests passed.
The unchanged schema2 checker preserves both accepted source scopes after their
catalog-only review rebinds: **2 accepted / 183**, with **181 pending** and all
183 reviewed. Softmax remains partial progress; the 41 numerical rules are intact.

The actual global audit exited 1. Its only failures are scope_review (181 pending),
comparator_preflight and library_replay (official comparator unavailable). Every
other gate passed. Whole-corpus CI, official replay and GPU correctness are not
claimed. No push was retried after the prior authentication failure.

All 951 previous public records, 58 supplemental contracts, 180 unrelated scope
rows and the original metadata baseline remain unchanged. The new catalog has
953 public declarations and 60 supplemental contracts (54 execution, six protocol)
across 16 tasks. Old accepted scope source/coverage/numerical bindings are intact.
The initial integration preparation failed while looking for an optional previous
numerical-followup file; its exact script/log are retained and the corrected
preparation adds no numerical follow-up or queue changes.
