# Quantization integer stage and mixed-output token softmax

Eleven modules add four partial execution contracts. The two quantization stages
now execute source-ordered fp32 multiplication, an explicit llrint-to-int64
callback and signed int8 store narrowing, with values and full memory frame.
Global covers the selected kernel; transpose still uses independent tile
coordinates rather than the original grouped pid map. The eight-bit readback
describes values in logical cells, not a checked physical byte-memory model.

Both token softmax variants now execute independent fp16/bf16/fp32 input/output
storage for all nine pairs. The complete fp32 row graph and destination cast
have successful execution, independent initial-state values and off-write frame.
Both public contracts retain complete erased-source equality.

Producer results include eleven successful final compilations, 26 named axiom
checks, one noncircularity check and seven module audits. Two selected quantization
signatures compiled offline to sm90; eighteen exact-original softmax frontend
routes generated connected SSA evidence. No GPU execution or native numerical
refinement is claimed. Root checked 681 frozen file hashes and read all sources.

All prior proofs, 938 catalog declarations, the original metadata baseline,
35 descriptive followups and 29 registered numerical rules remain unchanged.
Supplemental inventory becomes 45 execution and four protocol declarations
across 14 tasks. The four new statements have no exact numerical registration.
Host/grid composition and remaining source dtype routes are still required.

Whole-original-program acceptance remains 0/183, with all 183 pending. At this
integration base the scope ledger contains 85 confirmed gaps and 98 unreviewed
programs. Current integration outcomes will be recorded after actual checks.

## Observed integration validation

The eight-target integration build passed with 7,352 jobs. All 25 sharding
tests, catalog generation and both numerical collector checks passed. The
current catalog has 184 tasks, 942 public declarations, 3,560 task modules and
3,991 registered modules. The previous 938 declaration records and 45
supplemental declarations are unchanged, as are the other 179 scope rows.

The actual global audit exited 1. Its only failing gates were scope_review
(all 183 programs remain pending), comparator_preflight and library_replay
(the official comparator is unavailable). All other global gates passed.
No current comparator success, native execution or new numerical registration
is claimed. The original frozen metadata baseline remains byte-identical.

Two unsuccessful integration attempts are retained: initial cross-task source
bindings rejected by the catalog, and a ledger rebind started before catalog
generation finished. Neither altered existing proofs or accepted any task.
