# TritonBench-G current-standard end-to-end review (2026-10-21)

183 selected task revisions were reviewed from their actual Python/kernel source
and input contract through the independent mathematical target, execution/IO
connection, actual FP program pair, numerical evidence and file roles.
Sixteen agents reviewed whole task groups; the root independently checked the
current main dependency closures and ran a fresh per-task Lean entry check.

**Whole-task completion remains 76 completed, 0 numerical deferrals, 107 pending.**
Among the 76 completed tasks, 74 use strict/structural evidence and 2 use an
already admitted FP32 DIV-MUL-RCP rule. No new numerical experiment or deferred
numerical obligation was introduced. Passing 183 entry checks is not 183 completed
program pairs. The [single task checklist](../../../tritonbench_g/task_progress.md)
records the individual remaining blockers.

The review made these substantive scope corrections:

- `isfinite_kernel`: the main Correct entry now instantiates the existing generic
  source theorem on Real inputs, whose independent finite predicate is true.
  The all-word IEEE exponent-classification theorem remains valid under its
  original name in `Misc/WordCorrect.lean`. The actual FP32/FP64 pair is unchanged.
- `matmul_leakyrelu`: the former Correct theorem covers the contracted candidate,
  whereas selected Python uses the original dot/add source. That theorem is
  preserved in `KernelOnly/Misc/CandidateCorrect.lean`; the main Correct file
  explicitly records the missing original-source Real IO bridge. The task was
  already pending and remains pending. This is program-proof work, not a
  numerical assumption.
- `int_scaled_matmul`: the independent signed8 dot/signed32-wrap mathematical
  theorem remains primary, with its required integer precision. The scaled
  FP32 round/mul/totalSigned formula is preserved as an ordinary auxiliary
  theorem in `Misc/NativeCorrect.lean`; an independent ideal-Real scaling/source
  bridge is still missing. This task was already pending and remains pending.

Small optional-result extractions were completed for square_matrix, masked_add,
rowwise_reduction and flash_attn. Current README descriptions, moved links,
the f8 integer multiplication transformation and add_value's strict ADD-COMMUTE
classification were corrected. The stale live numerical-summary claim of 183
completed tasks was corrected; pinned historical reports were not rewritten.
Selected Python code and admitted input domains were not expanded or restricted.

The l2_norm_bwd wrapper accepts metadata `[2**31,1]` and submits grid.x=2**31,
beyond the native single-axis launch bound. A metadata-only replay of the actual
wrapper AST records this without a large allocation or GPU run. It is separately
listed in [TODO](../../../TODO.md), distinct from deferred grid-product overflow;
it does not invalidate the mathematical per-row kernel theorem.

All 183 current primary + explicitly tracked auxiliary entry checks passed;
the primary source dependency closures contain zero Misc modules. FP headline
types and all main theorem axioms were checked. Changed/affected modules were
built locally, while applicable internal proof objects were reused. This is not
independent full-source replay or GPU validation. Baseline receipt gaps,
historical dependency hashes, the stopped optional MathCaller rebuild and old
optional reader collisions remain explicitly documented limitations.

The [per-task audit](audit.json) contains each actual main/pair declaration,
the reviewed formula/source connection, numerical scope, concrete blockers,
fixes, source hashes and the new check evidence. Full owner observations and
receipts remain referenced rather than copied into another historical tree.
