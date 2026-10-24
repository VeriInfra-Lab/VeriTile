# Optional proof layout (2026-10-21)

All 183 selected TritonBench-G revisions now separate their main entries from
optional results. `Correct.lean` states independent Real correctness;
`FPEquiv.lean` states an actual two-program FP equivalence when one is proved.
Helpers needed by those entries remain in `Proofs/`. Optional native results,
Tensor/host applications, extra candidate applications and audits are under
`Misc/`, including `Misc/NativeCorrect.lean` and `Misc/TensorBridge.lean`.
The source import closure of every selected primary entry contains no `Misc/`
module. Shared contract/IO definitions remain available without importing an
optional application. Declaration namespaces remain stable when files move.

706 files were relocated. All previously tracked auxiliary files have
new `Misc/` paths. The former attention_llama `PublicAudit.lean` and
matmul_dequant_int4 `Whole.lean` aggregates are now optional entries.
Other optional files are listed as `misc_entries` separately
from the tracked joint-import set.

The 183 fresh per-task primary-plus-tracked-auxiliary checks passed, including
actual pair-headline types, unchanged pair declaration names and axiom audits.
Owners also built moved/changed current modules in their approved environments.
Detailed preservation, shared-definition extractions and minimal connection
adaptations are recorded in the sixteen group reports linked by the audit.
Some historical optional readers have duplicate declarations or cached generated
equation collisions; they are kept separate rather than made a new acceptance
requirement. This pass does not claim that every historical reader can be
jointly imported.
The independent optional `rms_matmul_rbe_scalar_domain/Misc/MathCaller.lean` build was stopped
after a long run; it has no source consumers and is outside the tracked import
set. Its source/audits are preserved, and no successful relocated build is
claimed. Group 09 records the limitation; it is not a numerical-experiment
deferral and does not affect current main-entry checks.

Completion remains **76 completed / 107 pending**: 36 retained actual pairs,
40 repaired actual pairs, 49 missing proof connections, 57 missing targets and
one incompatible pair. Directory moves add no FP32/native or host correctness
requirement, no numerical rule and no experiment. Independent full-source
replay and GPU validation were not run.

- [Per-task layout and current validation](audit.json)
- [Old-to-new path map](path-map.json)
- [Pinned prior semantic review](../fpequiv-pairs/README.md)
- [Current task checklist](../../../tritonbench_g/task_progress.md)

Prior semantic and numerical reports retain their original paths and hashes.
The relocation map locates their files; it does not refresh their experimental
or proof-replay claims.
