# cross_entropy1

`Kernels.lean` contains all 12 forward and 15 backward statements from the
selected original Python functions, plus the two existing integer column-index
commutation candidates. The contents of `triton { ... }` are copied Python
function bodies: original casts, branches, masks and operation order remain.
Caller types, integer values, constexpr values and allocation-relative pointer
offsets are bound outside the bodies. `jit-correspondence.json` checks all four
complete source pairs, including their caller bindings.

`Correct.lean` exposes shared `ProgramIO.Implements`/`ImplementsOn` statements
for the unsplit forward formula, the split forward block contribution and the
backward gradient. The original conditions remain explicit: unsplit coverage,
valid or ignored labels, class start and total class count; split nonempty block;
and the no-smoothing parameter condition. Backward has no added row-bound or
positive-width condition. Real semantics still use `WithBot Real` internally for
masked negative infinity and observe ordinary Real outputs.

`IO.lean` names the canonical source, the complete logical logit row, signed
integer label, runtime smoothing/square-scale/total-classes register bindings,
row/block addresses and every output. Backward includes the saved LSE and the
upstream scalar, whose input mask is disabled on ignored labels. Gradient/input
aliasing remains supported. Forward keeps only the original LSE/logits and
LSE/loss separation requirements.

`Misc/NativeCorrect.lean` uses shared `ProgramIO.Observes`: arbitrary symbolic source inputs
are evaluated under the existing FP32 primitive and compiler/reduction model.
The independent ordered expressions live in `Proofs/OrderedReference.lean`; they never execute
the source or read interpreter memory/registers. Native results retain ignored
labels, all FP32 words, masked tails, compiler contraction, and runtime scalar
words. The exact original word-seeded entry/ABI statements are retained in
`Misc/RawNative.lean`. No new floating rewrite, numerical assumption or
experiment is introduced.

`Proofs/KernelSafety.lean` and `Proofs/KernelTailSafety.lean` prove safety of the
actual source trace, including the guarded signed-label gather and conditional
upstream load. `Proofs/SourceResults.lean` reuses the existing complete executions
to establish exact typed cells and the full exterior frame. The generic IO and
Real adapters compose these results; the former raw Real specifications remain
ordinary results in `Proofs/RawCorrect.lean`.

`Misc/TensorBridge.lean` is unchanged and retains the complete actual-Tensor Real and
native applications. Python, `Contract.lean`, original precision and ABI are
unchanged. `Kernels.lean` now directly quotes both complete sources and the
existing exact integer-index candidates. Runtime scalar types use external `CallSignature` declarations. The signed
`total_classes` call value retains its integer-to-floating conversion.
`Proofs/SourcePreservation.lean` retains the historical expansion only as a
comparison snapshot and checks the current complete statement inventories. Each statement preserves the established single-program
scope with explicit row/block IDs; it adds no wrapper reduction or full-grid claim.

## Implementation pairs (2026-10-21)

`FPEquiv.lean` compares the complete repaired forward and backward original
programs with their existing integer column-index commutation candidates.
`PairIO.lean` fixes the same runtime scalar arguments, regional IO, row/block
IDs, every output and native payload carrier. `Proofs/SourcePair.lean` proves
exact complete-state execution equality and source access-safety equivalence
using only signed integer multiplication commutativity.

`Proofs/PairEquivalence.lean` composes those facts with the established source
execution results. Both actual invocations succeed and are safe under the
unchanged IO bounds; all active output cells and both complete exterior frames
agree. Forward keeps the original positive block and LSE separation premises;
backward retains ignored-label upstream suppression and in-place aliasing.
There is no finite-FP restriction, floating rewrite, experimental assumption or
new numerical validation obligation. The original primitive/compiler models and source FP32 operation order remain.
The current forward loss store includes the destination conversion omitted by
the old expansion; `KernelNative.lossValue` and the independent ordered native
reference now include that conversion. Scalar floating branches also use the
current typed lowering. No equality with the old AST or floating graph, arbitrary
cast identity, or compiler-schedule invariance across different graphs is assumed.
The full current executions, safety, independent Real results and complete
original/candidate FP pairs are revalidated against these operations.

The independent `Correct.lean` formulas and complete input conditions remain.
Optional source-to-native results keep their full names and statements in
`Misc/NativeCorrect.lean`; the main proof does not import `Misc`.

Optional proof applications and reader audits are now under `Misc/`. Their
namespaces, full theorem statements, proof bodies and prior acceptance status
are preserved. `Correct.lean`, `FPEquiv.lean` and their required proof helpers
do not import these optional entries. This directory move adds no correctness
claim or numerical validation.

## Current validation

Source, proof replay, trust-audit and environment-limit evidence is recorded in
`bench/review-checkpoints/2026-10-27/native-cross-entropy1-function-bodies/`.
The public input domains, independent mathematical formulas and source/candidate
comparison scope are unchanged. No numerical law, input restriction or GPU
experiment is added. The repository-wide migration and human review are pending.
