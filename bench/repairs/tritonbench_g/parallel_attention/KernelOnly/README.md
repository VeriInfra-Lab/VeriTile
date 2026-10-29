# Kernel-only parallel attention packet

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/Public.lean`, `Misc/PublicV2.lean`, `Misc/FPEquivV2.lean`.

Current source-pair completion (2026-10-21): `FPEquiv.fused_score_equiv` compares the actual complete original and fused candidate for forward or backward. Forward combines the diagonal score square and its immediately following mask into `tl.where(m_s, b_s * b_s, 0)`. DQ combines its diagonal dot and mask into one assignment. These remove intermediate same-name assignments, with no arithmetic reassociation or precision change. The complete backward parent calls the candidate DQ, retains the barrier and reuses the original DKV. No measured speedup is claimed.

`CandidateSyntax.lean` and `ShellSyntax.lean` contain ordinary direct quotations. `Fused.shell_body`, `fusedShell_body` and `shell_original` connect the complete readable backward shells to the retained DQ/barrier/DKV computation. The main pair uses the original `IOData.nativeJobs` runner and its per-stage native algebras. These stages model the existing compiler interpretation and do not assert separate physical launches. Exact stage run and safety equalities lift to the original complete grid, all active typed output cells and both exterior frames. The native descriptor retains its unrestricted shape/flag domain, every symbolic input and raw entry image; no finite-value guard or Real ShapeContract is added. `R=[]`; no numerical rewrite rule or experiment is needed. Existing optional native observations remain in `Misc/NativeCorrect.lean`.

This packet concerns the actual forward O/Z stores and the backward DQ/DK/DV
stores. DQ and DK are **partial value-shard scratch results**. DV is complete
for its stored value coordinate. Host shard sums, normalization, autograd,
allocation, compiler resource limits, and byte heaps are outside these claims.
The historical repair and original files are retained independently.

`Misc/Public.lean` is the review entry. `Correct.lean` exposes successful full-grid
Real source execution with independent O/Z and partial-gradient targets.
`Misc/NativeCorrect.lean` retains full-grid execution of the same shared quotations under
the staged native interpretation, all active output words, actual Tensor
logical observations and complete tagged-memory/physical-word frames.

## Source and mathematical path

`SourceSyntax.lean` and `QuerySyntax.lean` are readable shared Triton quotations.
The generic forward, DQ and DKV proofs derive complete execution, including
every store. `Combined.lean` executes the parent's DQ/barrier/DKV sequence.
`Grid.lean` enumerates the three actual independent program coordinates;
`GridComposition.lean` derives input preservation, unique ownership and the
whole-grid frame from the local executions. Success is a conclusion, never a
premise. Padded lanes do not appear in the store footprint.

`Reference.lean` defines the independent Real partial gradients. The scale is
the incoming ABI parameter σ throughout. It is not identified with an ideal
inverse square root. `Coverage.lean` supplies every logical O/DV coordinate,
every DQ/DK value-shard scratch coordinate, every Z row, and empty B/H/T grids.
Empty grids preserve the entire state without invoking any program.

## Native interpretation and compiler boundary

`NativeStage.lean` stages source expressions before scalar projection. Dots
retain their producer role, input provenance, scalar operands and live seed.
The fixed interpretation uses `sourcePlans`, the recognized local descriptor,
seeded scalar FMA chains, coefficient operand order/mask placement, and the
fused-square normalizer tree. It never assumes rounded multiplication plus
addition equals FMA. Floating input words do not select compiler routes.

`NativeReference.lean` defines the independent targets through pure array
recurrences and staged expression evaluation. The targets do not call the
source executor, source quotations or output stores. They are not a separate
formal equivalence to the historical `NativeGraph` module. The Lean theorems
establish the staged interpretation; applying it to a compiled binary requires
the actual artifact admission and the ordinary local routing/ABI contract in
`../checks/COMPILER-CONTRACT.md`. This packet does not prove the Python
recognizer, a compiler, or all unseen specializations correct.

The retained recognizer checks all scalar chains, their operands and seeds,
coefficient DAGs, normalizer DAGs, and compiler options. Its eight specialization
pairs are saved evidence, not a restriction on the shape-parametric Real
proofs and not an admission of all other shapes. The new staged transcription
must be reviewed against that local mapping; old receipts are not mislabeled
as checks of new Lean code. The no-scale public native interface explicitly
requires σ = 1. Scaled interfaces keep the actual ABI σ.

`PrimitiveLaws` contains fixed-arity scalar instruction agreements only. There
is no dot, completed reduction, target array, whole output, memory frame or
successful-run numerical hypothesis. Tensor inputs are read through actual
`VeriTile.Torch.FP32.Tensor` objects and API laws. Output observations use the
actual data pointer including storage offset, contiguous and resolved-view
metadata, and preserve exterior, unaligned and inactive physical words.

## Evidence and review

The packet's freeze manifest records source and build receipt hashes. The
task-local source guard pins the four repaired Python kernel ASTs and the
frozen Lean transcription files. This is a transcription drift guard, not a
claim that hashing proves semantic equivalence.

Existing `../validation.json`, `../cuda-lowering.json`,
`../lowering-binding.json`, and `../checks/mutation-guards.json` are reused:
11 CPU-interpreter cases, 10 empty cases, 12 rejected domains, four lazy-negative
controls, 16 actual-binder SM90 compilations, eight paired structural admissions,
and the saved compiler mutation controls. No GPU execution is claimed.
No numerical or compiler experiment was rerun to produce this packet.

All public Lean conclusions are checked with `#axiomsClean`; independent
targets have explicit `#specNonCircular` audits. The independent review should
inspect the concrete source-to-staged-expression mapping as well as the theorem
statements and saved build hashes before accepting the native boundary.
