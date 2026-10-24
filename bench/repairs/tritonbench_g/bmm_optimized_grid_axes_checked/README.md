# BMM with live grid-axis checks

This independent sibling changes only `_validate_inputs` in the frozen
`bmm_optimized_repaired.py`. For a nonempty output, it additionally requires
`batch <= 65535` and `cdiv(N, 32) <= 65535` (equivalently `N <= 2097120`).
The latter is the exact intersection of the CUDA grid.y bound across the
unchanged configurations: TILE_N=32 occurs and is the smallest N tile.
`Contract.axes_exact` proves this intersection. There is no new grid-product,
shared-resource, segmented-launch, or device-specific guard.

The original package and its frozen counterexample/review packets remain
unchanged. Its two admitted witnesses—batch 65536, and N 8388481—are rejected
by this sibling. This is an intentional domain restriction, not correctness
on the original admitted domain. Existing signed-32-bit checks remain intact.
Batch/M/N zero retains the original empty return and bypasses only the new
check. K=0 with nonempty output still launches and receives both axis checks.

The complete kernel, twelve configuration/stage occurrences, heuristics,
contiguous calls, FP32 inputs/carry/output, and `allow_tf32=False` are unchanged.
An AST comparison verifies that every original validator statement is retained
and that the new conditional is the only added Python AST node.

## Proof connection

`Kernels.lean` contains one readable source quotation. Its eight constexpr mask
specializations omit dead `mask=None` assignments and use an unmasked memory
operation for those branches. Nontrivial mask shapes and broadcasts, grouping
and short groups, pointer updates, loop, zero fill, and store remain explicit.
Tuple PID assignment is split into two scalar statements.

`IO.lean` exposes shared `ProgramIO` with finite contiguous A/B/O channels,
their exact capacities and flat row-major windows, all channel masks, no private
scratch, and the complete three-axis launch schedule. `Shape` retains the
batch/M/N/K dimensions. Each CTA installs the original program/grid IDs and
starts with fresh registers. A/B/O are distinct regional allocations; the
optional Tensor preparation retains caller-input storage aliases.

The actual quotation executes through shared Structural semantics. The bounded
`Proofs/SourceGraph.lean` carrier records the ordered operands of a source dot;
the actual accumulator addition consumes that record with its live old carry.
`GraphExecution` and `GraphRun` prove equality to the original
`Proofs/Execution.lean` interpretation for every source branch, loop iteration
and arbitrary input graph/carry. No alternate source AST or final-value callback
is supplied. `GraphValues` and `GraphTransport` connect the fixed graph observer
to the original complete `Bridge.grid`, including the full observed cell memory.
This retains the inspected compiler-profile boundary; it introduces no numerical
law equating unfused addition with fused accumulation.

The prefix, loop invariant, body, termination, and epilogue proofs establish
successful execution from arbitrary initial registers, without a successful-run
premise. Operands and writes are connected to the existing BMM model before
reusing its grouping, coverage, arbitrary CTA order, Real sum, padded native
FMA chain, and memory proofs. The grid interpreter computes each CTA from the
memory left by preceding CTAs. `ReadGrid` reuses these execution results for
arbitrary typed input reads without imposing all-offset cell-tag assumptions.
`GraphSafety` and `GraphWholeSafety` prove the actual evaluated load/store
addresses are backed, using the source's masks, swizzle, loop states and finite
channel windows. `IOComplete` joins safety, successful execution, every logical
output and exact exterior cell preservation in the shared judgment.

`Proofs/Tensor.lean` binds actual `Torch.TensorMetadata` and ordinary storage
views to the caller model. Both inputs use one allocation-indexed physical
reader, preserving storage offsets, zero strides, and aliases. The existing
contiguous API materialization supplies the exact logical preparation. The
fresh returned Tensor has the source shape, contiguous strides, FP32 dtype,
CUDA device, resolved flag, and data pointer. Physical output readback comes
from source-store cells; every caller allocation and every byte outside the
output interval is preserved. Native words retain all four bytes and require
no finite-only, no-NaN, or algebraic FMA hypotheses.

`Correct.lean` exposes the sole Real specification `kernel_real`:
`Spec.Real ((realIO ...).Implements (realFormula ...))`. `Misc/NativeCorrect.lean`
retains `kernel_native` as an ordinary theorem: the actual source IO implements
an independent padded scalar FMA formula. These results retain any complete CTA
permutation, all original configurations and stage occurrences, empty outputs,
K=0 and every native word. The public statements contain no raw state, runner,
post-state or cell fields. The native observation retains the original pointwise
equality of scalar `fma.rn.f32`; no numerical premise or experiment is added.

`FPEquiv.lean` proves the complete `original ≡[[]] quotientRemainder` pair.
The integer-CSE candidate replaces `pid % num_CTA_per_group` with
`pid - (pid // num_CTA_per_group) * num_CTA_per_group`, exposing the quotient
already used by `group_id`. This gives the compiler a shared division
expression; no measured speedup is claimed. All floating operations and the
selected live-FMA interpretation remain identical. The exact Nat identity and
non-underflow bound hold even for a zero divisor.

Both sources are ordinary complete quotations for all eight constexpr mask
branches. `Proofs/SourcePreservation.lean` proves complete original interface
and AST equality to the archived prior source. `PairConnection.source_pair`
proves whole-state execution equality and actual access-safety equivalence;
`PairIO.equivalent` lifts these to any complete three-axis CTA permutation,
including empty outputs and K=0, equal tagged observations and both exterior
memory frames. No finite-value guard or numerical rule is needed.

The existing dot-first source was considered first. Under the current
asymmetric graph routing it would zero-seed the dot before a separate add,
whereas the selected original uses a live fused carry. Its earlier finite-site
proof therefore does not directly establish the selected all-word comparison.
It is not used as a new numerical premise. The native observation remains an
independent optional theorem.

`TensorBridge.checked_real` and `TensorBridge.checked_native` retain the prior
complete checked Tensor claims, including physical bytes, allocation and device
identities, and the existing live-axis bounds. The old raw statements are ordinary
`TensorBridge.kernel_real` and `TensorBridge.kernel_native` aliases backed by
`Misc/RawCompatibility.lean`. `LegacyIO.from_io` explicitly consumes the shared
IO completeness proof, transports its result to the original source grid, and
preserves its complete output/frame scope. Both checked Tensor results use this
connection. Lazy negation remains rejected by the original validator.

## Evidence and boundaries

Evidence lives in `.veritile-cache/kernel-evidence/bmm_optimized_grid_axes_checked`.
The metadata controls cover both axis endpoints, both frozen witnesses, K=0,
all three empty-output cases, and retention of the old empty-input index checks.
The CPU controls cover 43 wrapper trials: 40 actual source interpreter launches
and three empty returns. They cover all twelve config/stage occurrences, tails,
full K tiles, zero K, transposes, offsets, broadcasts, and aliases. Five rejection
classes pass.

Offline CUDA SM80 compilation uses the actual Triton signature binder,
Autotuner configs, and Heuristics functions. Twenty profiles cover all twelve
config/stage occurrences, all eight mask combinations, and live K=0. Requested
stages, warps, CTA count and fusion options are checked against compiler
metadata. The eighty retained TTIR/TTGIR/LLIR/PTX artifacts show scalar
`fma.rn.f32` and no MMA path; the inspected TTIR carries the live old accumulator
into the dot operation.

These controls are finite source/compiler observations. They are not a GPU
execution result or a universal compiler-conformance theorem. PyTorch metadata,
allocation, contiguous materialization, device context restoration, source-to-
CUDA lowering, and scalar ISA numerical interpretation remain stated external
implementation boundaries. They add no admission exclusions or result-equality
premises. No additional host or primitive implementation project is claimed.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.
The retained raw compatibility applications are also under `Misc/`; the main
readers do not import them merely to re-export their optional conclusions.
