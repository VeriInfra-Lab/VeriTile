# ksoftmax live grid-y guard candidate

This separate candidate adds the missing live `Q <= 65535` CUDA grid-y guard to
both `softmax` and `softmax_backward`. The exact Python delta is two lines in the
shared `_extent` helper. Both wrappers already return on `numel() == 0` before
calling that helper, so their empty domain and pointer-query bypass are unchanged.
Existing `B < 2**31` checks already bound grid-x. No grid-product bound is added.
For example, live `(B,Q,K)=(65536,65535,1)` remains admitted despite its grid product
exceeding signed 32-bit range; product/runtime compatibility remains the separately
deferred scope identified by the root, rather than a new candidate input gate.

The full module after deleting those two lines is byte-for-byte identical to the
frozen predecessor. Both JIT bodies, arithmetic, causal/mask choices, 64-bit index
casts, FP32 types, launch arguments, autotune configurations and fusion options are
unchanged. The guard-only Python revision left the predecessor JITs unchanged. Historical
Lean and CPU receipts describe their recorded snapshots; the current Lean source
migration is described below. The original packet and independent launch defect
supplement remain historical evidence.

`Contract.lean` follows the actual checks and guard ordering, including original
Tensor/layout/device/rank/mask/extent/alias conditions. It proves admission iff the
original domain plus `0 < numel -> Q <= 65535`, exact empty-domain equivalence, and
positive, architecturally supported independent launch axes from admission. Flags
still require exact Python bools. No floating-value condition narrows admission.

`Kernels.lean` aliases the shared complete source quotations. `IO.lean` now
exposes the actual full B×Q launch schedule, FP32 source precision, padded
reduction width D, active K-lane output mask, causal reads and absent/QK/BK
mask windows. Shared addresses across rows are bound simultaneously. The
selected empty launch remains separate from the always-executed schedules
used by the existing reversal theorem.

`Correct.lean` uses the shared `ProgramIO.ImplementsOn` judgment for the
independent forward, ordinary-Real backward and extended log-backward
formulas. `FPEquiv.lean` has one primary `original ≡[[]] reversed` specification for
the two complete source row orders. The shared `ProgramIO.Implementation` view
retains actual invocation schedules and output observers, and its strict
structural certificate is the existing `ProgramIO.Equivalent` proof. `Misc/Native.lean` retains the auxiliary
`AddressedProgramIO.Implements` native graph result on all input words and the
empty-grid implementation result as ordinary theorems, with their original
`FPEquiv` declaration names. Empty grids still require only `B*Q*K=0`, with no
padded-depth positivity or coverage assumption. Input expression labels retain
their original region and address.
Both judgments require actual source access safety, successful execution,
complete typed outputs and the full exterior cell frame. No initial state,
runner, cell access or existential post-state appears in the public statements.

`Proofs/AccessSafety.lean` follows each actual branch and load/store mask.
`Proofs/Interface.lean` connects the shared schedule to the existing source
runner and reuses its complete row execution, value and frame proofs.
`Proofs/RealIO.lean` reuses the established mathematical identities;
`Proofs/NativeIO.lean` also proves the old raw-word binding and reference are
specializations of the new interface. Original raw statements remain in
`Proofs/RawReal.lean` and `Proofs/RawNative.lean`. The reader audit is in
`Misc/InterfaceChecks.lean`.

`Misc/TensorBridge.lean` imports the auxiliary native entry and retains every complete Tensor
pointer/output/byte/frame result and the preserved-input-byte application.
Its checked adapters still conclude supported live axes from the candidate
input contract, using the same existing source execution/value/frame basis.
Neither primary entry imports the bridge.

The inherited mathematical domains describe where the independent real formulas
are defined; they are not Python admission tests. The native endpoints retain all
FP32 patterns. Ordinary Tensor facts, selected compiler/ABI/native primitives and
validated metadata stability remain model boundaries. The current candidate has
zero numerical transformations and zero required numerical experiments. The
historical eight `LocalExact`/`NumericDomain` bridges remain optional primitive
model-conformance interfaces, as recorded in the frozen classification supplement.
No full-precision floating-point-equals-real claim is introduced.

The original candidate validation covered five Lean modules and source/guard boundary checks.
`check_candidate.py` verifies the exact byte and AST delta, then executes the actual
five wrapper/helper ASTs using metadata-only Tensor fixtures and launch capture:
176 boundary cases and eight checks that invalid flags/metadata still fail before
empty return. This is not kernel arithmetic, allocation, compiler or GPU evidence.
Existing CPU and offline-lowering evidence is retained and pinned without reruns.

## Earlier IO format validation

The earlier IO-format pass built its supporting modules, both primary entries,
the auxiliary native entry, Tensor bridge and task-root audit. It does not rerun frozen
historical evidence, compiler experiments or GPU tests. Its Python, Kernels and Contract pins describe that earlier snapshot. The original mathematical/native scopes,
selected compiler ABI/profile and all Tensor conclusions are retained.

The 2026-10-21 pair review preserves the full original row-order domain,
including K=0 execution in the schedule theorem and the independent selected
empty-launch helper. No floating operation, source precision, mask, observation,
or numerical rule changes. The previous IO judgment is also available as the
ordinary `reversed_schedule_io_equivalent` compatibility theorem in `Misc/Native.lean`.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.

## Complete Python function bodies

The source authority is `bench/repairs/tritonbench_g/ksoftmax_triton/Kernels.lean`.
Its `forward` and `backward` quotations preserve the selected Python functions
statement for statement, including docstrings, all six `.to(tl.int64)` calls,
optional mask tests, the backward `IS_FP16` branch, causal zero selection and
unmodified loads/stores. `jit-correspondence.json` maps both complete functions;
no hand-built AST or source macro stands in for either function body.

FP32 pointer types, signed INT64 scalar ABI, actual stride/K values and constexpr
flags/widths are caller metadata outside the quotations. `SourceParameters.lean`
contains the same absent/QK/BK `MaskMode` declarations previously shared by the
mathematics. `caller_mask_value` proves that the finite optional string supplied
to the raw body is exactly each admitted mode's original Python value. The
public `KnownMode` and Tensor contracts are unchanged. The selected FP32 host
passes `IS_FP16=False`; the pasted backward body retains the source branch.

The frontend specializes None/string tests from actual finite caller choices.
The complete source execution witnesses are in shared `Proofs/FP.lean`; natural
address identities are proved after executing the visible INT64 casts. Shared
`Proofs/RowValue.lean` connects actual stores to the independent ordered formulas.
A same-dtype FP32 store adds no extra numerical cast. The shared
`ksoftmax_triton/Proofs/PrecisionProfile.lean` proves preservation of the four
previous Native graph and Real row targets. The selected
`ksoftmax_triton_grid_y_checked/Proofs/PrecisionProfile.lean` separately freezes
the previous ordered IO formulas and proves five row/target/native-formula
equalities for all inputs, flags and reduction plans. Only the generic
`orderedRow` formula loses the historical extra store casts; memory views,
input domains and schedules are unchanged. None of these proofs assumes an
identity law for an arbitrary algebra.

The primary FP comparison still reverses the complete row schedule. Forward and
backward are two source functions, not the two sides of that comparison.
`Correct.lean`, `FPEquiv.lean`, `Contract.lean` and the original Python files retain
their public specifications and input domains. Execution, access safety, output,
source/byte frames and the actual native ABI connections remain proof obligations
of the complete programs.

The original benchmark's optional typed candidate witness uses the direct whole
interpreter-run equality for integer address multiplication. This avoids asking
a countermodel to satisfy unrelated certified numerical rules. Both original
and candidate failures, successful states and untouched storage are covered by
that structural equality.

For source correspondence and the selected entries, run:

```sh
python3 -m bench.jit_correspondence bench/repairs/tritonbench_g/ksoftmax_triton_grid_y_checked/jit-correspondence.json
python3 -m unittest bench.tests.test_ksoftmax_body
lake build bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Correct bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.FPEquiv bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Misc.TensorBridge bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.PrecisionProfile
```

Full canonical-module replay, current source hashes, trust checks and cumulative
build results belong to the corresponding review checkpoint. Earlier CPU and
compiler results above describe their recorded snapshots; no new GPU or
numerical experiment accompanies this source migration. All necessary Lean and
Python source files are tracked in Git; project build caches are derived artifacts.
