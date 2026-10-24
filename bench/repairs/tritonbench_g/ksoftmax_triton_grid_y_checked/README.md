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
unchanged. The candidate does not modify the original selected source or its 45
completed Lean modules and receipts. The original packet and independent launch
defect supplement remain historical evidence; root owns canonical selection.

`Contract.lean` follows the actual checks and guard ordering, including original
Tensor/layout/device/rank/mask/extent/alias conditions. It proves admission iff the
original domain plus `0 < numel -> Q <= 65535`, exact empty-domain equivalence, and
positive, architecturally supported independent launch axes from admission. Flags
still require exact Python bools. No floating-value condition narrows admission.

`Kernels.lean` aliases the frozen complete source quotations. `IO.lean` now
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

## Current format validation

The IO-format pass builds the exact new supporting modules, both primary entries,
the auxiliary native entry, Tensor bridge and task-root audit. It does not rerun frozen
historical evidence, compiler experiments or GPU tests. Python, Kernels and
Contract pins remain unchanged. The original mathematical/native scopes,
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

## Source-only organization (2026-10-22)

The shared repair `bench/repairs/tritonbench_g/ksoftmax_triton/Kernels.lean` now
contains the complete ordinary selected FP32 forward and backward quotations.
The selected `Kernels.lean` transparently reuses that public source authority,
which no longer imports its actual source from proof helpers. Complete-function
`rfl` checks against the pre-cleanup definitions preserve every flag, Option
String value, mask, cast, stride and output. The historical generic source API,
other precision profiles and 216-profile lexical checker remain unchanged.
The existing main pair, mathematical/native domains, schedules, safety, frames,
proof counts and numerical obligations are preserved.
