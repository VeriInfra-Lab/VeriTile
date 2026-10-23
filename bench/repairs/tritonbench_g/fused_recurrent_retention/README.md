# Fused recurrent retention repair

The retained repair makes backward consume the optional final-state cotangent
and use the actual strides of do and that cotangent. Forward arithmetic, tile
choices, public H0 detach, and host sums are unchanged. The original Python is
unchanged. Importing the candidate runs no fixture.

## Executable admitted domain

The wrapper requires positive B/H/T/DK/DV and FP32 ordinary strided CUDA tensors
on one device. Both launches enter `torch.cuda.device(q.device)` and restore the
caller device afterward. q/k/v and optional H0 must have canonical contiguous strides,
including singleton axes. BK=min(DK,32) and BV=min(DV,32) must be powers of two;
DK/DV at least 32 may have arbitrary tails. Caller read-only aliases are allowed.
Actual do and optional final-state cotangent may be nonnegative strided or
expanded views, with active indices inside their real backing allocations.
The output flag must be a Python bool. A final cotangent requires a returned
final state. Higher-order backward rejects; repeated first-order backward is
supported subject to ordinary saved-tensor version checks.

All checks run before the corresponding source launch. A symbolic expression
checker follows the source's signed32 scalar/index expression trees and every
formed pointer, including masked lanes and the final forward/reverse pointer
movements. It proves recursive integer safety, grid limits, signed64 byte
displacements, alignment, and absolute unsigned64 pointer windows. The same
window checks run on actual new scratch/final-state allocations before launch.
The checker does not enumerate the launch grid. Tensor observations bind to the
actual object, storage, offset, capacity, dimensions, strides, dtype and device.

T=0 rejects explicitly. Empty tensors expose null launch pointers, while the
retained backward header forms negative reverse pointers even with zero loop
iterations. This restriction establishes the formed-pointer contract. No
nonfinite-data exclusion is imposed on otherwise admitted FP32 tensors.

## Whole result and independent targets

For each head, beta=1-2^(-5-head), scale=DK^(-1/2), and detached supplied-or-zero
H0, the independent Real recurrence is:

```
H_(t+1) = beta * H_t + k_t v_t^T
o_t = scale * q_t^T H_(t+1)
```

It returns O[B,H,T,DV] and optional HT[B,H,DK,DV]. The independent dense adjoint
starts from the actual optional HT cotangent, adds each output's contribution,
then decays while traversing backward. The target includes dQ/dK/dV and the
final-only contribution. Backward returns exactly `(dq, dk, dv, None, None)`;
the public initial state remains detached. Caller inputs are unchanged.

`Kernels.lean`, `AddressSource.lean`, `AddressCheck.lean`, `APISource.lean`, and
`Contract.lean` contain source/metadata declarations without proof-helper
imports. Proof helpers consume the same declarations and do not import public
readers. Historical public proof families remain available through helper
modules; the original readers are thin imports of unchanged declarations.

`Proofs/Contract.lean` derives ForwardScope and BackwardScope from executable
checks. These scopes are not premises of the guarded public calls.
`Proofs/Host.lean` binds actual allocator object identities separately from
storage identities, returns the actual sum/final-state handles, and keeps the
saved q/k/v/detached-H0 context. Autograd's returned requires-grad metadata is
an ordinary observed handle property. `TensorBridge.checked_quoted_returned`
covers forward and a subsequently admitted backward with the returned context.
`TensorBridge.checked_quoted_fp_returned` proves complete source-word equality and
the returned objects/gradient tuple. Guard rejection retains the pre-call
memory, including a later backward rejection.

`Misc/TensorBridge.lean` retains the optional `TensorBridge.checked_returned` and
`TensorBridge.checked_fp_returned` checked-host applications, plus the quoted
and host byte/frame corollaries under `TensorBridge`. The Real and auxiliary native forward/backward entries use only raw kernel
geometry, stride arrays, arithmetic plans and initial memory. Their complete
Tensor applications remain in the bridge; neither main imports it.

`Proofs/CompleteFrame.lean` proves the complement of the exact source write
ranges is unchanged, including every unrelated allocation and padding inside
output allocations. Byte observations preserve all four raw bytes per FP32
word. Source scratch and returned reductions remain separate actual stores.
The public frame theorems also apply to rejected calls.

## Exact FP boundary

The native model preserves raw source words, including NaN payload/sign and
signaling bits; it inserts no NaN normalization. The source graph retains
`head * 1.0` and the operand order `+0 + seed`. A separately defined software
FP target follows the selected reductions, contractions and host sum trees.
The Real target uses exact reciprocal square root and exact powers of two;
the FP host-scale operation instead models Python binary64 power followed by
FP32 launch conversion. No FP-equals-Real assertion is made.

Seven fixed-arity native numerical correspondences remain conditional: add,
subtract, multiply, FMA, unsigned32-to-FP32 cast, exp2-FTZ, and host scale.
The exp2/host-scale models are fixed before input values exist. There is no
whole-dot, reduction, kernel or result equality premise. These universal scalar
leaves are not established by finite tests.

Ordinary runtime boundaries are metadata/byte observations, allocation
freshness and returned handles, saved-tensor versions, source load/store and
stream binding, and compiler/runtime-selected value-independent instruction
plans. The selected carry graph contracts `beta*old + key*value` as
`fma(beta,old,mul(key,value))`, and the reverse update as `fma(q*scale,do,old)`;
this operand routing must match the observed compiler graph, as must dot and
host-sum plans. The proof does not assert a CUDA/compiler refinement or cover
allocation exceptions, aborted submissions, or later primitive failures as
completed results. Source launches are modeled after successful completion.

## Validation

All 42 repair Lean modules compile. The closure includes public axiom and
noncircularity audits and executable
nonempty admission witnesses, including expanded final cotangents. CPU replay
executes the actual unchanged candidate kernel ASTs over physical tensor
storage. It passes 27 rejection controls and 6 complete admissions, each with a
repeated backward, across 108 source-program executions. Cases include 33/65
tails, true read-only q/k/v aliases, strided/expanded do and final cotangents,
H0 detach, and a nonzero final-only dK/dV. An independent differentiable dense
recurrence checks outputs and all gradients within floating tolerance. Caller
storage and all allocation canaries are checked exactly.

The replay is finite CPU evidence, not native numerical conformance. No native
GPU run or compiler conformance measurement was performed.

### CUDA launch-device observation

The installed `triton.runtime.jit.JITFunction.run` obtains
`driver.active.get_current_device()` and its current stream, selects
`self.device_caches[device]`, then calls `kernel.run` on that stream. Equal input
devices alone do not select this launch device. The wrapper therefore enters
q's device context around each source launch. The formal `current` parameter
is the effective device inside that context. A CPU control runs the expanded
cotangent case on logical cuda:1 while the caller starts on cuda:0, checks each
launch observes cuda:1, and checks the context restores cuda:0 afterward.

## Public IO and Tensor applications

`Correct.lean` exposes the complete forward and backward grids through the
shared `ProgramIO.Observes` judgment. `retentionIO` lists all six logical input
channels, five possible output channels, source invocations, scalar scale ABI,
three grid axes, windows, strides and optional-channel masks. There is no
private scratch. The mathematical `formula` is a finite partial sum over
independent dense prefix/suffix formulas. These are the actual kernel partial
outputs before the separate library reductions, plus the optional final state.

The inputs retain the original arbitrary `Staged.Node` carrier. The pure output
observer takes `Node.value`; it does not restrict inputs to leaves or alter
execution. `Misc/NativeCorrect.lean` retains the ordinary forward/backward native
observations with the original
FP32 model, exact FMA/reduction topology and seven existing primitive scalar
laws. It does not assert an optimization or native-equals-Real equality, and
introduces no numerical assumption or new experimental obligation.
`FPEquiv.lean` compares the complete original and compare-right source grids below.

`Proofs/SourceSafety*` proves actual prefix, branch and loop accesses safe;
`IOGeometry`, `IOBackwardBounds`, and `IOSchedule` derive complete-grid safety
from the shared IO bounds. `IOInputs` binds only valid, enabled logical inputs.
`LogicalCongruence` connects these inputs to independent recurrences without
requiring unused-memory equality. `PublicResult` reuses the existing
`Quoted.grid_execution_memory` theorem for successful complete execution,
exact typed output observations and the entire exterior frame. `MathResult`
connects those observations to the independent mathematical partial sums.

The previous stronger raw execution statements remain ordinary theorems in
`Proofs/KernelRealResult.lean` and `Proofs/KernelNativeResult.lean`.
`KernelInterface.layout_correspondence` retains exact source layout transport.
`TensorBridge` preserves all earlier Tensor results, including library
reductions, final cotangents, detached initial state, byte observations and
frames, using the same complete quotation execution foundation.

The format revision is checked by targeted Lean builds of all new helpers,
both public entries, TensorBridge and Reader, plus public-surface and axiom
checks. Historical CPU replay and GPU/numerical status above are unchanged;
no replay or numerical experiment is claimed by this format revision.

### Complete implementation equivalence (2026-10-21)

`Kernels.lean` now contains the four complete ordinary forward/backward `triton`
quotations. The old macro templates and placeholder-scale statement removal are
replaced by readable source and an explicit `tl.param scale : tl.float32`; the
actual scalar argument remains bound by `PublicData.job`. The four full source
functions are definitionally equal to their preserved historical versions in
`Misc/SourcePreservation.lean`.

The existing compare-right candidate changes only the two key-mask statements
from `index < DK` to `DK > index`. The complete candidate Python module is
`fused_recurrent_retention_compare_right.py`. The whole-module AST comparison
checks that these are its only changes. This reuses the historical comparison
orientation candidate; no performance improvement is claimed.

`compare_right_equiv` states `Pair.original ... ≡[[]] Pair.compareRight ...`
for either original mode and its complete three-axis grid. `Proofs/Pair.lean`
proves the exact local natural comparison equality, full kernel state equality
and safety, then every invocation and complete schedule. Original success,
all active typed output cells and the full exterior frame transfer to the
candidate. Both original forward and backward precision, loop order, runtime
scale, arbitrary incoming/final-gradient strides, reduction plans, seed/final
flags and arbitrary `Staged.Node` input tags remain admitted. No finite-value
restriction, floating numerical rewrite, experimental rule or deferred
numerical validation is introduced. Host reductions remain in their existing
optional scope, while the independent Real main theorems retain the complete
kernel partial-output scope stated above.
