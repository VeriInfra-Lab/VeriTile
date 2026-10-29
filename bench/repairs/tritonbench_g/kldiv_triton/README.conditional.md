# kldiv_triton repair candidate — additive revision002

This is a separate, reviewable repair of the original production APIs. It is not
an acceptance or edit of the original task. The candidate source is
`kldiv_triton_repaired.py`; the original 36 task files remain unchanged.

Original Python SHA256:
`43e6fb718ccf938d5b90d785159bd44febe7f3bdd7e3151c22fc79f18988aedf`.
Candidate Python SHA256:
`a441bca1e20330ce95c74d247a55c3d844a7c40f6159a97237619dfd9feb44ae`.
The separate candidate omits the original import-time GPU test harness.
The complete original-to-candidate diff is retained in the evidence package.

## Behavior and mathematical target

Forward prediction values are log probabilities. For raw targets the independent
Real term is `0` at target zero, and otherwise `t * (log(t) - y)`. For log targets
it is `exp(t) * (t-y)`. `none` returns all logical terms. `sum` returns their full
sum, `mean` divides by `rows*cols`, and `batchmean` divides by `rows`.

The original per-chunk reduction/store loop is retained. Reduced output storage
has shape `(rows, ceil(cols/block))`, so every actual chunk store is backed.
Both inputs undergo ordinary coordinate-preserving `resolve_neg().contiguous()`.
`contiguous()` can return the original descriptor when rows are empty or the
column dimension is a singleton, even when column stride is not one.
`Index.UnitColumns` therefore requires `rows = 0 ∨ cols ≤ 1 ∨ colStride = 1`.
`UnitColumns.active` proves the kernel address equality only for active rows and
columns. The storage and chunk proofs use active-coordinate equality, never an
equality of inactive row-function values. Fresh outputs and explicit
contiguous-format clones retain their stronger unit-column-stride contract.
Raw-target zero terms and padded lanes are explicitly selected to zero. The
actual source block is `max(1,min(16384,next_power_of_2(cols)))`, including width
zero. Allocation and context use the actual input owner.

Backward snapshots target and cotangent before writes, computes a separate dense
work buffer, then copies into the actual supplied input under `no_grad`. Its
logical gradient is `-t` or `-exp(t)`. This API has no reduction parameter; callers
supply normalization through the cotangent. The exact gcd guard rejects distinct
logical destination elements sharing a physical word; it does not require a
dense destination. An actual scalar-one cotangent returns the supplied object.
Every other accepted cotangent returns the actual broadcast product, whose rank
can exceed two. A broadcast error can occur after the supplied input has already
been written; the failure proofs retain that prefix.

## Profile and domain

The selected production profile is FP32 prediction, target, destination, work,
cotangent and returned numerical values; CUDA compilation is pinned to installed
Triton 3.7.1 with explicit SM90 target (emitted `sm_90a`). There is no float64
production conversion and no GPU execution or conformance claim. Other dtypes
and precision-promotion profiles are not certified by these entries.

Logical matrices are arbitrary two-dimensional Torch views with ordinary
nonnegative stride metadata, offsets, full backing allocations and logical
negative-view flags. Read views may overlap. Destination overlap is handled by
the source guard. Empty axes, multiple chunks, tails, aliases between any caller
roles, singleton/zero strides, unused row gaps and partial trailing allocation
bytes are included. There is no fixture-size restriction or newly imposed
native-safe shape cap. Invalid rank/reduction lookup is an ordinary pre-dispatch
Python error with no tensor write; the modeled entry domain has rank two and
one of the four named modes. Shape/device mismatch and destination overlap are
explicit modeled errors.

Ordinary Torch/runtime operations have local coordinate, allocation, context,
stream, shape, copy and failure-frame contracts in `Proofs/Runtime.lean`.
They do not supply kernel outputs or successful source execution. Reduction
uses the actual Torch-selected `ReductionPlan`, including operand order and
padding, rather than an opaque whole-sum equality. Broadcast dimensions/index
maps are explicit. `copyInto` failures may affect destination words but preserve
their byte complement. Successful context exits restore their entry state even
on body failure; failed exits retain their actual failure state.

`Launch.Contract` records the ordinary signed-64 stride metadata and successful
native launch-ID boundary (`rows <= 2^31`). A setup rejection is an ordinary
error. This is a launcher boundary, not a user shape guard or numerical axiom.
For every active row, the column bound follows from actual four-byte backing.
Current offsets, inactive-lane offsets and terminating loop-index increments fit
signed64; every active pointer and all four bytes are proved backed. An unused
one-past output pointer is not dereferenced.

## Public proofs and supporting results

`Correct.lean` and `FPEquiv.lean` are independent public entries; neither imports
the other. Helpers never import public reader entries.

| Result | Checked theorem or definition |
|---|---|
| Whole Real forward, including all ordinary failure frames | `Correct.forward` |
| Whole Real backward, supplied-object writes and actual return | `Correct.backward` |
| Independent Real KL total and normalization | `Math.loss`, `Math.total`, `Math.reduced`, `RealHost.forward_target` |
| Whole FP32 raw-word and four-byte outcomes | `FPEquiv.forward`, `FPEquiv.backward` |
| Optional local-primitive reference equivalence | `FPEquiv.forward_reference`, `FPEquiv.backward_reference` |
| Constructed full typed loop execution and output readback | `Tile.sourceRun`, `Tile.source_element`, `Tile.source_chunk`, `Backward.sourceRun`, `Backward.source_cell` |
| Distinct physical storage and interpreter carriers | `Storage.Adapter`, `Storage.forwardValue`, `Storage.backwardValue` |
| Full host snapshot/result composition | `Host.runForward_success`, `Host.runBackward_success` |
| All ordinary failure-prefix memory frames | `Host.runForward_frame`, `Host.runBackward_complement` |
| Context outcome states and restoration | `Host.finallyDevice_state`, `Host.finallyGrad_state`, `Host.device_restored`, `Host.grad_restored` |
| Actual row execution at current memory, arbitrary valid row orders | `Grid.forward_rows`, `Grid.backward_rows` |
| All-grid readback and full byte complement | `Grid.scheduled_readback`, `Grid.scheduled_bytes` |
| Coordinate address binding and materialization separation | `Grid.matrix_store_address`, `Grid.prepared_separate` |
| Exact nonoverlap guard | `Layout.injective`, `Physical.matrix_unique`, `Physical.supplied_copyback` |
| All 15 source block profiles and fixed-width bounds | `Launch.block_profile`, `Launch.prepared_indices`, `Index.active_pointer` |

Native storage contains `BitVec 32`, never provenance-tagged interpreter nodes.
Loads embed words as untagged atoms, and stores project the result of a checked
source execution. Only source literal `-1` carries the tag used to model the
compiled XOR sign operation. In this pinned compiler profile log-target backward
uses `sub.f32` from positive zero. Forward subtraction and multiplication stay
separate. `Native.lossOrder` selects the exact frozen loss operand order: raw
target is difference-first at all blocks; log target is factor-first for blocks
1 through 256 and difference-first for blocks 512 through 16384. The unchanged
source AST is interpreted by `forwardAlgebra` / `forwardModel`, whose loss
multiply explicitly lowers source factor/difference arguments into that native
order. `Storage.nativeForward` carries the selected block and flag into the
whole forward proof. This is an explicit local compiler-profile interpretation,
not a floating-point commutativity premise or a reinterpretation of the Real
target. `loss_order_profiles` covers all 15 source blocks; `loss_order_observable`
shows an abstract noncommutative primitive distinguishes the two orders.
Backward retains `Storage.native` and its original literal-only XOR lowering. Exp exposes multiplication by word `0x3fb8aa3b` then `ex2.approx`.
The installed `tl.log` polynomial/FMA/FTZ intrinsic is a pinned local scalar port,
not silently replaced by Real.log or a guessed single lg2 instruction.

The native reduction graph keeps the actual local fold starting at slot zero,
ascending subsequent slots, descending warp butterflies and descending shared
warp-leader butterflies. Each shuffle addition keeps self as the first operand.
Host sum uses its separately supplied actual addition plan. Real and FP32 empty
mean conventions are deliberately distinct.

## Remaining boundaries

Structural execution, host orchestration, source loops, snapshots, broadcast
return, supplied writeback, grid readback, fixed-width active addressing and
full memory frames are proved. Numerical conformance remains explicitly deferred
through seven local raw-word ports in `Native.Obligations` (add, subtract,
multiply, division, installed log intrinsic, exp2 approximation, equality) and
four local Torch ports in `NativeHost.Obligations` (sum addition, product,
equality, division by the actual Python natural scalar). The latter includes
actual scalar conversion and zero/large denominator behavior. Triton division
is present in the generic algebra interface but is not used by this kernel.
No deferred predicate states a whole-output, whole-reduction, index, memory,
protocol or source-success property. Both sides of every reference obligation use the same selected native operand
order. No commutativity or finite-only premise is added. Raw nonfinite payloads and signed zeros
remain in the word graph; no blanket IEEE NaN-payload identity is asserted.
Ordinary library/allocator/launcher contracts and compiler-to-device conformance
are not re-proved here. No GPU device was queried or launched.

## Evidence and freeze

Revision002 is additive and writes only
`/root/.codex/visualizations/2026/10/16/01a14539-f72c-7c83-aa71-3a0014d6e60a/repair-kldiv_triton-revision002`.
The earlier `repair-kldiv_triton-001/final001` package and its failed independent
review remain intact. Their hashes and complete histories are referenced by the
new manifest. Python, Kernels.lean, and the independent Real targets are unchanged.

New evidence:

- `metadata-controls001`: 224 actual production-host-AST CPU Torch controls,
  with capture-only kernels. Includes shape (2,1) stride (1,7), shape (0,4)
  stride (19,7), zero widths, singleton zero strides, negative views, and
  noncontiguous views across both flags and all reductions. All active addresses
  match; 256 prepared-input instances violate the old unconditional stride law.
  These controls check actual preparation metadata and caller bytes, not GPU
  numerical execution.
- `native-loss-order001`: all 120 frozen forward PTX profiles and all 752 loss
  multiplication instructions. Source locations and register def-use identify
  difference/factor order, paired prediction/target slots, and unchanged exp
  scaling order. 84 profiles are difference-first and 36 factor-first. The
  original pinned compiler receipt and exact artifacts are reused by hash.
- `source-binding003`: refreshed exact normalized Python/Lean kernel-body AST
  equality, full host-effect mapping, final-source pins, and explicit revised
  metadata/native-model mapping.
- `attempts`: complete new private producer history, including the sandbox
  executable-discovery failure. All final owned Lean sources have successful
  producer builds and their axiom audits.

The unchanged-source evidence in final001 remains applicable and is referenced
explicitly: 297 candidate CPU kernel controls, 9 original counterexamples,
1,764 exact layout cases; 150 offline compiled profiles and 8,160 binder cases;
90 native reduction-order audits; 274 host failure-prefix controls; and 1,936
broadcast metadata cases. This revision does not rerun or consume any old
reader. No GPU device is queried or launched.

`final002/manifest.json` freezes the revised source, complete declared dependency
source/object closure, canonical external families, exact successful producers,
new evidence, and both old/new histories. The 36 original task files are checked
unchanged. No shared library, original proof, registration, progress, approval,
integration, commit or push is modified. Independent review and acceptance
remain with root.
