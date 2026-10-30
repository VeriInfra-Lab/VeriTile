# batched_vecmat_mult: kernel result with finite int64 addressing

The selected Python source is `batched_vecmat_mult_int64.py`. It repairs the
original signed-int32 address arithmetic by widening N, K and both program IDs
before multiplication. CUDA FP32 allocation, floating expressions, block/tuning
parameters and the single-launch grid are unchanged. The original Python stays
separate. Large-grid product overflow and segmented launch compatibility remain
outside this task.

## Target and source precision

For every valid output coordinate,

```
out[m,n] = sum(k = 0 .. K-1, A[m,k] * B[m,n,k]).
```

A has shape `[M,K]`, B `[M,N,K]`, and output `[M,N]`. Loads, multiplication,
reduction, live carry and stores are FP32. Real means the mathematical
interpretation of the source graph; it does not claim exact native rounding.
`K=0` writes zero to every output, and zero M or N gives an empty grid.

`Kernels.lean` contains the complete selected Python function body, including
both dimension casts and both program-ID INT64 casts. The original and existing
block-first candidate each retain every load, broadcast, reduction, transpose,
carry update and store. Pointer types, runtime dimension bindings and constexpr
block sizes are outside the pasted bodies. `candidate_kernels.py` is explicitly
authored from the pre-existing carry-addition candidate; the selected original
Python and its wrapper are unchanged.

The shared `SourceGraph.source` directly references this actual source.
Constructive signed-register prologue, loop and tail proofs establish its
unchanged independent FP32 operation graph, complete memory result and exterior
frame. The finite-int64 interpreter then refines this same source under the
existing integer-agreement conditions. The optional `Misc/HistoricalCorrect`
stream contract is explicitly bound to its historical cast-free source; current
Real, FP, grid, native and Tensor results use the selected source.

## Input checks and launch facts

All eight scalar arguments must be builtin Python integers and nonnegative.
Bool, integer subclasses, floats and other objects are rejected before any
allocation. `Contract.lean` retains the exact unchanged classification and
normalization to natural metadata.

Three CUDA FP32 allocations precede the original ordered remainder/assert
stage: K/BK, M/BM, N/BN. `KernelDomain.KernelContract` describes exactly the
successful stage: positive blocks and whole-block divisibility.
`checkKernel_iff` proves this characterization. It is not a new Python guard:
zero blocks still cause modulo exceptions, and allocation failures still
precede the assertions. Invalid resource/compiler tuning can fail setup.
The kernel theorem does not promise that all classified scalar calls launch.

`KernelBinding.Allocated` binds each actual core FP32 Tensor to its shape,
storage, physical element map and data pointer. Dense CUDA layout, absent
negative-view flag, aligned nonempty pointers and representable capacity are
ordinary facts of these internal allocations, not extra caller restrictions.
`Binding` records the common device and fresh output separation from both live
input backing storages. A and B need not be mutually disjoint.

The necessary int64 storage bounds are derived from these actual allocation
capacities. `LaunchABI` retains the existing arange/program-ID/warp facts; no
new i32-safe dimension cap or grid-product compatibility condition is added.

## Public IO and Tensor results

`IO.lean` explicitly declares A[M,K], B[M,N,K], output[M,N], their row-major
addresses, no scratch, and the complete floor-divided two-dimensional launch
schedule. Each invocation runs the complete selected source with fresh registers
and its actual program/grid IDs. The Structural `.real` channels carry FP32
operation expressions; they do not change the source's FP32 precision.

`Correctness.batched_vecmat_grid_real` uses the shared `ProgramIO.Observes`
judgment and the independent full-K formula `sum_k A[m,k] * B[m,n,k]`.
Its mathematical interpretation retains arbitrary input expressions and the
original arbitrary word-to-Real decoder. Whole M/N/K block coverage and
separate output/input regions remain explicit conditions.

`FPEquivalence.batched_vecmat_grid_native` is an ordinary auxiliary theorem in
`Misc/NativeCorrect.lean` giving a complete native observation
under the original reduction-tree and contraction model, also using the shared
IO judgment. It observes the independent source-order recurrence after graph
contraction through FP32 mul/add/fma. It preserves floor K-loop behavior without
adding K divisibility. Arbitrary expressions include all prior address-tagged
word inputs and mismatched-tag defaults. This is a source-to-native model
observation; no alternate kernel or floating rewrite is asserted.
`IOCorrect.native` additionally provides the addressed Word-input specialization.
`KernelProfile.Shape` still erases payloads before selecting a reduction tree.

`Proofs/KernelSafety` proves actual source access safety for the complete
prologue, all K-loop iterations and final unmasked store. `IOSafety` derives it
from the logical windows for every actual grid invocation. `IOSchedule` and
`IOResults` reuse the existing source execution and independent memory results;
`IOCorrect` connects successful execution, every output and the exact tagged
exterior frame. Empty dimensions and K=0 retain their original behavior.

The finite-int64 `IntegerAgreement` premise remains visible on the Real
specification and the retained ordinary native theorem. Its exact source/finite-executor equality, full raw outputs and
frames remain as ordinary theorems in `Proofs/KernelRealResults.lean` and
`Proofs/KernelNativeResults.lean`. Their original complete statements are
preserved, together with all four `Misc/TensorBridge` theorems: source/finite
execution, data-pointer/access binding, initial/final physical realization,
input-storage frames and Python-check correspondence.

Physical heap addresses count FP32 words. Actual byte pointers are proved to
be four times the word base; finite source offsets reach the same input words
and every output store has its complete four-byte window in bounds. Nonzero
storage offsets and arbitrary initial output contents remain supported. This
task does not claim an additional byte-codec refinement theorem.

The public files are independent, with operational proofs under `Proofs/`.
Their import closures do not depend on reader entries or TensorBridge. Old
wrapper/protocol results remain optional historical helpers. No Python source,
input guard, source precision, model boundary or experiment status changed in
this IO-format pass.

## Original defect and numerical classification

For M=32784, N=512, K=128 and blocks `(16,32,64)`, program `(2048,0)` loads
B offset `32768*512*128 = 2**31`, within the declared extent. The original
signed-int32 expression yields `-2**31`; the selected repair yields `2**31`.
That original source defect remains separately classified.

The current public graph theorem has no `Local.PrimitiveLaw` or arbitrary dot
oracle premise. The older historical theorem used that additional local-port
contract; it is not a numerical experiment TODO for this kernel-only result.
All raw FP32 words remain admitted. Primitive floating semantics, Tensor API
facts and correspondence to a selected compiler/native graph profile are
explicit model/platform boundaries. Arbitrary backend profiles are not
certified. No new floating transform or approximation requires deferred
numerical experiments here.

## Validation

`check_contract.py` checks exact original/repair AST correspondence, the
unchanged wrapper after its eight-scalar precheck, 40 pre-allocation rejections,
ordered allocation/modulo/assert failures, and 28 actual Triton CPU-interpreted
launches. Cases include multiple programs/K blocks, empty dimensions and K=0.
All Tensor windows use nonzero backing offsets. Both complete input storages
and output prefix/suffix sentinels are checked, as is the original overflow
counterexample. Small integer inputs make these execution cases exact.

The frozen delivery records task-local Lean receipts, specification and
standard-base axiom audits, source hashes and the CPU log. CPU interpretation
is not GPU/compiler reduction/FMA conformance. Native GPU execution and an
independent cold rebuild are reported separately; neither is claimed here.


Public-entry review (2026-10-21): `batched_vecmat_grid_native` is retained with its
original name as an ordinary theorem in `Misc/NativeCorrect.lean`.
`FPEquiv.lean` now proves `Equivalence.block_first_strict`, the complete actual
`original ≡[[]] blockFirst` pair. `Kernels.lean` retains the existing candidate as a complete raw body:
only `vecmat + reducedBlock` becomes
`reducedBlock + vecmat`. All loads, broadcasts, reduction trees, transpose,
iteration order, stores and floor-grid invocations remain explicit.

The logical input cells still admit arbitrary source expressions, including
compound expressions. Both output observers apply the existing explicit
contraction pass, evaluate its FP32 operations, and wrap the resulting word in
an expression leaf. This pure observation does not change input encoding or
binding. The proof establishes that a contracted carry starts at zero and
subsequently has add/FMA form, so it never competes as a product for contraction.
Both programs therefore retain the same product and FMA operands. The separate
addition case reuses strict all-word `FP.ScalarOpsLaws.add_comm`; no
fused/separate identity, finite-value guard or experimental rule is assumed.

Task-local `Proofs/Pair*.lean` helpers prove complete candidate execution and
access safety, then connect every actual current-memory grid invocation.
Both schedules succeed, all tagged observed outputs agree, and both raw
exterior frames hold. Input A/B may alias each other; the retained separation
of the output region is the only alias condition. Empty grids, K=0 and arbitrary
source-structure reduction profiles are retained. These are model certificates,
without a hardware-conformance claim.
