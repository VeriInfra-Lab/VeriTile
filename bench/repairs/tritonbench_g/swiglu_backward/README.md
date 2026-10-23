# Checked FP32 SwiGLU backward kernel

The current public entries verify the complete selected JIT kernel grid. They
cover every legal order, all six `BLOCK_N` values (32 through 1024), both
`RECOMPUTE_OUTPUT` branches, tail masks, strided rows, paired chunks, and empty
batches. The wrapper is the boundary for allocation/reshape receipts and the
executable admission checks. The JIT body is unchanged from the original
benchmark. No extra copies or arithmetic changes were introduced.

`Kernels.lean` contains complete ordinary FP32 `triton` quotations for the
selected original and its existing index candidate.
`KernelProof.selected_quotation` retains exact identity with the canonical
source selection. X, Y, DOUT, DX, DY, and active OUT have FP32 storage;
the three explicit `.to(tl.float32)` loads and native arithmetic order remain.

The independent real specification uses

```
s(x) = 1 / (1 + exp(-x))
forward(x,y) = x*s(x)*y
DX(x,y,d) = d*y*s(x)*(1+x*(1-s(x)))
DY(x,d) = d*x*s(x)
```

The original mathematical helpers prove positivity of the denominator and both
partial derivatives of `forward`, then identify the supplied-upstream VJP.
`swiglu_backward_kernel_correct` proves successful execution and every effective
output against these formulas, with the complete regional frame. The retained
TensorBridge theorem also supplies the exterior aligned physical-word frame. This is the Real interpretation; it does not claim
that rounded FP32 output equals an exact real derivative.

`swiglu_backward_kernel_fp32` proves successful execution and equality to a
separately defined ordered native graph on all raw 32-bit input words. With
`x`, `y`, and `d` denoting the native casts of the three loaded words, its graph is

```
s = sigmoid(x)
DX = mul(mul(mul(s, fma(sub(1,s), x, 1)), y), d)
DY = mul(mul(s,x), d)
OUT = mul(mul(s,x), y)       # only when recomputing
```

The ternary FMA is retained; there is no split-FMA identity, reassociation,
multiplication commutation theorem, exact-sigmoid approximation claim, or
finite-only value guard. Operand order also determines which NaN behavior the
selected primitive model supplies. The source-graph interpreter and this native
reference are separate definitions, and a non-circularity audit checks that the
reference does not depend on the interpreter or kernel execution.

The retained TensorBridge native theorem additionally proves all four concrete little-endian bytes
of every output and the complete exterior byte frame. Its entry-byte assumption
is only ordinary FP32 load/byte decoding at actual input pointers. The theorem
uses that binding to recover the original logical Tensor input bytes. The
physical writeback takes values from the successfully executed DSL state;
it never receives an expected output as a premise. Row gaps, masked lanes,
inactive OUT, unrelated storage, and other devices retain their entry bytes.

The `Storage` receipt binds bounded allocation cells to one physical reader per
snapshot and device. `prepared_reads`, `initial_read`, and `input_binding` connect
original Tensor logical indexing, the existing normalization/reshape receipts,
source regional loads, and the actual signed-i32/64-bit-pointer addresses. Bounds
and alignment are proved only for accesses that actually occur. Distinct
storage objects are not assumed to have disjoint physical addresses.

## Admission repair

The previous checked wrapper compared `untyped_storage().data_ptr()` for
equality. Two distinct storage views can have different starts yet overlap:
FP32 tensors of shape `(2,128)` wrapping the same byte buffer at offsets 0 and 4
pass that old check. The unchanged source program `(0,0)` with block 32 writes
a live input through the shifted gradient view. The reproducible CPU source-AST
witness is recorded in `overlap_counterexample.json`; it is an admission/memory
defect, not a floating-point numerical issue.

The wrapper now compares absolute live enclosing byte intervals for gradient
versus XY/DOUT and active OUT versus XY/DOUT/gradient. This rejects partial
overlap between distinct storage objects, permits read-only aliases, and permits
disjoint slices of one storage object. Enclosing intervals conservatively include
row gaps. All other guards remain: strided FP32 CUDA payloads without lazy
negative flags, positive even paired-input width, original shape/stride assertions,
common device, signed-i32 bounds, nonoverlapping destination rows, and active
pointer alignment. Empty batches still avoid pointer queries, and inactive OUT
has no payload or address requirement.

## Validation and boundaries

Targeted Lean builds include both public entries and every new helper, with
standard-base-only axiom audits. The retained host entries live in
`Proofs/LegacyCorrect.lean` and `Proofs/LegacyFPEquiv.lean`; host execution and
autotuning are not part of the new public kernel theorems.

`check_contract.py` checks exact JIT AST identity, retained original wrapper
statements, 16 supplied-output paths across six configurations, allocation
branches, 12 original rejection cases, the shifted-storage defect, and accepted
read-only/disjoint-storage aliases. Its CPU executor uses real backing storage
with a CUDA metadata facade; CUDA execution is unavailable and was not run.
The FP32 compiler lowering and instruction primitive meanings are explicit model
boundaries. They are not numerical experiments or a claimed fresh compiler/GPU
validation. This revision introduces no numerical transformation and has no
deferred numerical-experiment obligation.

The independent Real correctness declaration uses the shared `ProgramIO.Implements`
judgment. The primary FP entry compares the two actual complete source programs
by `original ≡[[]] indexCommuted`. `IO.lean` exposes three full-matrix input channels, the DX/DY/optional
OUT channels, their row strides, the forward-output mask, and the complete
source launch schedule. Numeric dimensions, ownership/separation, positive block
size and arbitrary permutation of the full two-dimensional grid remain public.
`Correct.lean` shows the independent analytic formula; `Misc/NativeCorrect.lean` retains the
independent native recurrence. Neither main statement exposes execution states
or cell-level proof plumbing, and both have statement-surface audits.

`Proofs/SourceSafety.lean` checks the actual source loads and stores in both
recompute branches. `Proofs/IOImplements.lean` propagates that safety through
every actual launch state and composes `Whole.grid_result` with logical input
binding, all enabled outputs and the exact exterior frame. The former raw
entry results remain ordinary theorems in `Proofs/KernelRealResult.lean` and
`Proofs/KernelNativeResult.lean`.

The unchanged TensorBridge theorems preserve Prepared/Storage, checked physical
pointers, logical Tensor reads and native byte results. Both levels reuse
`Whole.grid_result`: `KernelProof.run_eq` identifies the quoted grid, while
prepared input reads and physical commit lemmas supply the Tensor correspondence.
No host guard, source dtype, runtime ABI or numerical assumption changed. Scoped
Lean builds are format/integration checks; they are not independent proof replay
or GPU/native experiments.

The 2026-10-21 pair review reuses the original task's existing index-commuted
FP32 source. Only the six natural row-address products change operand order.
The actual floating casts, sigmoid, multiplication and observed native FMA
patterns remain intact. `Proofs/SourceEquivalence.lean` proves whole source and
schedule equality plus the candidate's actual access safety;
`Proofs/GridEquivalence.lean` reuses the original full-grid implementation and
frames for every allowed permutation and both recompute branches. The shared
`ProgramIO.Implementation` view retains the concrete schedules and observers.
No numerical rule or finite-word restriction is added. The old
`swiglu_backward_kernel_fp32` implementation theorem keeps its name and statement
as an ordinary theorem in `Misc/NativeCorrect.lean`.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.
The primary equivalence uses the unchanged `swigluNativeIO` abbreviation from
`Proofs/NativeIO.lean`; the auxiliary native formula and theorem remain in
`Misc/NativeCorrect.lean`.

The 2026-10-22 source-only cleanup replaces the selected repair's generated
FP32 source with two complete ordinary quotations. `Proofs/SourcePreservation.lean`
proves full function equality for `kernel` and `commutedKernel` against their
former generic canonical selections, including the public pointer signature
and all universally quantified strides, dimensions and recomputation flags.
Both branches, casts, sigmoid, arithmetic order and ordered masked stores are
unchanged. The generic canonical alias and unselected storage profiles remain
available with their prior scope. This cleanup changes no source contract,
native observer, full-grid theorem, existing proof status or numerical rule.
