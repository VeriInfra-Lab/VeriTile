# logsumexp_fwd: checked kernel proofs

The selected repaired Python source and its guards are unchanged in this
kernel-only revision. The original benchmark remains separate and unchanged:
SHA-256 `8d5d3a23a334d4718b4d43373ae234f694a8813b50734dde79b9d878cf303550`.

## Mathematical target and scope

For input row `r`, segment `b`, last dimension `D`, and block width `B`, the
kernel writes scratch element `[r,b]` with the independent Real formula

```
log(sum(exp(scale * x[r, b*B+j])))  over 0 <= j < B, b*B+j < D
```

When scale is absent, the summand uses `x` directly. `B` is the original
`min(next_power_of_2(D), 65536)`. This is a per-block formula: the subsequent
Torch `z.logsumexp(-1)` and optional FP64 output conversion are outside the
required kernel theorem. Empty input/output grids execute no kernel program.

`Correct.logsumexp_kernel_correctness` proves successful execution of the shared
source for every complete ordering of the `(N, ceil(D/B))` grid, every scratch
output, and preservation of all tagged memory cells outside those outputs.
`KernelMath.block` contains no execution, maximum, shift or floating backend.
Existing stable-logsumexp mathematical lemmas are reused to connect the actual
source to this formula. Real semantics do not assert exact native arithmetic.

## Source precision and floating result

`Kernels.rowBlock` is the readable original typed DSL source used by the Real proof
and retained native theorem; `Kernels.rowBlockCandidate` is the complete optimized
source compared by the main pair. Input, scratch storage, scaling, max/sum reductions, subtraction,
`tl.exp`, `tl.log` and final addition retain FP32. The runtime scale register
is bound to its actual FP32 ABI word; the standard `tl.param` declaration supplies
register typing without emitting an assignment. Program IDs retain the
source's signed-int64 casts.

The directly related candidate `logsumexp_fwd_optimized.py` removes one full-vector
`tl.where` in the unscaled specialization. The masked load already fills every
inactive lane with negative infinity. The scaled specialization retains mask
restoration, including for arbitrary nonfinite scales. This optimization preserves
all FP32 arithmetic, reduction engines, casts, source masks and launch order; no
performance measurement is claimed.

`FPEquiv.source_equivalence` compares the actual original and candidate schedules
for every complete row/block permutation. Its empty experimental rule set follows
from exact source execution equality in the existing `Engine` model, together with
both source-safety proofs, successful executions, every observed output and the
complete exterior frame. All native words remain admitted. No floating arithmetic
rewrite is used.

The ordinary `FPEquiv.logsumexp_native_correctness` theorem in
`Misc/NativeCorrect.lean` relates actual successful source
execution to an independent ordered word graph. Each program's maximum and
sum tree is explicit. Every scalar operation is interpreted by the same
arbitrary backend on both sides; all FP32 words, including NaNs, infinities,
subnormals and signed zeros, remain admitted. No exp identity, reassociation,
rounding law or native NaN-payload convention is assumed.

There are **zero numerical experiment deferrals**. The old seven scalar
conformance equalities in `Native.LocalObligations` remain available only for
an optional reference-backend specialization; primitive semantics are modeling
boundaries, not per-task empirical assumptions. No claim equates this source
semantics theorem with executed GPU code.

## Input assertions and actual Tensor storage

The preserved Python checks require rank at least one, contiguous strided FP32
CUDA input, no unresolved negative view, fewer than `2**31` elements, and
four-byte alignment for nonempty input. Scale is `None` or a Python float;
output dtype is `None`, FP32 or FP64. The leading shape product is also below
`2**31`. No new input copy, dtype combination or numerical-value guard is added.
`TensorBridge.python_checks_iff_preconditions` proves the metadata checks agree with
`InputContract`; dynamic Tensor classification precedes this typed interface.

`KernelBinding.Input` connects that same descriptor to the shared
`VeriTile.Torch.FP32.Tensor` storage laws. Public input bindings read the actual
physical window, and proved direct-read lemmas identify those words with
logical Tensor values, including nonzero storage offsets. Pointer/alignment
lemmas use the observed `data_ptr` and backed allocation, not an unrelated
abstract base. `KernelBinding.Output` records the ordinary FP32 scratch
allocation, shape, device and observed-pointer relation. Existing `Machine`
lemmas supply masks, widened index calculations, access bounds and finite byte
addresses. No program-output premise is part of these storage laws.

The full tagged-cell frame is proved in the structural interpreter. Ordinary
Tensor/allocator storage binding and compilation to native memory operations
remain explicit API boundaries; no physical-byte GPU interpreter is claimed.
Memory-model rationale and references are in `documents/GpuMemoryModel.md`
and the shared `VeriTile/Torch/FP32.lean` API documentation.

## Original defects and retained repair

The original source is not correct for its unrestricted advertised inputs.
For example, a length-three zero row uses a padded fourth lane. With scale
`-1.0`, padded `-inf` becomes `+inf`, the maximum is `+inf`, and subtracting it
produces a NaN contribution; the expected answer is `log(3)`. Scale `0.0`
also poisons that padded lane. An all-`-inf` row produces `-inf - -inf` rather
than the expected `-inf` result. Rank-one input also exposes the original
`view(*shape[:-1])` call's empty argument list.

The already-selected repair restores masked lanes after scaling, uses zero
shift for either infinite maximum, handles empty extents and uses tuple-form
`view` for scalar output. Those production changes are preserved, not newly
expanded here. Acceptance of this repair does not prove the unrestricted
original benchmark correct.

## Organization and verification

Required public entries are `Correct.lean` and the two-program `FPEquiv.lean`.
The Real proof and optional native theorem share `Kernels.lean` and helpers in
`Proofs/`. Historical whole-wrapper results are
preserved separately in `Proofs/WrapperCorrect.lean` and
`Proofs/WrapperFPEquiv.lean`; required public imports contain no host
orchestration proof. Core metadata and memory semantics are reused.

`check_contract.py` executes the actual repaired Python body with a bounded
NumPy memory adapter and CPU Torch. It checks each scratch block against an
independent Torch formula before row aggregation, reversed complete program
order, nonzero storage offsets, input/output guard sentinels, all six retained
warp choices, empty/rank-one shapes, tails, the block cap, exceptional words,
output casts and input rejections. This is CPU source replay with CUDA mocked,
not native Triton or GPU execution. Native lowering inspection is available
through `check_cuda_lowering.py`; it is separate from kernel proof acceptance.

`Misc/TensorBridge.lean` contains the optional Tensor/check correspondence theorems;
`Correct.lean` retains the independent Real specification, while `FPEquiv.lean`
publishes the complete original/candidate implementation pair.

The Real entry and optional native theorem use the shared `ProgramIO` judgment. `IO.lean` displays
all input/output windows and the exact source invocation for every job in the
chosen full-grid permutation, including its actual scale register and native
reduction engine. `Proofs/AccessSafety.lean` checks the quoted source accesses;
`Proofs/Interface.lean` carries that safety through every actual predecessor
state and connects the existing full-grid execution, outputs and exact frame.

The Real input condition identifies ordinary reals embedded in the extended
source carrier; it is precisely the earlier mathematical input domain and is
neither a new Python guard nor an empirical assumption. The native interface
retains all raw FP32 words. `Proofs/RealExecution.lean` and
`Proofs/NativeExecution.lean` retain the former raw statements as ordinary
helper theorems. Both original Tensor applications still use those shared
execution foundations, with the same block-only scope and no final Torch row
reduction claim.


The prior missing-pair status is resolved by the unscaled mask-elimination
candidate and its complete source/grid proof. The historical source-to-native
theorems remain in `Misc/NativeCorrect.lean` with unchanged domains and statements.
The complete historical original source AST is preserved by the standard runtime
parameter declaration, as checked in the campaign preservation artifact.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.

`IO.lean` is the shared memory/schedule descriptor and `PairIO.lean` contains only the pair data. Candidate execution, safety and equivalence proofs are in `Proofs/PairIO.lean`, preserving their original declaration names.
