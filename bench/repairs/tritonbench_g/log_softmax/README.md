# log_softmax: kernel proofs for the separate repair

The original benchmark is unchanged (SHA-256
`ea7ae1e49d9dac09bb0fd692bebc73af80710370a05365f204f0a4213809ddaf`).
Its forward `log(exp(delta) / sum(exp(delta)))` can underflow before the log:
FP32 input `[0, -1000]` produces a negative-infinite second result instead of a
finite value near `-1000`. Its backward masked gradient load has no `other`,
so undefined padding enters the sum. The original proof's concrete padding
witness changes active results from `(-4, -2)` to `(-14, -12)` when two padded
gradient lanes change from zero to five. These remain original-source defects;
proofs of the separate repaired source do not certify the original.

The existing Python repair computes `delta - log(sum(exp(delta)))`, explicitly
pads backward gradients with zero and saved outputs with negative infinity,
and widens/clamps pointer indices. This revision keeps its Python code and
accepted input domain unchanged. All BM choices 1/2/4/8 and warp branches
4/8/16 remain. The existing `.contiguous()` calls are retained for input and
upstream gradient; callers are not newly required to provide contiguous data.

## Public kernel scope

- `Kernels.lean` contains readable FP32 forward/backward Triton DSL sources and
  their retained integer row-product commutations. It contains no handwritten
  AST or mathematical output implementation.
- `Correct.forward_grid_correctness` proves the independent full-row formula
  `x_i - log(sum_j exp(x_j))`; `backward_grid_correctness` proves
  `g_i - exp(saved_y_i) * sum_j g_j`. Both construct successful complete source
  grids, prove actual masked source-access safety, give every logical output
  and preserve all exterior typed memory through shared `ProgramIO.Implements`.
  `IO.lean` declares dense input/output windows and the full launch schedule,
  and adapts the independent row formulas from `Proofs/Real.lean` to coordinates
  and finite channels.
  The optional ordinary theorems `TensorBridge.forward_tensor_correctness` and
  `TensorBridge.backward_tensor_correctness` in `Misc/TensorBridge.lean` reuse the same
  proven full source executions with actual pointer reads
  and the existing contiguous conversion's local element law. They construct
  the mathematical initial state from FP32 words under an explicit
  `Word -> Real` interpretation, and state the formula on the original logical
  Tensor values. They assume neither an arbitrary Real input binding nor a
  successful kernel/output result. Saved backward values are observed from the
  actual saved Tensor at kernel entry; gradients refer to the caller's logical
  gradient before its contiguous conversion.
- `FPEquiv.forward_index_equiv` and `backward_index_equiv` prove successful
  execution and complete typed-memory equality for the actual source pairs,
  under arbitrary unchanged floating primitives, casts and reducers.
  The auxiliary `forward_native_grid` and `backward_native_grid` theorems in
  [NativeEquivalence.lean](Misc/NativeEquivalence.lean) cover the full selected grids and
  all native-observed result words and the exterior frame through shared
  `ProgramIO.ObservedEquivalent`. Both actual source traces are proved safe.
  Both execute the
  canonical source into symbolic graphs and use the same explicit reduction
  layout, native instruction interpretation and fusion schedule. All FP32
  words are admitted, including NaNs, infinities, signed zero and subnormals.
- Wrapper checks are connected by `checkInputs_iff` and `checkBackward_iff`.
  Wrapper return objects, autotuning callbacks and autograd implementation are
  not new public proof obligations. Applicable previous results remain in
  `Proofs/HostCorrect.lean`, `Proofs/HostFPEquiv.lean` and their existing helpers.

The checked domain remains strided FP32 CUDA, resolved negative views,
`numel < 2**31`, and four-byte-aligned nonempty pointers. Dimension is an
`operator.index` value in `[-rank, rank)`. Output dtype is None or FP32.
Nonempty calls require `N <= 131072` and `K <= 65535`; empty shapes skip launch
and need neither launch-only bound. Backward gradients match the saved output
shape/device; the saved output is contiguous. No value-based Python guard is
added. Real inputs are a mathematical interpretation, not an assertion that
FP32 has no NaN/Inf or that native arithmetic equals Real arithmetic.

`Proofs/Kernel.lean` evaluates every statement of the canonical source.
`KernelGrid.lean` composes those successful executions, derives coverage from
the actual two-dimensional launch and proves the full frame. It uses the
existing scatter algebra. `KernelMath.lean` connects execution to the retained
independent mathematics. `TensorBinding.lean` uses shared Torch `TensorLaws` to
connect observed pointers and logical elements, including the original
contiguous conversion's local element-preservation contract. `Address` and
`Machine` retain exact signed64 index and four-byte pointer bounds.
`NativeGrid.prepared_input` and `direct_input` bind the actual source graph's
input leaves to the caller's logical FP32 values through one absolute pointer
memory; input aliases do not require distinct allocations.
For the mathematical model, `Binding.PointerInput` uses the observed
`Tensor.dataPtr` directly; `dense_address` connects that pointer to shared
storage laws. `prepared_real_read` and `direct_real_read` derive the input
premises of the reused kernel mathematics for `Binding.realState`. The explicit
word interpretation is a mathematical abstraction, not an IEEE decoder or a
new Python value guard.

## Numerical and implementation boundaries

There are **no numerical transformation assumptions or deferred numerical
experiments** for the retained integer-index candidate. Exp/log, basic FP32
arithmetic, cast semantics, and the unchanged reduction/FMA lowering are
ordinary primitive/compiler model boundaries. The previous seven primitive
conformance ports remain optional model-refinement results in `Proofs/FP.lean`;
they are not numerical-experiment TODOs or premises of the public equivalence.
`Proofs/NativeGrid.lean` reuses the existing softmax native graph observer.
It retains source precision, casts, loads and complete scalar reduction trees.
Its `backward_fused` theorem interprets the actual adjacent `g - p * sum` form
using `fmaRN` when that contraction is selected. The public full-grid theorem
preserves native observations for any fixed selection; it never equates split
multiply/subtract to FMA. Supplying a compiler's actual tree, operand order and
fusion selection remains the explicit lowering-model boundary.

Source transcription, successful ordinary compiler/runtime submission,
Torch contiguous element preservation and fresh output allocation remain
ordinary source/library boundaries. Triton runtime grid-product compatibility
and segmented launches remain outside this task's expansion scope.

## Validation

The existing CPU wrapper/body harness passes 81 cases, including all retained
BM choices, warp boundaries, stable-underflow examples, nonfinite rows,
noncontiguous inputs/gradients, empty dimensions, rejection before allocation,
saved-output identity and input/exterior frames. This is CPU execution with
mocked CUDA metadata, not GPU execution or native numerical validation.

The current public kernel proofs pass targeted Lean builds, non-circular
statement checks and standard-base-only axiom audits. Both public source
equivalences report no FP assumptions. Targeted validation and current file
hashes are recorded in `/tmp/log-softmax-kernel-delivery.json`.
The preserved historical FP helper and host corollaries also replay
successfully against the refreshed shared dependencies.
No GPU execution or independent native replay is claimed.

`Misc/TensorBridge.lean` also contains the Python-check and physical-access binding
helpers. The primary `Correct.lean` and `FPEquiv.lean` entries do not import it.

## Public IO and optional Tensor interfaces

The two `Correct` headlines contain only M/N/K, the retained row-block choice,
the existing topology-only reduction plan, shared IO and independent logical
formulas. They expose no interpreter state, runner, cell expression or final
state witness. Forward has one flat M*N*K input channel; backward has the
saved-value and upstream channels. Both have one complete output channel and
no private scratch. Empty dimensions keep the original empty schedule.

`NativeIO.lean` retains all source load/cast/operation nodes, the complete
canonical grids and the integer-index candidate. [NativeEquivalence.lean](Misc/NativeEquivalence.lean) applies the shared
output-only native observer to arbitrary symbolic input terms, including every
original FP32 word leaf. The selected reduction layout, instruction semantics
and fusion schedule stay explicit. The general arbitrary-region one-program
source equivalences remain public as well.

`Proofs/KernelSafety.lean` follows the real evaluated source statements and
proves every active load/store address is backed by its declared region.
`Proofs/IOBinding.lean` connects the actual launch sequence and exact exterior
frame. `Proofs/PublicIO.lean` reuses the existing complete grid mathematics;
`Proofs/NativePublicIO.lean` reuses the original exact symbolic source equality.
No new kernel execution proof or numerical law is substituted for those results.

`TensorBridge` retains all former guarded and actual Tensor applications.
Operational versions of the old main statements are ordinary theorems in
`Proofs/RawCorrect.lean` and `Misc/RawNative.lean`; the Tensor and public IO
adapters share that identified execution foundation. Supporting modules do
not import either reader entry, and neither reader imports `TensorBridge`.

Targeted IO-format build records are in
`.veritile-cache/public-io-format-20261020/batch12-logsoftmax-final001.log`.
The Python source, `Kernels.lean`, and `Contract.lean` are unchanged. No GPU run,
independent native replay, new numerical assumption or numerical experiment
was added in this revision.

Source-only cleanup replaces the forward/backward whole-source macros with four
complete ordinary FP32 `triton` quotations for the retained originals and
index-commuted candidates. Complete function identities against the previous
sources hold by `rfl`; all precision, masks, clamped addresses, casts, reductions,
source order, domains, bindings and theorem declarations are retained. This
organization adds no numerical, dtype, native, host or hardware requirement and
does not recount the existing proof completion.
