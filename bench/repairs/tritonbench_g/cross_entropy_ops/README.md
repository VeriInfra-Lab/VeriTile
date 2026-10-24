# cross_entropy_ops: selected kernel proof

Current FP result (2026-10-21): complete forward and backward original
programs satisfy `≡[[]]` with the existing integer column-index commutation
candidates, adapted to the selected runtime ABI and all source repairs.
`Kernels.lean` directly quotes all four complete programs; shared `tl.param`
declarations add no instruction. `Proofs/SourcePreservation.lean` proves
complete equality with each historical source expansion.

`PairIO.lean` fixes the original logical IO, runtime scalars, row/block IDs,
source FP32 casts and instruction order. `SourcePair.same_run/same_safe` use
strict `Int.mul_comm` to connect actual full programs. `PairEquivalence` retains
success, actual access safety, every typed output and both complete exterior
frames. Split mode writes LSE/loss and frames all z-loss cells; unsplit observes
all three outputs. The original all-word domain, signed64 labels, ignored
branches, postscale masks and allowed gradient/input aliases remain. No floating
rewrite, experimental rule or new numerical validation obligation is introduced.
`Misc/NativeCorrect.lean` retains the full source/native observations as optional
ordinary theorems. Independent Real targets keep their complete original scope.

The ledger-selected `cross_entropy_ops_repaired.py` is unchanged (SHA256
`5f106a09beaeb4807b9a8f6409e0f015478be518e8a31d9ca889a32f9cd7e768`). Its guards
delegate to the unchanged JIT function objects in `cross_entropy_ops_conditional.py`
(SHA256 `b690a712bd968686ee78818e81dbbb21a8d15331e132eb1c994d1ed3eb2b4ba5`).
The original benchmark remains a separate defective source.

The public `Correct.lean` and `Misc/NativeCorrect.lean` now expose shared IO judgments over
logical channels, kernel geometry and runtime scalar ABI values. `IO.lean`
shows the source invocation, signed-label and FP32 logit channels, physical
element offsets/strides, all three forward outputs, the backward output mask,
and the absence of private scratch. The shared judgment proves actual source
access safety, successful execution, typed logical results and the exact
exterior cell frame. These statements cover every launch coordinate, empty grids vacuously,
all tails, smoothing branches, ignored rows and split/unsplit forward.
`Correct.forward_unsplit`, `Correct.forward_split` and `Correct.backward` expose
the independent Real results. `FPEquiv.native_forward` and
`FPEquiv.native_backward` expose the ordered native observations.
The native addressed-input encoder preserves the original source region/offset
labels used by compiler metadata, including the original signed-label region
read convention. It admits every raw FP32 word and imposes no finite-input guard.
`TensorBridge.real_forward`, `TensorBridge.real_backward`,
`TensorBridge.native_forward` and `TensorBridge.native_backward` retain the
complete prior Tensor applications, including the allowed in-place gradient view. They add no allocator, device-context, autograd
or Torch aggregation implementation proof. Historical wrapper and resident
results remain in `Misc/Legacy/HostCorrect.lean` and `Misc/Legacy/HostFPEquiv.lean`.
The old raw Real statements remain in `Proofs/KernelInterface.lean`; the old raw
native statements are retained in `Misc/RawNative.lean`. `Proofs/AccessSafety.lean`
checks the actual source trace, including the conditional label gather and
ignored upstream branch. `Proofs/LogicalExecution.lean` reuses the established
whole-source execution proofs; `Proofs/Interface.lean`, `RealIO.lean` and
`NativeIO.lean` assemble the public results. Supporting modules do not import
either public entry.

## Independent mathematical specification

For `u[j] = x[j] * a`, `L = log(sum(exp(u)))`, smoothing `s`, square scale `z`,
and target `t`, the unsplit forward outputs are:

- LSE: `L`, including ignored rows.
- Loss: `L - (1-s)*u[t] - s*sum(u)/C + z*L*L`, or zero when ignored.
- Logging z-loss: `z*L*L`, or zero when ignored.

Split forward stores each block's own log-sum-exp and
`-s*sum(block u)/C - (target in block ? (1-s)*u[t] : 0)`; ignored contributions
are zero. It does not write its z-loss allocation. The full frame includes that
allocation. Host aggregation is outside this kernel endpoint; the retained
independent `Spec.local_lse_composition` explains how block LSEs compose.

Backward takes its actual supplied LSE `L` and upstream `g`. Every active class
stores `(g*a)*((1+2*z*L)*exp(x[j]*a-L)-indicator(j=t)*(1-s)-s/C)`, or literal zero
when ignored. `gradient_from_lse` identifies this formula with the independent
cross-entropy gradient when its LSE argument is the mathematical row LSE.

In `TensorBridge`, the ideal Real initial state is constructed from the same
bound Tensor words and int64 labels. The input-only scalar projection links the three runtime
scalar words to their chosen Real interpretation. This mathematical result is
separate from floating-point accuracy or exact FP32-to-Real equality.

## Inputs, precision and memory

The unchanged checked wrapper admits CUDA, strided FP32 contiguous logits
`[R,C]`, contiguous int64 labels `[R]`, common device, no lazy negative bits,
positive `C`, valid or ignored labels, finite Python float `s` in `[0,1]`, finite
Python float `a` including zero/negative values, and finite nonnegative Python
float `z`. It checks signed-int64 ignored index, exact bool in-place flag,
`process_group=None`, `R <= 2^31-1`, and
`ceil(C/min(next_power_of_2(C),4096)) <= 65535`. Active pointers have natural
alignment. Offset views, shared backing and empty row batches remain admitted.
The actual forward block cap is 65536; backward's cap is 4096.

Backward upstream `[R]` may be strided or broadcast. The existing source clones
labels and upstream into separate snapshots; these are ordinary preparation
laws, not new host proofs. Backward can reuse the logit allocation.

`KernelContract` reuses the established descriptor-to-byte/pointer/index
bindings. `KernelBinding` additionally connects them to shared
`VeriTile.Torch.FP32.Tensor`, its actual `physicalIndex`, Tensor laws and backing
storage. No universal backing-base alignment is imposed. The Tensor bridge
composes these adapters and reads actual output Tensor elements.
Labels remain connected to signed-int64 backing bytes, not an independent
unconstrained target function. Memory frames quantify over all modeled tagged
allocation cells, including padding and unrelated storage.

`Kernels.lean` has all 14 forward and 16 backward top-level source statements.
The three scalar parameters are raw FP32 ABI registers; total_classes is an
int64 register converted at its actual floating use. Casts, multiplication
operand order, both post-scale masks, the unused smooth_positive assignment,
separate z-loss expression, and conditional stores are retained.

The native expression interpretation fixes explicit sum/max reduction trees,
primitive instruction meanings, FMA contraction and operand order through a
compiler schedule. `forward_square_fused` and `backward_correction_fused` expose
actual FMA observations. There is no FMA-versus-separate-operation equality,
whole-reduction equality, kernel-result assumption, or local numerical peephole
obligation. Primitive/compiler/ABI semantics are model boundaries. The current
kernel endpoints introduce zero deferred numerical experiments and make no GPU
or compiler-implementation conformance claim.

## Checks and original defects

`check_contract.py` passes 34 rejection/failure cases and 20 CPU interpreter,
autograd and backing-frame cases, including 4097/65537 tails, local splitting,
negative/zero scaling, saved-version failure and aliased/in-place inputs.
`Proofs/check_kernel_source.py` checks all three source pins and the source/DSL
statement inventory, reproduces original zero/negative-scale padding NaNs and
ignored-row NaN gradients, and checks the selected repair's three outputs and
exact ignored zero words. These are CPU source controls, not native numerical
conformance experiments. Python, guards, shared core, and ledger are unchanged.

This format pass uses the isolated overlay driver in
`.veritile-cache/public-io-format-20261020/workspace_lean.py`. Its task logs are
`batch13-crossentropy-*.log`, with exact module receipts and source/dependency
hashes in that new overlay. Both main entries have independent-formula checks,
axiom checks and strict public-statement allowlists. Python, `Kernels.lean`,
`Contract.lean`, all old Tensor theorem statements and historical acceptance
remain unchanged. No independent replay or GPU experiment was run for this pass.

Optional proof layout (2026-10-21): `Misc/NativeCorrect.lean`, `Misc/KernelAudit.lean`, `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.
Historical wrapper readers are physically retained in `Misc/Legacy/HostCorrect.lean`, `Misc/Legacy/HostFPEquiv.lean`. They remain optional historical interfaces and are checked separately from the current primary/auxiliary import union.

Additional optional reader paths: `Misc/RawNative.lean`. Their declaration bodies are retained. Historical readers remain separate from current primary/auxiliary checks.
