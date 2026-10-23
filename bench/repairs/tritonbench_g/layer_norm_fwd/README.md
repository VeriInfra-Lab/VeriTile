# layer_norm_fwd: checked inputs and whole returned storage

The independent target forms each row's input by optional row scaling,
seven-round Philox dropout, optional second input, and residual addition in
source order. Dropout uses the row seed narrowed from int64 to uint32 and
`rand(seed, column) > p`; the second stream starts at seed row `M + row`.
Kept values are divided by `1-p`. LayerNorm subtracts the population mean;
RMSNorm uses zero mean. Both use the population squared-deviation sum and
`1 / sqrt(variance + eps)`, then weight and optional bias. A second affine
output, unnormalized residual output, statistics, seeds and both split dropout
masks retain their original return behavior.

Inputs, arithmetic, reductions, load casts and statistics use FP32. The actual
requested `out_dtype` controls Y/Y1 stores. `residual_dtype` controls the
unnormalized output unless a supplied FP32 residual overrides it. Seeds remain
int64 and masks bool. All six original warp configurations (1/2/4/8/16/32),
three stages, default fusion, padded lanes and reduction/contraction structure
are retained. Destination conversions are explicit native store operations.

## Runtime contract

`_check_inputs` and `Admission.InputDomain` admit exactly the same input profile:

- Rank-two strided FP32 X, same-device CUDA FP32 weights and optional reads;
  X/residual/X1 shape `(M,N)` and strides `(max(N,1),1)`, parameter vectors
  shape `(N,)` and stride 1, rowscale shape `(M,)` and stride 1.
- Offset zero, no lazy-negative flag, and four-byte alignment for nonempty
  tensors. Empty tensors do not query their pointer. Read/read aliasing is
  allowed. The original X1/rowscale exclusion remains.
- Builtin finite positive `eps`; builtin finite `dropout_p` in `[0,1)`;
  exact bool normalization and mask flags. Raw binary64 values, including
  dropout signed zero, are retained in the Lean request and native ABI.
- Output dtype is None or an actual Torch dtype. The same applies to
  residual_dtype only when no residual is supplied. With a residual, the
  original override ignores that argument. Ordinary allocation/compiler
  failures for particular destination dtypes remain observable.

The original wrapper follows the new guard unchanged. No input transfers,
resolve_neg calls, clones or layout conversions remain. The only JIT repair
widens program_id to signed64 before pointer products. The actual device
context still selects X's device and restores the prior device. Allocations,
RNG generation, width check, autotuning and mask splitting retain source order.
No import-time benchmark execution is retained in the repair candidate.

`M=0` is retained as the original empty launch. `N=0` is admitted by the input
profile, produces the actual `BLOCK_N=0`, and follows the ordinary compiler
arange failure. `N>16384` retains the original post-allocation width error.
No positive-width premise is substituted for those source outcomes.

## Lean composition

`Kernels.lean` is source-only. `Source.kernel` preserves the complete selected
source transcription, and `source_exact` binds it to the retained source
execution proof. The signed64 repair is connected through the native address
and ABI proofs. Source-only historical reduction syntax also supports the
shared WholeReduction helper without changing the original benchmark source.

`Admission.checkInputs_iff` proves the exact guard/domain equivalence.
`Checked.prepare` and `Checked.surround` execute the trimmed source host.
`Checked.run` performs admission before that execution. The old proof's two
materialization operations are removed with local identity laws for
nonnegative, same-device tensors. `Checked.admitted` proves equality of the
complete state and result, including failure states; only no-op trace events
are erased. No kernel/output equation is an adapter premise.

`Equivalence.layer_norm_fwd_repaired_requested_storage_returns` derives the
historical DefaultInputs profile from this guard and composes the actual
successful checked call with the complete native returned-memory contract.
The actual final state is explicitly equal to the proved state. That contract
contains all eight returned coordinate observations, emitted target stores,
requested conversions, the selected compiler provenance, all actual benchmark
predecessor states, both seed/mask halves, incoming/current-stream readiness,
read-input preservation and the full caller backing-byte frame.

`Correctness.same_call_real_return` composes the same actual call with both
that native contract and the latest independent arbitrary-Real interpretation.
Its ideal physical memory and scalar interpretation are separate from native
payload decoding. It does not assert exact equality of floating computation
and unrounded Real arithmetic. The ideal result is evaluated from the original
logical inputs and generated discrete seeds, with returned identities and
frames derived from the same host call.

The retained numerical boundary consists of local host binary64 conversion,
constants, add/subtract/multiply/divide, ordered FMA, square-root approximation
and FTZ, comparison, Philox uniform conversion and each actual destination
store conversion. Complete reductions, addresses, grids, autotune history,
allocation flow and returned outputs are proved structure, not numerical
callbacks. The ordinary Tensor/allocator/compiler/stream API binding remains
explicit. There is no universal proof of PyTorch/Triton/runtime internals,
native GPU numerical conformance, or FP-to-Real error bound.

## Provenance and checks

The retained latest review is
`2026-10-17/layer-norm-fwd-real-repair-integration/repair-review.json`, which
accepted both the whole native wrapper and the same-call arbitrary-Real proof.
Historical candidate SHA256:
`293e170c8b2c80da9f806ef3d80aa11ba61df6629d5d2945d390b09db87fb9a8`.
The current compiler key uses the repaired candidate's current SHA256; historic
GPU/compiler evidence is not relabeled as evidence for this host revision.
The historical helper closure is restored under `Proofs`, including the latest
Real revision, without copying evidence archives or a new acceptance framework.

The full 145-module historical proof closure, new source/admission/trimmed-host
bridges, and both public entries compile. Public axiom audits remain within
the standard Lean base. The AST check also verifies the current candidate SHA
against `Proofs/CompilerProfile.lean`.

`python bench/repairs/tritonbench_g/layer_norm_fwd/check_contract.py` passes:
29 rejected calls, 20 admitted CPU source executions, original zero-column
and over-width failures, and a sparse signed32-overflow control. Exact AST
checks show the single row cast and otherwise unchanged kernel/host operations.
Cases cover both norms, optional streams, independent Philox masks, split-view
identity, input aliases, nondefault CUDA device selection/restoration, repeated
launches, empty rows, tails, requested floating/integer/bool stores and residual
dtype override. CUDA, allocations and compilation are mocked; source execution
uses actual CPU tensor backing storage. These are regression checks, not GPU
numerical evidence.
