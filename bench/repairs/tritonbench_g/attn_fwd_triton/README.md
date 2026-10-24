# attn_fwd_triton

The repair keeps the accepted causal two-loop kernel and its precision. It
checks the actual caller views, uses them directly, allocates BF16 `O` and the
original unused FP32 `M`, and returns the allocated `O`. Empty tensors return
after both allocations without checking scale lengths or launching.

The direct contract accepts ordinary resolved FP32 strided CUDA tensors on one
device, matching rank-four Q/K/V shapes, and independent nonnegative strides.
Read-only aliases and broadcast strides are allowed. Scales must already be
contiguous, with enough elements for the packed query/key block scales. No
`resolve_neg`, packing, flattening copy, or scalar normalization is added.

Executed guards check source launch axes, tile dimensions, a signed-64 envelope
for padded lanes and final unused pointer advancement, natural alignment, and
absolute unsigned-64 addresses. Backed allocation spans and the fresh output's
nonoverlapping layout use ordinary Tensor/allocation interpretation. No numerical
fact is needed for structural safety.

The public files are `Kernels.lean`, `Contract.lean`, `Correct.lean`, and
`FPEquiv.lean`; proof implementation is under `Proofs/`.

- `Checked.whole_real` proves the independent causal normalized base-2 weighted
  sum for every output coordinate, complete source execution, physical cell
  readback/frame, and all-grid access safety.
- `Checked.trace_safe` derives complete source trace safety from the successful
  checks and ordinary backed storage, without a numerical correspondence premise.
- `Checked.whole_native` starts from the actual caller bytes. FP32 loads read
  four little-endian bytes through the caller's Tensor descriptors and actual
  source strides. The selected source graph retains TF32 score dots, FP32
  scales/accumulation, FP16 P/V casts, the FP16 PV-dot result, its FP32 widening,
  the two separate native FMA updates, FP32 division, and BF16 conversion.
  The executed tail proves the BF16 output tag. Every returned element is
  serialized as two bytes and read through the actual allocated Tensor's
  pointer and strides. Object/storage identity, dtype, shape, device and strides
  are retained; every exterior physical byte and every original input allocation
  byte is preserved.
- `Checked.empty_native_return` proves actual object return and the entire byte
  frame for every empty dimension, including D=0 with nonempty B/H/N.

The source-to-recorded SM90 instruction graph is the ordinary pinned compiler/ISA
contract supported by the existing profile audit. Lean verifies graph execution
and the complete returned memory contract; it does not prove compiler internals
or native arithmetic equal to Real arithmetic. Local instruction semantics are
explicit in `NativeGraph.Operations`; any numerical relation is separate and
transports to the actual returned Tensor via `Checked.returned_relation`.
The auxiliary reciprocal rewrite still requires its explicit `div_mul_rcp` law.

Validation: 39 host-contract cases exercise the actual helper/wrapper branches,
allocation retention, direct caller objects, pointer checks and accepted kernel
AST digest. Twenty-one CPU Triton interpreter controls inspect all output cells
and exterior storage, including tails, independent strides, aliases and signed
scales. These controls do not establish native GPU numerical conformance.
The original defect witnesses and source modules are retained. Original Python
SHA-256: `de0fb466adb7844b34f53e44225c25411faad342e4192e8f9d5bbe71f154df12`.

TODO: the shared Triton launcher signed grid-product compatibility issue remains
deferred by the user. No task-specific grid-product guard is added. The task
retains per-axis/source bounds and the ordinary successful runtime delivery
boundary. Native numerical validation remains separate from structural closure.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `f9af84483b6ec2cc7463c8f3b25577ec3570e338bd65830516765c9e3a56a760`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(q.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.
