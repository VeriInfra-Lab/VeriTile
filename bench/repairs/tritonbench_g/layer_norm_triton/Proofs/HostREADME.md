# layer_norm_triton: checked whole forward and backward

For every normalized row, the independent target is
`mu = sum(x)/N`, `v = sum((x-mu)^2)/N`, `r = 1/sqrt(v+eps)`,
`z = (x-mu)*r`, and `y = z*w+b`. Its backward target is
`dx = (dy*w-z*mean(dy*w*z)-mean(dy*w))*r`, with `dw = sum_rows(dy*z)`
and `db = sum_rows(dy)`. The mathematical specification uses the original
logical inputs and statistics proved from the actual preceding forward call.
It is an arbitrary-Real interpretation on that same call route, rather than
a decoding of native payloads or an assertion that rounded FP32 equals Real
arithmetic. Division and square root are total; no general calculus theorem at
singular inputs is asserted.

All source loads, statistics, arithmetic, reductions, stores, gradients and
partial carries have the actual default FP32 precision. Native formulas retain
all UInt32 payloads, including NaNs, infinities, subnormals and signed zeros.
The exact binary64 epsilon bits enter the retained ABI conversion. Every N=1
operation, division, padding, reduction tree, operand order, FMA choice and
intermediate partial-storage cast remains explicit.

## Exact admission

`_check_inputs` and `Contract.InputContract` select the following domain:

- X, weight and bias are strided, contiguous FP32 CUDA tensors on one device,
  with no lazy-negative flag and four-byte alignment when nonempty. A nonzero
  storage offset is allowed, as are Torch-contiguous singleton/empty strides.
  Read/read aliases are allowed. Empty tensors do not query their pointer.
- `normalized_shape` is a builtin integer, tuple or list of nonnegative builtin
  integers. It is nonempty, describes a trailing suffix of X, and exactly
  matches both affine tensor shapes. Rank-one X and multiple normalized axes
  are retained. Invalid Python argument types map to rejection.
- Epsilon is a builtin finite strictly positive float. Its binary64 bits are
  passed unchanged; native conversion may still underflow. There is no
  finite-tensor-content restriction or nonzero native-denominator assumption.
- Backward requires a contiguous FP32 CUDA dY on the saved input's device,
  exactly matching the original input shape. A broadcast/noncontiguous incoming
  gradient is rejected. Autograd saved-tensor/version access runs before this
  gradient guard, matching the source order.

The width cap remains the source operation
`B = min(16384, next_power_of_2(N))`, followed by `N > B` failure before
allocation. The guard does not replace that failure with a new width domain.
Zero leading or normalized dimensions skip both kernels; affine gradients use
the actual zero allocations. Warp and group branches remain unchanged: warps
are `min(max(B//256,1),8)`, and groups use 256/128/96/64 at the original
1024/4096/8192 thresholds.

The checked source uses only same-device `contiguous().view(...)`; admission
makes contiguous an identity. No transfers, resolve_neg calls or clones remain.
Returned affine gradients are views on their actual same-device allocations.
`Checked.forward_storage` derives input-backing identity and the original
affine-device relationships from the guard and ordinary contiguous/view laws.
Device context selection and restoration, all allocation order, save/access
operations, five autograd return slots, and the source launch options remain.

## Repair and proof composition

The retained JIT repair widens address arithmetic to signed64 and replaces the
backward spin-lock publication scheme with one program per original modulo
group. That program visits its rows in increasing order, preserves the first
contribution without an added zero, and retains each partial FP32 cast. Scratch
has shape `[min(G,M),N]`; the final kernel retains streaming BM=32/BN=128
reduction and masked loads/stores. The three JIT kernel ASTs are pinned to the
latest reviewed repair. The original benchmark Python is unchanged.

`Kernels.lean` contains only the complete current Python AST and immutable
source hashes. `bind_source.py --check` verifies that transcription. Models,
compiler interfaces, source interpretation, complete grids, storage readback,
saved-state, runtime/stream and mathematical proofs live under `Proofs`.
Shared FastLayernorm preparation helpers now describe the same guarded
contiguous/view operations. Affine-return/readiness proofs are adapted to
view-only returns without changing their numerical graphs.

The strongest public entries are:

- `Correctness.layer_norm_repaired_forward_real_checked` and
  `layer_norm_repaired_backward_real_checked`: actual guarded output objects,
  actual final state, complete independent forward/DX/DW/DB formulas, and the
  real saved-state composition.
- `Equivalence.layer_norm_repaired_forward_fp32_checked` and
  `layer_norm_repaired_backward_fp32_checked`: all returned native cells,
  complete ordered compiler-selected graphs, returned shapes/devices/identities,
  stream readiness and restored device owner. Separate checked frame entries
  preserve caller backing bytes.

`Checked.forward_admitted` and `backward_admitted` derive their exact input
contracts from successful checked calls and prove equality to the complete
host outcome. `Contract` proves exact boolean admission equivalences, including
dynamic argument rejection. Ordinary Tensor observations bind metadata to the
same handles; caller membership/liveness and saved-version receipts describe
ordinary storage and autograd behavior. They contain no LayerNorm result or
saved-statistic correctness premise. Signed64 bounds follow the ordinary
successful storage allocations, not extra caller size restrictions.

The remaining `Numerics.Conformance` boundary is local target-bit realization
of scalar constants/counts/add/sub/mul/div/FMA/sqrt, epsilon conversion and the
selected primitive behavior. Compiler selection binds actual object bytes,
source hash, ABI, defaults, reduction trees, swaps, masks and casts. Whole
outputs, scratch reductions, successful programs, and saved-statistic equations
are not numerical callbacks. Ordinary compiler/runtime/Torch API bindings and
global grid-product/PID/segmentation compatibility remain explicit boundaries;
no runtime-internals or native GPU conformance claim is made.

## Provenance and validation

The latest retained review is
`2026-10-18/layer-norm-triton-whole-final001-repair-integration/repair-review.json`.
Its candidate SHA256 was
`f923e78f8abf743f502c6ffd35990566660c309847fb2fd4d77e0d2e37bb18f5`.
The current guarded candidate SHA256 is
`6f2f1898b81f0f51580f26c12c22fb8989cec2ed61a69a6e33f2d3a5aa9f20a7`.
The compiler source pin uses the current hash. The closure is reused without
copying historical evidence, relabeling GPU runs, or changing an acceptance
ledger or count.

The retained closure, new admission/storage bridges and public entries compile,
with axiom audits restricted to the standard Lean base.

`python bench/repairs/tritonbench_g/layer_norm_triton/check_contract.py` passes
30 rejected calls and 17 admitted CPU source forward/backward executions.
It covers all group branches, multiple normalized axes, rank one, empty axes,
N=1/tails, multi-row groups, nonzero offsets, aliases, nonfinite payloads,
nondefault device restoration, saved-version failure and the original width
failure. It executes the original forward source to expose its ignored
normalized suffix, and checks a signed32 overflow control without a large
allocation. CUDA, allocations and compilation are mocked; actual kernel source
runs over CPU tensor backing storage. These are regression checks, not GPU
numerical evidence.
