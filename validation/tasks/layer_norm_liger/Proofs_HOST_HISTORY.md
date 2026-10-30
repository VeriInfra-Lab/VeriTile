# Liger layer normalization: FP32 forward and backward

This candidate retains the original contiguous preparation, forward/backward
kernels, SM-count grouping, saved statistics, Torch partial-gradient sums, and
four autograd return slots. It applies the accepted forward-variance and
backward-padding fixes and makes input admission and native integer arithmetic
explicit. The original benchmark Python is unchanged.

## Contract and independent mathematics

X, W and B are strided FP32 CUDA tensors on the same device, with no unresolved
negative view flag. X is nonempty and has rank at least one; its final dimension
N lies in `1..65536`. W and B each have shape `[N]`. Nonempty data pointers are
four-byte aligned. Epsilon is a Python int/float for which the finiteness check
succeeds and is finite and positive. Bool follows Python's int classification.
There is no finite-value restriction on Tensor elements.

The original `ensure_contiguous` behavior is preserved. Noncontiguous inputs are
made contiguous, and X is viewed as `[rows,N]`; contiguous offset views remain
supported. The added checks run before the decorator copies each Tensor. The
implementation does not add device transfers, dtype expansion, negative-view
resolution, or a new unconditional clone policy. The ordinary contiguous/view
API contracts preserve the original logical values and saved object regions.

For each logical row, the independent Real target is population mean and
variance, `inverse = 1/sqrt(variance+epsilon)`, then
`Y = (X-mean)*inverse*W+B`. The backward target is

- `xhat = (X-mean)*inverse`, `u = W*dY`;
- `dX = (u - xhat*mean(xhat*u) - mean(u))*inverse`;
- `dW[col] = sum_rows(dY[row,col]*xhat[row,col])`;
- `dB[col] = sum_rows(dY[row,col])`.

Backward consumes the actual saved Mean/RSTD cells produced by forward. It
writes separate dX and per-program dW/dB buffers, then returns the actual Torch
axis-0 sum/cast results. Autograd returns `(dX,dW,dB,None)`. Every output, saved
statistic and accumulator retains FP32 precision. The upstream Tensor is checked
for FP32/layout/device/logical flags/alignment and then made contiguous; its
shape must have the saved element count and final width. Ordinary autograd
supplies the forward output shape, so dX has the original input shape. The
internal backward helper preserves the incoming shape when it differs but has
the same count and width. Incoming dY may alias another saved read-only input or
Y; fresh backward allocations separate the writes.

## Concrete repairs

The original variance reduction included centered padded lanes. For
`X=[2.25,3,3.75]`, unit W, zero B and epsilon `0.625`, it returned
`[-0.375,0,0.375]`; the logical result is `[-0.75,0,0.75]`. The candidate masks
those variance contributions to zero. Backward also masks padded xhat before its
reductions and partial accumulation.

Extents, strides, program IDs and columns are widened before native products.
Backward group size and loop step are explicit int64 values. Scalar arguments
use `tl.cast`, including when Triton specializes the integer one as constexpr.
Rows per program use exact integer ceiling division. No artificial small-size
limit is added. Ordinary successful FP32 input and partial-buffer allocations
bound `rows*N` and `programs*N` by `2^62`; the proof derives exact signed-int64
row products, rounded group endpoints, padded offsets and pointer increments.
It includes prelude arithmetic for programs whose row groups are empty.

The forward settings remain next-power-of-two blocks, at most 65536, with
4/8/16/32 warps at the original thresholds. The HIP branch uses 16 instead of
32 at the largest threshold. The source backward launch does not pass the
calculated warp count; its Triton default of four warps is preserved. Native
logical-grid delivery is an ordinary runtime condition. The shared large-grid
launcher issue remains separate under `documents/bench/TODO.md`.

## Proof organization and public results

`Kernels.lean` contains only both source definitions and source parameters.
Historical helper proofs are retained under `Proofs/`, with their namespaces and
useful declaration names preserved. `Proofs/BackwardKernels.lean` now owns the
source-split theorem. All historical whole-grid, partial readback, selected-tree,
FP32-word, saved-statistic, frame and independent-gradient results are reused.

`Contract.lean` names the exact supported domain. Its helper proves both
forward and backward Python-check equivalences. `Settings` proves coverage and
the maximum block bound for the actual next-power-of-two choice. `NativeIndex`
proves the machine integer equalities; addressing/coverage is not a numerical
assumption. `Wrapper` constructs the actual row strides, group count, launch
parameters, saved region identities and eight allocations/sum outputs, and
proves the historical whole-program premises from these choices and generic
freshness. Future dY need not be disjoint from earlier forward outputs.

`Preparation` models the original successful contiguous/view API calls. Its
laws describe only ordinary Tensor metadata, logical copy preservation and FP32
storage capacity; no kernel/output result is an API premise. The kernel input
memory binding identifies the resulting Tensor's actual cells. `LogicalInputs`
projects the independent mathematical formula back to the original Tensors.

Public entries:

- `Correctness.layer_norm_original_inputs`: exact admission, successful ordinary
  preparation, storage-derived native bounds, complete actual forward/backward
  host results, and Y's mathematical formula over the original logical inputs.
- `Correctness.layer_norm_roundtrip_real`: complete source-generated wrapper
  parameters, all rows, saved producer statistics, dX and actual reduced dW/dB,
  returned metadata and the retained memory frame.
- `Correctness.layer_norm_backward_real`: independent complete backward result
  at the actual callback state after cotangent preparation.
- `Equivalence.layer_norm_source_fp32`: actual complete FP32-word source result,
  including both grids, partial sums/casts and returned descriptors.
- `Equivalence.layer_norm_selected_plan`: complete selected trees and local FMA
  choices, preserving source structure with no reduction/output equality oracle.

The public checked domain is precisely `Contract.InputContract` plus the actual
cotangent check when backward is involved. Successful Tensor/allocator and
compiler/control/typed-memory APIs remain explicit ordinary boundaries. A
successful device query must supply a positive SM count, as checked in Python.

## Numerical deferrals and validation

Native GPU numerical conformance and FP-to-Real accuracy are not claimed. The
later shared numerical batch must bind the local FP32 zero/literal/epsilon/count
conversion, add/subtract/multiply/divide/negate/FMA/rsqrt/casts, the separate Torch
sum addition/cast profile, and the local source/configuration schedule to the
actual compiler/device. Complete trees and all source control/memory/reduction
structure are already proved; no row/kernel/loop/launch/output law is deferred.
The selected plan remains explicit structural data; this work does not claim a
complete native compiler schedule extraction or HIP GPU execution.

- All 42 task-local Lean modules compile; the public entries and retained
  audits use only the standard logical base. Existing non-circularity checks
  cover the independent mathematics and actual axis-sum/cast semantics.
- `check_source.py` passes ten complete actual-source forward/autograd/backward
  executions over actual CPU storage, including noncontiguous X/W/B, expanded
  dY, empty groups, tail widths, original input preservation, one store per
  output cell, and independent PyTorch forward/gradient comparison. It checks
  fifteen guard rejections, actual CPU misaligned storage, the exact original
  padding counterexample, all forward warp thresholds/HIP selection, and the
  actual backward default. CUDA metadata and launch are adapted.
- `check_lowering.py --output-dir /tmp/liger-lowering` passes all 18 offline
  SM80 profiles for both kernels, every forward warp threshold and maximum
  block, backward's actual four-warp default, i32/u32/i64 ABIs and scalar-one
  specialization, including the largest backward block of 65536.
- No CUDA device was available. CPU source tests and offline compilation do
  not establish native numerical conformance or runtime launch applicability.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `81e64aaea77ee7b45073440fc67d51b2e6dae041356e7285098399771d8e47d8`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(X.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.
