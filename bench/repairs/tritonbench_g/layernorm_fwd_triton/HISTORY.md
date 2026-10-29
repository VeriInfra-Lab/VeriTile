# Historical wrapper-level revision

The material below records the earlier wrapper-level proof and its historical numerical classification. It is not the current kernel-only theorem scope.

# Three-pass strided FP32 layer normalization

The separate candidate fixes the ignored inner strides and widens native index
arithmetic. It retains the original three passes, block128, default launch
settings, two-dimensional grid, explicit FP32 loads/accumulators/output cast,
and `1 / sqrt(variance + eps)` operation order. The original benchmark Python
is unchanged.

## Input contract and result

X is a strided FP32 CUDA Tensor of shape `[S,H,N]`; W is a strided FP32 CUDA
Tensor of shape `[H,N]` on the same device. Unresolved negative views are
rejected. Each nonempty input data pointer must be four-byte aligned. Epsilon
is a Python int/float whose finiteness query succeeds and is finite and
positive; Python bool follows int semantics. Tensor values may be nonfinite.

Empty dimensions, contiguous and noncontiguous views, nonzero storage offsets,
stride-zero/broadcast inputs, input aliasing, and noncanonical singleton
strides are supported by the same direct strided kernel. These are represented
by ordinary valid Torch metadata and storage bounds. The candidate adds no
copies, dtype/device transfers, byte-gather materialization, negative-view
resolution, or segmented/grid-capped launch. It allocates `empty_like(X)` and
returns that exact Tensor, with X's shape, FP32 dtype, and device.

For each logical `(sequence,head)` row, the independent mathematical target is

```
mean = sum_col(X) / N
variance = sum_col((X - mean)^2) / N
Y = ((X - mean) / sqrt(variance + eps)) * W
```

The Real interpretation quantifies arbitrary original Real input memory and
epsilon independently of FP words. There is no output coordinate when a
dimension is zero. The FP theorem instead preserves the complete selected
three-pass recurrence, actual source scalar ABI conversion, reduction trees,
FMA/separate accumulation choices, sqrt/reciprocal order and FP32 store.

## Repairs and source binding

The original shape `[1,2,2]` with strides `[4,1,2]` makes two programs write
physical cell1 and leave cell3 unwritten. The repaired loads and stores multiply
columns by the actual X/W/Y inner strides. Tests execute this original
counterexample and the repaired source on real CPU storage.

Program IDs, all eight strides, the column count, loop zero/step and lanes are
int64 before arithmetic. Scalar casts use `tl.cast`, including constexpr-one
specialization. Active element and byte addresses are proved exact from actual
Tensor/storage bounds. The fresh FP32 output allocation also bounds column
loop arithmetic, padded lane addition and final increment. Masked inactive
pointer products may wrap; no memory event observes them. The retained
word-arithmetic lemma handles compiler pointer strength reduction. No artificial
small-shape guard is introduced.

The original logical grid remains `(S,H)`, including empty grids. Successful
logical-grid delivery is an ordinary runtime condition. The shared Triton
large-grid launcher issue remains deferred under `bench/TODO.md`; there is no
claim that every admitted shape launches successfully on the installed runtime.

`Kernels.lean` contains only the source and source parameters. Applicable
historical address, streaming, reduction, whole-grid, compiler-control and
readback proofs retain their declaration names under `Proofs/`. The host
preparation proof now describes the single actual allocation and original
input handles. It does not reinterpret a copying route as an identity call.

`Contract.lean` names the exact Python admission domain. Both public entries
expose `python_checks_iff_preconditions`. The ordinary metadata binding ties
these checks to the actual Tensor descriptor and data pointer; the scalar
query interface observes the same epsilon subsequently packed into the ABI.
Original storage objects and offsets are retained, and the complete backing
bytes are included in the caller frame. Output alignment and freshness follow
from ordinary successful allocator laws. Local compiler recipes describe
nodes, masks, addresses and tree-leaf coverage, never a desired output equality.

Public results include:

- `Correctness.layernorm_fwd_triton_repaired_real_returns`: every logical
  returned coordinate satisfies the independent formula on original inputs;
  the arbitrary-Real and actual-byte caller frames are preserved.
- `Equivalence.layernorm_fwd_triton_repaired_fp32_returns`: actual ordered FP32
  result, returned allocation, selected object/ABI, readiness, device-context
  restoration and caller frame.
- `Equivalence.layernorm_fwd_triton_repaired_all_outcome_frame`: admitted calls
  preserve caller bytes through API/compiler/launch/context-exit failures.
- `Equivalence.layernorm_fwd_triton_tensor_binding`: exact returned shape,
  dtype/device, original X/W identities, alignment of all active addresses,
  and preservation of every byte of both original backing storages.

`Proofs/Binding.call_native_addresses` attaches the native pointers to those
same original/returned logical coordinates. The historical revision005 Storage
helper remains available for dependent tasks; its optional byte-gather
representation definitions do not add a gather to this candidate.

## Validation and deferred numerical work

All 27 task-local Lean modules compile. Public and helper audits report only
the standard logical base. The 13 actual-source wrapper/kernel executions
cover empty/singleton/tail/multiple-chunk shapes, strided and offset inputs,
broadcasts, input aliases and huge unused singleton strides; each returned
logical cell is written once and inputs remain unchanged. Sixteen guard
rejections include actual misaligned byte-offset storage and oversized Python
integer epsilon; bool epsilon is admitted. Independent PyTorch comparisons
pass, and the exact original write collision is reproduced.

`check_lowering.py --output-dir /tmp/layernorm-fwd-lowering` passes five offline
SM80 profiles with Triton3.7.1: i32/u32/i64 integer ABIs plus all-zero and
all-one scalar specialization. The emitted accesses are scalar `.b32` loads
and stores; int64 loop and product lowering is checked. FP32-epsilon profiles
emit `fma.rn.f32`, `div.full.f32`, and `sqrt.approx.ftz.f32`.

No CUDA device was available. Native GPU numerical conformance and FP-to-Real
accuracy remain unclaimed. The later numerical batch must validate the local
zero/one/count/epsilon conversion, add/subtract/multiply/FMA/division/sqrt
semantics, including NaNs, infinities, signed zero, subnormals and selected
flush behavior. Complete schedule extraction for every compiler configuration
also remains an explicit compiler boundary. Addressing, masks, loop coverage,
host wiring, returned metadata and memory frames are already proved and are
not numerical deferrals.
