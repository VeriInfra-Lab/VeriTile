# flash_decode2_llama: complete guarded Tensor call

`flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, O, block_seq)`
mutates the exact supplied FP32 output Tensor and returns `None`. Llama keeps exactly `D` in `{16, 32, 64, 128}`, `BD = D`, and the original unmasked Mid loads and output stores.
The source keeps its sequential summary loop, FP32 expression order, scalar
exponential, final division and default fusion behavior. All JIT AST nodes match
the previously accepted repair: the only repair arithmetic changes from the
original source remain the int64 promotions of both program IDs and the loaded
int32 length. This pass adds no launch segmentation or shared launcher-product
compatibility claim.

## Mathematical target and precision

For each batch `b`, let `K[b] = ceil(Length[b] / block_seq)`. The independent
Real target is

```
Out[b,h,d] = sum(j < K[b], exp(Log[b,h,j]) * Mid[b,h,j,d])
             / sum(j < K[b], exp(Log[b,h,j]))
```

The inputs are the supplied stage-1 summaries. Relating those summaries to raw
token attention requires a separate stage-1 consistency result and is not part
of this stage-2 API. The Real interpreter uses `Option Real` with `none` as
negative infinity and proves the online recurrence equals this unshifted
weighted formula. Arbitrary independent Real storage values supply the finite
summary entries; a native completed output is never a Real premise.

The separate native proof preserves raw FP32 words, source operand order,
load boundaries, sequential updates, FMA choices and all final byte stores.
Compiler choices inspect expression structure and cannot inspect payload words
or provide a result. Numerical refinement uses exactly seven local scalar
families: add, subtract, multiply, full divide, maximum, scalar exp32 and RN FMA.
Scalar exp32 retains the native rounded log2(e)-multiply/approximate-exp2
interpretation boundary. Constants and loads are fixed. No whole-loop,
reduction, output-valued, or exact-Real-equals-FP32 assumption is used.

## Checked domain

The accepted scope remains ordinary strided contiguous CUDA tensors at storage
offset zero, without negative/conjugate view flags, on one device. Mid, Log and
output are FP32; Length is signed int32. Shapes are `Mid[B,H,S,D]`,
`Log[B,H,S]`, `Length[B]`, and `Out[B,H,D]`, with positive `D`.
`block_seq` is a built-in positive signed-int32 Python integer. Canonical
calculated strides are supplied to the kernel. Actual observed Tensor strides
are retained in the binding, so harmless noncanonical singleton or empty-axis
strides do not become extra admission restrictions. This scoped contract does
not claim general noncontiguous or nonzero-offset layout support.

Geometry checks bound batch/head launch axes, lane width and formed signed64
byte offsets. Active pointers are explicitly checked for four-byte alignment
and for uint64 spans covering formed masked addresses. These checks do not
assume that every possible external allocation is aligned. Full storage-interval
checks separate output from all inputs and separate int32 Length storage from
FP32 summaries. Mid and Log may alias read-only. Interval checks also reject
partially overlapping imported storage objects with different base pointers.

When `B,H > 0`, the wrapper reads each signed int32 length and checks
`0 < Length[b] <= S * block_seq`. The named `InputContract` includes these
contents, not just tensor metadata. The Lean checker has a proved equivalence
to that full contract, including dynamic argument rejection. Native length
loads decode signed int32 before natural loop transport; sign extension and
the dynamic signed64 ceiling calculation are proved.

Empty batch or head axes return `None` without a kernel launch, pointer query,
length-content read or owner-device context change. The exact supplied output
object, its metadata/strides and all memory remain unchanged. Positive shape,
dtype/device/layout and the named zero-offset/lazy-flag scope still apply.

## Actual Tensor and memory proof

`Proofs/TensorBinding.lean` under the Phi task binds the supplied FP32 inputs and
output to shared `VeriTile.Torch.FP32.Tensor` objects and their ordinary API
laws. Length has a separate signed-int32 binding with the same four-byte storage
unit. Each named byte-memory allocation is tied to the actual Tensor storage
identity, base and capacity. Allocation names preserve storage identity, including
aliases. Checked contiguity and zero offset derive physical coordinates; no
payload packing, snapshot, temporary output or copyback is inserted.

`Proofs/NativeSafety.lean` derives active word bounds and alignment, exact formed
uint64 pointers, signed source-offset bounds, dynamic iteration capacity and
actual Tensor coordinates. Both public Tensor guarantees include these facts.
The native execution performs concrete little-endian four-byte stores to the
supplied output allocation and preserves every other caller byte. The Real
execution performs the complete typed source program and transports its output
view to that same allocation name; its frame includes all other storage cells.

`Correct.lean` and `FPEquiv.lean` are independent public entries for this task.
They expose the actual source variant, full checked-call Tensor result and exact
empty-call identity. Shared source/loop/grid/Real/native helpers live under
`flash_decode2_phi/Proofs`; neither public entry imports the other.
`Kernels.lean` is source-only and imports no task proof. All source/host/grid,
finite-address and memory obligations are proved. Ordinary Torch metadata,
allocation identity and same-device stream completion are explicit API
boundaries. Native compiler scalar/FMA interpretation remains the local
numerical boundary; this proof does not guarantee compilation resources or
assert a new universal runtime-compatibility result.

## Validation

Run `python bench/repairs/tritonbench_g/flash_decode2_llama/check_contract.py`.
The shared CPU harness executes this task's actual host and JIT AST. It passes
11 summary/tail/singleton/empty/alias cases and 30 rejection controls. Cases
cover int32 maximum length/block size, all retained width forms, incomplete
summary counts, inactive padding NaNs, output tail canaries, shared read-only
summaries, exact output mutation/None return, owner-context restoration and
empty-call pointer/content silence. Rejection controls cover shape, dtype,
device, layout, offset/view flags, argument types, invalid signed lengths,
alignment/span and overlapping storage. Input bytes and exterior output bytes
are preserved. This is CPU source validation, not GPU numerical execution.

Both public entries and the changed shared proof closure compile with
standard-base axiom audits and noncircular target checks. The JIT AST SHA-256,
excluding source locations, is `291edeb85857e096f069b1dbe4ac55ef65a9afb388825d3c20f40300d281da02`; it equals the historical accepted
repair AST and is checked by the harness.
