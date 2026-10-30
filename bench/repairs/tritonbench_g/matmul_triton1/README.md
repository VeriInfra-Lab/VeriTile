# matmul_triton1

The selected Python implementation remains
`revision001/matmul_triton1_repaired.py` (SHA256
`705bfb2cecafd2644cfe60922a0eebce2e110d0221343c7e87b273c14bbfe62a`).
The complete mask-hoisted candidate is described below. The original
benchmark and every revision001 file are preserved.

Start with `Kernels.lean`, `Contract.lean`, `Correct.lean`, `Misc/NativeCorrect.lean`, and
`Misc/Public.lean`. Correct and optional NativeCorrect use the same readable Triton
quotation. `Kernel.source_eq` proves that quotation equals the previously
checked selected source. Register renaming, integer addition order and moving
the pure output-pointer expression to the epilogue retain that source's graph.
The source's `z += tl.dot(..., allow_tf32=False)` uses its compiler-selected
live accumulator; this is explicitly bound to the default SM90 profile below.

The input domain is the existing compatible contiguous rank-two FP32 domain
on cuda:0 with resolved negative flags, FP32 default output dtype, signed-64
metadata and aligned nonempty pointers. Empty M/N, K=0, contiguous offsets,
overlapping/read-read input aliases, irrelevant singleton strides, NaNs and
infinities remain admitted. The ordinary output Tensor contract supplies its
actual shape, device, data pointer, storage offset, capacity, fresh identity
and separation from complete input backing allocations. A representable
successful grid is a launch boundary, not an added Python guard. The wrapper
connection uses only its existing admission and metadata facts; no new
allocator or host lifecycle proof is claimed.

In `VeriTile.Bench.Repairs.MatmulTriton1`:

- `matmul_triton1_real` derives the full independent mathematical sum
  `C[i,j] = sum(k<K, A[i,k]*B[k,j])` from actual logical input values under
  the Real arithmetic interpretation.
- `matmul_triton1_native` derives the exact native reference: ascending K
  lanes, all masked zero padding, and the live FP32 scalar FMA carry.
- `IONative.fp32_if_local` transports to the IEEE software reference
  using only the existing local zero and single-FMA correspondence laws.

The retained raw results derive successful finite64 and complete typed-source grid
execution, equality of their entire final memory, independent outputs and the
complete typed frame. The retained TensorBridge results add the physical post
and all actual input/output Tensor observations and frames. Native bridge
results include the actual Torch logical output, little-endian
FP32 bytes, every input backing-storage byte, and every exterior output byte.
The memory is a common aligned FP32-word reader with byte projections, not a
separate byte allocator. No output array, successful run, whole-dot equality
or completed-output condition is supplied as a premise.

`Proofs/Source.lean` composes the two executions. `Backing.lean` proves common
initial/final backing, initial image identity, store commutation and input
storage preservation. `Coverage.lean` proves all source stores are in bounds,
all output elements are covered, physical byte writers are unique, actual
finite data pointers match, and each complete source program realizes the
ordered physical stores. `Bytes.lean` supplies byte observations/frames;
`References.lean` connects the independent references; `Boundary.lean` proves
existing guard admission, empty-grid/post identity and K=0 values. Public
axiom and reference noncircularity audits pass.

The prior selection validation used the reviewed local-first-v2 scoped Lean
builder, with eleven new or updated task-module receipts. Its inherited
revision001/core sources and cached objects were pinned separately.
Two necessary offline compilations of the exact selected source use its
omitted/default options: SM90, 4 warps, 3 stages, fixed 32x32x32 tiles, i32 and
i64 metadata. Both emit eight live carry chains with 32 ascending-K
`fma.rn.f32` operations each, and no TF32/MMA/FTZ FMA opcode. These checks do
not execute a GPU or establish native numerical conformance. Historical
stages=2 compiler evidence remains historical and is not default=3 evidence.
The saved 27 rejection/16 CPU-interpreted cases are reused, not rerun.

Compiler lane/address/synchronization mapping, actual GPU instruction
conformance (including exceptional values/subnormals/NaN payloads) and ordinary
Torch storage/launch API laws remain explicit local boundaries. The Real
result does not assert floating-point equality to exact reals. Frozen evidence
is under `.veritile-cache/task-deliveries/matmul_triton1/authorized-resumption-v1`;
acceptance and integration belong to the independent review and root process.

`Correct.lean` and `Misc/NativeCorrect.lean` expose the shared `ProgramIO.Implements`
judgment. `IO.lean` names A[M,K], B[K,N] and C[M,N], their exact source strides,
the selected FP32 quotation and the complete row-major grid. There is no
private scratch or in-place output alias. Input/input aliases remain allowed.
Canonical strides, finite index/storage bounds and region separation remain
explicit. Real uses exactly K terms; native uses the original ordered padded
dot recurrence and explicit scalar instruction model.

`Proofs/IOSafetySteps.lean` and `IOSafety.lean` prove the actual masked loads,
dynamic loop and stores safe from declared IO windows using the existing source
invariants. `IOGridSafety` carries that safety through the complete launch
schedule. `IOResults` and `IOCorrect` reuse the complete finite/shared execution
proofs to obtain successful execution, all logical outputs and every exterior
cell. `KernelRealResults` and `KernelNativeResults` retain the old raw statements,
including equality of both executors' entire memories and both frames.

`Misc/TensorBridge.lean` retains all three former complete Tensor applications and
`TensorBridge.NativePost`, with unchanged Tensor/byte/input-allocation observations.
Both layers reuse the same `FiniteSource.grid_correct` and `Core.grid_correct`
proofs; the bridge's existing `Kernel.execute` supplies the parameter correspondence.
`IOTensorBinding` derives the new IO input binding from the existing concrete
Tensor binding and proves both new targets agree with the same Tensor formulas.
The conditional IEEE zero/FMA transport remains an ordinary helper theorem in
`IONative`; it is an existing primitive-model boundary, not a new transformation
assumption. This organization pass changes no Python, kernel source or contract,
and performs no new GPU or numerical experiment.


`FPEquiv.source_pair_equiv` compares the complete selected source with
`matmul_triton1_mask_hoisted.py`. The candidate computes `row_mask` and
`column_mask` before the K loop, replacing the repeated row/column comparisons
in both masked loads and reusing them in the final store. This removes two
vector comparisons from each K iteration. No measured speedup is claimed.
The source and candidate are complete ordinary `triton` quotations in
`Kernels.lean`; the earlier row-product candidate remains available.

The shared complete `ProgramIO` judgment retains exactly the canonical-stride,
finite-index, storage, geometry and output-separation premises. From the
original logical input binding and IO bounds, both full grids are safe and
succeed, every typed output cell agrees, and both exact exterior raw frames
hold. `Proofs/MaskExecution.lean` extends the existing source invariant with
the two cached masks and proves the actual complete candidate follows the
same masked-load and ordered-dot recurrence. `MaskSafety.lean` proves every
candidate access safe. `Pair.lean` lifts the memory equality across every
actual launch without equating private registers.

No numerical rewrite rule, new input restriction or experiment is required.
All uint64 casts, FP32 loads and live IEEE dot, padded K lanes, pointer updates,
validation, empty/K=0 branches, complete grid and wrapper are preserved. The
independent `Correct.lean` and original complete native/Tensor applications
remain unchanged. `Pair.implements_transfer` carries any independently proved
original logical formula, safety and exact frame to the candidate.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/Public.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。
