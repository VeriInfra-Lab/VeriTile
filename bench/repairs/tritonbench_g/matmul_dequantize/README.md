# `matmul_dequantize`: finite kernel IO proof entries

The canonical source remains `matmul_dequantize_repaired.py`, SHA256
`8aead7bf90ca37c06d10382702dd47072db7614a55b33e0940c2a33da1f64283`, as selected
by `bench/tritonbench_g/task_progress.json` → `contract_revision`. This revision
changes no Python, guard, dtype, admitted layout, allocator or launch behavior.
The historical repair is still precisely the four s2 K-tail load masks; the
GPTQ and standalone dequantization source bodies are unchanged from the benchmark.

## Target, precision and proof scope

Decode Q and Z as eight four-bit fields in each signed INT32 storage word. The
independent mathematical targets are

```
D[k,n] = (q[k,n] - z[k/group,n]) * scale[k/group,n]
C[m,n] = sum_{k=0}^{K-1} A[m,k] * D[k,n]
```

`Correct.lean` proves these formulas in Real semantics. `Misc/NativeCorrect.lean` proves
successful full kernel execution and equality to an independent source-order
word/instruction reference and the complete typed memory frame. The ordinary
TensorBridge applications additionally prove every output Tensor byte and
the physical exterior frame. It uses one explicit ISA on both sides. Neither theorem claims FP=Real.
No additional implementation or artificial source rewrite is introduced.

| Kernel entry prefix | Selected source and precision | Coverage |
| --- | --- | --- |
| `gptq_kernel_*` | FP16 A/scales/output, FP32 dot accumulation | Both BM64/BN256 and BM128/BN128 configurations, both group branches |
| `s2_kernel_*` | FP32 A/scales; FP16 dot operands/output; FP32 accumulator | Literal SPLIT_K=1, four K-tail masks, wrapped M/N loads and output tail mask |
| `dequant16_kernel_*` | FP16 scales/output; packed signed INT32 Q/Z | All K/N masked tails; also the Triton kernel used by s1 |
| `dequant32_kernel_*` | FP32 scales/output; packed signed INT32 Q/Z | All K/N masked tails |

The suffix is `correct` in `Correct.lean` and `equiv` in `Misc/NativeCorrect.lean`.
All entries accept an arbitrary permutation of the complete actual grid.
The FP16 dequantization Tensor bridge explicitly permits `.dequant16` and `.s1`; the
subsequent `torch.mm` in s1 remains an external library consumer. The public
results contain no wrapper invocation, returned-object behavior or Torch
implementation proof obligation. They assume neither successful execution nor
an expected output.

## Shared source and reused proof chain

`Kernels.lean` exposes precision-selected aliases of the single readable
quotations in `Whole001/Kernels.lean` and its original source imports.
`Proofs/KernelSources.lean` proves exact source identities for all four routes.
There is no second handwritten AST. Both Real and native grid proofs execute
these same quotations.

The existing `Whole001/Proofs/` source, loop, grid, mathematical-result, signed
index and instruction-graph proofs are reused unchanged. The new
`Misc/KernelResult.lean` connects them directly to actual Tensor coordinates,
whole-memory serialization, every output byte, coverage and typed/byte frames.
`Proofs/KernelTensor.lean` connects the original byte words to the ordinary
shared `VeriTile.Torch.FP32.Tensor.logicalValue`, including resolved negative
view flags. FP16 and packed INT32 retain their existing typed storage bindings.
No new byteheap interpreter or allocator model is introduced.

The prior wrapper/consumer entry sources are preserved byte-for-byte under
`Misc/LegacyCorrect.lean` and `Misc/LegacyFPEquiv.lean`. The old README and
public source snapshots are pinned in the delivery evidence. These preserved
results are historical reuse, not an additional acceptance requirement.

## Input contract and memory correspondence

The existing checked domain is retained: resolved rank-two strided CUDA
Tensors on one device, positive dimensions, canonical contiguous strides,
nonzero storage offsets permitted, checked storage capacity/pointer origin,
alignment and uint64 pointer spans, and separate complete live allocations.
`group_size` is a positive Python integer, excluding bool. K/N are multiples of
8 and group divides K; GPTQ additionally requires K/group multiples of 32 and
N divisible by 256. s2 and standalone dequantization keep their masked tails.
All floating words remain admitted; no finite or nonzero payload guard is added.

`Contract.lean` retains the executable metadata, exact source-expression,
active-allocation and formed-pointer checks, with `checkArguments_iff`.
`ABI.SourceExact` remains in the ordinary TensorBridge conclusions. It covers the actual
nonnegative signed-int32 expressions, including inactive lanes and final
pointer increments. Ordinary Tensor bindings connect metadata, actual data
pointer, allocation identity/base/capacity, storage offset and physical index.
Only live route Tensors are bound; unused slots in the old six-descriptor model
are ghost metadata and require no runtime Tensor or allocation.

The reused byte bridge loads original little-endian allocation bytes at the
actual storage offsets. Its serializer takes executed state values at source
store addresses. The independent expected memory takes reference expressions
constructed from the initial inputs. The public native result equates them,
then reads each byte through the output Tensor's own physical index. Output
prefix/suffix canaries and all memory outside the output stores are preserved.

## Primitive and compiler boundaries

`Native.WordISA` supplies scalar literal/conversion/arithmetic semantics and
one output lane of a fixed SM80 `m16n8k16` instruction. `Native.lower` composes
these leaves: FP16 multiply/FMA for GPTQ dequantization, FP32 multiply and half
casts for s2, the requested standalone multiply precision, FP32 live dot
accumulators, and FP16 stores. `Native.Plan` records compiler operand routing
and multiply orientation. Full source execution, K-block composition, output
coordinates, stores and frames are proved above these leaves.

These primitive/cast/FTZ/special-value and compiler-realization contracts are
model boundaries. They are **not numerical experiments**, and the new public
entries no longer require the legacy `Numerics actual target` equality port.
There is no numerical rewrite assumption or deferred experiment in this
revision. Universal compiler refinement and GPU execution are not claimed.
The historical comments referring to deferred numerical interpretation remain
inside the preserved old modules and do not classify this revision.

Model references:

- [Triton scalar/tensor type promotion](https://triton-lang.org/main/python-api/triton-semantics.html#type-promotion).
- [Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html),
  [store](https://triton-lang.org/main/python-api/generated/triton.language.store.html),
  and [dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).
- NVIDIA PTX [load/store alignment](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands),
  [conversion](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt),
  [multiply](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul),
  [half FMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-fma),
  and [matrix multiply-accumulate](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).
- Tensor API references for pointers, storage offsets, contiguous coordinates
  and lazy negation are cited in `VeriTile/Torch/FP32.lean`.

## Targeted validation

Evidence is pinned under `.veritile-cache/kernel-evidence/matmul_dequantize/`.
The unchanged canonical Python and CPU checker permit reuse of the root's
receipt `/tmp/veritile-dequant-control-vshmst9j/evidence/cpu-controls.json`:
11 actual Triton-interpreter launches, zero unsafe accesses, five additional
guard admissions and 26 rejections, source AST comparison, actual caller-byte
frames and offset canaries. This uses an identified CUDA-metadata/CPU-allocation
shim and CPU Torch.mm; it is not GPU or native numerical evidence.

`probe_actual_binder.py` intercepts actual canonical wrapper launches, passes
their real argument lists/configurations through the installed Triton 3.7.1
binder and `_pack_args`, and compiles for the explicit offline SM80 target.
All ten samples pass, covering both GPTQ configurations/group branches, s2
M=1/group=1 and K40/N136 tails, both dequantization precisions/group=1, and
nonzero pointer offsets. The probe preserves implicit scalar=1 and pointer
alignment specialization. It does not substitute a hand-picked ASTSource
signature. CPU allocation views provide metadata/pointers; no GPU driver
launch is represented.

Saved TTIR/TTGIR/LLVM/PTX artifacts confirm the sampled operation profiles:
FP16 FMA, FP32 mul/cvt, requested dequantization precision and live-seed
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. This is bounded profile
inspection, not a full dataflow reconstruction or universal compiler proof.
Independent replay and GPU testing remain separately reported by the reviewer.
Triton runtime grid-product compatibility remains deferred to the shared
follow-up after the main benchmark work.

`Misc/TensorBridge.lean` contains the ordinary `TensorBridge.python_checks_iff_preconditions`,
`TensorBridge.actual_fp32_logical_read`, and `TensorBridge.actual_tensor_byte_read`
theorems. The four Real specifications and four ordinary native kernel theorems
keep their primary names in `Correct.lean` and `Misc/NativeCorrect.lean`; neither entry imports the bridge.

The main entries use shared `ProgramIO.Implements` (Real) and
`ProgramIO.Observes` (native). No main statement takes a raw `State` or exposes
`run`, memory cells, or a post-state. `IO.lean` describes standalone dequant16
and dequant32; `FirstIO.lean` and `SecondIO.lean` describe the two matrix kernels.
Their finite typed channels expose source addresses, masks, storage dtypes,
actual PID installation, arbitrary complete-grid permutations, and no scratch.
Real targets are the independent matrix/dequantization formulas above. Native
targets are the independent ordered expression recurrences over the same
logical inputs, evaluated by the original `Native.Plan` and `Native.WordISA`.
All symbolic `Native.Expr` inputs are admitted, including non-leaf expressions;
there is no new restriction to finite floats or literal word leaves.

The public native GPTQ theorem retains arbitrary original block and group
parameters. Its finite windows include A's row-only mask, unmasked packed
weights, and unmasked scale/zero loads. In noGroups mode, scale/zero preloads
remain backed even when the K loop is empty. Pointer increments retain their
original integer quotient order. The stricter blockInner=32, exact K coverage
and group assumptions belong only to the existing Real matrix-product theorem.
S2 likewise preserves arbitrary native steps/blockInner, original modulo row
and column addressing, all four K masks and all original casts. Its Real
theorem retains blockInner=32 and the original ceil(K/32) iteration count.
Input regions may alias exactly as before; simultaneous input bindings enforce
consistent values at repeated addresses. Output/input separation is unchanged.

`Proofs/FirstSafety.lean`, `SecondSafety.lean`, `Dequant32Safety.lean` and
`Dequant16Safety.lean` prove access bounds on the actual quoted sources and
actual loop successor states. The corresponding input/result helpers connect
finite operands to the existing complete-grid execution foundations, prove
independent formula equality, and retain exact exterior cell frames.
`inputsOfState_binding` for each route explicitly supplies finite inputs for
every original arbitrary state. Safety is a proved conclusion, never a
numerical assumption or an extra source-success premise.

The exact former raw statements and proofs are ordinary theorems in
`Misc/KernelRealResult.lean` and `Misc/KernelNativeResult.lean`, relocated by
namespace only apart from replacing the `specification` keyword. All complete
TensorBridge statements and bodies are unchanged; only its two imports point
to those helpers. Both levels reuse exactly the original FirstGrid, SecondGrid,
DequantGrid16 and DequantGrid32 foundations and the existing byte/Tensor
correspondence. `Misc/Review.lean` owns combined entry/Tensor audits. Neither main
entry imports TensorBridge, and supporting proofs do not import either main.

Current targeted build receipts and the public-format report are recorded in
`.veritile-cache/public-io-format-20261020/batch04-report.json`. Existing source,
contract and historical evidence pins above remain unchanged. This format
revision performs no GPU experiments or independent cold replay.

## Implementation pairs (2026-10-21)

`FPEquiv.lean` now exposes four actual `original ≡[[]] candidate` results:
`gptq_scalar_address_equiv`, `s2_scalar_address_equiv`,
`dequant32_scalar_address_equiv` and `dequant16_scalar_address_equiv`.
Each compares complete ordinary quotations from `Kernels.lean` through
`CandidateIO.lean`, on the same public IO and arbitrary complete-grid permutation.

For GPTQ and S2, the candidate precomputes
`GROUP_SIZE_M * ceil(N / BLOCK_SIZE_N)` from source parameters instead of
multiplying the previously computed runtime `num_pid_n`. The complete Python
candidate `matmul_dequantize_scalar_address.py` specializes `N` with
`tl.constexpr` and declares this group extent `tl.constexpr`. Every original
shape remains accepted; compilation may specialize by N. This removes one
runtime scalar multiplication from grouped PID mapping. No measured hardware
speedup is claimed.

For both standalone dequantization precisions, output address formation changes
from `(pid_k*BK+i)*stride_k + (pid_n*BN+j)*stride_n` to
`pid_k*(BK*stride_k) + pid_n*(BN*stride_n) + i*stride_k + j*stride_n`.
The block origins are computed scalarly, reducing vector arithmetic while
retaining every original source/output stride. No contiguity premise is added.
All source loads, signed words, casts, FP16/FP32 arithmetic, FP32 dot accumulation,
tail masks, NO_GROUPS branches, loop bounds, output dtypes and autotunes remain.

`Proofs/ScalarAddress.lean` proves exact full-state equality after each changed
pure prefix, equality of the complete unchanged remainder, and source trace
safety equivalence. These results lift through every actual invocation and the
full arbitrary-order grid, with all outputs and exterior typed-cell frames.
The public pair theorems hold for arbitrary typed scalar/dot backends, without
floating laws, new input restrictions or numerical experiments. Their premises
are the existing geometry, output/input separation and grid permutation only.
The independent four Real formulas in `Correct.lean` remain unchanged, as do the
ordinary native observations in `Misc/NativeCorrect.lean`.

Optional proof applications and reader audits are now under `Misc/`. Their
namespaces, full theorem statements, proof bodies and prior acceptance status
are preserved. `Correct.lean`, `FPEquiv.lean` and their required proof helpers
do not import these optional entries. This directory move adds no correctness
claim or numerical validation.
